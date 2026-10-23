
import asyncio
import json
import logging
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import edge_tts
import httpx
from edge_tts import SubMaker
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

APP_VERSION = "3.0.0"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
).rstrip("/")
GROQ_CHAT_MODEL = os.getenv(
    "GROQ_CHAT_MODEL", "openai/gpt-oss-120b"
)
GROQ_WHISPER_MODEL = os.getenv(
    "GROQ_WHISPER_MODEL", "whisper-large-v3"
)

WORK = Path(os.getenv("TZ_RECAP_WORK_DIR", "jobs"))
WORK.mkdir(parents=True, exist_ok=True)
STATE_FILE = WORK / "jobs_state.json"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tz-recap")

app = FastAPI(
    title="TZ Recap API",
    version=APP_VERSION,
    description=(
        "Video transcription, Burmese translation, voice-over, "
        "SRT subtitles and MP4 generation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICES = {
    "thiha": "my-MM-ThihaNeural",
    "သီဟ": "my-MM-ThihaNeural",
    "thiha · male (burmese)": "my-MM-ThihaNeural",
    "nilar": "my-MM-NilarNeural",
    "နီလာ": "my-MM-NilarNeural",
    "nilar · female (burmese)": "my-MM-NilarNeural",
}

DEFAULT_VOICE = "my-MM-ThihaNeural"

STYLE_GUIDANCE = {
    "fast recap": (
        "Use energetic movie-recap narration and clear transitions. "
        "Never invent plot events."
    ),
    "cinematic": (
        "Use cinematic storytelling and suspenseful transitions "
        "without inventing events."
    ),
    "natural": (
        "Use simple, conversational Burmese that sounds natural when spoken."
    ),
    "horror": (
        "Use suspenseful horror narration without adding events "
        "absent from the source."
    ),
    "neutral": (
        "Use clear, neutral Burmese suitable for subtitles and voice-over."
    ),
}

JOB_LOCK = threading.Lock()
JOBS: dict[str, dict[str, Any]] = {}
ACTIVE_STATUSES = {
    "queued", "transcribing", "translating", "voice", "rendering"
}


def save_state():
    try:
        with JOB_LOCK:
            snapshot = json.dumps(
                JOBS, ensure_ascii=False, indent=2
            )
        temporary = STATE_FILE.with_suffix(".tmp")
        temporary.write_text(snapshot, encoding="utf-8")
        temporary.replace(STATE_FILE)
    except Exception:
        logger.exception("Unable to save job state")


def load_state():
    if not STATE_FILE.exists():
        return

    try:
        saved = json.loads(
            STATE_FILE.read_text(encoding="utf-8")
        )
        if isinstance(saved, dict):
            for job_id, job in saved.items():
                if not isinstance(job, dict):
                    continue

                if job.get("status") in ACTIVE_STATUSES:
                    job["status"] = "error"
                    job["error"] = (
                        "Server restarted during this job. "
                        "Please submit the video again."
                    )

                JOBS[job_id] = job
    except Exception:
        logger.exception("Unable to load saved jobs")


load_state()


def set_job(job_id: str, **updates):
    with JOB_LOCK:
        JOBS.setdefault(job_id, {}).update(updates)
    save_state()


def job_snapshot(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else {}


def run_command(command, timeout=900):
    try:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Required program not installed: {command[0]}. "
            "Check that ffmpeg is installed."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Processing timed out. Try a shorter video."
        ) from exc


def require_api_key():
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it in Render Environment "
            "and redeploy."
        )


def clean_text(text):
    text = (text or "").replace("\ufeff", "").replace("\x00", "")
    return "\n".join(
        line.strip() for line in text.splitlines()
    ).strip()


def normalize_mode(value):
    mode = (
        (value or "translate")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if mode in {
        "recap", "movie_recap", "movierecap", "summary"
    }:
        return "recap"

    if mode in {
        "voiceover", "voice_over", "rewrite"
    }:
        return "voiceover"

    return "translate"


def normalize_source_language(value):
    value = (value or "zh").strip().lower()

    if value in {"", "auto", "automatic", "detect"}:
        return None

    allowed = {
        "zh", "en", "ja", "ko", "th", "vi", "id",
        "ms", "hi", "fr", "de", "es", "pt", "ru",
        "ar", "my",
    }

    return value if value in allowed else "zh"


async def transcribe(video_path, folder, source_language="zh"):
    require_api_key()

    audio_path = folder / "source_audio.mp3"

    result = run_command([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-b:a", "64k", str(audio_path),
    ])

    if result.returncode != 0 or not audio_path.exists():
        raise RuntimeError(
            f"Could not extract audio: {(result.stderr or '')[-1600:]}"
        )

    form_data = {
        "model": GROQ_WHISPER_MODEL,
        "response_format": "json",
    }

    language_code = normalize_source_language(source_language)
    if language_code:
        form_data["language"] = language_code

    async with httpx.AsyncClient(timeout=300) as client:
        with audio_path.open("rb") as audio:
            response = await client.post(
                f"{GROQ_BASE_URL}/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}"
                },
                data=form_data,
                files={
                    "file": (
                        audio_path.name,
                        audio,
                        "audio/mpeg",
                    )
                },
            )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Speech recognition failed "
            f"({response.status_code}): {response.text[:1200]}"
        )

    transcript = clean_text(response.json().get("text", ""))

    if not transcript:
        raise RuntimeError("No speech was detected in the video.")

    return transcript


async def write_script(transcript, options):
    require_api_key()

    mode = normalize_mode(
        str(options.get("mode", "translate"))
    )
    target_language = str(
        options.get("lang", "Burmese")
    ).strip() or "Burmese"

    style = str(
        options.get("style", "Fast Recap")
    ).strip().lower()

    style_hint = STYLE_GUIDANCE.get(
        style, STYLE_GUIDANCE["fast recap"]
    )

    if target_language.lower() in {
        "burmese", "myanmar", "my", "မြန်မာ"
    }:
        language_rule = (
            "Output natural, fluent Myanmar Burmese in Unicode only. "
            "Do not use romanized Burmese."
        )
    else:
        language_rule = f"Output in {target_language}."

    if mode == "translate":
        task = (
            "Translate the complete source transcript into natural, "
            "easy-to-understand Burmese. Preserve the original meaning, "
            "important details, speaker identity and event order. "
            "Repair fragmented speech into sensible sentences. "
            "Use appropriate Myanmar honorifics for royal or historical "
            "dialogue. Do not summarize, invent details, explain, or add "
            "a heading. Return only the translation."
        )
    elif mode == "voiceover":
        task = (
            "Rewrite the transcript as smooth spoken narration while "
            "preserving its meaning and events. Do not add facts, scenes, "
            "motives, or dialogue not supported by the source. "
            "Return only the finished voice-over script."
        )
    else:
        task = (
            "Write an engaging movie recap based only on the source "
            "transcript. Keep important plot events in order and use "
            "clear transitions. Do not invent missing scenes or facts. "
            "Return only the narration."
        )

    prompt = f"""
You are an experienced Chinese-to-Myanmar translator
and movie voice-over writer.

{language_rule}

Rules:
- Understand the source meaning; do not translate mechanically word by word.
- Preserve who is speaking and who is being addressed.
- For Chinese royal or historical dialogue, use natural Burmese forms
  such as "အရှင်မင်းကြီး" only when the context supports them.
- Use fluent Burmese grammar and natural punctuation.
- Do not output English explanations or a "Translation:" heading.
- Do not repeat the source transcript.
- Never invent details to fill gaps.
- Style guidance: {style_hint}

Task:
{task}

Source transcript:
{transcript[:24000]}
""".strip()

    payload = {
        "model": GROQ_CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise translator. Meaning and natural "
                    "target-language expression come first. Never invent "
                    "details. Return only the requested output."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens": 6000,
    }

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Translation failed ({response.status_code}): "
            f"{response.text[:1200]}"
        )

    choices = response.json().get("choices", [])
    if not choices:
        raise RuntimeError("AI returned an empty script.")

    script = clean_text(
        choices[0].get("message", {}).get("content", "")
    )

    if not script:
        raise RuntimeError("AI returned an empty script.")

    return script


def make_fallback_srt(script):
    chunks = [
        line.strip() for line in script.splitlines()
        if line.strip()
    ]

    if not chunks and script.strip():
        chunks = [script.strip()]

    def stamp(ms):
        hours, remainder = divmod(ms, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        seconds, millis = divmod(remainder, 1000)
        return (
            f"{hours:02}:{minutes:02}:"
            f"{seconds:02},{millis:03}"
        )

    result = []
    cursor = 0

    for index, chunk in enumerate(chunks, 1):
        duration = max(2200, min(7000, len(chunk) * 85))
        result.extend([
            str(index),
            f"{stamp(cursor)} --> {stamp(cursor + duration)}",
            chunk,
            "",
        ])
        cursor += duration

    return "\n".join(result)


async def create_voice(script, folder, voice_name):
    key = voice_name.strip().lower()
    voice = VOICES.get(
        key, VOICES.get(voice_name.strip(), DEFAULT_VOICE)
    )

    audio_path = folder / "voice.mp3"
    srt_path = folder / "subtitles.srt"

    maker = SubMaker()
    tts = edge_tts.Communicate(script, voice)

    with audio_path.open("wb") as output:
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                output.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                try:
                    maker.create_sub(
                        (chunk["offset"], chunk["duration"]),
                        chunk["text"],
                    )
                except Exception:
                    pass

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError("TTS did not return any audio.")

    try:
        subtitles = maker.generate_subs()
    except Exception:
        subtitles = ""

    if not subtitles.strip():
        subtitles = make_fallback_srt(script)

    srt_path.write_text(subtitles, encoding="utf-8")
    return audio_path, srt_path


def probe_duration(path):
    result = run_command([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ], timeout=60)

    try:
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def render_video(video, audio, output, aspect):
    filters = {
        "9:16": (
            "scale=720:1280:force_original_aspect_ratio=increase,"
            "crop=720:1280"
        ),
        "16:9": (
            "scale=1280:720:force_original_aspect_ratio=increase,"
            "crop=1280:720"
        ),
        "1:1": (
            "scale=1080:1080:force_original_aspect_ratio=increase,"
            "crop=1080:1080"
        ),
    }

    vf = filters.get(aspect, filters["9:16"])
    source_duration = probe_duration(video)
    audio_duration = probe_duration(audio)

    speed = 1.0
    if source_duration > 0 and audio_duration > 0:
        speed = max(
            0.5, min(2.0, source_duration / audio_duration)
        )

    setpts = f"setpts={1.0 / speed:.6f}*PTS"

    result = run_command([
        "ffmpeg", "-y",
        "-i", str(video),
        "-i", str(audio),
        "-vf", f"{vf},{setpts}",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "160k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-movflags", "+faststart",
        str(output),
    ], timeout=1800)

    if (
        result.returncode != 0
        or not output.exists()
        or output.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Video rendering failed: {(result.stderr or '')[-2200:]}"
        )


async def process_job(job_id):
    job = job_snapshot(job_id)
    if not job:
        return

    folder = Path(job["folder"])
    video = folder / "in.mp4"
    options = job.get("options", {})

    try:
        set_job(
            job_id,
            status="transcribing",
            progress=10,
            message="ဗီဒီယိုအသံကို စာသားပြောင်းနေသည်",
        )

        transcript = await transcribe(
            video,
            folder,
            str(options.get("source_language", "zh")),
        )

        (folder / "transcript.txt").write_text(
            transcript, encoding="utf-8"
        )

        set_job(
            job_id,
            transcript=transcript,
            status="translating",
            progress=35,
            message="မြန်မာဘာသာပြန်နေသည်",
        )

        script = await write_script(transcript, options)

        (folder / "script.txt").write_text(
            script, encoding="utf-8"
        )

        set_job(
            job_id,
            script=script,
            status="voice",
            progress=55,
            message="မြန်မာ AI အသံနှင့် SRT ဖန်တီးနေသည်",
        )

        audio, srt = await create_voice(
            script, folder, str(options.get("voice", "သီဟ"))
        )

        set_job(
            job_id,
            status="rendering",
            progress=75,
            message="MP4 ဗီဒီယို တည်ဆောက်နေသည်",
        )

        output = folder / "result.mp4"

        render_video(
            video, audio, output,
            str(options.get("aspect", "9:16")),
        )

        set_job(
            job_id,
            status="done",
            progress=100,
            message="ပြီးပါပြီ",
            result_url=f"/api/jobs/{job_id}/result",
            script_url=f"/api/jobs/{job_id}/script",
            srt_url=f"/api/jobs/{job_id}/srt",
            transcript_url=f"/api/jobs/{job_id}/transcript",
            audio_url=f"/api/jobs/{job_id}/audio",
            created_video_size=output.stat().st_size,
        )

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        set_job(
            job_id,
            status="error",
            progress=100,
            message="လုပ်ဆောင်မှု မအောင်မြင်ပါ",
            error=str(exc),
        )


@app.get("/")
async def root():
    return {
        "name": "TZ Recap API",
        "version": APP_VERSION,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    ffmpeg_ok = subprocess.call([
        "sh", "-c", "command -v ffmpeg >/dev/null 2>&1"
    ]) == 0

    return {
        "status": "ok",
        "version": APP_VERSION,
        "groq_configured": bool(GROQ_API_KEY),
        "ffmpeg_available": ffmpeg_ok,
    }


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    settings: str = Form("{}"),
):
    allowed_extensions = {
        ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"
    }

    filename = (video.filename or "upload.mp4").lower()

    if Path(filename).suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: MP4, MOV, MKV, WEBM, M4V, AVI.",
        )

    try:
        data = json.loads(settings or "{}")
        if not isinstance(data, dict):
            raise ValueError()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="settings must be a valid JSON object.",
        )

    job_id = uuid.uuid4().hex[:12]
    folder = WORK / job_id
    folder.mkdir(parents=True, exist_ok=True)
    video_path = folder / "in.mp4"
    total = 0

    try:
        with video_path.open("wb") as target:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break

                total += len(chunk)

                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Maximum upload size is 100 MB.",
                    )

                target.write(chunk)
    finally:
        await video.close()

    if total == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty.",
        )

    # Default mode is translation, not recap.
    options = {
        "mode": normalize_mode(
            str(data.get("mode", "translate"))
        ),
        "voice": str(data.get("voice", "သီဟ")),
        "style": str(data.get("style", "Fast Recap")),
        "aspect": str(
            data.get("aspect_ratio", data.get("aspect", "9:16"))
        ),
        "lang": str(
            data.get("language", data.get("lang", "Burmese"))
        ),
        "source_language": str(
            data.get("source_language", "zh")
        ),
    }

    with JOB_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "တန်းစီထားသည်",
            "folder": str(folder),
            "filename": filename,
            "options": options,
        }

    save_state()
    background_tasks.add_task(process_job, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/jobs/{job_id}",
        "script_url": f"/api/jobs/{job_id}/script",
        "srt_url": f"/api/jobs/{job_id}/srt",
        "result_url": f"/api/jobs/{job_id}/result",
        "transcript_url": f"/api/jobs/{job_id}/transcript",
        "mode": options["mode"],
    }


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_snapshot(job_id)

    if not job:
        raise HTTPException(
            status_code=404, detail="Job not found."
        )

    return {
        key: value
        for key, value in job.items()
        if key not in {
            "folder", "options", "transcript", "script"
        }
    }


def artifact(job_id, filename, media_type, download_name):
    job = job_snapshot(job_id)

    if not job:
        raise HTTPException(
            status_code=404, detail="Job not found."
        )

    path = Path(job["folder"]) / filename

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} is not ready yet.",
        )

    return FileResponse(
        path,
        media_type=media_type,
        filename=download_name,
    )


@app.get("/api/jobs/{job_id}/script")
async def get_script(job_id: str):
    return artifact(
        job_id, "script.txt",
        "text/plain; charset=utf-8",
        "myanmar_script.txt",
    )


@app.get("/api/jobs/{job_id}/transcript")
async def get_transcript(job_id: str):
    return artifact(
        job_id, "transcript.txt",
        "text/plain; charset=utf-8",
        "source_transcript.txt",
    )


@app.get("/api/jobs/{job_id}/srt")
async def get_srt(job_id: str):
    return artifact(
        job_id, "subtitles.srt",
        "application/x-subrip; charset=utf-8",
        "myanmar_subtitles.srt",
    )


@app.get("/api/jobs/{job_id}/result")
async def get_result(job_id: str):
    return artifact(
        job_id, "result.mp4",
        "video/mp4",
        "tz_recap_result.mp4",
    )


@app.get("/api/jobs/{job_id}/audio")
async def get_audio(job_id: str):
    return artifact(
        job_id, "voice.mp3",
        "audio/mpeg",
        "myanmar_voice.mp3",
    )
