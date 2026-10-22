import asyncio
import json
import logging
import os
import subprocess
import threading
import uuid
from pathlib import Path

import edge_tts
import httpx
from fastapi import (
    BackgroundTasks, FastAPI, File, Form,
    HTTPException, UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="TZ Recap API", version="2.2.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tz-recap")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK = Path("jobs")
WORK.mkdir(parents=True, exist_ok=True)
STATE_FILE = WORK / "job_state.json"

JOBS = {}
JOB_LOCK = threading.RLock()

MAX_UPLOAD = 100 * 1024 * 1024

VOICES = {
    "Thiha · Male (Burmese)": "my-MM-ThihaNeural",
    "Nilar · Female (Burmese)": "my-MM-NilarNeural",
    "သီဟ": "my-MM-ThihaNeural",
    "နီလာ": "my-MM-NilarNeural",
}

STYLES = {
    "Fast Recap": "ဇာတ်လမ်းကို မြန်မြန်ဆန်ဆန်၊ စာကြောင်းတိုတိုဖြင့် ပြောပါ။",
    "Story Mode": "ဇာတ်လမ်းပြောပြသလို သဘာဝကျပြီး ဆွဲဆောင်မှုရှိအောင် ရေးပါ။",
    "Dramatic": "စိတ်လှုပ်ရှားဖွယ် ရုပ်ရှင် recap ပုံစံဖြင့် ရေးပါ။",
    "Summary": "အဓိကအချက်များကို ရှင်းလင်းစွာ အကျဉ်းချုပ်ပါ။",
}

ACTIVE_STATUSES = {
    "queued", "transcribing", "writing_script",
    "generating_voice", "rendering",
}


# ---------------- JOB STORAGE ----------------

def save_jobs_unlocked():
    temp = STATE_FILE.with_suffix(".tmp")
    temp.write_text(
        json.dumps(JOBS, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(STATE_FILE)


def update_job(job_id, **changes):
    with JOB_LOCK:
        JOBS.setdefault(job_id, {})
        JOBS[job_id].update(changes)
        save_jobs_unlocked()


def load_jobs():
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            JOBS.update(data)
    except (OSError, ValueError):
        logger.exception("Could not load job state")


load_jobs()

for job in JOBS.values():
    if job.get("status") in ACTIVE_STATUSES:
        job.update(
            status="error",
            progress=0,
            error="Server restarted during processing. Please upload again.",
        )

with JOB_LOCK:
    try:
        save_jobs_unlocked()
    except OSError:
        logger.exception("Could not save job state")


# ---------------- COMMAND HELPERS ----------------

def run(cmd, timeout=900):
    logger.info("Running command: %s", cmd[0])

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        details = result.stderr or result.stdout or "Command failed"
        raise RuntimeError(details[-2000:])

    return result


def duration(path):
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ], timeout=60)

    try:
        value = float(result.stdout.strip())
    except (ValueError, TypeError):
        raise RuntimeError("Could not read media duration")

    if value <= 0:
        raise RuntimeError("Invalid media duration")

    return value


# ---------------- GROQ TRANSCRIPTION ----------------

def transcribe(video, folder):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Render Environment တွင် GROQ_API_KEY မရှိပါ။")

    audio = folder / "audio_small.mp3"

    run([
        "ffmpeg", "-y",
        "-threads", "1",
        "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-b:a", "32k",
        str(audio),
    ], timeout=900)

    with open(audio, "rb") as f:
        response = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            data={"model": "whisper-large-v3"},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            timeout=300,
        )

    if response.is_error:
        raise RuntimeError(
            f"Groq transcription error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    transcript = response.json().get("text", "").strip()
    if not transcript:
        raise RuntimeError("ဗီဒီယိုထဲမှ အသံစာသား မရရှိပါ။")

    return transcript


# ---------------- TRANSLATION / RECAP ----------------

def write_script(transcript, opts):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY မသတ်မှတ်ထားပါ။")

    lang = opts["lang"]

    if opts["mode"].lower() == "translate":
        task = (
            f"Translate this transcript into {lang}. "
            "Preserve meaning, names, and event order. "
            "For Burmese, use natural Burmese Unicode. "
            "Return only the translation."
        )
    else:
        task = (
            f"Write a compelling movie recap in {lang}. "
            f"{STYLES.get(opts['style'], STYLES['Fast Recap'])} "
            "Preserve the original event order. "
            "Use natural Burmese Unicode when appropriate. "
            "Return narration only."
        )

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-oss-120b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert movie translator and recap writer. "
                        "Preserve meaning. Do not invent events."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{task}\nTarget language: {lang}\n\n"
                        f"Transcript:\n{transcript[:20000]}"
                    ),
                },
            ],
            "temperature": 0.4,
            "max_tokens": 6000,
        },
        timeout=180,
    )

    if response.is_error:
        raise RuntimeError(
            f"Groq writing error {response.status_code}: "
            f"{response.text[:1000]}"
        )

    script = (
        response.json()
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not script:
        raise RuntimeError("AI မှ စာသားမရရှိပါ။")

    return script


# ---------------- VOICE + SRT ----------------


async def create_voice(script, voice, audio_path, srt_path):
    if not script.strip():
        raise RuntimeError("Narration စာသားအလွတ်ဖြစ်နေပါတယ်။")

    last_error = None

    for attempt in range(3):
        try:
            if audio_path.exists():
                audio_path.unlink()

            communicate = edge_tts.Communicate(
                script.strip(), voice
            )
            submaker = edge_tts.SubMaker()
            audio_size = 0

            with open(audio_path, "wb") as out:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        out.write(chunk["data"])
                        audio_size += len(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        submaker.feed(chunk)

            if audio_size == 0:
                raise RuntimeError("TTS အသံဒေတာ မရရှိပါ။")

            # Use Edge TTS subtitles when available.
            srt_text = submaker.get_srt().strip()

            # Fallback: generate subtitles if Edge TTS
            # did not return any subtitle cues.
            if not srt_text:
                import re

                parts = re.split(
                    r"(?<=[။.!?])\s*",
                    script.strip()
                )
                parts = [p.strip() for p in parts if p.strip()]

                # Split long sentences into readable pieces.
                segments = []
                for part in parts:
                    while len(part) > 100:
                        cut = part.rfind(" ", 0, 100)
                        if cut < 30:
                            cut = 100
                        segments.append(part[:cut].strip())
                        part = part[cut:].strip()
                    if part:
                        segments.append(part)

                if not segments:
                    segments = [script.strip()]

                audio_seconds = duration(audio_path)
                weights = [max(1, len(x)) for x in segments]
                total_weight = sum(weights)
                elapsed = 0.0
                entries = []

                def stamp(seconds):
                    ms = max(0, int(seconds * 1000))
                    hours = ms // 3600000
                    minutes = (ms % 3600000) // 60000
                    secs = (ms % 60000) // 1000
                    millis = ms % 1000
                    return (
                        f"{hours:02}:{minutes:02}:"
                        f"{secs:02},{millis:03}"
                    )

                for index, (text, weight) in enumerate(
                    zip(segments, weights), start=1
                ):
                    start = elapsed
                    elapsed += audio_seconds * weight / total_weight
                    end = min(audio_seconds, elapsed)

                    entries.append(
                        f"{index}\n"
                        f"{stamp(start)} --> {stamp(end)}\n"
                        f"{text}\n"
                    )

                srt_text = "\n".join(entries).strip()

            if not srt_text:
                raise RuntimeError("SRT စာတန်း မဖန်တီးနိုင်ပါ။")

            srt_path.write_text(
                srt_text + "\n",
                encoding="utf-8"
            )

            if not srt_path.is_file() or srt_path.stat().st_size == 0:
                raise RuntimeError("SRT ဖိုင်အလွတ်ဖြစ်နေပါတယ်။")

            logger.info(
                "SRT created: %s bytes",
                srt_path.stat().st_size
            )
            return

        except Exception as exc:
            last_error = exc

            if audio_path.exists():
                audio_path.unlink()

            logger.warning(
                "TTS attempt %s failed: %s",
                attempt + 1, exc
            )

            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Edge TTS မအောင်မြင်ပါ: {last_error}")


# ---------------- VIDEO PROCESSING ----------------

def process(job_id, video, opts):
    folder = WORK / job_id

    try:
        update_job(job_id, status="transcribing", progress=15)
        transcript = transcribe(video, folder)

        update_job(job_id, status="writing_script", progress=40)
        script = write_script(transcript, opts)

        (folder / "transcript.txt").write_text(
            transcript, encoding="utf-8"
        )
        (folder / "script.txt").write_text(
            script, encoding="utf-8"
        )

        update_job(job_id, status="generating_voice", progress=65)

        voice = VOICES.get(opts["voice"], "my-MM-ThihaNeural")
        audio = folder / "voice.mp3"
        srt = folder / "subs.srt"

        asyncio.run(create_voice(script, voice, audio, srt))

        update_job(job_id, status="rendering", progress=85)

        video_duration = duration(video)
        audio_duration = duration(audio)
        speed = audio_duration / video_duration

        aspect_filters = {
            "9:16": (
                "scale=720:1280:force_original_aspect_ratio=decrease,"
                "pad=720:1280:(ow-iw)/2:(oh-ih)/2"
            ),
            "16:9": (
                "scale=1280:720:force_original_aspect_ratio=decrease,"
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2"
            ),
            "1:1": (
                "scale=720:720:force_original_aspect_ratio=decrease,"
                "pad=720:720:(ow-iw)/2:(oh-ih)/2"
            ),
        }

        aspect = opts["aspect"]

        # Scale first to reduce processing cost; adjust playback speed.
        vf = f"setpts={speed:.6f}*PTS"
        if aspect in aspect_filters:
            vf += "," + aspect_filters[aspect]
        else:
            vf += ",scale=720:1280:force_original_aspect_ratio=decrease"
            vf += ",pad=720:1280:(ow-iw)/2:(oh-ih)/2"

        output = folder / "out.mp4"

        run([
            "ffmpeg", "-y",
            "-threads", "1",
            "-filter_threads", "1",
            "-i", str(video),
            "-i", str(audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-vf", vf,
            "-c:v", "libx264",
            "-threads", "1",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-shortest",
            "-movflags", "+faststart",
            str(output),
        ], timeout=900)

        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("MP4 ဖိုင် မထုတ်နိုင်ပါ။")

        update_job(
            job_id,
            status="done",
            progress=100,
            script_url=f"/api/jobs/{job_id}/script",
            srt_url=f"/api/jobs/{job_id}/srt",
            result_url=f"/api/jobs/{job_id}/result",
        )

        logger.info("Job %s completed", job_id)

    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        try:
            update_job(
                job_id,
                status="error",
                progress=0,
                error=f"{type(exc).__name__}: {str(exc)[:1500]}",
            )
        except Exception:
            logger.exception("Could not save job error")


# ---------------- API ROUTES ----------------

@app.get("/")
def home():
    index = Path("index.html")
    if index.is_file():
        return FileResponse(index)

    return {
        "name": "TZ Recap API",
        "version": "2.2.0",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_key_configured": bool(os.environ.get("GROQ_API_KEY")),
    }


@app.post("/api/jobs")
async def create_job(
    bg: BackgroundTasks,
    video: UploadFile = File(...),
    settings: str = Form("{}"),
):
    if not video.filename:
        raise HTTPException(400, "Video file is required.")

    if video.content_type and not video.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a video file.")

    try:
        data = json.loads(settings)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid settings JSON.")

    if not isinstance(data, dict):
        raise HTTPException(400, "Settings must be a JSON object.")

    job_id = uuid.uuid4().hex[:12]
    folder = WORK / job_id
    folder.mkdir(parents=True, exist_ok=False)
    video_path = folder / "in.mp4"

    size = 0
    try:
        with open(video_path, "wb") as dest:
            while True:
                chunk = await video.read(256 * 1024)
                if not chunk:
                    break

                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(
                        413,
                        "Video exceeds 100 MB. Please upload a smaller video.",
                    )
                dest.write(chunk)

    except Exception:
        if video_path.exists():
            video_path.unlink()
        try:
            folder.rmdir()
        except OSError:
            pass
        raise
    finally:
        await video.close()

    if size == 0:
        raise HTTPException(400, "Uploaded video is empty.")

    opts = {
        "mode": str(data.get("mode", "Recap")),
        "voice": str(data.get("voice", "သီဟ")),
        "style": str(data.get("style", "Fast Recap")),
        "aspect": str(data.get("aspect_ratio", data.get("aspect", "9:16"))),
        "lang": str(data.get("language", data.get("lang", "Burmese"))),
    }

    update_job(
        job_id,
        status="queued",
        progress=5,
        created_video_size=size,
    )

    bg.add_task(process, job_id, video_path, opts)

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/jobs/{job_id}",
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return JSONResponse(
                status_code=404,
                content={"status": "not_found", "error": "Job not found."},
            )
        return dict(job)


@app.get("/api/jobs/{job_id}/script")
def job_script(job_id: str):
    path = WORK / job_id / "script.txt"
    if not path.is_file():
        raise HTTPException(404, "Script is not ready.")
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename="recap_script.txt",
    )


@app.get("/api/jobs/{job_id}/srt")
def job_srt(job_id: str):
    path = WORK / job_id / "subs.srt"
    if not path.is_file():
        raise HTTPException(404, "SRT is not ready.")
    return FileResponse(
        path,
        media_type="application/x-subrip",
        filename="myanmar_recap.srt",
    )


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    path = WORK / job_id / "out.mp4"
    if not path.is_file():
        raise HTTPException(404, "MP4 is not ready.")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename="recap.mp4",
    )
