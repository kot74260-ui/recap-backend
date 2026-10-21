
import asyncio
import json
import os
import subprocess
import uuid
from pathlib import Path

import edge_tts
import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="TZ Recap API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK = Path("jobs")
WORK.mkdir(exist_ok=True)

JOBS = {}

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


def run(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or "Command failed")[-3000:]
        )
    return result


def duration(path):
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path)
    ])
    value = float(result.stdout.strip())
    if value <= 0:
        raise RuntimeError("Video or audio duration is invalid.")
    return value


def transcribe(video, folder):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Render Environment တွင် GROQ_API_KEY မရှိပါ။")

    audio = folder / "audio_small.mp3"

    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-b:a", "32k", str(audio)
    ])

    with open(audio, "rb") as f:
        response = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": "whisper-large-v3"},
            files={"file": ("audio.mp3", f, "audio/mpeg")},
            timeout=300,
        )

    if response.is_error:
        raise RuntimeError(
            f"Groq transcription error {response.status_code}: "
            f"{response.text[:1500]}"
        )

    transcript = response.json().get("text", "").strip()
    if not transcript:
        raise RuntimeError("ဗီဒီယိုထဲမှ အသံစာသား မရရှိပါ။")

    return transcript


def write_script(transcript, opts):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY မသတ်မှတ်ထားပါ။")

    lang = opts["lang"]
    if opts["mode"].lower() == "translate":
        task = (
            f"Translate the following transcript into {lang}. "
            "Preserve the original meaning, names and sequence. "
            "If the target language is Burmese, use natural Burmese Unicode."
        )
    else:
        task = (
            f"Write a compelling movie recap narration in {lang}. "
            f"{STYLES.get(opts['style'], STYLES['Fast Recap'])} "
            "Keep the story events in the correct order. "
            "Use natural Burmese Unicode when the target is Burmese. "
            "Return narration only, without headings or explanations."
        )

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-oss-120b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert movie translator and recap writer. "
                        "Preserve meaning and never invent missing events."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{task}\n"
                        f"Target language: {lang}\n\n"
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
            f"{response.text[:1500]}"
        )

    data = response.json()
    script = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not script:
        raise RuntimeError("AI မှ narration စာသား မရရှိပါ။")

    return script



async def create_voice(script, voice, audio_path, srt_path):
    if not script or not script.strip():
        raise RuntimeError("AI narration စာသား အလွတ်ဖြစ်နေပါတယ်။")

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        communicate = edge_tts.Communicate(
            script.strip(),
            voice
        )
        submaker = edge_tts.SubMaker()
        audio_size = 0

        with open(audio_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                    audio_size += len(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    submaker.feed(chunk)

        if audio_size == 0:
            raise RuntimeError(
                "Edge TTS မှ အသံဒေတာ မရရှိပါ။ "
                f"အသုံးပြုထားသော voice: {voice}"
            )

        srt_path.write_text(
            submaker.get_srt(),
            encoding="utf-8"
        )

    except Exception as exc:
        if audio_path.exists():
            audio_path.unlink()

        raise RuntimeError(
            f"Edge TTS အသံထုတ်လုပ်မှု မအောင်မြင်ပါ။ "
            f"Voice={voice}; Error={type(exc).__name__}: {exc}"
        ) from exc



def process(job_id, video, opts):
    folder = WORK / job_id

    try:
        JOBS[job_id].update(status="transcribing", progress=15)
        transcript = transcribe(video, folder)

        JOBS[job_id].update(status="writing_script", progress=40)
        script = write_script(transcript, opts)

        (folder / "transcript.txt").write_text(
            transcript, encoding="utf-8"
        )
        (folder / "script.txt").write_text(
            script, encoding="utf-8"
        )

        JOBS[job_id].update(status="generating_voice", progress=65)

        voice = VOICES.get(opts["voice"], "my-MM-ThihaNeural")
        audio = folder / "voice.mp3"
        srt = folder / "subs.srt"

        asyncio.run(create_voice(script, voice, audio, srt))

        JOBS[job_id].update(status="rendering", progress=85)

        video_duration = duration(video)
        audio_duration = duration(audio)

        # Keep the video running for the full narration duration.
        vf = f"setpts={audio_duration / video_duration:.6f}*PTS"

        aspect = opts["aspect"]
        aspect_filters = {
            "9:16": (
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
            ),
            "16:9": (
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
            ),
            "1:1": (
                "scale=1080:1080:force_original_aspect_ratio=decrease,"
                "pad=1080:1080:(ow-iw)/2:(oh-ih)/2"
            ),
        }

        if aspect in aspect_filters:
            vf += "," + aspect_filters[aspect]

        output = folder / "out.mp4"

        run([
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(audio),
            "-vf", vf,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output),
        ])

        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError("MP4 ဖိုင် မထုတ်နိုင်ပါ။")

        JOBS[job_id].update(
            status="done",
            progress=100,
            script_url=f"/api/jobs/{job_id}/script",
            srt_url=f"/api/jobs/{job_id}/srt",
            result_url=f"/api/jobs/{job_id}/result",
        )

    except Exception as exc:
        JOBS[job_id].update(
            status="error",
            progress=0,
            error=str(exc)[:3000],
        )


@app.get("/")
def home():
    index = Path("index.html")
    if index.exists():
        return FileResponse(index)
    return {
        "name": "TZ Recap API",
        "status": "online",
        "docs": "/docs",
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
        settings_data = json.loads(settings)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid settings JSON.")

    job_id = uuid.uuid4().hex[:12]
    folder = WORK / job_id
    folder.mkdir(parents=True, exist_ok=False)

    video_path = folder / "in.mp4"

    try:
        size = 0
        with open(video_path, "wb") as dest:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > 500 * 1024 * 1024:
                    raise HTTPException(
                        413, "Video is larger than the 500 MB limit."
                    )
                dest.write(chunk)
    except Exception:
        if video_path.exists():
            video_path.unlink()
        raise
    finally:
        await video.close()

    if size == 0:
        raise HTTPException(400, "Uploaded video is empty.")

    opts = {
        "mode": settings_data.get("mode", "Recap"),
        "voice": settings_data.get("voice", "သီဟ"),
        "style": settings_data.get("style", "Fast Recap"),
        "aspect": settings_data.get("aspect", "Auto"),
        "lang": settings_data.get("lang", "Burmese"),
    }

    JOBS[job_id] = {"status": "queued", "progress": 5}
    bg.add_task(process, job_id, video_path, opts)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={"status": "not_found", "error": "Job not found."},
        )
    return job


@app.get("/api/jobs/{job_id}/script")
def job_script(job_id: str):
    path = WORK / job_id / "script.txt"
    if not path.is_file():
        raise HTTPException(404, "Script is not ready.")
    return FileResponse(
        path, media_type="text/plain; charset=utf-8",
        filename="recap_script.txt",
    )


@app.get("/api/jobs/{job_id}/srt")
def job_srt(job_id: str):
    path = WORK / job_id / "subs.srt"
    if not path.is_file():
        raise HTTPException(404, "SRT is not ready.")
    return FileResponse(
        path, media_type="application/x-subrip",
        filename="myanmar_recap.srt",
    )


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    path = WORK / job_id / "out.mp4"
    if not path.is_file():
        raise HTTPException(404, "MP4 is not ready.")
    return FileResponse(
        path, media_type="video/mp4", filename="recap.mp4"
    )
