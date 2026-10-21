import asyncio, json, os, subprocess, uuid
from pathlib import Path

import anthropic
import edge_tts
import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORK = Path("jobs"); WORK.mkdir(exist_ok=True)
JOBS = {}
claude = anthropic.Anthropic()

VOICES = {
    "Thiha · Male (Burmese)": "my-MM-ThihaNeural",
    "Nilar · Female (Burmese)": "my-MM-NilarNeural",
}
STYLES = {
    "Fast Recap": "ပြန်ကြားရန် မြန်မြန်ဆန်ဆန်၊ စာကြောင်းတိုတိုများဖြင့် အကျဉ်းချုပ်ပါ။",
    "Story Mode": "ဇာတ်လမ်းပြောပြသလို ဆွဲဆောင်မှုရှိရှိ ပြန်ပြောပါ။",
    "Dramatic": "စိတ်လှုပ်ရှားဖွယ် ပြင်းပြင်းထန်ထန် လေသံဖြင့် ပြောပါ။",
    "Summary": "အဓိကအချက်များကိုသာ ရှင်းရှင်းလင်းလင်း အကျဉ်းချုပ်ပါ။",
}


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(out.stdout.strip())


def transcribe(video, d):
    mp3 = d / "audio_small.mp3"
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(mp3)])
    with open(mp3, "rb") as f:
        r = httpx.post("https://api.groq.com/openai/v1/audio/transcriptions",
                       headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
                       data={"model": "whisper-large-v3"}, files={"file": ("a.mp3", f, "audio/mpeg")},
                       timeout=300)
    r.raise_for_status()
    return r.json()["text"]


def process(job_id, video, opts):
    d = WORK / job_id
    try:
        JOBS[job_id].update(status="transcribing", progress=15)
        transcript = transcribe(video, d)

        JOBS[job_id].update(status="writing script", progress=40)
        if opts["mode"] == "Translate":
            task = f"အောက်ပါစာသားကို {opts['lang']} သို့ တိတိကျကျ ဘာသာပြန်ပါ။"
        else:
            task = f"အောက်ပါ ဗီဒီယိုစာသားကို {opts['lang']} ဘာသာဖြင့် recap ရေးပါ။ {STYLES.get(opts['style'], '')}"
        msg = claude.messages.create(
            model="claude-sonnet-4-5", max_tokens=2000,
            messages=[{"role": "user", "content": f"{task}\nပြန်ပေးရာတွင် narration စာသားသာ ပေးပါ။\n\n{transcript[:20000]}"}])
        script = msg.content[0].text.strip()
        (d / "script.txt").write_text(script, encoding="utf-8")

        JOBS[job_id].update(status="generating voice", progress=65)
        voice = VOICES.get(opts["voice"], "my-MM-ThihaNeural")
        audio = d / "voice.mp3"

        async def tts():
            com = edge_tts.Communicate(script, voice)
            sub = edge_tts.SubMaker()
            with open(audio, "wb") as f:
                async for ch in com.stream():
                    if ch["type"] == "audio":
                        f.write(ch["data"])
                    elif ch["type"] in ("WordBoundary", "SentenceBoundary"):
                        sub.feed(ch)
            (d / "subs.srt").write_text(sub.get_srt(), encoding="utf-8")
        asyncio.run(tts())

        JOBS[job_id].update(status="rendering", progress=85)
        vd, ad = duration(video), duration(audio)
        vf = f"setpts={ad / vd:.4f}*PTS"
        ar = {"9:16": "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1",
              "16:9": "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1",
              "1:1": "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:-1:-1"}.get(opts["aspect"])
        if ar:
            vf += "," + ar
        if opts["captions"]:
            style = "FontName=Padauk,FontSize=%d,Outline=2,Alignment=2" % int(22 * opts["scale"] / 100)
            vf += f",subtitles={d/'subs.srt'}:force_style='{style}'"
        out = d / "out.mp4"
        run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-vf", vf,
             "-map", "0:v", "-map", "1:a", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(out)])
        JOBS[job_id].update(status="done", progress=100)
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


@app.post("/api/jobs")
async def create_job(bg: BackgroundTasks, video: UploadFile = File(...), settings: str = Form("{}")):
    s = json.loads(settings)
    opts = dict(mode=s.get("mode", "Recap"), voice=s.get("voice", ""), style=s.get("style", "Fast Recap"),
                aspect=s.get("aspect", "Auto"), lang=s.get("lang", "Burmese"),
                captions=s.get("cap", True), scale=int(s.get("scale", 109)))
    job_id = uuid.uuid4().hex[:10]
    d = WORK / job_id; d.mkdir()
    path = d / "in.mp4"
    path.write_bytes(await video.read())
    JOBS[job_id] = {"status": "queued", "progress": 5}
    bg.add_task(process, job_id, path, opts)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return JOBS.get(job_id, {"status": "not_found"})


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str):
    return FileResponse(WORK / job_id / "out.mp4", media_type="video/mp4", filename="recap.mp4")
    @app.get("/")
def home():
    return FileResponse("index.html")
