"""Clip Mate server. Run with:  python -m uvicorn backend.main:app  (see run scripts)."""
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import pipeline, social  # noqa: E402

DATA = ROOT / "data"
(DATA / "projects").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Clip Mate")
JOB_LOCK = threading.Lock()   # one heavy job at a time keeps a laptop responsive
IO_LOCK = threading.RLock()
PROJECTS = {}


def pdir(pid):
    return DATA / "projects" / pid


def save(p):
    with IO_LOCK:
        (pdir(p["id"]) / "project.json").write_text(json.dumps(p, indent=2), encoding="utf-8")


def load_all():
    for f in (DATA / "projects").glob("*/project.json"):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if p.get("state") in ("queued", "downloading", "transcribing", "selecting", "rendering"):
            p["state"], p["message"] = "error", "Interrupted when the app was closed. Delete and try again."
        for c in p.get("clips", []):
            if c.get("status") == "rendering":
                c["status"] = "error"
        PROJECTS[p["id"]] = p


load_all()


def log(p, text):
    p.setdefault("log", []).append({"t": time.time(), "text": text})
    save(p)


def set_state(p, state, message="", progress=None):
    p["state"], p["message"] = state, message
    if progress is not None:
        p["progress"] = round(progress, 3)
    save(p)


def clip_url(pid, clip):
    return f"/media/projects/{pid}/clips/{clip['file']}"


# ------------------------------------------------------------------ job
def process(pid):
    p = PROJECTS[pid]
    d = pdir(pid)
    s = p["settings"]
    try:
        set_state(p, "queued", "Waiting for the previous job to finish", 0)
        with JOB_LOCK:
            # 1. ingest
            set_state(p, "downloading", "Getting the video", 0.02)
            src, title = pipeline.ingest(
                d, upload_path=p["source"].get("path"), url=p["source"].get("url"),
                on_progress=lambda f: set_state(p, "downloading", "Downloading from the link", 0.02 + 0.13 * f))
            info = pipeline.probe(src)
            p.update(title=p["title"] if p["source"]["kind"] == "upload" and p["title"] else title,
                     info=info, source_file=src.name)
            log(p, f"Got the video · {int(info['duration'] // 60)}:{int(info['duration'] % 60):02d}")
            # 2. transcribe
            set_state(p, "transcribing", "Listening to every word", 0.15)
            tr = pipeline.transcribe(
                src, os.getenv("WHISPER_MODEL", "small"),
                on_progress=lambda f: set_state(p, "transcribing", "Listening to every word", 0.15 + 0.40 * f))
            (d / "transcript.json").write_text(json.dumps(tr), encoding="utf-8")
            log(p, f"Transcribed · {len(tr['words']):,} words")
            # 3. pick moments
            set_state(p, "selecting", "Finding the strongest moments", 0.55)
            picks, warnings = pipeline.select_clips(
                tr, info["duration"], s["num_clips"], s["min_len"], s["max_len"], s["profile"],
                os.getenv("CLIP_LLM_MODEL", "claude-sonnet-5-5"),
                on_progress=lambda f: set_state(p, "selecting", "Finding the strongest moments", 0.55 + 0.10 * f))
            p["warnings"] = warnings
            log(p, f"Picked {len(picks)} strong moment{'' if len(picks) == 1 else 's'}")
            p["clips"] = [{"id": f"c{i + 1}", "start": c["start"], "end": c["end"], "score": round(c["score"]),
                           "title": c["title"], "reason": c["reason"], "status": "queued", "version": 0,
                           "file": None, "published": {}, "reframe": s["reframe"], "caption_style": s["caption_style"]}
                          for i, c in enumerate(picks)]
            save(p)
            # 4. render
            n = max(len(p["clips"]), 1)
            for i, clip in enumerate(p["clips"]):
                set_state(p, "rendering", f"Rendering clip {i + 1} of {len(p['clips'])}", 0.65 + 0.35 * i / n)
                render_one(p, clip, tr["words"])
                log(p, f"Rendered clip {i + 1} · {clip['title'][:48]}")
            log(p, f"{len(p['clips'])} clip{'' if len(p['clips']) == 1 else 's'} ready")
            set_state(p, "done", "", 1.0)
    except Exception as e:  # surface the real reason in the UI
        traceback.print_exc()
        p["error"] = str(e)
        set_state(p, "error", str(e))


def render_one(p, clip, words):
    pid, d = p["id"], pdir(p["id"])
    clip["status"] = "rendering"
    save(p)
    try:
        clip["version"] += 1
        out_name = f"{clip['id']}_v{clip['version']}.mp4"
        (d / "clips").mkdir(exist_ok=True)
        pipeline.render_clip(d / p["source_file"], d / "clips" / out_name, d / "work" / clip["id"],
                             clip["start"], clip["end"], words, p["info"], clip["reframe"], clip["caption_style"])
        for old in (d / "clips").glob(f"{clip['id']}_v*.mp4"):
            if old.name != out_name:
                old.unlink(missing_ok=True)
        clip["file"], clip["status"] = out_name, "ready"
        clip.pop("error", None)
    except Exception as e:
        clip["status"], clip["error"] = "error", str(e)
        traceback.print_exc()
    save(p)


# ------------------------------------------------------------------ api
@app.get("/api/status")
def status():
    return {
        "ffmpeg": shutil.which(pipeline.FFMPEG) is not None,
        "cuda": pipeline.has_cuda(),
        "whisper_model": os.getenv("WHISPER_MODEL", "small"),
        "llm": pipeline.llm_provider(),
        "youtube": social.youtube_status(),
    }


@app.get("/api/projects")
def list_projects():
    items = sorted(PROJECTS.values(), key=lambda p: p["created"], reverse=True)
    return [{k: p.get(k) for k in ("id", "title", "state", "progress", "created")} for p in items]


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    p = PROJECTS.get(pid)
    if not p:
        raise HTTPException(404, "Project not found")
    out = {**p, "clips": [{**c, "url": clip_url(pid, c) if c.get("file") else None} for c in p.get("clips", [])]}
    return out


@app.post("/api/projects")
def create_project(
    file: Optional[UploadFile] = File(None),
    url: str = Form(""),
    num_clips: int = Form(5),
    min_len: int = Form(25),
    max_len: int = Form(60),
    profile: str = Form("auto"),
    reframe: str = Form("crop"),
    caption_style: str = Form("pop"),
):
    url = url.strip()
    if bool(file and file.filename) == bool(url):
        raise HTTPException(400, "Add either a video file or a link, not both.")
    if min_len < 10 or max_len > 180 or min_len >= max_len:
        raise HTTPException(400, "Clip length must be between 10 and 180 seconds, with min below max.")
    pid = uuid.uuid4().hex[:10]
    d = pdir(pid)
    d.mkdir(parents=True)
    if file and file.filename:
        dest = d / ("upload" + Path(file.filename).suffix.lower())
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        source = {"kind": "upload", "path": str(dest), "name": file.filename}
        title = Path(file.filename).stem
    else:
        source, title = {"kind": "url", "url": url}, url
    p = {"id": pid, "title": title, "created": time.time(), "state": "queued", "message": "", "progress": 0,
         "source": source, "clips": [], "warnings": [],
         "settings": {"num_clips": max(1, min(num_clips, 15)), "min_len": min_len, "max_len": max_len,
                      "profile": profile, "reframe": reframe, "caption_style": caption_style}}
    PROJECTS[pid] = p
    save(p)
    threading.Thread(target=process, args=(pid,), daemon=True).start()
    return {"id": pid}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    if pid not in PROJECTS:
        raise HTTPException(404, "Project not found")
    PROJECTS.pop(pid)
    shutil.rmtree(pdir(pid), ignore_errors=True)
    return {"ok": True}


class RenderReq(BaseModel):
    start: float
    end: float
    reframe: Optional[str] = None
    caption_style: Optional[str] = None


@app.post("/api/projects/{pid}/clips/{cid}/render")
def rerender(pid: str, cid: str, req: RenderReq):
    p = PROJECTS.get(pid)
    clip = next((c for c in (p or {}).get("clips", []) if c["id"] == cid), None)
    if not clip:
        raise HTTPException(404, "Clip not found")
    dur = p["info"]["duration"]
    start, end = max(0.0, req.start), min(dur, req.end)
    if end - start < 5 or end - start > 180:
        raise HTTPException(400, "A clip must be between 5 and 180 seconds long.")
    clip.update(start=round(start, 2), end=round(end, 2),
                reframe=req.reframe or clip["reframe"], caption_style=req.caption_style or clip["caption_style"],
                status="queued", published={})
    save(p)

    def job():
        with JOB_LOCK:
            words = json.loads((pdir(pid) / "transcript.json").read_text(encoding="utf-8"))["words"]
            render_one(p, clip, words)

    threading.Thread(target=job, daemon=True).start()
    return {"ok": True}


# ------------------------------------------------------------------ youtube
@app.get("/api/auth/youtube/login")
def youtube_login(request: Request):
    try:
        return RedirectResponse(social.youtube_auth_url(str(request.url_for("youtube_callback"))))
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/oauth/youtube/callback", name="youtube_callback")
def youtube_callback(request: Request, state: str = "", error: str = ""):
    if error:
        return RedirectResponse(f"/?youtube=denied")
    try:
        social.youtube_finish_auth(state, str(request.url))
    except Exception as e:
        raise HTTPException(400, str(e))
    return RedirectResponse("/?youtube=connected")


@app.post("/api/auth/youtube/disconnect")
def youtube_disconnect():
    social.youtube_disconnect()
    return {"ok": True}


class PublishReq(BaseModel):
    project_id: str
    clip_id: str
    title: str
    description: str = ""
    privacy: str = "private"


@app.post("/api/publish/youtube")
def publish_youtube(req: PublishReq):
    p = PROJECTS.get(req.project_id)
    clip = next((c for c in (p or {}).get("clips", []) if c["id"] == req.clip_id), None)
    if not clip or clip.get("status") != "ready":
        raise HTTPException(400, "This clip is not ready to publish.")
    try:
        res = social.youtube_upload(pdir(p["id"]) / "clips" / clip["file"], req.title, req.description,
                                    req.privacy if req.privacy in ("private", "unlisted", "public") else "private")
    except Exception as e:
        raise HTTPException(400, f"YouTube upload failed: {e}")
    clip.setdefault("published", {})["youtube"] = res["url"]
    save(p)
    return res


# ------------------------------------------------------------------ static
app.mount("/media", StaticFiles(directory=DATA), name="media")
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
