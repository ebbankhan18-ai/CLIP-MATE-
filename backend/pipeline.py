"""Clip Mate pipeline: ingest -> transcribe -> pick moments -> render vertical clips with captions."""
import json
import os
import re
import subprocess
from pathlib import Path

FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE = os.getenv("FFPROBE_BIN", "ffprobe")


# ---------------------------------------------------------------- helpers
def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").strip().splitlines()[-8:])
        raise RuntimeError(f"{Path(cmd[0]).name} failed:\n{tail}")
    return p.stdout


def probe(path):
    out = run([FFPROBE, "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)])
    j = json.loads(out)
    s = j["streams"][0]
    return {"width": int(s["width"]), "height": int(s["height"]), "duration": float(j["format"]["duration"])}


def has_cuda():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


# ---------------------------------------------------------------- 1. ingest
def ingest(pdir, upload_path=None, url=None, on_progress=None):
    """Return (source_path, title)."""
    if url:
        import yt_dlp

        def hook(d):
            if d.get("status") == "downloading" and on_progress:
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                if total:
                    on_progress(min(d["downloaded_bytes"] / total, 1.0))

        opts = {
            "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(pdir / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [hook],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        files = [p for p in pdir.glob("source.*") if p.suffix not in (".part", ".ytdl")]
        if not files:
            raise RuntimeError("Download finished but no video file was found.")
        return files[0], (info.get("title") or "YouTube video")
    return Path(upload_path), Path(upload_path).stem


# ---------------------------------------------------------------- 2. transcribe
def transcribe(src, model_name, on_progress=None):
    """Word-level transcript: {"language", "words": [{w,s,e}], "sentences": [{s,e,text}]}"""
    from faster_whisper import WhisperModel

    cuda = has_cuda()
    model = WhisperModel(model_name, device="cuda" if cuda else "cpu",
                         compute_type="float16" if cuda else "int8")
    segments, info = model.transcribe(str(src), word_timestamps=True, vad_filter=True)
    words = []
    for seg in segments:
        for w in (seg.words or []):
            t = (w.word or "").strip()
            if t and w.start is not None and w.end is not None:
                words.append({"w": t, "s": round(w.start, 3), "e": round(w.end, 3)})
        if on_progress and info.duration:
            on_progress(min(seg.end / info.duration, 1.0))
    return {"language": info.language, "words": words, "sentences": build_sentences(words)}


def build_sentences(words, max_words=28, max_gap=0.8):
    out, cur = [], []

    def flush():
        if cur:
            out.append({"s": cur[0]["s"], "e": cur[-1]["e"], "text": " ".join(x["w"] for x in cur)})
            cur.clear()

    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > max_gap or len(cur) >= max_words):
            flush()
        cur.append(w)
        if re.search(r"[.!?]$", w["w"]):
            flush()
    flush()
    return out


# ---------------------------------------------------------------- 3. pick moments
PROFILE_HINTS = {
    "auto": "Infer the content type yourself and judge accordingly.",
    "podcast": "Conversation/interview. Favor strong opinions, stories, surprising claims, and disagreements. The clip must make sense without the rest of the episode.",
    "vlog": "Vlog/lifestyle. Favor emotional beats, funny moments, reveals, and clear before/after payoffs.",
    "tutorial": "Tutorial/educational. Favor one complete tip or insight with a clear result. Start with the problem or the promise.",
    "gaming": "Gameplay/commentary. Favor big reactions, clutch plays, fails, and funny commentary.",
    "sports": "Sports/commentary. Favor decisive plays, hype reactions, and key explanations.",
    "lecture": "Lecture/talk. Favor memorable explanations, analogies, and quotable lines.",
}

SYSTEM_PROMPT = """You are a senior short-form video editor who cuts long videos into clips for Shorts/Reels/TikTok.
You choose moments that stand alone: a strong hook in the first 3 seconds, one complete idea or story, and a satisfying end.
You never start mid-sentence or end mid-thought. You respond with JSON only."""


def llm_provider():
    """Which model picks the clips: 'anthropic', 'ollama' (free, local) or '' (basic heuristic)."""
    want = os.getenv("LLM_PROVIDER", "").strip().lower()
    if want == "ollama" or (not want and not os.getenv("ANTHROPIC_API_KEY") and os.getenv("OLLAMA_MODEL")):
        return "ollama"
    if want in ("", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return ""


def _ask_llm(prompt, provider, model):
    if provider == "ollama":
        import urllib.request
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        body = json.dumps({"model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"), "stream": False,
                           "options": {"num_ctx": 8192, "temperature": 0.3},
                           "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(host + "/api/chat", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read())["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Could not get an answer from Ollama at {host}. Is Ollama running, and did you run "
                               f"'ollama pull {os.getenv('OLLAMA_MODEL', 'qwen2.5:7b-instruct')}'? ({e})")
    import anthropic
    msg = anthropic.Anthropic().messages.create(model=model, max_tokens=2500, system=SYSTEM_PROMPT,
                                                messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _llm_json(prompt, provider, model):
    last = None
    for _ in range(2):
        text = _ask_llm(prompt, provider, model)
        try:
            a, b = text.index("["), text.rindex("]")
            return json.loads(text[a:b + 1])
        except Exception as e:  # retry once
            last = e
    raise RuntimeError(f"The model did not return valid JSON: {last}")


def _closest(sentences, t, key):
    return min(range(len(sentences)), key=lambda i: abs(sentences[i][key] - t))


def _snap(sentences, start, end, min_len, max_len, duration):
    i, j = _closest(sentences, start, "s"), _closest(sentences, end, "e")
    if j < i:
        return None
    while j > i and sentences[j]["e"] - sentences[i]["s"] > max_len:
        j -= 1
    while j + 1 < len(sentences) and sentences[j]["e"] - sentences[i]["s"] < min_len:
        j += 1
    s = max(sentences[i]["s"] - 0.15, 0.0)
    e = min(sentences[j]["e"] + 0.35, duration)
    return (s, e) if e - s >= min_len * 0.6 else None


def _overlap(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def _heuristic(sentences, min_len, max_len):
    """No-API fallback: score windows by speech density and expressive punctuation."""
    target = (min_len + max_len) / 2
    cands = []
    for i in range(len(sentences)):
        j = i
        while j + 1 < len(sentences) and sentences[j]["e"] - sentences[i]["s"] < target:
            j += 1
        span = sentences[j]["e"] - sentences[i]["s"]
        if span < min_len * 0.6:
            continue
        text = " ".join(x["text"] for x in sentences[i:j + 1])
        score = len(text.split()) / span * 20 + 6 * (text.count("?") + text.count("!"))
        cands.append({"start": sentences[i]["s"], "end": sentences[j]["e"], "score": min(score, 99),
                      "title": " ".join(text.split()[:9]), "reason": "Heuristic pick (no LLM key set)."})
    return cands


def select_clips(transcript, duration, n, min_len, max_len, profile, model, on_progress=None):
    sentences = transcript["sentences"]
    if not sentences:
        raise RuntimeError("No speech was detected in this video, so there is nothing to clip yet.")
    warnings, cands = [], []

    provider = llm_provider()
    if provider:
        size, overlap = 600, 60
        starts = list(range(0, int(duration), size - overlap)) or [0]
        for k, c0 in enumerate(starts):
            chunk = [x for x in sentences if c0 <= x["s"] < c0 + size]
            if len(chunk) >= 3:
                lines = "\n".join(f"[{x['s']:.1f}-{x['e']:.1f}] {x['text']}" for x in chunk)
                prompt = (
                    f"Content type: {PROFILE_HINTS.get(profile, PROFILE_HINTS['auto'])}\n\n"
                    f"From the transcript below, pick up to {min(6, max(3, n))} best clips, each {min_len}-{max_len} seconds long.\n"
                    "Rules:\n"
                    "- start must equal the start time of a line; end must equal the end time of a line (numbers in brackets).\n"
                    "- Open with a hook (a claim, question, tension or surprise). Cut any slow warm-up.\n"
                    "- One complete idea or story; it must make sense with no outside context.\n"
                    "- score 0-100 = how likely it is to hold attention. Be strict; most clips are below 80.\n"
                    'Return ONLY a JSON array: [{"start":0.0,"end":0.0,"score":0,"title":"catchy title, max 70 chars","reason":"one sentence"}]\n\n'
                    f"TRANSCRIPT:\n{lines}"
                )
                for c in _llm_json(prompt, provider, model):
                    try:
                        cands.append({"start": float(c["start"]), "end": float(c["end"]),
                                      "score": float(c.get("score", 50)),
                                      "title": str(c.get("title", "Clip"))[:100],
                                      "reason": str(c.get("reason", ""))})
                    except (KeyError, TypeError, ValueError):
                        continue
            if on_progress:
                on_progress((k + 1) / len(starts))
    else:
        warnings.append("No AI model is set up, so a basic heuristic picked the clips. Set up Ollama (free) or an Anthropic key for much better results.")
        cands = _heuristic(sentences, min_len, max_len)

    snapped = []
    for c in cands:
        se = _snap(sentences, c["start"], c["end"], min_len, max_len, duration)
        if se:
            snapped.append({**c, "start": round(se[0], 2), "end": round(se[1], 2)})
    snapped.sort(key=lambda c: -c["score"])
    chosen = []
    for c in snapped:
        if all(_overlap((c["start"], c["end"]), (x["start"], x["end"])) < 0.3 for x in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    return chosen, warnings


# ---------------------------------------------------------------- 4. captions
CAPTION_STYLES = {
    "pop":   dict(font="Arial", size=84, bold=1, primary="&H00FFFFFF", hi="&H0000E5FF", outline="&H00000000", ol=7, border=1, upper=True),
    "clean": dict(font="Arial", size=70, bold=1, primary="&H00FFFFFF", hi="&H0080DE4A", outline="&H00000000", ol=4, border=1, upper=False),
    "karaoke": dict(font="Arial", size=76, bold=1, primary="&H00FFFFFF", hi="&H00A6F26D", outline="&H00000000", ol=6, border=1, upper=False, karaoke=True),
    "box":   dict(font="Arial", size=72, bold=1, primary="&H00FFFFFF", hi="&H0000E5FF", outline="&HAA000000", ol=14, border=3, upper=False),
}


def _ts(t):
    cs = max(int(round(t * 100)), 0)
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _group_words(words):
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= 3 or w["s"] - cur[-1]["e"] > 0.7 or sum(len(x["w"]) for x in cur) + len(w["w"]) > 22):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,;:]$", w["w"]):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def make_ass(words, start, end, style_name, path):
    st = CAPTION_STYLES.get(style_name, CAPTION_STYLES["pop"])
    clip_words = [{"w": re.sub(r"[{}\\]", "", w["w"]), "s": w["s"] - start, "e": w["e"] - start}
                  for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.05]
    if st["upper"]:
        for w in clip_words:
            w["w"] = w["w"].upper()
    lines = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 2",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{st['font']},{st['size']},{st['hi'] if st.get('karaoke') else st['primary']},{st['primary']},{st['outline']},&H00000000,{st['bold']},0,0,0,100,100,0,0,{st['border']},{st['ol']},0,2,90,90,430,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for g in _group_words(clip_words):
        if st.get("karaoke"):  # one line per phrase, words fill with colour as they are spoken
            parts = []
            for i, w in enumerate(g):
                nxt = g[i + 1]["s"] if i + 1 < len(g) else w["e"]
                parts.append("{\\kf" + str(max(int(round((nxt - w["s"]) * 100)), 1)) + "}" + w["w"])
            lines.append(f"Dialogue: 0,{_ts(g[0]['s'])},{_ts(g[-1]['e'] + 0.15)},Cap,,0,0,0,,{' '.join(parts)}")
            continue
        for i, w in enumerate(g):
            t0 = w["s"]
            t1 = g[i + 1]["s"] if i + 1 < len(g) else w["e"] + 0.15
            if t1 <= t0:
                t1 = t0 + 0.1
            parts = []
            for k, x in enumerate(g):
                if k == i:
                    pop = "\\fscx110\\fscy110" if st["border"] == 1 else ""
                    parts.append("{\\1c" + st["hi"] + "&" + pop + "}" + x["w"] + "{\\r}")
                else:
                    parts.append(x["w"])
            lines.append(f"Dialogue: 0,{_ts(t0)},{_ts(t1)},Cap,,0,0,0,,{' '.join(parts)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 5. reframe + render
def detect_face_x(src, start, end):
    """Median horizontal face position (0..1) over the clip, or 0.5 if none found."""
    try:
        import cv2
        import numpy as np
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:  # OpenCV missing or without the Haar detector: fall back to a centered crop
        return 0.5
    cap = cv2.VideoCapture(str(src))
    xs, t, step = [], start, max(0.5, (end - start) / 40)
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        sc = 480 / h
        gray = cv2.cvtColor(cv2.resize(frame, (int(w * sc), 480)), cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            xs.append((x + fw / 2) / gray.shape[1])
        t += step
    cap.release()
    return float(np.median(xs)) if xs else 0.5


def render_clip(src, out_path, work_dir, start, end, words, info, reframe="crop", caption_style="pop"):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    make_ass(words, start, end, caption_style, work_dir / "captions.ass")

    sw, sh = info["width"], info["height"]
    if reframe == "blur" and sw / sh > 9 / 16 + 0.02:
        graph = ("[0:v]split=2[a][b];"
                 "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5[bg];"
                 "[b]scale=1080:-2[fg];"
                 "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,ass=captions.ass[v]")
    else:
        cw = min(sw, int(sh * 9 / 16) // 2 * 2)
        x = 0
        if cw < sw:
            cx = detect_face_x(src, start, end)
            x = int(min(max(cx * sw - cw / 2, 0), sw - cw))
        graph = (f"[0:v]crop={cw}:{sh}:{x}:0,scale=1080:1920:force_original_aspect_ratio=increase,"
                 "crop=1080:1920,setsar=1,ass=captions.ass[v]")

    enc = os.getenv("VIDEO_ENCODER", "libx264")
    venc = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21"] if enc == "h264_nvenc"
            else ["-c:v", enc, "-preset", "veryfast", "-crf", "20"])
    cmd = [FFMPEG, "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(Path(src).resolve()),
           "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
           *venc, "-pix_fmt", "yuv420p", "-r", "30",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
           "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(Path(out_path).resolve())]
    run(cmd, cwd=str(work_dir))  # cwd lets the ass filter use a plain relative path (Windows-safe)
