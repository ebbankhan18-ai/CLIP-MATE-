# Clip Mate

Turn one long video into vertical clips with captions, then publish from the same page.
Everything runs on your own computer. Your videos never leave it, except what you choose to
upload to a platform and the transcript text sent to the AI that picks the moments.

## 1. Install the basics (once)

- **Python 3.10 to 3.12** (python.org). On Windows, tick "Add Python to PATH".
- **ffmpeg**
  - Windows: `winget install Gyan.FFmpeg` (then reopen the terminal)
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## 2. Start it

- Windows: double-click `run.bat`
- macOS / Linux: `./run.sh`

The first run creates a virtual environment and installs everything (a few minutes), then opens
http://localhost:8000. The first transcription also downloads the speech model once (about 500 MB for `small`).

## 3. Choose the AI that picks the clips (this is what makes clip quality good)

**Free, runs on your laptop (Ollama):** install Ollama from ollama.com, open a terminal and run
`ollama pull qwen2.5:7b-instruct` (about 5 GB; use `qwen2.5:3b-instruct` on a low-memory laptop).
In `.env` remove the `#` before `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=...`, save, restart.

**Paid, better quality (Anthropic):** paste your API key after `ANTHROPIC_API_KEY=`.

With neither, the app still works, but clips are picked by a basic heuristic and will be much weaker.

## 4. Connect YouTube (so you can publish from the page)

1. Go to https://console.cloud.google.com and create a project.
2. APIs & Services > Library > enable **YouTube Data API v3**.
3. APIs & Services > OAuth consent screen: choose External, fill the basics, and add **your own Google account as a Test user**.
4. APIs & Services > Credentials > Create credentials > OAuth client ID > **Desktop app**. Download the JSON.
5. Save it in this folder as `client_secret.json`, restart the app, and click **Connect YouTube** in the header.

Important: a new Google project that has not passed Google's compliance audit can only upload
**private** videos, whatever visibility you pick. For your own channel that is fine: upload as
private, then switch to public in YouTube Studio. The audit is only needed if you want other
people to publish publicly through your app.

## Using it

1. Drop a video or paste a link, choose the options, press **Find clips**.
2. Watch progress; clips appear as they finish. The red marks on the timeline show where each clip came from.
3. Not perfect? Change Start/End seconds and press **Re-render**.
4. **Download** or open **Publish** to send it to YouTube.

## Settings (.env)

| Setting | What it does |
|---|---|
| `WHISPER_MODEL` | `tiny` / `base` / `small` / `medium` / `large-v3`. Bigger is more accurate and slower. |
| `VIDEO_ENCODER` | `libx264` everywhere. `h264_nvenc` on an NVIDIA GPU for much faster rendering. |
| `CLIP_LLM_MODEL` | Model that picks the moments. |

## Troubleshooting

- "ffmpeg missing" chip: install ffmpeg and reopen the terminal.
- Slow transcription: use `WHISPER_MODEL=base`, or use an NVIDIA GPU (CUDA drivers installed).
- Link downloads fail: `pip install -U yt-dlp` inside the virtual environment (sites change often).
- Captions look wrong: the font is Arial. If it is missing on your system, edit `CAPTION_STYLES` in `backend/pipeline.py`.

## What is where

```
backend/pipeline.py   download, transcribe, pick moments, captions, reframe, render
backend/main.py       local server, job queue, API
backend/social.py     publishing (YouTube today)
frontend/index.html   the web page
data/                 your projects, clips and the YouTube login token (stays local)
```

## Not built yet (next steps)

- **TikTok**: its API keeps posts private until TikTok audits your app. Sending clips to your TikTok *drafts* (you press publish in the app) is the practical route.
- **Instagram Reels**: needs a Business/Creator account linked to a Facebook Page, a Meta app, and a public video URL (a free tunnel such as Cloudflare Tunnel can provide one while it uploads).
- Smoother face tracking (the crop follows the median face position for the whole clip today), filler-word and silence trimming, editable captions.
