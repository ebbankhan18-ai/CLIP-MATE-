@echo off
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
if not exist .env copy .env.example .env >nul
start "" http://localhost:8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
