@echo off
title Indic Translator - Backend (FastAPI)
cd /d "%~dp0"

echo [*] Launching Offline Indic Translator Backend...

:: Enforce strict offline operation
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set OFFLINE_MODE=1

call .venv\Scripts\activate.bat
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info
pause
