@echo off
setlocal enabledelayedexpansion
title Offline Indic Translator - Setup

echo ============================================================
echo   Offline Indic Translator: Automated Setup (Windows 11)
echo ============================================================

cd /d "%~dp0"

echo.
echo [*] Step 1: Installing dependencies...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_dependencies.ps1"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [*] Step 2: Downloading offline AI models...
.venv\Scripts\python.exe download_models.py --all
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Model download failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [*] Step 3: Running verification diagnostics...
.venv\Scripts\python.exe verify_models.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Model verification failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   SETUP COMPLETE!
echo   Run 'start_application.bat' to launch the translator.
echo ============================================================
pause
