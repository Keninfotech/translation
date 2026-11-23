@echo off
title Offline Indic Translator
cd /d "%~dp0"

echo ============================================================
echo   Starting Offline Indic Translator (HP Omen 16 RTX 4070)
echo ============================================================

:: Start Backend Server
start "Indic Translator Backend" cmd /c "run_backend.bat"

:: Wait 2 seconds for backend to initialize
timeout /t 2 /nobreak >nul

:: Start Frontend Server
start "Indic Translator Frontend" cmd /c "run_frontend.bat"

:: Wait 2 seconds and open browser
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5173

echo.
echo Application started! Both backend and frontend are running.
echo Keep the backend and frontend command windows open.
echo.
pause
