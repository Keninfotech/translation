# PowerShell dependency installation script for Offline Indic Translator
# Windows 11 / NVIDIA RTX 4070 Setup

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Offline Indic Translator: Automated Windows Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/6] Checking Python installation..." -ForegroundColor Yellow
$pythonExe = ""
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py -3.11"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} else {
    Write-Error "Python 3.11 not found on PATH. Please install Python 3.11 from python.org."
}
Write-Host "Using Python: $pythonExe" -ForegroundColor Green

# 2. Check Node.js and npm
Write-Host "`n[2/6] Checking Node.js and npm..." -ForegroundColor Yellow
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js not found on PATH. Please install Node.js 20+ from nodejs.org."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "npm not found on PATH."
}
$nodeVer = node --version
Write-Host "Node.js version: $nodeVer" -ForegroundColor Green

# 3. Check NVIDIA GPU and CUDA
Write-Host "`n[3/6] Checking NVIDIA GPU & CUDA capability..." -ForegroundColor Yellow
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    Write-Host "NVIDIA GPU detected. GPU acceleration enabled." -ForegroundColor Green
} else {
    Write-Host "NVIDIA GPU not detected. Application will run in CPU fallback mode." -ForegroundColor Yellow
}

# 4. Create and configure virtual environment
Write-Host "`n[4/6] Setting up Python virtual environment (.venv)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Invoke-Expression "$pythonExe -m venv .venv"
    Write-Host "Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment .venv already exists." -ForegroundColor Green
}

$venvPython = ".venv\Scripts\python.exe"
$venvPip = ".venv\Scripts\pip.exe"

# Upgrade pip
& $venvPython -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4
Write-Host "`nInstalling PyTorch with CUDA 12.4 support..." -ForegroundColor Yellow
& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install backend dependencies
Write-Host "`nInstalling backend requirements..." -ForegroundColor Yellow
& $venvPython -m pip install -r backend\requirements.txt

# 5. Setup IndicTransToolkit
Write-Host "`n[5/6] Verifying IndicTransToolkit..." -ForegroundColor Yellow
& $venvPython -c "from IndicTransToolkit import IndicProcessor; print('IndicTransToolkit ready!')"

# 6. Install frontend dependencies
Write-Host "`n[6/6] Installing frontend npm packages..." -ForegroundColor Yellow
Set-Location frontend
npm install
npm run build
Set-Location ..

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Dependencies successfully installed!" -ForegroundColor Green
Write-Host "  Next: Run 'python download_models.py' to download model files." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
