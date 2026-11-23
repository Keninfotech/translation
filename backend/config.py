import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.getenv('MODEL_ROOT', str(ROOT / 'models')))

# STT Configuration
STT_PATH = Path(os.getenv('STT_PATH', str(MODEL_ROOT / 'whisper')))

# MT Direction Checkpoints (IndicTrans2)
MT_DIR = Path(os.getenv('MT_PATH', str(MODEL_ROOT / 'indictrans2')))
MT_PATHS = {
    'en-indic': Path(os.getenv('MT_EN_INDIC_PATH', str(MT_DIR / 'en-indic'))),
    'indic-en': Path(os.getenv('MT_INDIC_EN_PATH', str(MT_DIR / 'indic-en'))),
    'indic-indic': Path(os.getenv('MT_INDIC_INDIC_PATH', str(MT_DIR / 'indic-indic'))),
}
# Legacy single MT_PATH fallback if someone specifies a custom directory
MT_PATH = MT_PATHS['en-indic'] if not MT_DIR.is_file() else MT_DIR

# VAD Configuration
VAD_PATH = Path(os.getenv('VAD_PATH', str(MODEL_ROOT / 'vad' / 'silero_vad.onnx')))

# TTS Configuration
VOICES_DIR = Path(os.getenv('VOICES_DIR', str(MODEL_ROOT / 'voices')))
TTS_DIR = Path(os.getenv('TTS_DIR', str(MODEL_ROOT / 'tts')))

# Piper Executable: check virtualenv first, then system PATH
venv_piper = ROOT / '.venv' / 'Scripts' / 'piper.exe'
default_piper = str(venv_piper) if venv_piper.exists() else 'piper'
PIPER_EXE = os.getenv('PIPER_EXE', default_piper)

# Hardware & Inference Configuration
DEVICE = os.getenv('DEVICE', 'auto')  # 'cuda', 'cpu', or 'auto'
COMPUTE_TYPE = os.getenv('COMPUTE_TYPE', 'float16')  # 'float16', 'int8', 'float32'
MAX_AUDIO_BYTES = int(os.getenv('MAX_AUDIO_BYTES', str(25 * 1024 * 1024)))

# Offline Environment Enforcement
OFFLINE_MODE = os.getenv('OFFLINE_MODE', '0') == '1'
if OFFLINE_MODE:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
