#!/usr/bin/env python3
"""
Model download utility for Offline Indic Translator.
Downloads verified open-source models for STT, MT, TTS, and VAD directly into local directories.
"""

import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"

def log(msg: str):
    print(f"[ModelDownloader] {msg}", flush=True)

def download_vad():
    log("Checking Silero VAD...")
    vad_dir = MODEL_ROOT / "vad"
    vad_dir.mkdir(parents=True, exist_ok=True)
    vad_file = vad_dir / "silero_vad.onnx"
    if vad_file.exists() and vad_file.stat().st_size > 1_000_000:
        log(f"Silero VAD already downloaded ({vad_file.stat().st_size} bytes).")
        return
    import urllib.request
    url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    log(f"Downloading Silero VAD from {url}...")
    urllib.request.urlretrieve(url, str(vad_file))
    log("Silero VAD download complete.")

def download_stt(model_id: str = "Systran/faster-whisper-small"):
    log(f"Downloading STT model ({model_id}) into models/whisper...")
    stt_dir = MODEL_ROOT / "whisper"
    stt_dir.mkdir(parents=True, exist_ok=True)
    if (stt_dir / "model.bin").exists() and (stt_dir / "config.json").exists():
        log(f"STT model already present at {stt_dir}.")
        return
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_id,
        local_dir=str(stt_dir),
        local_dir_use_symlinks=False,
        max_workers=4,
    )
    log("STT model download complete.")

def download_mt(directions=None):
    if directions is None:
        directions = ["en-indic", "indic-en", "indic-indic"]

    repo_map = {
        "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
        "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
        "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
    }

    from huggingface_hub import snapshot_download

    for direction in directions:
        repo_id = repo_map.get(direction)
        if not repo_id:
            continue
        mt_dir = MODEL_ROOT / "indictrans2" / direction
        mt_dir.mkdir(parents=True, exist_ok=True)
        if (mt_dir / "config.json").exists() and (mt_dir / "model.safetensors").exists():
            log(f"IndicTrans2 [{direction}] already present at {mt_dir}.")
            continue
        log(f"Downloading IndicTrans2 [{direction}] from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(mt_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.pt"],  # Prefer safetensors to save bandwidth
            max_workers=4,
        )
        log(f"IndicTrans2 [{direction}] download complete.")

def download_piper_voices(langs=None):
    if langs is None:
        langs = ["en", "hi", "ml", "te", "mr", "bn"]

    voices_dir = MODEL_ROOT / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)

    piper_map = {
        "en": ("en/en_US/lessac/medium/en_US-lessac-medium.onnx", "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"),
        "hi": ("hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx", "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json"),
        "ml": ("ml/ml_IN/arjun/medium/ml_IN-arjun-medium.onnx", "ml/ml_IN/arjun/medium/ml_IN-arjun-medium.onnx.json"),
        "te": ("te/te_IN/padmavathi/medium/te_IN-padmavathi-medium.onnx", "te/te_IN/padmavathi/medium/te_IN-padmavathi-medium.onnx.json"),
        "mr": ("mr/mr_IN/google/medium/mr_IN-google-medium.onnx", "mr/mr_IN/google/medium/mr_IN-google-medium.onnx.json"),
        "bn": ("bn/bn_BD/google/medium/bn_BD-google-medium.onnx", "bn/bn_BD/google/medium/bn_BD-google-medium.onnx.json"),
    }

    import urllib.request
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"

    for lang in langs:
        if lang not in piper_map:
            continue
        onnx_rel, json_rel = piper_map[lang]
        dest_onnx = voices_dir / f"{lang}.onnx"
        dest_json = voices_dir / f"{lang}.onnx.json"

        if not dest_onnx.exists():
            log(f"Downloading Piper voice for [{lang}]...")
            urllib.request.urlretrieve(base_url + onnx_rel, str(dest_onnx))
        if not dest_json.exists():
            urllib.request.urlretrieve(base_url + json_rel, str(dest_json))
        log(f"Piper voice for [{lang}] ready.")

def download_mms_tts(langs=None):
    if langs is None:
        langs = ["kn", "ta", "gu", "pa", "or"]

    mms_map = {
        "kn": "facebook/mms-tts-kan",
        "ta": "facebook/mms-tts-tam",
        "gu": "facebook/mms-tts-guj",
        "pa": "facebook/mms-tts-pan",
        "or": "facebook/mms-tts-ori",
    }

    from huggingface_hub import snapshot_download

    for lang in langs:
        repo_id = mms_map.get(lang)
        if not repo_id:
            continue
        tts_dir = MODEL_ROOT / "tts" / lang
        tts_dir.mkdir(parents=True, exist_ok=True)
        if (tts_dir / "config.json").exists() and (tts_dir / "vocab.json").exists():
            log(f"MMS-TTS model for [{lang}] already present at {tts_dir}.")
            continue
        log(f"Downloading MMS-TTS model for [{lang}] from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(tts_dir),
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        log(f"MMS-TTS model for [{lang}] ready.")

def main():
    parser = argparse.ArgumentParser(description="Download offline models for Indic Translator")
    parser.add_argument("--all", action="store_true", help="Download all models (STT, MT, TTS, VAD)")
    parser.add_argument("--vad", action="store_true", help="Download Silero VAD")
    parser.add_argument("--stt", action="store_true", help="Download Faster-Whisper STT model")
    parser.add_argument("--mt", action="store_true", help="Download IndicTrans2 MT models")
    parser.add_argument("--tts", action="store_true", help="Download Piper and MMS TTS voices")
    parser.add_argument("--stt-model", default="Systran/faster-whisper-small", help="Hugging Face model ID for STT")
    args = parser.parse_args()

    if not (args.all or args.vad or args.stt or args.mt or args.tts):
        args.all = True

    log("Starting model download process...")
    if args.all or args.vad:
        download_vad()
    if args.all or args.stt:
        download_stt(args.stt_model)
    if args.all or args.mt:
        download_mt()
    if args.all or args.tts:
        download_piper_voices()
        download_mms_tts()
    log("All requested models downloaded successfully.")

if __name__ == "__main__":
    main()
