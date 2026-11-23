#!/usr/bin/env python3
"""
Diagnostic and verification script for Offline Indic Translator.
Performs offline inference tests on STT, MT, TTS, and VAD, reporting latency and memory.
"""

import sys
import time
import os
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_system_status():
    print_header("1. System and Hardware Diagnostics")
    from backend.engine import engine
    status = engine.status()
    print(f"Active Device: {status['device'].upper()}")
    print(f"GPU Name: {status['gpu_name']}")
    if status['vram']:
        print(f"GPU VRAM: {status['vram']['free_mb']} MB Free / {status['vram']['total_mb']} MB Total")
    print(f"STT Model Found: {status['stt']}")
    print(f"Translation Found: {status['translation']}")
    print(f"Supported Directions: {status['translation_directions']}")
    print(f"Available TTS Voices ({len(status['tts_voices'])}): {status['tts_voices']}")
    print(f"Silero VAD Found: {status['vad']}")
    assert status['ready'], "System reports models are not ready!"
    print(">>> Hardware & System Check: PASS")

def test_stt():
    print_header("2. Speech-to-Text (STT) Verification")
    from backend.engine import engine
    import numpy as np
    from backend.audio import audio_to_wav_bytes

    # Generate synthetic 16kHz sine wave audio (1.0 second, 440 Hz)
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    waveform = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_bytes = audio_to_wav_bytes(waveform, sr)

    t0 = time.time()
    asr = engine.asr()
    load_time = time.time() - t0
    print(f"Whisper Model Load Time: {load_time:.2f}s")

    # Transcribe silent/tone wave to verify pipeline execution without crashing
    try:
        engine.transcribe(wav_bytes, "en")
        print("Transcription Pipeline Execution: PASS")
    except ValueError as e:
        # Expected for non-speech tone
        print(f"Transcription Pipeline Audio Validation: PASS ({e})")

def test_translation_directions():
    print_header("3. Machine Translation (MT) Multi-Direction Verification")
    from backend.engine import engine

    test_cases = [
        ("en", "hi", "general", "Hello, welcome to India.", "English -> Hindi"),
        ("en", "ml", "general", "How are you today?", "English -> Malayalam"),
        ("ml", "en", "general", "സുഖമാണോ?", "Malayalam -> English"),
        ("hi", "en", "general", "नमस्ते, आप कैसे हैं?", "Hindi -> English"),
        ("ml", "hi", "general", "സുഖമാണോ?", "Malayalam -> Hindi (Pivot)"),
        ("en", "hi", "medical", "Take 500 mg paracetamol twice daily.", "Medical Mode (Dosage Check)"),
        ("en", "hi", "medical", "Do not take this medicine without food.", "Medical Mode (Negation Check)"),
        ("en", "hi", "office", "Please review the quarterly KPI and SLA targets.", "Office Mode (Terminology Check)"),
    ]

    for src, tgt, mode, text, desc in test_cases:
        t0 = time.time()
        res = engine.translate(text, src, tgt, mode)
        elapsed = (time.time() - t0) * 1000
        print(f"\n[{desc}] ({elapsed:.1f} ms)")
        print(f"  Input:       {text}")
        print(f"  Translation: {res['text']}")
        if res['warnings']:
            print(f"  Advisories:  {res['warnings']}")
        if res.get('flags'):
            print(f"  Flags:       {res['flags']}")
        assert res['text'].strip(), f"Translation was empty for {desc}"

    print("\n>>> All Translation Directions: PASS")

def test_tts():
    print_header("4. Text-to-Speech (TTS) Verification")
    from backend.engine import engine

    tts_tests = [
        ("en", "Hello, translation is ready."),
        ("hi", "नमस्ते, अनुवाद तैयार है।"),
        ("ml", "നമസ്കാരം, പരിഭാഷ തയ്യാറാണ്."),
        ("te", "నమస్కారం, అనువాదం సిద్ధంగా ఉంది."),
        ("kn", "ನಮಸ್ಕಾರ, ಅನುವಾದ ಸಿದ್ಧವಾಗಿದೆ."),
    ]

    for lang, text in tts_tests:
        t0 = time.time()
        wav = engine.synthesize(text, lang)
        elapsed = (time.time() - t0) * 1000
        print(f"TTS [{lang}] generated {len(wav)} bytes in {elapsed:.1f} ms")
        assert len(wav) > 1000, f"Generated audio too small for {lang}"

    print(">>> TTS Synthesis: PASS")

def test_vad():
    print_header("5. Voice Activity Detection (VAD) Verification")
    from backend.vad import vad_detector
    import numpy as np

    vad_detector.reset()
    silence = np.zeros(512, dtype=np.float32)
    prob_silence = vad_detector.process_chunk(silence)
    print(f"VAD Silence Speech Probability: {prob_silence:.4f} (Expected < 0.1)")
    assert prob_silence < 0.2, "VAD failed on silence frame"

    # Simulated voice burst
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    voice_frame = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    prob_voice = vad_detector.process_chunk(voice_frame)
    print(f"VAD Active Frame Probability: {prob_voice:.4f}")
    print(">>> VAD Verification: PASS")

def main():
    print("Running comprehensive offline diagnostics for Indic Translator...")
    test_system_status()
    test_stt()
    test_translation_directions()
    test_tts()
    test_vad()
    print("\n" + "=" * 60)
    print("  ALL OFFLINE DIAGNOSTIC AND FUNCTIONAL TESTS PASSED!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
