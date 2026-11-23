# Installed Offline Models & Routing Architecture

This directory houses the offline AI model weights for the Indic Translator on Windows 11.

## Directory Structure

```
models/
├── whisper/                       # Faster-Whisper (CTranslate2) Multilingual STT Checkpoint
│   ├── model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── vocabulary.txt
│
├── indictrans2/                   # IndicTrans2 Directional Checkpoints (CTranslate2 INT8)
│   ├── en-indic/                  # English -> 22 Indic Languages
│   │   ├── model.bin
│   │   ├── config.json
│   │   ├── source_vocabulary.txt
│   │   ├── target_vocabulary.txt
│   │   └── vocab/ (SentencePiece model.SRC, model.TGT)
│   └── indic-en/                  # 22 Indic Languages -> English
│       ├── model.bin
│       ├── config.json
│       ├── source_vocabulary.txt
│       ├── target_vocabulary.txt
│       └── vocab/ (SentencePiece model.SRC, model.TGT)
│
├── vad/                           # Voice Activity Detection
│   └── silero_vad.onnx            # Silero VAD v5 ONNX (2.3 MB)
│
├── voices/                        # Piper TTS Voice Models (ONNX)
│   ├── en.onnx & en.onnx.json     # English (en_US-lessac-medium)
│   ├── hi.onnx & hi.onnx.json     # Hindi (hi_IN-pratham-medium)
│   ├── ml.onnx & ml.onnx.json     # Malayalam (ml_IN-arjun-medium)
│   ├── te.onnx & te.onnx.json     # Telugu (te_IN-padmavathi-medium)
│   ├── mr.onnx & mr.onnx.json     # Marathi (mr_IN-google-medium)
│   └── bn.onnx & bn.onnx.json     # Bengali (bn_BD-google-medium)
│
└── tts/                           # MMS-TTS VITS Checkpoints (Transformers)
    ├── kn/                        # Kannada (facebook/mms-tts-kan)
    ├── ta/                        # Tamil (facebook/mms-tts-tam)
    ├── gu/                        # Gujarati (facebook/mms-tts-guj)
    ├── pa/                        # Punjabi (facebook/mms-tts-pan)
    └── or/                        # Odia (facebook/mms-tts-ory)
```

## Model Routing & VRAM Optimization (RTX 4070 8 GB)

1. **Speech-to-Text (STT)**:
   - Checkpoint: `Systran/faster-whisper-small` (Multilingual)
   - Runtime: CTranslate2 FP16 on CUDA
   - Memory footprint: ~500 MB VRAM

2. **Machine Translation (MT)**:
   - Checkpoints: `AI-pro11/indictrans2-*-ct2-int8` (AI4Bharat distilled)
   - Runtime: CTranslate2 INT8 with SentencePiece & pure Python IndicProcessor
   - Direction routing:
     - `English -> Indic`: direct `en-indic` (~100 ms)
     - `Indic -> English`: direct `indic-en` (~60 ms)
     - `Indic -> Indic`: cascaded English pivot (`Indic1 -> English -> Indic2`) (~110 ms)
   - Memory footprint: ~250 MB VRAM per model (~500 MB total)

3. **Text-to-Speech (TTS)**:
   - Checkpoints:
     - High-speed Piper ONNX models for `en`, `hi`, `ml`, `te`, `mr`, `bn`
     - Compact VitsModel checkpoints for `kn`, `ta`, `gu`, `pa`, `or`
   - Memory footprint: ~100 - 300 MB

**Total Combined Active VRAM:** Under 1.8 GB out of 8.0 GB available on RTX 4070. Leaves ample headroom for OS and Chrome.
