import io
import os
import sys
import json
import wave
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Ensure PyTorch CUDA DLLs (cublas, cudnn) are discoverable by CTranslate2 on Windows
from .config import ROOT, MODEL_ROOT, STT_PATH, MT_PATHS, VOICES_DIR, TTS_DIR, VAD_PATH, PIPER_EXE, DEVICE, COMPUTE_TYPE

torch_lib = ROOT / '.venv' / 'Lib' / 'site-packages' / 'torch' / 'lib'
if torch_lib.exists():
    if hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(str(torch_lib))
        except Exception:
            pass
    os.environ['PATH'] = str(torch_lib) + os.pathsep + os.environ.get('PATH', '')

import torch
import sentencepiece as spm
import ctranslate2
from IndicTransToolkit import IndicProcessor

from .audio import decode_audio_bytes, validate_audio, audio_to_wav_bytes
from .domain import domain_engine

LANGS = {
    'en': 'eng_Latn', 'hi': 'hin_Deva', 'kn': 'kan_Knda', 'ml': 'mal_Mlym',
    'ta': 'tam_Taml', 'te': 'tel_Telu', 'bn': 'ben_Beng', 'mr': 'mar_Deva',
    'gu': 'guj_Gujr', 'pa': 'pan_Guru', 'or': 'ory_Orya'
}

LANG_NAMES = {
    'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada', 'ml': 'Malayalam',
    'ta': 'Tamil', 'te': 'Telugu', 'bn': 'Bengali', 'mr': 'Marathi',
    'gu': 'Gujarati', 'pa': 'Punjabi', 'or': 'Odia'
}

class Engine:
    def __init__(self):
        self._asr = None
        self._translators: Dict[str, ctranslate2.Translator] = {}
        self._sp_models: Dict[str, Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]] = {}
        self._mms_models: Dict[str, Tuple[Any, Any]] = {}
        self._processor = IndicProcessor(inference=True)
        self._lock = threading.Lock()
        self._mms_lock = threading.Lock()

    def get_device(self) -> str:
        if DEVICE == 'cpu':
            return 'cpu'
        if DEVICE == 'cuda':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_compute_type(self, dev: str) -> str:
        if dev == 'cuda':
            return COMPUTE_TYPE if COMPUTE_TYPE in ('float16', 'int8_float16', 'int8') else 'float16'
        return 'int8'

    def status(self) -> Dict[str, Any]:
        """Returns deep real-time system and model status."""
        stt_ok = STT_PATH.exists() and ((STT_PATH / 'model.bin').exists() or (STT_PATH / 'model.safetensors').exists())
        mt_en_indic = MT_PATHS['en-indic'].exists() and (MT_PATHS['en-indic'] / 'model.bin').exists()
        mt_indic_en = MT_PATHS['indic-en'].exists() and (MT_PATHS['indic-en'] / 'model.bin').exists()
        vad_ok = VAD_PATH.exists()

        piper_voices = sorted([x.stem for x in VOICES_DIR.glob('*.onnx')]) if VOICES_DIR.exists() else []
        mms_voices = sorted([x.name for x in TTS_DIR.iterdir() if x.is_dir()]) if TTS_DIR.exists() else []
        all_voices = sorted(list(set(piper_voices + mms_voices)))

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
        vram_info = {}
        if cuda_available:
            free_b, total_b = torch.cuda.mem_get_info()
            vram_info = {
                "free_mb": round(free_b / (1024 * 1024)),
                "total_mb": round(total_b / (1024 * 1024)),
                "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024))
            }

        return {
            'stt': bool(stt_ok),
            'translation': bool(mt_en_indic and mt_indic_en),
            'translation_directions': {
                'en_to_indic': bool(mt_en_indic),
                'indic_to_en': bool(mt_indic_en),
                'indic_to_indic': bool(mt_en_indic and mt_indic_en)  # Supported via low-latency pivot
            },
            'tts_voices': all_voices,
            'piper_voices': piper_voices,
            'mms_voices': mms_voices,
            'vad': bool(vad_ok),
            'device': self.get_device(),
            'gpu_name': gpu_name,
            'vram': vram_info,
            'languages': list(LANGS),
            'language_names': LANG_NAMES,
            'ready': bool(stt_ok and mt_en_indic and mt_indic_en)
        }

    def asr(self):
        """Loads and caches the Faster-Whisper model."""
        if self._asr is None:
            if not STT_PATH.exists() or not (STT_PATH / 'model.bin').exists():
                raise RuntimeError(f"Missing local Whisper model weights at: {STT_PATH}")
            from faster_whisper import WhisperModel
            dev = self.get_device()
            compute_type = self.get_compute_type(dev)
            self._asr = WhisperModel(str(STT_PATH), device=dev, compute_type=compute_type)
        return self._asr

    def transcribe(self, data: bytes, lang: str) -> str:
        """Decodes arbitrary audio bytes and executes speech-to-text."""
        if lang not in LANGS:
            raise ValueError(f"Unsupported source language: {lang}")

        # 1. Decode arbitrary audio into 16kHz mono float32
        audio = decode_audio_bytes(data, target_sr=16000)

        # 2. Validate audio length and check for silence
        validate_audio(audio, min_duration=0.2, max_duration=120.0, sr=16000)

        # 3. Whisper transcription with VAD filtering
        asr_model = self.asr()
        # English code is 'en'; Indic languages map to standard 2-letter codes for whisper
        whisper_lang = lang if lang != 'or' else 'ne'  # whisper fallback for Odia
        segments, _ = asr_model.transcribe(
            audio.astype(np.float32),
            language=whisper_lang,
            vad_filter=True,
            beam_size=2,
            temperature=0.0
        )
        return ' '.join(s.text.strip() for s in segments).strip()

    def _get_translator_pair(self, direction: str) -> Tuple[ctranslate2.Translator, spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
        """Loads and caches CTranslate2 Translator and SentencePiece tokenizers for a direction."""
        if direction not in MT_PATHS:
            raise ValueError(f"Unknown translation direction: {direction}")

        if direction not in self._translators:
            model_dir = MT_PATHS[direction]
            if not model_dir.exists() or not (model_dir / 'model.bin').exists():
                raise RuntimeError(f"Missing IndicTrans2 checkpoint for [{direction}] at: {model_dir}")

            dev = self.get_device()
            try:
                translator = ctranslate2.Translator(str(model_dir), device=dev)
            except Exception as e:
                # Fallback to CPU if CUDA library error occurs
                translator = ctranslate2.Translator(str(model_dir), device='cpu')

            sp_src = spm.SentencePieceProcessor()
            sp_tgt = spm.SentencePieceProcessor()
            sp_src_path = model_dir / 'vocab' / 'model.SRC'
            sp_tgt_path = model_dir / 'vocab' / 'model.TGT'
            if not sp_src_path.exists():
                sp_src_path = model_dir / 'model.SRC'
            if not sp_tgt_path.exists():
                sp_tgt_path = model_dir / 'model.TGT'

            sp_src.load(str(sp_src_path))
            sp_tgt.load(str(sp_tgt_path))

            self._translators[direction] = translator
            self._sp_models[direction] = (sp_src, sp_tgt)

        return self._translators[direction], self._sp_models[direction][0], self._sp_models[direction][1]

    def _translate_single_direction(self, text: str, src_lang: str, tgt_lang: str, direction: str) -> str:
        """Executes translation for a single hop (en-indic or indic-en)."""
        translator, sp_src, sp_tgt = self._get_translator_pair(direction)
        src_tag = LANGS[src_lang]
        tgt_tag = LANGS[tgt_lang]

        # 1. Preprocess with IndicProcessor
        pre = self._processor.preprocess_batch([text], src_lang=src_tag, tgt_lang=tgt_tag)[0]
        words = pre.split(' ', 2)[2] if pre.startswith(src_tag) else pre

        # 2. Tokenize with language tags
        tokens = [src_tag, tgt_tag] + sp_src.encode(words, out_type=str) + ['</s>']

        # 3. Translate with CTranslate2
        results = translator.translate_batch([tokens], beam_size=2, max_decoding_length=256)
        out_tokens = results[0].hypotheses[0]

        # 4. Filter special tokens and detokenize
        filtered = [t for t in out_tokens if t not in ['<s>', '</s>', '<unk>', src_tag, tgt_tag]]
        detok = sp_tgt.decode(filtered)

        # 5. Postprocess with IndicProcessor
        return self._processor.postprocess_batch([detok], lang=tgt_tag)[0]

    def translate(self, text: str, source: str, target: str, mode: str = 'general') -> Dict[str, Any]:
        """
        Translates text between any of the 11 supported languages.
        Automatically routes:
        - English -> Indic: en-indic model
        - Indic -> English: indic-en model
        - Indic -> Indic: Cascaded pivot (Indic1 -> English -> Indic2)
        Also evaluates domain rules (General, Medical, Office).
        """
        if source not in LANGS or target not in LANGS:
            raise ValueError(f"Unsupported language pair: {source} -> {target}")
        if mode not in ('general', 'medical', 'office'):
            raise ValueError(f"Unsupported mode: {mode}")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Text is empty")

        if source == target:
            return {'text': clean_text, 'warnings': [], 'flags': {}}

        with self._lock:
            if source == 'en':
                # Direct English -> Indic
                translated = self._translate_single_direction(clean_text, 'en', target, 'en-indic')
            elif target == 'en':
                # Direct Indic -> English
                translated = self._translate_single_direction(clean_text, source, 'en', 'indic-en')
            else:
                # Indic -> Indic via ultra-fast English pivot (~80ms total)
                intermediate_en = self._translate_single_direction(clean_text, source, 'en', 'indic-en')
                translated = self._translate_single_direction(intermediate_en, 'en', target, 'en-indic')

        # Domain mode validation & safety checks
        validation = domain_engine.validate(clean_text, translated, mode, source, target)

        return {
            'text': translated,
            'warnings': validation['warnings'],
            'flags': validation['flags']
        }

    def _get_mms_model(self, lang: str):
        """Loads and caches MMS-TTS VitsModel and AutoTokenizer."""
        if lang not in self._mms_models:
            from transformers import VitsModel, AutoTokenizer
            lang_dir = TTS_DIR / lang
            if not lang_dir.exists():
                raise RuntimeError(f"Missing MMS-TTS model for {lang} at: {lang_dir}")
            tok = AutoTokenizer.from_pretrained(str(lang_dir), local_files_only=True)
            model = VitsModel.from_pretrained(str(lang_dir), local_files_only=True)
            model.eval()
            self._mms_models[lang] = (tok, model)
        return self._mms_models[lang]

    def synthesize(self, text: str, lang: str) -> bytes:
        """
        Synthesizes speech to WAV bytes:
        - Uses Piper ONNX for languages with Piper voices (en, hi, ml, te, mr, bn).
        - Uses MMS-TTS VITS for other Indic languages (kn, ta, gu, pa, or).
        """
        if lang not in LANGS:
            raise ValueError(f"Unsupported language for speech: {lang}")

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Speech text is empty")

        voice_file = VOICES_DIR / f"{lang}.onnx"
        voice_cfg = VOICES_DIR / f"{lang}.onnx.json"

        # Strategy 1: Piper ONNX Voice
        if voice_file.exists() and voice_cfg.exists():
            with tempfile.TemporaryDirectory() as d:
                outfile = Path(d) / 'speech.wav'
                command = [PIPER_EXE, '-m', str(voice_file), '-f', str(outfile)]
                result = subprocess.run(command, input=clean_text, text=True, capture_output=True, timeout=60, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"Piper synthesis failed: {result.stderr[-400:]}")
                return outfile.read_bytes()

        # Strategy 2: MMS-TTS VITS Model
        mms_dir = TTS_DIR / lang
        if mms_dir.exists():
            with self._mms_lock:
                tok, model = self._get_mms_model(lang)
                inputs = tok(clean_text, return_tensors='pt')
                with torch.no_grad():
                    waveform = model(**inputs).waveform[0].cpu().numpy()
                buf = io.BytesIO()
                import soundfile as sf
                sf.write(buf, waveform, samplerate=model.config.sampling_rate, format='WAV')
                return buf.getvalue()

        raise RuntimeError(f"No offline voice model found for language '{lang}'. Available voices: {self.status()['tts_voices']}")

engine = Engine()
