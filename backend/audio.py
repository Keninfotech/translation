"""
Audio decoding, normalization, and DSP pipeline for Offline Indic Translator.
Supports in-memory decoding of browser WebM/Opus, WAV, MP3, OGG, and FLAC to 16 kHz mono float32.
"""

import io
import wave
import numpy as np
import av
import soundfile as sf
from typing import Tuple

def decode_audio_bytes(data: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Decodes arbitrary audio bytes (including WebM/Opus from browser MediaRecorder)
    into a 1D float32 NumPy array at target_sr (16000 Hz) mono.
    """
    if not data or len(data) < 100:
        raise ValueError("Audio data is too short or empty")

    # Strategy 1: In-memory PyAV decoding (handles WebM, Opus, Ogg, MP3, AAC, WAV)
    try:
        buf = io.BytesIO(data)
        container = av.open(buf)
        if not container.streams.audio:
            raise ValueError("No audio stream found in container")

        audio_stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='flt', layout='mono', rate=target_sr)
        frames = []

        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            for r in resampled_frames:
                frames.append(r.to_ndarray()[0])  # mono channel

        # Flush any remaining frames in resampler
        resampled_frames = resampler.resample(None)
        if resampled_frames:
            for r in resampled_frames:
                frames.append(r.to_ndarray()[0])

        if frames:
            audio = np.concatenate(frames).astype(np.float32)
            # Clip between -1.0 and 1.0
            return np.clip(audio, -1.0, 1.0)
    except Exception as av_err:
        # Strategy 2: Fallback to SoundFile
        try:
            buf = io.BytesIO(data)
            audio, sr = sf.read(buf, dtype='float32')
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                import scipy.signal
                num_samples = int(len(audio) * float(target_sr) / sr)
                audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
            return np.clip(audio, -1.0, 1.0)
        except Exception as sf_err:
            raise ValueError(f"Failed to decode audio: av_error={av_err}, sf_error={sf_err}") from av_err

def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Converts a float32 NumPy array into standard 16-bit PCM WAV bytes."""
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()

def is_silent(audio: np.ndarray, rms_threshold: float = 0.005) -> bool:
    """Calculates RMS energy to detect near-silent audio recordings."""
    if len(audio) == 0:
        return True
    rms = np.sqrt(np.mean(audio ** 2))
    return bool(rms < rms_threshold)

def validate_audio(audio: np.ndarray, min_duration: float = 0.2, max_duration: float = 120.0, sr: int = 16000):
    """Validates audio length and properties."""
    duration = len(audio) / float(sr)
    if duration < min_duration:
        raise ValueError(f"Recording is too short ({duration:.2f}s). Minimum is {min_duration:.1f}s.")
    if duration > max_duration:
        raise ValueError(f"Recording exceeds maximum duration ({duration:.2f}s > {max_duration:.1f}s).")
    if is_silent(audio):
        raise ValueError("Audio is silent or volume is too low to transcribe.")
