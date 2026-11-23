"""
Silero VAD (Voice Activity Detection) ONNX inference for real-time speech boundary segmentation.
"""

from pathlib import Path
import numpy as np
import onnxruntime as ort
from .config import VAD_PATH

class VADDetector:
    def __init__(self, model_path: Path = VAD_PATH, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.sample_rate = 16000
        self._session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def _get_session(self):
        if self._session is None:
            if not self.model_path.exists():
                raise RuntimeError(f"VAD model not found at {self.model_path}")
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=['CPUExecutionProvider'])
        return self._session

    def reset(self):
        """Resets the internal recurrent state between conversation utterances."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> float:
        """
        Processes a 512-sample (32ms) audio frame at 16kHz.
        Returns speech probability [0.0 - 1.0].
        """
        if len(chunk) != 512:
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)))
            else:
                chunk = chunk[:512]

        session = self._get_session()
        chunk_input = chunk.reshape(1, 512).astype(np.float32)
        sr_tensor = np.array(self.sample_rate, dtype=np.int64)

        try:
            out, self._state = session.run(
                None,
                {
                    'input': chunk_input,
                    'state': self._state,
                    'sr': sr_tensor
                }
            )
            return float(out[0][0])
        except Exception:
            # Fallback to simple RMS calculation if ONNX session fails
            rms = np.sqrt(np.mean(chunk ** 2))
            return 1.0 if rms > 0.01 else 0.0

    def is_speech(self, chunk: np.ndarray) -> bool:
        return self.process_chunk(chunk) >= self.threshold

vad_detector = VADDetector()
