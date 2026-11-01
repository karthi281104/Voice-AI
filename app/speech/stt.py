from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_SIZE = os.getenv("STT_MODEL", "small")
DEVICE = os.getenv("STT_DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16")

_model_instance = None
_model_load_failed = False


class STTUnavailableError(RuntimeError):
    """Raised when the STT model is not installed or failed to load."""


def get_stt_model():
    """Lazy initialize and cache the faster-whisper model instance."""
    global _model_instance, _model_load_failed
    if _model_load_failed:
        return None
    if _model_instance is None:
        try:
            from faster_whisper import WhisperModel
            try:
                _model_instance = WhisperModel(
                    MODEL_SIZE,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE,
                )
                logger.info(
                    "Initialized faster-whisper (%s) on %s (%s)",
                    MODEL_SIZE, DEVICE, COMPUTE_TYPE,
                )
            except Exception as cuda_exc:
                logger.warning(
                    "faster-whisper failed on %s/%s, falling back to CPU/int8: %s",
                    DEVICE, COMPUTE_TYPE, cuda_exc,
                )
                _model_instance = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
                logger.info("faster-whisper loaded on CPU (int8 fallback)")
        except ImportError:
            _model_load_failed = True
            logger.warning(
                "faster-whisper is not installed. STT is unavailable. "
                "Install with: pip install faster-whisper"
            )
            return None
        except Exception as exc:
            _model_load_failed = True
            logger.error("faster-whisper failed to load: %s", exc, exc_info=True)
            return None
    return _model_instance


def transcribe(
    audio: np.ndarray | str,
    language: str | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """
    Transcribe a 16 kHz audio buffer (float32 numpy array) or audio path using faster-whisper.
    Returns:
        {
            "text": str,
            "language": str,
            "language_probability": float,
            "error": str | None
        }
    """
    model = get_stt_model()
    if model is None:
        if raise_on_error:
            raise STTUnavailableError("faster-whisper is not installed or failed to load")
        return {
            "text": "",
            "language": language or "en",
            "language_probability": 0.0,
            "error": "stt_unavailable",
        }

    lang_code = language.split("-")[0] if language else None
    segments, info = model.transcribe(
        audio,
        beam_size=1,
        language=lang_code,
        vad_filter=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "language_probability": getattr(info, "language_probability", 1.0),
        "error": None,
    }

