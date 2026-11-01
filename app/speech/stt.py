from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_SIZE = os.getenv("STT_MODEL", "small")
DEVICE = os.getenv("STT_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")

_model_instance = None
_model_load_failed = False


class STTUnavailableError(RuntimeError):
    """Raised when the STT model is not installed or failed to load."""


def get_stt_model(force_reload: bool = False):
    """Lazy initialize and cache the faster-whisper model instance."""
    global _model_instance, _model_load_failed
    if force_reload:
        _model_instance = None
        _model_load_failed = False
    if _model_load_failed and not force_reload:
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
    global _model_instance
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
    try:
        segments, info = model.transcribe(
            audio,
            beam_size=1,
            language=lang_code,
            vad_filter=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except RuntimeError as r_err:
        err_str = str(r_err).lower()
        if "cublas" in err_str or "cuda" in err_str:
            logger.warning("CUDA execution failed (%s), falling back to CPU/int8", r_err)
            try:
                from faster_whisper import WhisperModel
                _model_instance = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
                segments, info = _model_instance.transcribe(
                    audio,
                    beam_size=1,
                    language=lang_code,
                    vad_filter=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
            except Exception as fallback_exc:
                logger.error("CPU/int8 STT fallback failed: %s", fallback_exc, exc_info=True)
                if raise_on_error:
                    raise STTUnavailableError("Speech recognition failed on CUDA and CPU fallbacks") from fallback_exc
                return {
                    "text": "",
                    "language": language or "en",
                    "language_probability": 0.0,
                    "error": "stt_unavailable",
                }
        else:
            if raise_on_error:
                raise STTUnavailableError("Speech recognition failed") from r_err
            logger.error("Speech recognition failed: %s", r_err, exc_info=True)
            return {
                "text": "",
                "language": language or "en",
                "language_probability": 0.0,
                "error": "stt_error",
            }

    return {
        "text": text,
        "language": getattr(info, "language", language or "en"),
        "language_probability": getattr(info, "language_probability", 1.0),
        "error": None,
    }
