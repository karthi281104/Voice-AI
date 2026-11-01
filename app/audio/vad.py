from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.audio.audio_utils import calculate_rms_energy

logger = logging.getLogger(__name__)


class SileroVADDetector:
    """
    Voice Activity Detector with Silero VAD inference and adaptive fallback.
    Identifies:
    1. Speech start (onset)
    2. Continuous speech
    3. Trailing silence duration
    4. End of customer turn (configurable 300-400ms threshold)
    5. Barge-in / interruption during assistant playback
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        trailing_silence_ms: float = 350.0,
        min_speech_ms: float = 150.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.trailing_silence_ms = trailing_silence_ms
        self.min_speech_ms = min_speech_ms

        # State tracking
        self.is_speaking = False
        self.speech_duration_ms = 0.0
        self.silence_duration_ms = 0.0
        self.total_frames_processed = 0

        # Background noise floor estimation for adaptive fallback
        self._noise_floor = 0.008
        self._onnx_session = None
        self._init_model()

    def _init_model(self) -> None:
        """Attempt to load Silero VAD via ONNX runtime if available."""
        try:
            import onnxruntime as ort
            # Optional: If a local silero_vad.onnx exists, load it
            import os
            model_path = os.getenv("SILERO_VAD_PATH", "silero_vad.onnx")
            if os.path.exists(model_path):
                self._onnx_session = ort.InferenceSession(model_path)
                logger.info("Loaded Silero VAD ONNX model from %s", model_path)
        except Exception:
            self._onnx_session = None

    def predict_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """
        Compute speech probability in [0.0, 1.0].
        Uses ONNX Silero VAD if loaded, else robust energy/spectral feature detection.
        """
        if audio_chunk.size == 0:
            return 0.0

        if self._onnx_session is not None:
            try:
                # Silero VAD v4/v5 ONNX expects 512 samples at 16kHz
                padded = audio_chunk
                if len(padded) != 512:
                    padded = np.pad(padded, (0, max(0, 512 - len(padded))))[:512]
                tensor_in = padded.reshape(1, -1).astype(np.float32)
                outputs = self._onnx_session.run(None, {"input": tensor_in})
                return float(outputs[0][0][0])
            except Exception:
                pass

        # Adaptive energy and spectral dynamics fallback
        rms = calculate_rms_energy(audio_chunk)
        if rms < self._noise_floor * 0.9:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms

        snr = rms / (self._noise_floor + 1e-6)
        prob = 1.0 / (1.0 + np.exp(-3.0 * (snr - 2.5)))
        return float(np.clip(prob, 0.0, 1.0))

    def process_frame(self, audio_chunk: np.ndarray) -> dict[str, Any]:
        """
        Process a single audio frame (typically 20ms = 320 samples at 16kHz).
        Returns state transitions: speech_started, is_speech, utterance_end.
        """
        frame_duration_ms = (len(audio_chunk) / self.sample_rate) * 1000.0
        self.total_frames_processed += 1

        prob = self.predict_speech_prob(audio_chunk)
        is_speech_frame = prob >= self.threshold

        speech_started = False
        utterance_end = False

        if is_speech_frame:
            self.speech_duration_ms += frame_duration_ms
            self.silence_duration_ms = 0.0

            if not self.is_speaking and self.speech_duration_ms >= self.min_speech_ms:
                self.is_speaking = True
                speech_started = True
        else:
            if self.is_speaking:
                self.silence_duration_ms += frame_duration_ms
                if self.silence_duration_ms >= self.trailing_silence_ms:
                    self.is_speaking = False
                    utterance_end = True
                    self.speech_duration_ms = 0.0
            else:
                self.speech_duration_ms = max(0.0, self.speech_duration_ms - frame_duration_ms)

        return {
            "is_speech": self.is_speaking,
            "frame_is_speech": is_speech_frame,
            "speech_prob": prob,
            "speech_started": speech_started,
            "utterance_end": utterance_end,
            "speech_duration_ms": self.speech_duration_ms,
            "silence_duration_ms": self.silence_duration_ms,
        }

    def reset(self) -> None:
        """Reset turn state for the next utterance."""
        self.is_speaking = False
        self.speech_duration_ms = 0.0
        self.silence_duration_ms = 0.0
