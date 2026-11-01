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
            import os
            model_path = os.getenv("SILERO_VAD_PATH", "silero_vad.onnx")
            if os.path.exists(model_path):
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 1
                self._onnx_session = ort.InferenceSession(
                    model_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                # Silero v4/v5 recurrent state tensors
                self._h = np.zeros((2, 1, 64), dtype=np.float32)
                self._c = np.zeros((2, 1, 64), dtype=np.float32)
                logger.info(
                    "Loaded Silero VAD ONNX model from %s (inputs: %s)",
                    model_path,
                    [i.name for i in self._onnx_session.get_inputs()],
                )
                self._using_energy_fallback = False
            else:
                logger.warning(
                    "Silero VAD ONNX model not found at '%s'. "
                    "Using energy-based VAD fallback. Set SILERO_VAD_PATH to enable neural VAD.",
                    model_path,
                )
                self._using_energy_fallback = True
        except ImportError:
            logger.warning(
                "onnxruntime not installed. Using energy-based VAD fallback. "
                "Install with: pip install onnxruntime"
            )
            self._onnx_session = None
            self._using_energy_fallback = True
        except Exception as exc:
            logger.error(
                "Silero VAD ONNX model failed to load: %s. Using energy-based fallback.",
                exc, exc_info=True,
            )
            self._onnx_session = None
            self._using_energy_fallback = True

    def predict_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """
        Compute speech probability in [0.0, 1.0].
        Uses ONNX Silero VAD if loaded (with correct v4/v5 input tensors),
        else uses robust energy/spectral feature detection.
        """
        if audio_chunk.size == 0:
            return 0.0

        if self._onnx_session is not None:
            try:
                # Silero VAD v4/v5: expects 512 samples at 16kHz
                chunk = audio_chunk
                if len(chunk) != 512:
                    chunk = np.pad(chunk, (0, max(0, 512 - len(chunk))))[:512]
                tensor_in = chunk.reshape(1, -1).astype(np.float32)
                sr_tensor = np.array(self.sample_rate, dtype=np.int64)

                # Try v4/v5 signature: input, sr, h, c -> output, hn, cn
                ort_inputs = {
                    "input": tensor_in,
                    "sr": sr_tensor,
                    "h": self._h,
                    "c": self._c,
                }
                outputs = self._onnx_session.run(None, ort_inputs)
                prob = float(outputs[0].squeeze())
                # Update recurrent state
                self._h = outputs[1]
                self._c = outputs[2]
                return float(np.clip(prob, 0.0, 1.0))
            except Exception as exc:
                logger.debug(
                    "Silero ONNX inference failed (%s), using energy fallback", exc
                )
                # Fall through to energy fallback

        # Energy-based fallback
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
