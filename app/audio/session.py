from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

import numpy as np

from app.audio.audio_utils import int16_to_float32
from app.audio.vad import SileroVADDetector

logger = logging.getLogger(__name__)


class VoiceSessionState:
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"


class VoiceSession:
    """
    Stateful real-time audio session for a connected WebSocket client.
    Handles:
    - Frame-level PCM buffering
    - Speech onset pre-roll buffer retention
    - Silero VAD state machine
    - Customer barge-in / interruption of AI generation & playback
    - Conversation history
    """

    def __init__(
        self,
        session_id: str,
        language: str = "en-IN",
        trailing_silence_ms: float = 350.0,
    ) -> None:
        self.session_id = session_id
        self.language = language
        self.state = VoiceSessionState.LISTENING

        self.vad = SileroVADDetector(
            sample_rate=16000,
            trailing_silence_ms=trailing_silence_ms,
        )

        # Pre-roll ring buffer (retains ~200ms = 10 frames of 20ms)
        self._preroll_buffer: deque[np.ndarray] = deque(maxlen=10)
        self._utterance_frames: list[np.ndarray] = []

        # Interruption & cancellation
        self.active_generation_task: asyncio.Task | None = None
        self.history: list[dict[str, str]] = []

    def set_language(self, language: str) -> None:
        if language in ("en-IN", "ta-IN"):
            self.language = language

    def interrupt(self, reason: str = "User speech detected") -> bool:
        """
        Immediately cancel any active generation/playback task.
        Returns True if session was in an active state (SPEAKING/PROCESSING/task running) that was interrupted.
        """
        was_active = (self.state in (VoiceSessionState.SPEAKING, VoiceSessionState.PROCESSING))
        if self.active_generation_task is not None and not self.active_generation_task.done():
            self.active_generation_task.cancel()
            self.active_generation_task = None
            was_active = True
            logger.info("Session %s interrupted (%s): generation cancelled", self.session_id, reason)

        self.state = VoiceSessionState.LISTENING
        return was_active

    def add_audio_frame(self, pcm_bytes: bytes) -> dict[str, Any]:
        """
        Process an incoming 16 kHz mono 16-bit PCM frame.
        Returns state transitions:
            {
                "speech_started": bool,
                "is_speaking": bool,
                "utterance_ready": bool,
                "interrupted": bool,
                "audio": np.ndarray | None
            }
        """
        audio_frame = int16_to_float32(pcm_bytes)
        if audio_frame.size == 0:
            return {
                "speech_started": False,
                "is_speaking": False,
                "utterance_ready": False,
                "interrupted": False,
                "audio": None,
            }

        vad_result = self.vad.process_frame(audio_frame)
        interrupted = False

        # Barge-in check: If AI is speaking/processing and customer starts speaking
        if self.state in (VoiceSessionState.SPEAKING, VoiceSessionState.PROCESSING):
            if vad_result["speech_started"] or (vad_result["is_speech"] and vad_result["speech_duration_ms"] > 100):
                interrupted = self.interrupt("Barge-in voice detected")

        if vad_result["speech_started"]:
            self.state = VoiceSessionState.LISTENING
            # Transfer pre-roll audio so the start of words is not clipped
            self._utterance_frames = list(self._preroll_buffer)
            self._utterance_frames.append(audio_frame)
            return {
                "speech_started": True,
                "is_speaking": True,
                "utterance_ready": False,
                "interrupted": interrupted,
                "audio": None,
            }

        if vad_result["is_speech"]:
            self._utterance_frames.append(audio_frame)
            return {
                "speech_started": False,
                "is_speaking": True,
                "utterance_ready": False,
                "interrupted": interrupted,
                "audio": None,
            }

        # User is not speaking right now
        self._preroll_buffer.append(audio_frame)

        if vad_result["utterance_end"] and len(self._utterance_frames) > 0:
            full_utterance = np.concatenate(self._utterance_frames)
            self._utterance_frames.clear()
            self.state = VoiceSessionState.PROCESSING
            return {
                "speech_started": False,
                "is_speaking": False,
                "utterance_ready": True,
                "interrupted": interrupted,
                "audio": full_utterance,
            }

        return {
            "speech_started": False,
            "is_speaking": False,
            "utterance_ready": False,
            "interrupted": interrupted,
            "audio": None,
        }

    def reset_utterance(self) -> None:
        self._utterance_frames.clear()
        self._preroll_buffer.clear()
        self.vad.reset()
        self.state = VoiceSessionState.LISTENING


# Active session registry
_sessions: dict[str, VoiceSession] = {}


def get_or_create_session(session_id: str, language: str = "en-IN") -> VoiceSession:
    if session_id not in _sessions:
        _sessions[session_id] = VoiceSession(session_id=session_id, language=language)
    return _sessions[session_id]


def remove_session(session_id: str) -> None:
    session = _sessions.pop(session_id, None)
    if session:
        session.interrupt("Session closed")
