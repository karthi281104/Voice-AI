from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """Abstract interface for streaming text-to-speech providers."""

    @abstractmethod
    async def synthesize_stream(
        self, text: str, language: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Yield audio chunks (e.g. PCM / MP3) as they become available.
        """
        raise NotImplementedError


class BrowserFallbackTTSProvider(TTSProvider):
    """
    Default milestone provider: delegates audio synthesis to the browser's
    native speech synthesis API via WebSocket events while server-side
    open-source Tamil/English streaming TTS is being evaluated.
    """

    async def synthesize_stream(
        self, text: str, language: str
    ) -> AsyncGenerator[bytes, None]:
        # Browser handles synthesis locally; yields empty audio chunks
        if False:
            yield b""


class MockStreamingTTSProvider(TTSProvider):
    """Generates synthetic 16 kHz PCM frames for testing audio streaming pipelines."""

    async def synthesize_stream(
        self, text: str, language: str
    ) -> AsyncGenerator[bytes, None]:
        import numpy as np
        # 16000 samples/sec * 0.1 sec = 1600 samples per chunk
        for _ in range(3):
            samples = np.zeros(1600, dtype=np.int16)
            yield samples.tobytes()


def get_tts_provider() -> TTSProvider:
    return BrowserFallbackTTSProvider()
