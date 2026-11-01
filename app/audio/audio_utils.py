from __future__ import annotations

import numpy as np


def int16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Convert raw 16-bit little-endian signed PCM bytes to a float32 array in [-1.0, 1.0].
    """
    if not pcm_bytes:
        return np.empty(0, dtype=np.float32)
    # Ensure byte length is even
    truncated_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
    int16_data = np.frombuffer(pcm_bytes[:truncated_len], dtype=np.int16)
    return (int16_data.astype(np.float32) / 32768.0).astype(np.float32)


def float32_to_int16(audio: np.ndarray) -> bytes:
    """
    Convert a normalized float32 array in [-1.0, 1.0] back to 16-bit signed PCM bytes.
    Clips values to prevent int16 overflow wrap-around.
    """
    if audio.size == 0:
        return b""
    clipped = np.clip(audio, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    return int16_data.tobytes()


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Fast linear interpolation resampler from orig_sr to target_sr.
    """
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive integers")
    if orig_sr == target_sr or audio.size == 0:
        return audio
    num_target_samples = int(round(len(audio) * target_sr / orig_sr))
    if num_target_samples == 0:
        return np.empty(0, dtype=np.float32)
    orig_indices = np.linspace(0, len(audio) - 1, num=len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num=num_target_samples)
    return np.interp(target_indices, orig_indices, audio).astype(np.float32)


def calculate_rms_energy(audio: np.ndarray) -> float:
    """
    Calculate the Root Mean Square (RMS) energy of an audio signal.
    """
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def chunk_pcm_frames(pcm_bytes: bytes, frame_samples: int = 320) -> list[bytes]:
    """
    Split 16-bit PCM bytes into fixed-sized frames.
    At 16 kHz mono, 320 samples = 20 ms = 640 bytes.
    """
    frame_bytes = frame_samples * 2
    chunks = []
    for i in range(0, len(pcm_bytes), frame_bytes):
        chunk = pcm_bytes[i:i + frame_bytes]
        if len(chunk) == frame_bytes:
            chunks.append(chunk)
    return chunks
