import unittest
import numpy as np

from app.audio.audio_utils import (
    int16_to_float32,
    float32_to_int16,
    resample_audio,
    calculate_rms_energy,
    chunk_pcm_frames,
)


class AudioUtilsTests(unittest.TestCase):
    def test_int16_to_float32_conversion(self):
        # 0, max, min in 16-bit signed
        int16_arr = np.array([0, 32767, -32768], dtype=np.int16)
        raw_bytes = int16_arr.tobytes()
        float_arr = int16_to_float32(raw_bytes)

        self.assertEqual(len(float_arr), 3)
        self.assertAlmostEqual(float_arr[0], 0.0, places=4)
        self.assertAlmostEqual(float_arr[1], 32767 / 32768, places=4)
        self.assertAlmostEqual(float_arr[2], -1.0, places=4)

    def test_float32_to_int16_roundtrip(self):
        original = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        pcm_bytes = float32_to_int16(original)
        recovered = int16_to_float32(pcm_bytes)

        self.assertEqual(len(recovered), len(original))
        for orig, rec in zip(original, recovered):
            self.assertAlmostEqual(orig, rec, delta=0.001)

    def test_resample_audio_ratio(self):
        # 48 kHz to 16 kHz should reduce length by factor of 3
        duration_sec = 0.5
        orig_sr = 48000
        target_sr = 16000
        orig_samples = int(orig_sr * duration_sec)
        audio_48k = np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, orig_samples)).astype(np.float32)

        resampled = resample_audio(audio_48k, orig_sr, target_sr)
        expected_len = int(target_sr * duration_sec)
        self.assertAlmostEqual(len(resampled), expected_len, delta=2)

    def test_calculate_rms_energy(self):
        silence = np.zeros(1600, dtype=np.float32)
        self.assertEqual(calculate_rms_energy(silence), 0.0)

        sine = np.sin(np.linspace(0, 100, 1600)).astype(np.float32)
        rms = calculate_rms_energy(sine)
        self.assertGreater(rms, 0.6)
        self.assertLess(rms, 0.8)

    def test_chunk_pcm_frames(self):
        # 1600 bytes = 800 samples. Split into 320 samples (640 bytes)
        raw = b"\x00" * 1600
        chunks = chunk_pcm_frames(raw, frame_samples=320)
        # 1600 / 640 = 2 full chunks (320 bytes leftover discarded)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 640)


if __name__ == "__main__":
    unittest.main()
