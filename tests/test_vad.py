import unittest
import numpy as np

from app.audio.vad import SileroVADDetector


class VADTests(unittest.TestCase):
    def setUp(self):
        # 16 kHz sample rate, 20ms frames = 320 samples
        self.vad = SileroVADDetector(
            sample_rate=16000,
            threshold=0.5,
            trailing_silence_ms=300.0,
            min_speech_ms=100.0,
        )

    def test_silence_does_not_trigger_speech(self):
        silence_frame = np.zeros(320, dtype=np.float32)
        for _ in range(10):  # 200 ms of silence
            res = self.vad.process_frame(silence_frame)
            self.assertFalse(res["is_speech"])
            self.assertFalse(res["speech_started"])
            self.assertFalse(res["utterance_end"])

    def test_speech_onset_and_continuation(self):
        # Sine wave representing vocalized signal (RMS ~ 0.35)
        speech_frame = (0.5 * np.sin(np.linspace(0, 50, 320))).astype(np.float32)

        speech_started_seen = False
        # Feed 10 frames = 200 ms of speech (min_speech_ms is 100 ms)
        for _ in range(10):
            res = self.vad.process_frame(speech_frame)
            if res["speech_started"]:
                speech_started_seen = True

        self.assertTrue(speech_started_seen)
        self.assertTrue(self.vad.is_speaking)

    def test_utterance_endpoint_after_trailing_silence(self):
        speech_frame = (0.5 * np.sin(np.linspace(0, 50, 320))).astype(np.float32)
        silence_frame = np.zeros(320, dtype=np.float32)

        # 1. Start speaking (150 ms)
        for _ in range(8):
            self.vad.process_frame(speech_frame)
        self.assertTrue(self.vad.is_speaking)

        # 2. Trailing silence (16 frames = 320 ms > 300 ms trailing silence threshold)
        utterance_end_seen = False
        for _ in range(16):
            res = self.vad.process_frame(silence_frame)
            if res["utterance_end"]:
                utterance_end_seen = True
                break

        self.assertTrue(utterance_end_seen, "Expected VAD to detect end of utterance after trailing silence")
        self.assertFalse(self.vad.is_speaking)


if __name__ == "__main__":
    unittest.main()
