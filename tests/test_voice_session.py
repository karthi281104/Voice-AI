import unittest
import numpy as np

from app.audio.session import VoiceSession, VoiceSessionState, get_or_create_session, remove_session
from app.audio.audio_utils import float32_to_int16


class VoiceSessionTests(unittest.TestCase):
    def setUp(self):
        self.session = VoiceSession("test-session-123", language="en-IN", trailing_silence_ms=200.0)

    def test_session_initial_state(self):
        self.assertEqual(self.session.state, VoiceSessionState.LISTENING)
        self.assertEqual(self.session.language, "en-IN")

    def test_preroll_and_utterance_flow(self):
        silence_frame = float32_to_int16(np.zeros(320, dtype=np.float32))
        speech_frame = float32_to_int16((0.5 * np.sin(np.linspace(0, 50, 320))).astype(np.float32))

        # 1. Send 5 frames of silence (populates pre-roll buffer)
        for _ in range(5):
            res = self.session.add_audio_frame(silence_frame)
            self.assertFalse(res["speech_started"])
            self.assertFalse(res["is_speaking"])

        # 2. Send 8 speech frames (triggers speech_started)
        speech_started_triggered = False
        for _ in range(8):
            res = self.session.add_audio_frame(speech_frame)
            if res["speech_started"]:
                speech_started_triggered = True

        self.assertTrue(speech_started_triggered)

        # 3. Send 12 silence frames (240ms > 200ms trailing silence)
        utterance_ready = False
        final_audio = None
        for _ in range(12):
            res = self.session.add_audio_frame(silence_frame)
            if res["utterance_ready"]:
                utterance_ready = True
                final_audio = res["audio"]
                break

        self.assertTrue(utterance_ready)
        self.assertIsNotNone(final_audio)
        self.assertGreater(len(final_audio), 320 * 8)  # Includes pre-roll + speech frames

    def test_barge_in_interruption_detection(self):
        # Set session to SPEAKING state (AI is currently answering)
        self.session.state = VoiceSessionState.SPEAKING

        speech_frame = float32_to_int16((0.6 * np.sin(np.linspace(0, 50, 320))).astype(np.float32))

        interrupted = False
        for _ in range(8):
            res = self.session.add_audio_frame(speech_frame)
            if res["interrupted"]:
                interrupted = True
                break

        self.assertTrue(interrupted, "Expected barge-in to trigger interruption when customer speaks during AI playback")
        self.assertEqual(self.session.state, VoiceSessionState.LISTENING)

    def test_session_lifecycle(self):
        s1 = get_or_create_session("lifecycle-test")
        self.assertIsNotNone(s1)
        remove_session("lifecycle-test")


if __name__ == "__main__":
    unittest.main()
