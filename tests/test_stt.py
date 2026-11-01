import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from app.speech.stt import transcribe


class STTTests(unittest.TestCase):
    def test_transcribe_interface_structure(self):
        dummy_audio = np.zeros(16000, dtype=np.float32)
        res = transcribe(dummy_audio, language="en-IN")

        self.assertIn("text", res)
        self.assertIn("language", res)
        self.assertIn("language_probability", res)
        self.assertIsInstance(res["text"], str)

    def test_transcribe_with_mocked_whisper(self):
        dummy_audio = np.zeros(16000, dtype=np.float32)

        mock_segment = MagicMock()
        mock_segment.text = "Hello clinic"

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("app.speech.stt.get_stt_model", return_value=mock_model):
            res = transcribe(dummy_audio, language="en-IN")
            self.assertEqual(res["text"], "Hello clinic")
            self.assertEqual(res["language"], "en")
            self.assertAlmostEqual(res["language_probability"], 0.98)


if __name__ == "__main__":
    unittest.main()
