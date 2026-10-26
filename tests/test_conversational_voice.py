import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Use an isolated test database
tmp = tempfile.TemporaryDirectory()
os.environ["VOICEDESK_DB"] = str(Path(tmp.name) / "test_stage2.db")

from fastapi.testclient import TestClient
from app.main import app, init_db

init_db()
client = TestClient(app)


class ConversationalVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = {
            "name": "Metro Dental Clinic",
            "description": "State-of-the-art dental clinic.",
            "address": "12 Mount Road, Chennai",
            "hours": "9:00 AM to 7:00 PM, Mon-Sat",
            "default_language": "en-IN",
            "services": [
                {
                    "name": "Teeth Whitening",
                    "price": "₹3500",
                    "duration": "60 minutes",
                    "description": "Laser teeth whitening treatment."
                }
            ],
            "doctors": [
                {
                    "name": "Dr. Sarah",
                    "specialty": "Cosmetic Dentist",
                    "availability": "Mon-Fri, 10 AM to 4 PM"
                }
            ],
            "faqs": [
                {
                    "question": "Is teeth whitening painful?",
                    "answer": "No, it is a completely pain-free procedure."
                }
            ]
        }
        res = client.put("/api/business", json=cls.profile)
        assert res.status_code == 200

    def test_chat_response_includes_response_time_ms(self):
        """Verifies that the server turnaround latency is measured and exposed."""
        res = client.post("/api/chat", json={
            "message": "What are your hours?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("response_time_ms", data)
        self.assertIsInstance(data["response_time_ms"], (int, float))
        self.assertGreaterEqual(data["response_time_ms"], 0.0)

    def test_streaming_endpoint_returns_tokens_and_done_event(self):
        """Verifies /api/chat/stream yields ndjson tokens and a final done event."""
        res = client.post("/api/chat/stream", json={
            "message": "How much is Teeth Whitening?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        lines = [line.strip() for line in res.text.split("\n") if line.strip()]
        self.assertGreater(len(lines), 1, "Expected multiple streamed chunks")

        tokens = []
        done_event = None
        for line in lines:
            event = json.loads(line)
            if event["type"] == "token":
                tokens.append(event["content"])
            elif event["type"] == "done":
                done_event = event

        full_streamed_text = "".join(tokens)
        self.assertIsNotNone(done_event, "Streaming did not yield a 'done' event")
        self.assertIn("response_time_ms", done_event)
        self.assertIn("₹3500", full_streamed_text)
        self.assertIn(done_event["engine"], ["llm", "rule_based"])

    def test_streaming_fallback_when_llm_is_unavailable(self):
        """Verifies /api/chat/stream gracefully falls back to rule engine if LLM fails."""
        with patch("httpx.get", side_effect=Exception("Ollama unreachable")):
            res = client.post("/api/chat/stream", json={
                "message": "When is Dr. Sarah available?",
                "language": "en-IN"
            })
            self.assertEqual(res.status_code, 200)
            lines = [line.strip() for line in res.text.split("\n") if line.strip()]
            events = [json.loads(line) for line in lines]
            done_event = next(e for e in events if e["type"] == "done")
            self.assertEqual(done_event["engine"], "rule_based")
            self.assertIn("Dr. Sarah", done_event["answer"])
            self.assertIn("10 AM", done_event["answer"])

    def test_streaming_with_conversation_history(self):
        """Verifies /api/chat/stream resolves pronouns using multi-turn conversation history."""
        history = [
            {"role": "user", "content": "Tell me about Dr. Sarah."},
            {"role": "assistant", "content": "Dr. Sarah is our Cosmetic Dentist."}
        ]
        res = client.post("/api/chat/stream", json={
            "message": "What days is she available?",
            "language": "en-IN",
            "history": history
        })
        self.assertEqual(res.status_code, 200)
        lines = [line.strip() for line in res.text.split("\n") if line.strip()]
        done_event = next(json.loads(line) for line in lines if json.loads(line)["type"] == "done")
        self.assertTrue(
            "Mon-Fri" in done_event["answer"] or "10 AM" in done_event["answer"] or "Dr. Sarah" in done_event["answer"],
            f"Expected doctor availability in streamed response, got: {done_event['answer']}"
        )

    def test_streaming_unknown_question_rejection(self):
        """Verifies unknown questions in streaming are properly rejected without hallucinations."""
        res = client.post("/api/chat/stream", json={
            "message": "Do you do brain surgery?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        lines = [line.strip() for line in res.text.split("\n") if line.strip()]
        done_event = next(json.loads(line) for line in lines if json.loads(line)["type"] == "done")
        ans_lower = done_event["answer"].lower()
        self.assertTrue(
            "can't confirm" in ans_lower or "cannot confirm" in ans_lower or "contact" in ans_lower,
            f"Expected rejection in streaming, got: {done_event['answer']}"
        )


if __name__ == "__main__":
    unittest.main()
