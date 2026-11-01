from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Use an isolated test database
tmp = tempfile.TemporaryDirectory()
os.environ["VOICEDESK_DB"] = str(Path(tmp.name) / "test_fallback.db")

from fastapi.testclient import TestClient
from app.main import app, init_db, save_profile, BusinessProfile, Service, Doctor, FAQ
from app.llm.ollama_client import LLMFailureReason

init_db()
client = TestClient(app)


class FallbackScenariosTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure a test business profile
        profile = BusinessProfile(
            name="Karthi Dental Clinic",
            description="Premium dental healthcare clinic in Chennai.",
            address="Chennai, Tamil Nadu",
            hours="Monday-Saturday, 9 AM-6 PM",
            default_language="en-IN",
            services=[
                Service(
                    name="Teeth Scaling",
                    price="₹1000",
                    duration="30 mins",
                    description="Plaque removal and dental polishing.",
                ),
                Service(
                    name="Root Canal",
                    price="₹4500",
                    duration="60 mins",
                    description="Endodontic tooth repair.",
                ),
            ],
            doctors=[
                Doctor(
                    name="Dr. Priya",
                    specialty="Dentist",
                    availability="Monday-Friday, 10 AM-4 PM",
                )
            ],
            faqs=[
                FAQ(
                    question="Do you accept credit cards?",
                    answer="Yes, all major credit cards and UPI are accepted.",
                )
            ],
        )
        save_profile(profile)

    def test_fast_path_greetings_never_hit_llm(self):
        """Greetings must resolve instantly via rule engine with no fallback reason."""
        with patch("app.main.answer_question_llm") as mock_llm:
            res = client.post("/api/chat", json={"message": "hi", "language": "en-IN"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "rule_based")
            self.assertEqual(data["source"], "greeting")
            self.assertIn("Karthi Dental Clinic", data["answer"])
            self.assertIsNone(data["fallback_reason"])
            mock_llm.assert_not_called()

    def test_fast_path_doctor_never_hits_llm(self):
        """Doctor queries resolve via rule engine with no fallback reason."""
        with patch("app.main.answer_question_llm") as mock_llm:
            res = client.post("/api/chat", json={"message": "who is the doctor?", "language": "en-IN"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "rule_based")
            self.assertEqual(data["source"], "doctor")
            self.assertIn("Dr. Priya", data["answer"])
            self.assertIsNone(data["fallback_reason"])
            mock_llm.assert_not_called()

    def test_fast_path_location_never_hits_llm(self):
        """Location queries resolve via rule engine."""
        with patch("app.main.answer_question_llm") as mock_llm:
            res = client.post("/api/chat", json={"message": "where is the clinic located?", "language": "en-IN"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "rule_based")
            self.assertEqual(data["source"], "business")
            self.assertIn("Chennai", data["answer"])
            self.assertIsNone(data["fallback_reason"])
            mock_llm.assert_not_called()

    def test_fast_path_hours_never_hits_llm(self):
        """Hours queries resolve via rule engine."""
        with patch("app.main.answer_question_llm") as mock_llm:
            res = client.post("/api/chat", json={"message": "what are your opening hours?", "language": "en-IN"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "rule_based")
            self.assertEqual(data["source"], "business")
            self.assertIn("9 AM-6 PM", data["answer"])
            self.assertIsNone(data["fallback_reason"])
            mock_llm.assert_not_called()

    def test_fast_path_plain_time_question_uses_business_hours(self):
        with patch("app.main.answer_question_llm") as mock_llm:
            res = client.post("/api/chat", json={"message": "what is the time?", "language": "en-IN"})
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "rule_based")
            self.assertEqual(data["source"], "business")
            self.assertIn("9 AM-6 PM", data["answer"])
            mock_llm.assert_not_called()

    def test_fast_path_generic_doctor_visit_question_uses_schedule(self):
        with patch("app.main.answer_question_llm") as mock_llm:
            res = client.post("/api/chat", json={
                "message": "when can I visit the doctor?",
                "language": "en-IN",
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "rule_based")
            self.assertEqual(data["source"], "doctor")
            self.assertIn("10 AM-4 PM", data["answer"])
            mock_llm.assert_not_called()

    def test_clinic_information_uses_saved_profile(self):
        response = client.post("/api/chat", json={"message": "Tell me about the clinic"})
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"]
        self.assertIn("Karthi Dental Clinic", answer)
        self.assertIn("Teeth Scaling", answer)
        self.assertIn("Chennai", answer)

    def test_generic_prices_list_all_services(self):
        response = client.post("/api/chat", json={"message": "Can I know the prices?"})
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"]
        self.assertIn("Teeth Scaling", answer)
        self.assertIn("₹1000", answer)
        self.assertIn("Root Canal", answer)
        self.assertIn("₹4500", answer)

    def test_unknown_service_price_does_not_invent_amount(self):
        response = client.post("/api/chat", json={"message": "How much does implant surgery cost?"})
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"].lower()
        self.assertIn("do not have", answer)
        self.assertNotIn("₹1000", answer)

    def test_appointment_request_asks_for_date_and_time(self):
        response = client.post("/api/chat", json={"message": "I want to book an appointment."})
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"].lower()
        self.assertIn("date", answer)
        self.assertIn("time", answer)
        self.assertIn("confirmed", answer)

    def test_appointment_requirement_is_not_overpromised(self):
        response = client.post("/api/chat", json={"message": "Is there any appointment needed?"})
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"].lower()
        self.assertIn("contact", answer)
        self.assertNotIn("appointment booked", answer)

    def test_tomorrow_hours_include_day_context(self):
        response = client.post("/api/chat", json={"message": "Are you open tomorrow?"})
        self.assertEqual(response.status_code, 200)
        answer = response.json()["answer"]
        self.assertIn("Monday-Saturday", answer)
        self.assertIn("Tomorrow is", answer)

    def test_complex_query_uses_llm_when_available(self):
        """Complex conversational query routes to LLM and returns engine='llm'."""
        with patch("app.main.chat_completion", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = (
                "Dr. Priya usually takes appointments between 10 AM and 4 PM on weekdays.",
                120.0,
                LLMFailureReason.SUCCESS,
            )
            res = client.post("/api/chat", json={
                "message": "in which the doctor arrives?",
                "language": "en-IN"
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "llm")
            self.assertEqual(data["source"], "llm")
            self.assertIn("Dr. Priya", data["answer"])
            self.assertIsNone(data["fallback_reason"])

    def test_complex_query_falls_back_when_llm_fails(self):
        """When LLM returns None / fails, system falls back to rule fallback with diagnostic reason."""
        with patch("app.main.chat_completion", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = (None, 50.0, LLMFailureReason.UNAVAILABLE)
            res = client.post("/api/chat", json={
                "message": "in which the doctor arrives?",
                "language": "en-IN"
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "rule_based")
            self.assertEqual(data["source"], "fallback")
            self.assertTrue("can't confirm" in data["answer"].lower() or "cannot confirm" in data["answer"].lower())
            self.assertEqual(data["fallback_reason"], "llm_unavailable_or_filtered")

    def test_streaming_fast_path(self):
        """Streaming endpoint fast path returns immediate done event with engine='rule_based'."""
        with patch("app.main.stream_chat_completion") as mock_stream:
            res = client.post("/api/chat/stream", json={"message": "hi", "language": "en-IN"})
            self.assertEqual(res.status_code, 200)
            lines = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
            self.assertTrue(any(l.get("type") == "token" for l in lines))
            done = next(l for l in lines if l.get("type") == "done")
            self.assertEqual(done["engine"], "rule_based")
            self.assertEqual(done["source"], "greeting")
            mock_stream.assert_not_called()

    def test_streaming_llm_success(self):
        """Streaming endpoint emits tokens and done event from LLM."""
        async def fake_stream(messages, model=None, options=None):
            yield {"type": "token", "content": "Hello! "}
            yield {"type": "token", "content": "How may I help?"}
            yield {
                "type": "done",
                "answer": "Hello! How may I help?",
                "engine": "llm",
                "source": "llm",
                "model": "qwen3:4b",
                "response_time_ms": 150.0,
                "ttft_ms": 30.0,
                "failure_reason": None,
            }

        with patch("app.main.stream_chat_completion", side_effect=fake_stream):
            res = client.post("/api/chat/stream", json={
                "message": "can you tell me if you have any appointment slots?",
                "language": "en-IN"
            })
            self.assertEqual(res.status_code, 200)
            lines = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
            tokens = [l["content"] for l in lines if l.get("type") == "token"]
            self.assertIn("Hello! ", tokens)
            done = next(l for l in lines if l.get("type") == "done")
            self.assertEqual(done["engine"], "llm")
            self.assertIsNone(done["fallback_reason"])

    def test_streaming_llm_failure_emits_fallback_with_reason(self):
        """Streaming endpoint gracefully handles failure and emits fallback done with reason."""
        async def fake_stream_fail(messages, model=None, options=None):
            yield {
                "type": "done",
                "answer": "",
                "engine": "llm",
                "source": "llm",
                "model": "qwen3:4b",
                "response_time_ms": 20.0,
                "ttft_ms": 0.0,
                "failure_reason": LLMFailureReason.UNAVAILABLE,
            }

        with patch("app.main.stream_chat_completion", side_effect=fake_stream_fail):
            res = client.post("/api/chat/stream", json={
                "message": "tell me something complex about insurance coverage?",
                "language": "en-IN"
            })
            self.assertEqual(res.status_code, 200)
            lines = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
            done = next(l for l in lines if l.get("type") == "done")
            self.assertEqual(done["engine"], "rule_based")
            self.assertEqual(done["source"], "fallback")
            self.assertEqual(done["fallback_reason"], LLMFailureReason.UNAVAILABLE.value)

    def test_voice_websocket_control_messages(self):
        """WebSocket voice endpoint supports set_language, interrupt, and clear control actions."""
        with client.websocket_connect("/ws/voice/test_ws_control") as ws:
            ws.send_text(json.dumps({"action": "set_language", "language": "ta-IN"}))
            ev = ws.receive_json()
            self.assertEqual(ev["event"], "language_updated")
            self.assertEqual(ev["language"], "ta-IN")

            ws.send_text(json.dumps({"action": "interrupt"}))
            ev2 = ws.receive_json()
            self.assertEqual(ev2["event"], "interrupted")

            ws.send_text(json.dumps({"action": "clear"}))
            ev3 = ws.receive_json()
            self.assertEqual(ev3["event"], "chat_cleared")


if __name__ == "__main__":
    unittest.main()
