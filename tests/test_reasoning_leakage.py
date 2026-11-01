import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Use an isolated test database
tmp = tempfile.TemporaryDirectory()
os.environ["VOICEDESK_DB"] = str(Path(tmp.name) / "test_reasoning.db")

from fastapi.testclient import TestClient
from app.main import app, init_db, build_system_prompt
from app.llm.ollama_client import clean_conversational_response, is_reasoning_text

init_db()
client = TestClient(app)


class ReasoningLeakageAndConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = {
            "name": "Karthi Dental Clinic",
            "description": "Leading dental care clinic.",
            "address": "45 Gandhi Road, Chennai",
            "hours": "Monday-Saturday, 9 AM-6 PM",
            "default_language": "en-IN",
            "services": [
                {
                    "name": "Teeth cleaning",
                    "price": "₹1000",
                    "duration": "30 minutes",
                    "description": "Professional teeth cleaning and scaling."
                },
                {
                    "name": "Root canal treatment",
                    "price": "₹4500",
                    "duration": "60 minutes",
                    "description": "Painless single-sitting root canal."
                }
            ],
            "doctors": [
                {
                    "name": "Dr. Priya",
                    "specialty": "Dentist",
                    "availability": "Monday-Saturday, 9 AM-6 PM"
                }
            ],
            "faqs": [
                {
                    "question": "Do I need an appointment?",
                    "answer": "Walk-ins are welcome, but appointments are recommended."
                }
            ]
        }
        res = client.put("/api/business", json=cls.profile)
        assert res.status_code == 200

    def test_strict_system_prompt_rules(self):
        """Verifies system prompt strictly instructs model against reasoning leakage."""
        prompt = build_system_prompt(self.profile, "en-IN")
        self.assertIn("virtual receptionist", prompt.lower())
        self.assertIn("karthi dental clinic", prompt.lower())
        # New prose prompt uses "never explain your reasoning" instead of "never reveal"
        self.assertTrue(
            "never explain" in prompt.lower() or "never reveal" in prompt.lower(),
            "Prompt must instruct model to not expose internal reasoning"
        )
        self.assertTrue(
            "reasoning" in prompt.lower() or "thinking" in prompt.lower(),
            "Prompt must contain word 'reasoning' or 'thinking' to discourage internal deliberation"
        )
        self.assertTrue(
            "1-3 sentences" in prompt.lower() or "1–3 sentences" in prompt.lower(),
            "Prompt must instruct model to be concise"
        )

    def test_cleaning_removes_think_tags(self):
        """Verifies clean_conversational_response removes think tags and preamble."""
        raw = "<think>\nLet me check the Business Profile JSON.\nThe user is asking about timings.\n</think>We are open Monday-Saturday, 9 AM-6 PM."
        cleaned = clean_conversational_response(raw)
        self.assertEqual(cleaned, "We are open Monday-Saturday, 9 AM-6 PM.")
        self.assertNotIn("<think>", cleaned)
        self.assertNotIn("Business Profile JSON", cleaned)

    def test_cleaning_removes_reasoning_preamble(self):
        """Verifies cleaning strips typical chain-of-thought phrases."""
        raw = "Okay, let me check the Business Profile JSON.\nI need to respond to the user.\nPossible response: Hello! Welcome to Karthi Dental Clinic. How can I help you?"
        cleaned = clean_conversational_response(raw)
        self.assertIn("Hello! Welcome to Karthi Dental Clinic. How can I help you?", cleaned)
        self.assertNotIn("Business Profile JSON", cleaned)
        self.assertNotIn("Possible response", cleaned)

    def test_is_reasoning_text_detector(self):
        """Verifies is_reasoning_text correctly flags internal deliberation lines."""
        self.assertTrue(is_reasoning_text("Let me check the Business Profile JSON"))
        self.assertTrue(is_reasoning_text("The caller is asking who is the doctor."))
        self.assertTrue(is_reasoning_text("I need to formulate a concise greeting."))
        self.assertTrue(is_reasoning_text("We are given the following profile information:"))
        self.assertFalse(is_reasoning_text("Hello! Welcome to Karthi Dental Clinic."))
        self.assertFalse(is_reasoning_text("Our clinic has Dr. Priya, a Dentist."))

    # Exact required conversations:
    def test_conversation_1_greeting_hi(self):
        """Test 1: User says 'hi' -> Hello! Welcome to Karthi Dental Clinic. How can I help you?"""
        res = client.post("/api/chat", json={"message": "hi", "language": "en-IN"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reply = data["answer"]
        self.assertIn("Hello! Welcome to Karthi Dental Clinic. How can I help you?", reply)
        self.assertNotIn("<think>", reply)
        self.assertNotIn("Business Profile", reply)

    def test_conversation_2_who_is_the_doctor(self):
        """Test 2: User says 'who is the doctor?' -> Our clinic has Dr. Priya, a Dentist."""
        res = client.post("/api/chat", json={"message": "who is the doctor?", "language": "en-IN"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reply = data["answer"]
        self.assertIn("Dr. Priya", reply)
        self.assertIn("Dentist", reply)
        self.assertNotIn("<think>", reply)
        self.assertNotIn("Let me check", reply)

    def test_conversation_3_timings(self):
        """Test 3: User says 'What are your timings?' -> Answer using the actual business profile. Never invent timings."""
        res = client.post("/api/chat", json={"message": "What are your timings?", "language": "en-IN"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reply = data["answer"]
        self.assertTrue("9 AM" in reply or "9:00 AM" in reply or "Monday-Saturday" in reply or "Monday to Saturday" in reply)
        self.assertNotIn("<think>", reply)
        self.assertNotIn("invent", reply.lower())

    def test_conversation_4_tamil_services(self):
        """Test 4: User says 'உங்க கிளினிக்ல என்னென்ன சேவைகள் இருக்கு?' -> Answer in Tamil using configured services."""
        res = client.post("/api/chat", json={"message": "உங்க கிளினிக்ல என்னென்ன சேவைகள் இருக்கு?", "language": "ta-IN"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reply = data["answer"]
        self.assertTrue(len(reply) > 10)
        # Verify dental services or Tamil keywords present
        self.assertTrue("Teeth cleaning" in reply or "பல்" in reply or "Root canal" in reply or "சேவை" in reply)
        self.assertNotIn("<think>", reply)

    def test_conversation_5_cardiologist(self):
        """Test 5: User says 'Do you have a cardiologist?' -> Politely clarify clinic does not list a cardiologist."""
        res = client.post("/api/chat", json={"message": "Do you have a cardiologist?", "language": "en-IN"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reply = data["answer"]
        self.assertTrue("do not have" in reply.lower() or "not have" in reply.lower() or "dental" in reply.lower() or "specialist" in reply.lower() or "not listed" in reply.lower())
        self.assertNotIn("<think>", reply)
        self.assertNotIn("Let me check", reply)

    def test_conversation_6_who_are_you(self):
        """Test 6: User says 'Who are you?' -> Identify yourself as the clinic's virtual receptionist."""
        res = client.post("/api/chat", json={"message": "Who are you?", "language": "en-IN"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        reply = data["answer"]
        self.assertIn("receptionist", reply.lower())
        self.assertIn("karthi dental clinic", reply.lower())
        self.assertNotIn("<think>", reply)


if __name__ == "__main__":
    unittest.main()
