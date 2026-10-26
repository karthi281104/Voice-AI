import json
import os
import tempfile
import unittest
from pathlib import Path

# Use a separate database for tests.
tmp = tempfile.TemporaryDirectory()
os.environ["VOICEDESK_DB"] = str(Path(tmp.name) / "test.db")

from fastapi.testclient import TestClient
from app.main import app, init_db

init_db()
client = TestClient(app)


class VoiceDeskAPITests(unittest.TestCase):
    def test_health_reports_database(self):
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_business_profile_can_be_saved_and_loaded(self):
        profile = {
            "name": "Test Clinic",
            "description": "A test clinic",
            "address": "Chennai",
            "hours": "Mon-Fri 9 AM-5 PM",
            "default_language": "en-IN",
            "services": [{"name": "Cleaning", "price": "₹1000", "duration": "30 min", "description": "Dental cleaning"}],
            "doctors": [{"name": "Dr. Asha", "specialty": "Dentist", "availability": "Mon-Fri mornings"}],
            "faqs": [{"question": "Do you accept walk-ins?", "answer": "Please call first."}]
        }
        saved = client.put("/api/business", json=profile)
        self.assertEqual(saved.status_code, 200)
        loaded = client.get("/api/business")
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.json()["name"], "Test Clinic")
        self.assertEqual(loaded.json()["services"][0]["price"], "₹1000")

    def test_chat_uses_saved_service_price(self):
        response = client.post("/api/chat", json={"message": "How much does Cleaning cost?", "language": "en-IN"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("₹1000", response.json()["answer"])
        self.assertEqual(response.json()["source"], "service")

    def test_unknown_question_does_not_invent_answer(self):
        response = client.post("/api/chat", json={"message": "Do you have an MRI machine?", "language": "en-IN"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("can't confirm", response.json()["answer"])

    def test_invalid_language_is_rejected(self):
        response = client.post("/api/chat", json={"message": "hello", "language": "fr-FR"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
