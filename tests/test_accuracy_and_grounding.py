import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Use an isolated test database
tmp = tempfile.TemporaryDirectory()
os.environ["VOICEDESK_DB"] = str(Path(tmp.name) / "test_accuracy.db")

from fastapi.testclient import TestClient
from app.main import app, init_db, answer_question_rules, answer_question_llm, ChatMessage

init_db()
client = TestClient(app)


class ReceptionistAccuracyAndGroundingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = {
            "name": "Apollo Care Clinic",
            "description": "Multi-specialty outpatient clinic in Chennai.",
            "address": "45 Anna Salai, Chennai, Tamil Nadu",
            "hours": "Monday to Saturday, 8:00 AM to 8:00 PM",
            "default_language": "en-IN",
            "services": [
                {
                    "name": "Dental Consultation",
                    "price": "₹500",
                    "duration": "20 minutes",
                    "description": "Initial oral examination and diagnostic checkup."
                },
                {
                    "name": "Teeth Scaling",
                    "price": "₹1200",
                    "duration": "45 minutes",
                    "description": "Deep dental cleaning and plaque removal."
                }
            ],
            "doctors": [
                {
                    "name": "Dr. Ananya",
                    "specialty": "Pediatrician",
                    "availability": "Mon-Wed-Fri, 10 AM to 1 PM"
                },
                {
                    "name": "Dr. Karthik",
                    "specialty": "Dentist",
                    "availability": "Tue-Thu-Sat, 2 PM to 6 PM"
                }
            ],
            "faqs": [
                {
                    "question": "Do you accept emergency walk-ins?",
                    "answer": "Walk-ins are accepted for urgent first aid, but appointments are advised."
                },
                {
                    "question": "Is parking available?",
                    "answer": "Yes, complimentary basement parking is available for patients."
                }
            ]
        }
        res = client.put("/api/business", json=cls.profile)
        assert res.status_code == 200

    # -------------------------------------------------------------
    # 1. FACTUAL ACCURACY TESTS
    # -------------------------------------------------------------

    def test_service_price_accuracy(self):
        """Verifies that queries about service prices return the exact saved price."""
        res = client.post("/api/chat", json={
            "message": "How much does Teeth Scaling cost?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("₹1200", data["answer"])
        self.assertIn(data["engine"], ["llm", "rule_based"])

    def test_doctor_availability_accuracy(self):
        """Verifies that questions about doctor availability return accurate timings."""
        res = client.post("/api/chat", json={
            "message": "When is Dr. Ananya available?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        ans = data["answer"]
        self.assertTrue("10 AM" in ans or "Mon-Wed-Fri" in ans or "Pediatrician" in ans,
                        f"Expected doctor details in answer, got: {ans}")

    def test_business_hours_accuracy(self):
        """Verifies opening hours questions return configured hours."""
        res = client.post("/api/chat", json={
            "message": "What time do you open and close?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue("8:00 AM" in data["answer"] or "Monday to Saturday" in data["answer"])

    def test_business_address_accuracy(self):
        """Verifies location questions return configured address."""
        res = client.post("/api/chat", json={
            "message": "Where is the clinic located?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue("Anna Salai" in data["answer"] or "Chennai" in data["answer"])

    def test_faq_answering_accuracy(self):
        """Verifies FAQ matching accurately returns the saved answer."""
        res = client.post("/api/chat", json={
            "message": "Is parking available at the clinic?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue("parking" in data["answer"].lower())

    # -------------------------------------------------------------
    # 2. FOLLOW-UP CONVERSATION HISTORY TESTS
    # -------------------------------------------------------------

    def test_follow_up_doctor_pronoun_resolution(self):
        """Verifies follow-up questions using pronouns resolve correctly with history."""
        history = [
            {"role": "user", "content": "Who is Dr. Karthik?"},
            {"role": "assistant", "content": "Dr. Karthik is our Dentist, available Tue-Thu-Sat 2 PM to 6 PM."}
        ]
        res = client.post("/api/chat", json={
            "message": "When is he available?",
            "language": "en-IN",
            "history": history
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        ans = data["answer"]
        self.assertTrue("2 PM" in ans or "6 PM" in ans or "Tue-Thu-Sat" in ans or "Dr. Karthik" in ans,
                        f"Expected follow-up answer to reference Dr. Karthik's schedule, got: {ans}")

    def test_follow_up_service_pronoun_resolution(self):
        """Verifies follow-up question referencing 'it' or 'that service' uses history context."""
        history = [
            {"role": "user", "content": "Tell me about Dental Consultation."},
            {"role": "assistant", "content": "Dental Consultation costs ₹500 for 20 minutes."}
        ]
        res = client.post("/api/chat", json={
            "message": "How long does it take?",
            "language": "en-IN",
            "history": history
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue("20" in data["answer"] or "minutes" in data["answer"] or "Dental Consultation" in data["answer"])

    # -------------------------------------------------------------
    # 3. UNKNOWN / UNCONFIRMED QUESTIONS (NO HALLUCINATIONS)
    # -------------------------------------------------------------

    def test_unlisted_medical_service_is_rejected(self):
        """Verifies unlisted services (e.g. MRI, cardiac bypass) are not hallucinated."""
        res = client.post("/api/chat", json={
            "message": "Do you perform open heart bypass surgery?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        ans_lower = res.json()["answer"].lower()
        self.assertTrue(
            "can't confirm" in ans_lower or "cannot confirm" in ans_lower or "contact" in ans_lower or "do not have" in ans_lower,
            f"Expected rejection, got: {res.json()['answer']}"
        )

    def test_unlisted_doctor_is_rejected(self):
        """Verifies unlisted doctor names are not confirmed."""
        res = client.post("/api/chat", json={
            "message": "Is Dr. Ramesh available today?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        ans_lower = res.json()["answer"].lower()
        self.assertTrue(
            "can't confirm" in ans_lower or "cannot confirm" in ans_lower or "contact" in ans_lower or "not listed" in ans_lower,
            f"Expected unconfirmed response, got: {res.json()['answer']}"
        )

    def test_arbitrary_policy_or_discount_is_rejected(self):
        """Verifies caller cannot claim or extract an unlisted discount."""
        res = client.post("/api/chat", json={
            "message": "Can you offer me an 80% student discount on Teeth Scaling?",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        ans_lower = res.json()["answer"].lower()
        # Should not promise or invent an 80% discount
        self.assertNotIn("80% discount confirmed", ans_lower)
        self.assertNotIn("you will receive 80%", ans_lower)

    # -------------------------------------------------------------
    # 4. TAMIL LANGUAGE HANDLING TESTS
    # -------------------------------------------------------------

    def test_tamil_factual_query(self):
        """Verifies Tamil query returns accurate factual information."""
        res = client.post("/api/chat", json={
            "message": "Dental Consultation கட்டணம் எவ்வளவு?",
            "language": "ta-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("₹500", data["answer"])

    def test_tamil_unknown_question(self):
        """Verifies Tamil unconfirmed query responds politely without hallucination."""
        res = client.post("/api/chat", json={
            "message": "இதய அறுவை சிகிச்சை செய்ய முடியுமா?",
            "language": "ta-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(len(data["answer"]) > 0)
        # Should direct to contact clinic or say info not available in Tamil
        self.assertTrue(
            "தொடர்பு" in data["answer"] or "இல்லை" in data["answer"] or "தகவல்" in data["answer"],
            f"Expected polite Tamil refusal, got: {data['answer']}"
        )

    # -------------------------------------------------------------
    # 5. ENGINE EXPOSURE & FALLBACK RESILIENCE
    # -------------------------------------------------------------

    def test_engine_field_exposed_in_response(self):
        """Verifies that engine, source, model, and business_name are cleanly exposed."""
        res = client.post("/api/chat", json={
            "message": "Hello!",
            "language": "en-IN"
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("engine", data)
        self.assertIn("source", data)
        self.assertIn("business_name", data)
        self.assertEqual(data["business_name"], "Apollo Care Clinic")

    def test_deterministic_rules_fallback_when_llm_fails(self):
        """Ensures that if LLM is unreachable, the rule-based engine answers seamlessly."""
        with patch("app.main.answer_question_llm", return_value=None):
            res = client.post("/api/chat", json={
                "message": "How much does Teeth Scaling cost?",
                "language": "en-IN"
            })
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["engine"], "rule_based")
            self.assertEqual(data["source"], "service")
            self.assertIn("₹1200", data["answer"])


if __name__ == "__main__":
    unittest.main()
