import sys
import os
sys.path.insert(0, os.path.abspath("."))

from app.main import get_profile, answer_question, answer_question_llm, answer_question_rules

profile = {
    "name": "Apollo Care Clinic",
    "description": "Multi-specialty neighborhood clinic.",
    "address": "45 Anna Salai, T. Nagar, Chennai",
    "hours": "Monday to Saturday, 8:00 AM to 8:00 PM",
    "default_language": "en-IN",
    "services": [
        {"name": "Teeth Scaling", "price": "₹1200", "duration": "30 minutes", "description": "Ultrasonic teeth cleaning."},
        {"name": "Dental Consultation", "price": "₹500", "duration": "20 minutes", "description": "Comprehensive oral checkup."}
    ],
    "doctors": [
        {"name": "Dr. Ananya", "specialty": "Pediatrician", "availability": "Mon-Wed-Fri, 10 AM to 1 PM"},
        {"name": "Dr. Karthik", "specialty": "Dentist", "availability": "Tue-Thu-Sat, 2 PM to 6 PM"}
    ],
    "faqs": []
}

print("Testing answer_question_llm:")
llm_res = answer_question_llm("How much does Teeth Scaling cost?", "en-IN", profile)
print("LLM result:", llm_res)

print("Testing answer_question_rules:")
rules_res = answer_question_rules("How much does Teeth Scaling cost?", "en-IN", profile)
print("Rules result:", rules_res)

print("Testing overall answer_question:")
ans = answer_question("How much does Teeth Scaling cost?", "en-IN", profile)
print("Overall:", ans)
