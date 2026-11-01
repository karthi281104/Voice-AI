import asyncio
import time
import httpx

async def test_receptionist_prompt():
    url = "http://127.0.0.1:11434/api/chat"
    
    # Prompt structured purely as a receptionist persona
    system_prompt = (
        "You are the virtual receptionist for Karthi Dental Clinic.\n"
        "Clinic details:\n"
        "- Name: Karthi Dental Clinic\n"
        "- Hours: Monday-Saturday, 9 AM-6 PM\n"
        "- Doctors: Dr. Priya, Dentist (Monday-Saturday, 9 AM-1 PM)\n"
        "- Services: Teeth cleaning (₹1500), Root canal treatment (₹4500)\n\n"
        "Guidelines:\n"
        "- Speak directly to the caller. Never include thoughts, analysis, or explanations.\n"
        "- Greeting: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
        "- Answers must be concise (1-3 sentences) using only the clinic details above.\n"
        "- If asked about something not in the clinic details (e.g., cardiologist), politely state that our clinic does not have that.\n"
        "- Respond in the language used by the caller (Tamil or English).\n"
    )

    tests = [
        ("hi", "Hello! Welcome to Karthi Dental Clinic. How can I help you?"),
        ("who is the doctor?", "Our clinic has Dr. Priya, a Dentist."),
        ("What are your timings?", "Our business hours are Monday-Saturday, 9 AM-6 PM."),
        ("உங்க கிளினிக்ல என்னென்ன சேவைகள் இருக்கு?", "Tamil answer"),
        ("Do you have a cardiologist?", "No cardiologist"),
        ("Who are you?", "Virtual receptionist"),
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for q, expected in tests:
            t0 = time.perf_counter()
            res = await client.post(url, json={
                "model": "qwen3:4b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q}
                ],
                "think": False,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 120,
                    "stop": ["Customer:", "User:", "\n\nUser:"]
                }
            })
            dur = (time.perf_counter() - t0) * 1000
            d = res.json()
            cnt = d.get("message", {}).get("content", "")
            print(f"[{dur:.0f}ms] Q: {q.encode('ascii', 'backslashreplace').decode()}")
            print("  RAW:", repr(cnt[:120]))

asyncio.run(test_receptionist_prompt())
