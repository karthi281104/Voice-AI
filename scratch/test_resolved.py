import asyncio
import time
import httpx

async def test_resolved():
    url = "http://127.0.0.1:11434/api/chat"
    system_prompt = (
        "You are the virtual receptionist for Karthi Dental Clinic.\n\n"
        "Clinic details:\n"
        "- Business: Karthi Dental Clinic\n"
        "- Hours: Monday-Saturday, 9 AM-6 PM\n"
        "- Doctors: Dr. Priya, Dentist (Available: Monday-Saturday, 9 AM-1 PM)\n"
        "- Services: Teeth cleaning (₹1500), Root canal treatment (₹4500)\n\n"
        "Receptionist Rules:\n"
        "- If the user says a greeting (like 'hi' or 'hello'), reply: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
        "- For factual questions, answer directly in 1-2 concise sentences using only the clinic details.\n"
        "- If the user asks 'Who are you?', identify yourself as the virtual receptionist for Karthi Dental Clinic.\n"
        "- If asked for an unlisted doctor or specialty (e.g., cardiologist), politely state that Karthi Dental Clinic does not have a cardiologist.\n"
        "- Answer in the user's language (Tamil or English).\n"
    )

    tests = [
        "hi",
        "who is the doctor?",
        "What are your timings?",
        "உங்க கிளினிக்ல என்னென்ன சேவைகள் இருக்கு?",
        "Do you have a cardiologist?",
        "Who are you?",
    ]

    async with httpx.AsyncClient(timeout=45.0) as client:
        for q in tests:
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
                    "num_predict": 300,
                }
            })
            dur = (time.perf_counter() - t0) * 1000
            d = res.json()
            cnt = d.get("message", {}).get("content", "")
            print(f"\n==================== Q: {q} ({dur:.0f}ms) ====================")
            if "</think>" in cnt:
                ans = cnt.split("</think>", 1)[1].strip()
                print("CLEAN ANSWER:", repr(ans))
            else:
                print("NO </think> - RAW:", repr(cnt[:120]))

asyncio.run(test_resolved())
