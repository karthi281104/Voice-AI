import asyncio
import json
import httpx

async def test_no_think():
    url = "http://127.0.0.1:11434/api/chat"
    system_prompt = (
        "/no_think\n"
        "You are the virtual receptionist for Karthi Dental Clinic.\n"
        "Rules:\n"
        "- Respond directly to the user as a receptionist.\n"
        "- Keep greetings to 1 short sentence: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
        "- Keep answers concise (1-3 sentences).\n"
        "- Use only the facts in the business profile.\n"
        "- If information is not in the profile, say you don't have that information.\n\n"
        "Business Profile:\n"
        "Name: Karthi Dental Clinic\n"
        "Hours: Monday-Saturday, 9 AM-6 PM\n"
        "Doctors: Dr. Priya, Dentist (Monday-Saturday, 9 AM-1 PM)\n"
        "Services: Teeth cleaning (Rs. 1500), Root canal treatment (Rs. 4500)\n"
    )

    tests = [
        "hi",
        "who is the doctor?",
        "What are your timings?",
        "உங்க கிளினிக்ல என்னென்ன சேவைகள் இருக்கு?",
        "Do you have a cardiologist?",
        "Who are you?",
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for q in tests:
            payload = {
                "model": "qwen3:4b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q}
                ],
                "think": False,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 100,
                }
            }
            res = await client.post(url, json=payload)
            d = res.json()
            msg = d.get("message", {})
            print(f"=== Q: {q.encode('ascii', 'backslashreplace').decode()} ===")
            print("THINKING:", repr(msg.get("thinking", "")))
            print("CONTENT:", repr(msg.get("content", "")))

asyncio.run(test_no_think())
