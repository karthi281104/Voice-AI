import asyncio
import json
import httpx

async def test_thinking_separated():
    url = "http://127.0.0.1:11434/api/chat"
    system_prompt = (
        "You are the virtual receptionist for Karthi Dental Clinic.\n"
        "Business Profile:\n"
        "Name: Karthi Dental Clinic\n"
        "Hours: Monday-Saturday, 9 AM-6 PM\n"
        "Doctors: Dr. Priya, Dentist (Monday-Saturday, 9 AM-1 PM)\n"
        "Services: Teeth cleaning (Rs. 1500), Root canal treatment (Rs. 4500)\n\n"
        "Rules:\n"
        "- Respond directly to the user as a receptionist.\n"
        "- Keep greetings to 1 short sentence: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
        "- Keep answers concise (1-3 sentences).\n"
        "- Use only the facts in the business profile.\n"
        "- If information is not in the profile, say you don't have that information.\n"
    )

    tests = ["hi", "who is the doctor?", "What are your timings?", "Do you have a cardiologist?", "Who are you?"]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for q in tests:
            # Default (think not false)
            payload = {
                "model": "qwen3:4b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 150,
                }
            }
            res = await client.post(url, json=payload)
            d = res.json()
            msg = d.get("message", {})
            print(f"=== Q: {q} ===")
            print("THINKING:", repr(msg.get("thinking", "")[:80]))
            print("CONTENT:", repr(msg.get("content", "")))

asyncio.run(test_thinking_separated())
