import asyncio
import httpx

async def print_full():
    url = "http://127.0.0.1:11434/api/chat"
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
        "hi",
        "who is the doctor?",
        "What are your timings?",
        "Do you have a cardiologist?",
        "Who are you?",
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for q in tests:
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
                    "num_predict": 250,
                }
            })
            d = res.json()
            cnt = d.get("message", {}).get("content", "")
            print(f"\n==================== Q: {q} ====================")
            print(cnt)

asyncio.run(print_full())
