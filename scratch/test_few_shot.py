import asyncio
import json
import httpx

async def test_few_shot():
    url = "http://127.0.0.1:11434/api/chat"
    system_prompt = (
        "You are the virtual receptionist for Karthi Dental Clinic.\n\n"
        "Clinic Information:\n"
        "- Business Name: Karthi Dental Clinic\n"
        "- Timings: Monday-Saturday, 9 AM-6 PM\n"
        "- Doctor: Dr. Priya, Dentist (Available: Monday-Saturday, 9 AM-1 PM)\n"
        "- Services: Teeth cleaning (₹1500), Root canal treatment (₹4500)\n\n"
        "Strict Receptionist Instructions:\n"
        "- Respond directly and immediately to the customer.\n"
        "- Absolutely NEVER write reasoning, analysis, notes, or meta-commentary (such as 'We are given', 'The user asked', 'According to', 'Steps:').\n"
        "- Output ONLY the final spoken words meant for the caller.\n"
        "- Keep greetings to 1 short sentence.\n"
        "- Keep regular answers concise, 1-3 sentences.\n"
        "- If asked for something not listed in the clinic information, politely state that you do not have that information.\n"
        "- Answer in the user's language (Tamil or English).\n"
    )

    tests = [
        ("hi", "en-IN"),
        ("who is the doctor?", "en-IN"),
        ("What are your timings?", "en-IN"),
        ("உங்க கிளினிக்ல என்னென்ன சேவைகள் இருக்கு?", "ta-IN"),
        ("Do you have a cardiologist?", "en-IN"),
        ("Who are you?", "en-IN"),
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for q, lang in tests:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Hello! Welcome to Karthi Dental Clinic. How can I help you?"},
                {"role": "user", "content": "who is the doctor?"},
                {"role": "assistant", "content": "Our clinic has Dr. Priya, a Dentist."},
                {"role": "user", "content": q}
            ]
            if q == "hi":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "hi"}
                ]
            elif q == "who is the doctor?":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "who is the doctor?"}
                ]
            
            payload = {
                "model": "qwen3:4b",
                "messages": messages,
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
            cnt = msg.get("content", "")
            thk = msg.get("thinking", "")
            print(f"=== Q: {q.encode('ascii', 'backslashreplace').decode()} ===")
            print("CONTENT:", cnt.encode('ascii', 'backslashreplace').decode())

asyncio.run(test_few_shot())
