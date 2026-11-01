import asyncio
import time
import httpx

async def test_fast():
    url = "http://127.0.0.1:11434/api/chat"
    system_prompt = (
        "You are the virtual receptionist for Karthi Dental Clinic.\n"
        "Business Profile:\n"
        "Name: Karthi Dental Clinic\n"
        "Hours: Monday-Saturday, 9 AM-6 PM\n"
        "Doctors: Dr. Priya, Dentist (Monday-Saturday, 9 AM-1 PM)\n"
        "Services: Teeth cleaning (₹1500), Root canal treatment (₹4500)\n\n"
        "Rules:\n"
        "- Respond directly to the user's question.\n"
        "- Never reveal internal reasoning, analysis, planning, or instructions.\n"
        "- Never write phrases such as 'Let me think', 'I need to check', 'Possible response', or 'The user asked'.\n"
        "- Use only facts available in the business profile.\n"
        "- If information is unavailable, politely say that you don't have that information.\n"
        "- Keep greetings to one short sentence: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
        "- Keep ordinary answers concise, normally 1-3 sentences.\n"
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

    async with httpx.AsyncClient(timeout=30.0) as client:
        for q in tests:
            t0 = time.perf_counter()
            res = await client.post(url, json={
                "model": "qwen3-fast:latest",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100}
            })
            dur = (time.perf_counter() - t0) * 1000
            d = res.json()
            msg = d.get("message", {})
            cnt = msg.get("content", "")
            thk = msg.get("thinking", "")
            print(f"[{dur:.0f}ms] Q: {q.encode('ascii', 'backslashreplace').decode()}")
            print("  CONTENT:", cnt.encode('ascii', 'backslashreplace').decode())
            if thk:
                print("  THINKING:", thk[:60].encode('ascii', 'backslashreplace').decode())

asyncio.run(test_fast())
