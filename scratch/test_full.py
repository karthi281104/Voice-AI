import asyncio
import time
import httpx

async def test_full_response():
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
        "- Never reveal internal reasoning or analysis.\n"
        "- Keep greetings to 1 short sentence: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
        "- Keep ordinary answers concise (1-3 sentences).\n"
        "- Use only the facts in the business profile.\n"
        "- If information is not in the profile, politely clarify you don't have that information.\n"
    )

    tests = ["hi", "who is the doctor?", "What are your timings?", "Do you have a cardiologist?", "Who are you?"]

    async with httpx.AsyncClient(timeout=45.0) as client:
        for q in tests:
            t0 = time.perf_counter()
            payload = {
                "model": "qwen3:4b",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q}
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 400,
                }
            }
            res = await client.post(url, json=payload)
            dur = (time.perf_counter() - t0) * 1000
            d = res.json()
            msg = d.get("message", {})
            cnt = msg.get("content", "")
            print(f"=== Q: {q} ({dur:.0f}ms) ===")
            if "</think>" in cnt:
                parts = cnt.split("</think>", 1)
                print("REASONING LENGTH:", len(parts[0]))
                print("FINAL ANSWER:", repr(parts[1].strip()))
            else:
                print("NO </think> FOUND in:", repr(cnt[:120]))

asyncio.run(test_full_response())
