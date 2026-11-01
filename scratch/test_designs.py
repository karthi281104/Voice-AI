import asyncio
import time
import httpx

async def test_prompt_designs():
    url = "http://127.0.0.1:11434/api/chat"
    
    # Design A: Clean Persona
    prompt_A = (
        "You are the receptionist at Karthi Dental Clinic.\n"
        "Clinic details:\n"
        "- Name: Karthi Dental Clinic\n"
        "- Hours: Monday-Saturday, 9 AM-6 PM\n"
        "- Doctor: Dr. Priya, Dentist (Available: Monday-Saturday, 9 AM-1 PM)\n"
        "- Services: Teeth cleaning (₹1500), Root canal treatment (₹4500)\n\n"
        "Instructions:\n"
        "Speak directly to the caller as the receptionist. Do not include any internal thoughts, reasoning, or explanations. Only state the final spoken response.\n"
        "Greetings: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
        "Answer questions in 1-2 sentences using only the clinic details above. If anything is not listed, say we don't have that information."
    )

    # Design B: Conversation history priming
    prompt_B = prompt_A

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("--- Testing Design A (Plain prompt) ---")
        for q in ["hi", "who is the doctor?", "What are your timings?", "Do you have a cardiologist?"]:
            t0 = time.perf_counter()
            res = await client.post(url, json={
                "model": "qwen3:4b",
                "messages": [
                    {"role": "system", "content": prompt_A},
                    {"role": "user", "content": q}
                ],
                "think": False,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 100}
            })
            dur = (time.perf_counter() - t0) * 1000
            cnt = res.json().get("message", {}).get("content", "")
            print(f"[{dur:.0f}ms] Q: {q} -> {cnt[:100]!r}")

        print("\n--- Testing Design B (With assistant example / few-shot in messages) ---")
        for q in ["hi", "who is the doctor?", "What are your timings?", "Do you have a cardiologist?"]:
            t0 = time.perf_counter()
            messages = [
                {"role": "system", "content": prompt_B},
            ]
            if q == "hi":
                messages.append({"role": "user", "content": "hi"})
            else:
                messages.extend([
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "Hello! Welcome to Karthi Dental Clinic. How can I help you?"},
                    {"role": "user", "content": q}
                ])
            res = await client.post(url, json={
                "model": "qwen3:4b",
                "messages": messages,
                "think": False,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 100}
            })
            dur = (time.perf_counter() - t0) * 1000
            cnt = res.json().get("message", {}).get("content", "")
            print(f"[{dur:.0f}ms] Q: {q} -> {cnt[:100]!r}")

asyncio.run(test_prompt_designs())
