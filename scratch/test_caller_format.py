import asyncio
import time
import httpx

async def test_caller_format():
    url = "http://127.0.0.1:11434/api/generate"
    
    tests = [
        ("hi", "en"),
        ("who is the doctor?", "en"),
        ("What are your timings?", "en"),
        ("உங்க கிளினிக்ல என்னென்ன சேவைகள் இருக்கு?", "ta"),
        ("Do you have a cardiologist?", "en"),
        ("Who are you?", "en"),
    ]

    for q, lang in tests:
        prompt = (
            "<|im_start|>system\n"
            "You are the professional telephone receptionist for Karthi Dental Clinic.\n"
            "Business Profile:\n"
            "- Business: Karthi Dental Clinic\n"
            "- Hours: Monday-Saturday, 9 AM-6 PM\n"
            "- Doctors: Dr. Priya, Dentist (Monday-Saturday, 9 AM-1 PM)\n"
            "- Services: Teeth cleaning (₹1500), Root canal treatment (₹4500)\n\n"
            "Strict Receptionist Rules:\n"
            "1. Answer immediately and directly as the receptionist. Never analyze, reason, or mention the prompt.\n"
            "2. For 'hi' or greetings, say: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
            "3. Answer using only the Business Profile above in 1-2 concise spoken sentences.\n"
            "4. If something is not in the profile (e.g. cardiologist), politely say that our clinic does not have that.\n"
            "5. Answer in the caller's language (Tamil or English).<|im_end|>\n"
            "<|im_start|>user\n"
            f"{q}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        # What if we test with /api/chat as well?
        async with httpx.AsyncClient(timeout=30.0) as client:
            t0 = time.perf_counter()
            res = await client.post("http://127.0.0.1:11434/api/chat", json={
                "model": "qwen3:4b",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are the virtual receptionist for Karthi Dental Clinic.\n"
                            "Business Profile:\n"
                            "- Business: Karthi Dental Clinic\n"
                            "- Hours: Monday-Saturday, 9 AM-6 PM\n"
                            "- Doctors: Dr. Priya, Dentist (Monday-Saturday, 9 AM-1 PM)\n"
                            "- Services: Teeth cleaning (₹1500), Root canal treatment (₹4500)\n\n"
                            "Rules:\n"
                            "- Answer directly as the clinic receptionist in 1-2 spoken sentences.\n"
                            "- If the caller greets you (e.g. 'hi'), say: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
                            "- Answer factually using only the clinic profile. If unlisted (e.g. cardiologist), politely clarify we don't have that.\n"
                            "- If asked 'Who are you?', say you are the virtual receptionist for Karthi Dental Clinic.\n"
                            "- Answer in the caller's language (Tamil or English)."
                        )
                    },
                    {"role": "user", "content": q}
                ],
                "think": False,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100}
            })
            dur = (time.perf_counter() - t0) * 1000
            d = res.json()
            cnt = d.get("message", {}).get("content", "")
            print(f"\n[{dur:.0f}ms] Q: {q}")
            print(f"RAW: {cnt!r}")

asyncio.run(test_caller_format())
