import asyncio
import time
import httpx

async def test_generate_raw():
    url = "http://127.0.0.1:11434/api/generate"
    
    prompt = (
        "<|im_start|>system\n"
        "You are the virtual receptionist for Karthi Dental Clinic.\n"
        "Business Profile: Karthi Dental Clinic. Hours: Mon-Sat 9AM-6PM. Doctor: Dr. Priya, Dentist.\n"
        "Rules: Speak directly as a receptionist. 1 short sentence.<|im_end|>\n"
        "<|im_start|>user\n"
        "hi<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "</think>\n"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        t0 = time.perf_counter()
        res = await client.post(url, json={
            "model": "qwen3:4b",
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 50}
        })
        dur = (time.perf_counter() - t0) * 1000
        d = res.json()
        print(f"[{dur:.0f}ms] Output: {d.get('response', '')!r}")

asyncio.run(test_generate_raw())
