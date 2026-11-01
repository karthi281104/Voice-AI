import asyncio
import time
import httpx

async def test_tags():
    url = "http://127.0.0.1:11434/api/chat"
    base_prompt = (
        "You are the virtual receptionist for Karthi Dental Clinic.\n"
        "Business Profile:\n"
        "Name: Karthi Dental Clinic\n"
        "Hours: Monday-Saturday, 9 AM-6 PM\n"
        "Doctors: Dr. Priya, Dentist (Monday-Saturday, 9 AM-1 PM)\n"
        "Services: Teeth cleaning (₹1500), Root canal treatment (₹4500)\n\n"
        "Rules:\n"
        "- Respond directly to the user's question.\n"
        "- Keep greetings to one short sentence: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'\n"
        "- Keep ordinary answers concise, normally 1-3 sentences.\n"
        "- Use only facts in business profile. If unavailable, politely state you don't have that information.\n"
    )

    tags = [
        ("<|think_off|>", f"<|think_off|>\n{base_prompt}"),
        ("/no_think", f"/no_think\n{base_prompt}"),
        ("prefilled assistant", base_prompt),
    ]

    async with httpx.AsyncClient(timeout=30.0) as client:
        for tag_name, sys_prompt in tags:
            print(f"\n==================== Testing {tag_name} ====================")
            for q in ["hi", "who is the doctor?", "What are your timings?", "Do you have a cardiologist?"]:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": q}
                ]
                if tag_name == "prefilled assistant":
                    # assistant already replied with empty think or closed think
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "<think>\n</think>\nHello! Welcome to Karthi Dental Clinic. How can I help you?"},
                        {"role": "user", "content": q}
                    ]

                t0 = time.perf_counter()
                res = await client.post(url, json={
                    "model": "qwen3:4b",
                    "messages": messages,
                    "think": False,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 100}
                })
                dur = (time.perf_counter() - t0) * 1000
                d = res.json()
                msg = d.get("message", {})
                cnt = msg.get("content", "")
                thk = msg.get("thinking", "")
                print(f"[{dur:.0f}ms] Q: {q}")
                print("  CONTENT:", cnt[:120].encode('ascii', 'backslashreplace').decode())
                if thk:
                    print("  THINKING:", thk[:60].encode('ascii', 'backslashreplace').decode())

asyncio.run(test_tags())
