import asyncio
import json
import httpx

async def test_ollama():
    url = "http://127.0.0.1:11434/api/chat"
    
    # Test 1: root think: False
    payload1 = {
        "model": "qwen3:4b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 100}
    }
    
    # Test 2: options think: False
    payload2 = {
        "model": "qwen3:4b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 100, "think": False}
    }
    
    # Test 3: both
    payload3 = {
        "model": "qwen3:4b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 100, "think": False}
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        print("--- Test 1 (root think: False) ---")
        try:
            r1 = await client.post(url, json=payload1)
            print("Status:", r1.status_code)
            d1 = r1.json()
            msg1 = d1.get("message", {})
            print("Keys in msg1:", list(msg1.keys()))
            print("msg1.thinking:", repr(msg1.get("thinking", "")))
            print("msg1.content:", repr(msg1.get("content", "")))
        except Exception as e:
            print("Error 1:", e)

        print("\n--- Test 2 (options think: False) ---")
        try:
            r2 = await client.post(url, json=payload2)
            print("Status:", r2.status_code)
            d2 = r2.json()
            msg2 = d2.get("message", {})
            print("Keys in msg2:", list(msg2.keys()))
            print("msg2.thinking:", repr(msg2.get("thinking", "")))
            print("msg2.content:", repr(msg2.get("content", "")))
        except Exception as e:
            print("Error 2:", e)

        print("\n--- Test 4: Streaming with think: False ---")
        payload_stream = {
            "model": "qwen3:4b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "think": False,
            "options": {"temperature": 0.2, "num_predict": 100, "think": False}
        }
        chunks = []
        async with client.stream("POST", url, json=payload_stream) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                c = json.loads(line)
                m = c.get("message", {})
                chunks.append((m.get("content", ""), m.get("thinking", "")))
        print(f"Received {len(chunks)} chunks.")
        for idx, (cnt, thk) in enumerate(chunks[:10]):
            print(f" Chunk {idx}: content={repr(cnt)}, thinking={repr(thk)}")
        if len(chunks) > 10:
            print(" ...")
            for idx, (cnt, thk) in enumerate(chunks[-5:], start=len(chunks)-5):
                print(f" Chunk {idx}: content={repr(cnt)}, thinking={repr(thk)}")

asyncio.run(test_ollama())
