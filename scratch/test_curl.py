import asyncio
import json
import httpx

async def test_curl_equivalent():
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": "qwen3:4b",
        "messages": [
            {"role": "user", "content": "hi"}
        ],
        "think": False,
        "stream": False
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(url, json=payload)
        d = res.json()
        print("Raw response:")
        print(json.dumps(d, indent=2))

asyncio.run(test_curl_equivalent())
