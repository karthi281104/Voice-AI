import asyncio
import json
import os
import re
import sys
import time
import httpx

sys.path.insert(0, os.path.abspath("."))

def clean_conversational_response(raw_text: str, profile_name: str = "our clinic") -> str:
    if not raw_text:
        return ""
    text = raw_text.strip()
    
    # 1. Strip think blocks
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</?think>", "", text).strip()

    # 2. Extract explicit response markers if reasoning leaked
    response_marker = re.search(
        r'(?:therefore,?\s*(?:the\s*)?response\s*(?:is)?:?|final\s*response:?|receptionist:?|response:?|the answer\s*(?:is)?:?)\s*["“]?([^"”\n\r]+(?:["”][^"”\n\r]*)?)',
        text,
        flags=re.IGNORECASE
    )
    if response_marker:
        candidate = response_marker.group(1).strip().strip('"\' ')
        if len(candidate) > 5 and not candidate.lower().startswith("we are"):
            return candidate

    # 3. Detect reasoning preamble lines and filter them
    reasoning_prefixes = (
        "we are given", "we are the receptionist", "the user said", "the user asks",
        "the caller", "let me check", "i need to respond", "possible response",
        "according to", "first, i need", "steps:", "option 1:", "option 2:",
        "double-checking", "let me re-read", "guidelines say", "in the caller's language",
        "we must respond", "we have to"
    )

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    clean_lines = []
    for line in lines:
        lower_line = line.lower()
        if any(lower_line.startswith(p) for p in reasoning_prefixes):
            q_match = re.search(r'["“]([^"”]{6,})["”]', line)
            if q_match:
                clean_lines.append(q_match.group(1).strip())
            continue
        if re.match(r"^\d+\.\s+", line):
            continue
        clean_lines.append(line)

    if clean_lines:
        text = " ".join(clean_lines).strip()
    else:
        quotes = re.findall(r'["“]([^"”]{6,})["”]', raw_text)
        if quotes:
            text = quotes[-1].strip()

    text = re.sub(r'^(?:Assistant|Receptionist|Response|Answer):\s*', '', text, flags=re.IGNORECASE).strip()
    return text.strip('"\' ')

async def test_stream_pipeline(q: str, lang: str, profile: dict):
    from app.main import build_system_prompt
    sys_prompt = build_system_prompt(profile, lang)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": q}
    ]
    
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": "qwen3:4b",
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 250, "think": False}
    }
    
    t_start = time.perf_counter()
    streamed_tokens = []
    reasoning_buffer = ""
    stream_started = False
    in_think_block = False
    first_token_time = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                if not content:
                    if chunk.get("done", False):
                        break
                    continue

                if "<think>" in content:
                    in_think_block = True
                if in_think_block:
                    if "</think>" in content:
                        in_think_block = False
                        content = content.split("</think>", 1)[-1]
                    else:
                        continue

                clean_piece = re.sub(r"</?think>", "", content)
                if not clean_piece:
                    continue

                if not stream_started:
                    reasoning_buffer += clean_piece
                    buf_low = reasoning_buffer.lower().strip()
                    starts_reasoning = any(
                        buf_low.startswith(p)
                        for p in ("we are given", "we are the receptionist", "the user said", "the user asks",
                                  "the caller", "let me check", "i need to respond", "possible response",
                                  "according to", "first, i need", "steps:", "okay, the user")
                    )
                    if starts_reasoning or (len(buf_low) < 15 and any(p.startswith(buf_low) for p in ("we are", "the user", "let me", "okay,"))):
                        resp_match = re.search(r'(?:therefore,?\s*(?:the\s*)?response\s*(?:is)?:?|response:?|receptionist:?)\s*["“]?([^"”\n\r]+)', reasoning_buffer, re.I)
                        if resp_match:
                            clean_text = resp_match.group(1).strip()
                            if clean_text:
                                stream_started = True
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                streamed_tokens.append(clean_text)
                        continue
                    else:
                        stream_started = True
                        clean_piece = reasoning_buffer
                        reasoning_buffer = ""

                if clean_piece:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    streamed_tokens.append(clean_piece)

                if chunk.get("done", False):
                    break

    dur = (time.perf_counter() - t_start) * 1000
    ttft = (first_token_time - t_start) * 1000 if first_token_time else dur
    final_text = clean_conversational_response("".join(streamed_tokens))
    if not final_text and reasoning_buffer:
        final_text = clean_conversational_response(reasoning_buffer)

    return final_text, ttft, dur

async def main():
    profile = {
        "name": "Karthi Dental Clinic",
        "description": "State-of-the-art dental care clinic.",
        "address": "12 Mount Road, Chennai",
        "hours": "Monday-Saturday, 9 AM-6 PM",
        "default_language": "en-IN",
        "services": [
            {"name": "Teeth cleaning", "price": "₹1500", "duration": "30 mins", "description": "Professional teeth cleaning."},
            {"name": "Root canal treatment", "price": "₹4500", "duration": "60 mins", "description": "Pain-free root canal treatment."}
        ],
        "doctors": [
            {"name": "Dr. Priya", "specialty": "Dentist", "availability": "Monday-Saturday, 9 AM-1 PM"}
        ],
        "faqs": [
            {"question": "Do I need an appointment?", "answer": "Walk-ins are welcome, but appointments are recommended."}
        ]
    }
    
    test_cases = [
        ("hi", "en-IN"),
        ("who is the doctor?", "en-IN"),
        ("What are your timings?", "en-IN"),
        ("உங்க கிளினிக்ல என்னென்ன சேவைகள் இருக்கு?", "ta-IN"),
        ("Do you have a cardiologist?", "en-IN"),
        ("Who are you?", "en-IN"),
    ]

    for q, lang in test_cases:
        ans, ttft, total = await test_stream_pipeline(q, lang, profile)
        print(f"\nQ: {q}")
        print(f"TTFT: {ttft:.0f}ms | Total: {total:.0f}ms")
        print(f"Answer: {ans}")

asyncio.run(main())
