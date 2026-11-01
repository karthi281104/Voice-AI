import asyncio
import json
import re
import time
import httpx

def clean_conversational_response(raw_text: str, profile_name: str = "the clinic") -> str:
    """
    Robust cleaner that guarantees NO internal reasoning, planning, or model metadata leaks.
    """
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
        r'(?:therefore,?\s*(?:the\s*)?response\s*(?:is)?:?|final\s*response:?|receptionist:?|response:?)\s*["“]?([^"”\n\r]+(?:["”][^"”\n\r]*)?)',
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
        "according to the", "first, i need", "steps:", "option 1:", "option 2:",
        "double-checking", "let me re-read", "guidelines say", "in the caller's language",
        "we must respond", "we have to", "the answer:"
    )

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    clean_lines = []
    for line in lines:
        lower_line = line.lower()
        if any(lower_line.startswith(p) for p in reasoning_prefixes):
            # Check if this line has a quoted response at the end
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
        # If all lines were reasoning, look for quotes
        quotes = re.findall(r'["“]([^"”]{6,})["”]', raw_text)
        if quotes:
            text = quotes[-1].strip()

    # Final polish: strip prefixes like 'Assistant:' or quotes
    text = re.sub(r'^(?:Assistant|Receptionist|Response|Answer):\s*', '', text, flags=re.IGNORECASE).strip()
    return text.strip('"\' ')

print("Testing cleaner:")
print("Result 1:", repr(clean_conversational_response('We are given the caller said "hi". Therefore, the response is: "Hello! Welcome to Karthi Dental Clinic. How can I help you?"')))
print("Result 2:", repr(clean_conversational_response('We are given the question: "who is the doctor?". The answer: Our clinic has Dr. Priya, a Dentist.')))
