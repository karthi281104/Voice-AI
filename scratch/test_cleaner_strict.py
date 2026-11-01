import re

def is_reasoning_text(text: str) -> bool:
    if not text:
        return True
    low = text.lower()
    indicators = (
        "we are given", "let me check", "the user asks", "the user said",
        "the caller", "okay, let me", "okay, the user", "step 1", "step 2",
        "should be:", "rules:", "guidelines", "concise (1-3 sentences)",
        "i need to respond", "possible response", "translates to",
        "first, i need", "without a greeting", "double-checking",
        "we must respond", "we have to", "the answer:", "according to"
    )
    return any(ind in low for ind in indicators)

def clean_conversational_response(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.strip()
    
    # 1. If </think> is present, check text after </think>
    if "</think>" in text:
        after_think = text.rsplit("</think>", 1)[-1].strip()
        if after_think and not is_reasoning_text(after_think):
            return after_think.strip('"\' ')
    
    # 2. Check for explicit quoted response after "Therefore the response is:"
    match = re.search(
        r'(?:therefore,?\s*(?:the\s*)?response\s*(?:is)?:?|response:?|receptionist:?)\s*["“]([^"”\n\r]{5,})["”]',
        text,
        re.I
    )
    if match:
        cand = match.group(1).strip()
        if not is_reasoning_text(cand):
            return cand.strip('"\' ')
            
    # 3. Check if whole text is clean without reasoning
    text_no_tags = re.sub(r"</?think>", "", text).strip()
    if not is_reasoning_text(text_no_tags):
        return text_no_tags.strip('"\' ')

    # 4. If all else is reasoning, return empty so system safely falls back
    return ""

test_strings = [
    # 1. Clean answer
    "Hello! Welcome to Karthi Dental Clinic. How can I help you?",
    # 2. Reasoning before think tag
    "Okay, let me think.\n</think>\nHello! Welcome to Karthi Dental Clinic. How can I help you?",
    # 3. Pure reasoning leaked
    "We are given the user said 'hi'. The rules say: concise (1-3 sentences). Therefore, the response is: \"Hello! Welcome to Karthi Dental Clinic. How can I help you?\"",
    # 4. Pure reasoning without answer
    "Okay, let me try to figure out how to respond to this user's question. The user is asking in Tamil: ...",
    # 5. Garbage extracted lines like 'rules:'
    "rules:"
]

for s in test_strings:
    res = clean_conversational_response(s)
    print("IN:", repr(s[:60]))
    print("OUT:", repr(res))
    print()
