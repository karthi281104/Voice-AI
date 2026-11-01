import re

def clean_conversational_response(raw_text: str) -> str:
    """
    Strips internal thoughts, reasoning tags, and chain-of-thought preambles
    leaving only the direct, user-facing receptionist response.
    """
    text = raw_text.strip()
    
    # 1. Split on closing think tag if present
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    
    # 2. Remove any remaining XML think blocks or standalone tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"</?think>", "", text).strip()
    
    # 3. Detect chain-of-thought preambles if model leaked reasoning without tags
    reasoning_indicators = (
        "we are given", "we are the receptionist", "the user said", "the user asks",
        "the caller", "let me check", "i need to respond", "possible response",
        "according to the", "first, i need", "steps:", "option 1:", "option 2:",
        "double-checking", "let me re-read", "guidelines say"
    )
    
    has_reasoning = any(ind in text.lower() for ind in reasoning_indicators)
    
    if has_reasoning:
        # Check if there is an explicit "Response:" or "Therefore, the response is:"
        match = re.search(r'(?:therefore,?\s*(?:the\s*)?response\s*(?:is)?:?|response:?|receptionist:?)\s*["“]?([^"”\n\r]+)["”]?', text, flags=re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if len(extracted) > 5:
                return extracted.strip('"\' ')
        
        # Check if there are quoted sentences in the text (often the final answer)
        quotes = re.findall(r'["“]([^"”]{5,})["”]', text)
        if quotes:
            # Return the last substantial quoted string
            for q in reversed(quotes):
                q_clean = q.strip()
                if not any(ind in q_clean.lower() for ind in ("hi", "who is", "what are", "we are")):
                    return q_clean
                if any(w in q_clean for w in ("Welcome", "Dr.", "hours", "clinic", "வணக்கம்")):
                    return q_clean
            return quotes[-1].strip()

        # Split into lines and discard reasoning lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_lines = []
        for line in lines:
            if any(line.lower().startswith(ind) for ind in reasoning_indicators):
                continue
            if re.match(r"^\d+\.\s+", line):
                continue
            if line.lower().startswith("in the caller's language"):
                continue
            clean_lines.append(line)
        
        if clean_lines:
            text = " ".join(clean_lines).strip()

    # Clean any leftover prefixes like "Response:" or "Receptionist:"
    text = re.sub(r'^(?:Assistant|Receptionist|Response|Answer):\s*', '', text, flags=re.IGNORECASE).strip()
    text = text.strip('"\' ')
    return text

# Test with our raw outputs from earlier:
samples = [
    """We are given the caller said "hi", so we follow the rule: 
   If the caller greets you (e.g. 'hi'), say: 'Hello! Welcome to Karthi Dental Clinic. How can I help you?'

We must answer in the caller's language. The caller used English ("hi"), so we respond in English.

Therefore, the response is: 
   "Hello! Welcome to Karthi Dental Clinic. How can I help you?"

But note: the""",
    """We are given the user's question: "who is the doctor?"

According to the business profile:
- Doctors: Dr. Priya, Dentist (Monday-Saturday, 9 AM-1 PM)

We are to answer directly as the clinic receptionist in 1-2 spoken sentences.

The answer: Our clinic has Dr. Priya, a Dentist.""",
    """Okay, the user just said "hi". Let me check the guidelines. 

First, I need to greet them properly. 
</think>

Hello! Welcome to Karthi Dental Clinic. How can I help you?""",
    """We are given the caller's question: "What are your timings?"
 The clinic details state: Hours: Monday-Saturday, 9 AM-6 PM
 
 Therefore, the response is: "Our business hours are Monday-Saturday, 9 AM to 6 PM." """
]

for s in samples:
    print("IN:")
    print(s[:60] + "...")
    print("OUT:", repr(clean_conversational_response(s)))
    print("-" * 40)
