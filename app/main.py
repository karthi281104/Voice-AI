from __future__ import annotations

import json
import os
import re
import sqlite3
import time

import httpx
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DB_PATH = Path(os.getenv("VOICEDESK_DB", str(ROOT / "voicedesk.db")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))

app = FastAPI(title="VoiceDesk AI MVP", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class Service(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    price: str = Field(default="Not provided", max_length=80)
    duration: str = Field(default="Not provided", max_length=80)
    description: str = Field(default="", max_length=500)


class Doctor(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    specialty: str = Field(default="", max_length=120)
    availability: str = Field(default="Not provided", max_length=250)


class FAQ(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1, max_length=1000)


class BusinessProfile(BaseModel):
    name: str = Field(default="My Clinic", min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    address: str = Field(default="", max_length=500)
    hours: str = Field(default="", max_length=300)
    default_language: str = Field(default="en-IN", pattern=r"^(en-IN|ta-IN)$")
    services: list[Service] = Field(default_factory=list, max_length=100)
    doctors: list[Doctor] = Field(default_factory=list, max_length=100)
    faqs: list[FAQ] = Field(default_factory=list, max_length=200)


class ChatMessage(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    language: str = Field(default="en-IN", pattern=r"^(en-IN|ta-IN)$")
    history: list[ChatMessage] = Field(default_factory=list, max_length=30)


class ChatResponse(BaseModel):
    answer: str
    engine: str = Field(default="rule_based", description="Engine used: 'llm' or 'rule_based'")
    source: str = Field(description="Source identifier: 'llm', 'service', 'doctor', 'faq', 'business', 'greeting', or 'fallback'")
    model: str | None = Field(default=None, description="Active LLM model if engine is llm")
    business_name: str
    response_time_ms: float = Field(default=0.0, description="Server turnaround time in milliseconds")


def check_ollama_status() -> dict[str, Any]:
    """Check if Ollama server is reachable and if the configured model is ready."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=2.5)
        if response.status_code == 200:
            payload = response.json()
            models = [str(m.get("name", "")) for m in payload.get("models", [])]
            model_found = any(
                m == OLLAMA_MODEL or m.startswith(f"{OLLAMA_MODEL}:") or m.split(":")[0] == OLLAMA_MODEL.split(":")[0]
                for m in models
            )
            return {
                "connected": True,
                "base_url": OLLAMA_BASE_URL,
                "configured_model": OLLAMA_MODEL,
                "model_available": model_found,
                "available_models": models,
            }
    except Exception as exc:
        return {
            "connected": False,
            "base_url": OLLAMA_BASE_URL,
            "configured_model": OLLAMA_MODEL,
            "model_available": False,
            "error": str(exc),
        }
    return {
        "connected": False,
        "base_url": OLLAMA_BASE_URL,
        "configured_model": OLLAMA_MODEL,
        "model_available": False,
    }


def db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session():
    conn = db_connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def profile_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "name": row["name"],
        "description": row["description"],
        "address": row["address"],
        "hours": row["hours"],
        "default_language": row["default_language"],
        "services": json.loads(row["services_json"]),
        "doctors": json.loads(row["doctors_json"]),
        "faqs": json.loads(row["faqs_json"]),
    }


def init_db() -> None:
    with db_session() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS business_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                address TEXT NOT NULL DEFAULT '',
                hours TEXT NOT NULL DEFAULT '',
                default_language TEXT NOT NULL DEFAULT 'en-IN',
                services_json TEXT NOT NULL DEFAULT '[]',
                doctors_json TEXT NOT NULL DEFAULT '[]',
                faqs_json TEXT NOT NULL DEFAULT '[]'
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO business_profile
            (id, name, description, address, hours, default_language,
             services_json, doctors_json, faqs_json)
            VALUES (1, 'ABC Dental Clinic',
                    'A sample dental clinic. Replace this information with your business details.',
                    'Chennai, Tamil Nadu',
                    'Monday-Saturday, 9 AM-6 PM',
                    'en-IN',
                    '[{"name":"Teeth cleaning","price":"₹1500","duration":"30 minutes","description":"Cleaning and polishing"}]',
                    '[{"name":"Dr. Priya","specialty":"Dentist","availability":"Monday-Saturday, 9 AM-1 PM"}]',
                    '[{"question":"Do you accept walk-ins?","answer":"Please call the clinic before visiting so the team can confirm availability."}]')
        """)


def get_profile() -> dict[str, Any]:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM business_profile WHERE id = 1").fetchone()
    if row is None:
        raise RuntimeError("Business profile has not been initialized.")
    return profile_to_dict(row)


def save_profile(profile: BusinessProfile) -> dict[str, Any]:
    data = profile.model_dump()
    with db_session() as conn:
        conn.execute("""
            INSERT INTO business_profile
            (id, name, description, address, hours, default_language,
             services_json, doctors_json, faqs_json)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              name=excluded.name,
              description=excluded.description,
              address=excluded.address,
              hours=excluded.hours,
              default_language=excluded.default_language,
              services_json=excluded.services_json,
              doctors_json=excluded.doctors_json,
              faqs_json=excluded.faqs_json
        """, (
            data["name"], data["description"], data["address"], data["hours"],
            data["default_language"],
            json.dumps(data["services"], ensure_ascii=False),
            json.dumps(data["doctors"], ensure_ascii=False),
            json.dumps(data["faqs"], ensure_ascii=False),
        ))
    return get_profile()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def keyword_score(query: str, candidate: str) -> int:
    # Simple baseline matching; replace with evaluated multilingual retrieval later.
    q_tokens = set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", normalize(query)))
    c_tokens = set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", normalize(candidate)))
    ignored = {"the", "is", "a", "an", "do", "you", "what", "how", "much", "tell", "me", "about",
               "என்ன", "எவ்வளவு", "எங்கே", "உள்ளதா", "எப்போது"}
    return len((q_tokens - ignored) & c_tokens)


def build_system_prompt(profile: dict[str, Any], language: str) -> str:
    lang_instruction = (
        "Respond in natural, polite Tamil (தமிழ்). Use Tamil script only."
        if language == "ta-IN" else
        "Respond in natural, polite Indian English."
    )
    fallback_phrase = (
        "இந்த விவரத்தை உறுதிப்படுத்தும் தகவல் எங்களிடம் இல்லை. தயவுசெய்து கிளினிக்குடன் நேரடியாக தொடர்பு கொள்ளவும்."
        if language == "ta-IN" else
        "I cannot confirm that from the business information available. Please contact the clinic directly to verify."
    )
    return f"""You are VoiceDesk, an AI voice receptionist for '{profile.get("name", "our clinic")}'.

STRICT GROUNDING & BEHAVIOR RULES:
1. Business Profile JSON (The ONLY source of truth):
{json.dumps(profile, ensure_ascii=False)}

2. Strict Factuality & Unknown Questions Handling:
- Answer ONLY using facts explicitly stated in the Business Profile above.
- NEVER invent, extrapolate, or guess prices, services, doctors, procedures, addresses, hours, or appointment availability.
- If the caller asks about ANY service, doctor, price, procedure, or policy NOT confirmed in the profile above (for example: unlisted surgeries, MRI, CT scans, cosmetic procedures, unlisted discounts), you MUST say:
"{fallback_phrase}"
- Doctor availability is descriptive information only; you cannot book, cancel, or modify calendar appointments.

3. Spoken Voice Conciseness:
- Language: {lang_instruction}
- You are speaking aloud over the phone/browser: keep your reply to 1-2 short spoken sentences (maximum 40 words).
- Be conversational and direct.
- Put any internal reasoning inside <think>...</think> tags. Output only the final spoken response after </think>.
"""


def answer_question_llm(
    message: str,
    language: str,
    profile: dict[str, Any],
    history: list[ChatMessage] | None = None,
) -> tuple[str, str, str, str | None] | None:
    system_prompt = build_system_prompt(profile, language)
    ollama_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if history:
        for item in history[-6:]:
            if item.content.strip():
                ollama_messages.append({"role": item.role, "content": item.content.strip()})
    ollama_messages.append({"role": "user", "content": message.strip()})

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": ollama_messages,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 300,
                },
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            payload = response.json()
            msg_obj = payload.get("message", {})
            answer = str(msg_obj.get("content", "")).strip()
            if not answer and "thinking" in msg_obj:
                answer = str(msg_obj.get("thinking", "")).strip()
            if "</think>" in answer:
                answer = answer.rsplit("</think>", 1)[-1].strip()
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

            # Sanity check: if model leaked reasoning preamble like "Step 1:" or "We are given a query"
            if re.search(r"^(Step \d|We are given|The user asks|Let me check)", answer, flags=re.IGNORECASE):
                # Try to extract the last paragraph or line
                lines = [l.strip() for l in answer.splitlines() if l.strip()]
                candidate = lines[-1] if lines else ""
                if any(w in candidate for w in ("வணக்கம்", "தொடர்பு", "இல்லை", "confirm", "contact", "₹", "Rs")):
                    answer = candidate
                else:
                    return None

            if answer:
                return answer, "llm", "llm", OLLAMA_MODEL
    except Exception:
        pass
    return None


def answer_question_rules(
    message: str,
    language: str,
    profile: dict[str, Any],
    history: list[ChatMessage] | None = None,
) -> tuple[str, str, str, str | None]:
    """Deterministic fallback engine when LLM is unavailable or offline."""
    q = normalize(message)
    ta = language == "ta-IN"
    hello = "வணக்கம்!" if ta else "Hello!"
    not_sure = (
        "இந்த விவரத்தை உறுதிப்படுத்தும் தகவல் எங்களிடம் இல்லை. தயவுசெய்து கிளினிக்குடன் நேரடியாக தொடர்பு கொள்ளவும்."
        if ta else
        "I can't confirm that from the business information available. Please contact the business to verify."
    )

    # Context resolution from previous history turn if pronouns or follow-up references exist
    combined_query = q
    if history:
        pronouns = ("he", "she", "her", "him", "his", "they", "them", "it", "that", "doctor", "service",
                    "அவர்", "அவரது", "டாக்டர்", "சேவை")
        if any(re.search(rf"\b{p}\b", q) for p in ("he", "she", "her", "him", "his", "they", "them", "it", "that")) or any(p in q for p in ("அவர்", "அவரது", "டாக்டர்", "சேவை")):
            recent_context = " ".join(item.content for item in history[-2:])
            combined_query = f"{recent_context} {q}"

    # 1. Greetings
    if re.search(r"\b(hello|hi|hey|vanakkam)\b", q) or any(w in q for w in ("வணக்கம்", "ஹலோ", "காலை வணக்கம்")):
        greeting_text = (
            f"{hello} Thank you for contacting {profile['name']}. How can I assist you today?"
            if not ta else
            f"{hello} {profile['name']} தொடர்பு கொண்டதற்கு நன்றி. நான் உங்களுக்கு எப்படி உதவலாம்?"
        )
        return greeting_text, "rule_based", "greeting", None

    # 2. FAQs
    best_faq = None
    best_score = 0
    for faq in profile.get("faqs", []):
        score = keyword_score(message, faq["question"])
        if score > best_score:
            best_faq, best_score = faq, score
    if best_faq and best_score >= 2:
        return best_faq["answer"], "rule_based", "faq", None

    # 3. Doctors (Direct name query or pronoun follow-up)
    for doc in profile.get("doctors", []):
        doc_name = normalize(doc["name"])
        doc_tokens = set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", doc_name)) - {"dr", "doctor", "டாக்டர்"}
        is_doc_mentioned = any(t in normalize(combined_query) for t in doc_tokens) or (doc_name in normalize(combined_query))
        if is_doc_mentioned:
            if any(w in q for w in ("when", "time", "timing", "available", "availability", "schedule", "நேரம்", "எப்போது")):
                ans = (
                    f"{doc['name']} is available {doc['availability']}."
                    if not ta else
                    f"{doc['name']} கிடைக்கும் நேரம்: {doc['availability']}."
                )
                return ans, "rule_based", "doctor", None
            details = [doc["name"]]
            if doc.get("specialty"):
                details.append(f"Specialty: {doc['specialty']}" if not ta else f"சிறப்பு: {doc['specialty']}")
            if doc.get("availability") and doc["availability"] != "Not provided":
                details.append(f"Availability: {doc['availability']}" if not ta else f"நேரம்: {doc['availability']}")
            ans = ". ".join(details) + "."
            return ans, "rule_based", "doctor", None

    # 4. Services & Pricing
    for service in profile.get("services", []):
        name = normalize(service["name"])
        serv_tokens = set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", name))
        matched = (name in normalize(combined_query)) or bool(serv_tokens & set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", normalize(combined_query))))
        if matched:
            if any(w in q for w in ("price", "cost", "charge", "fee", "rate", "much", "விலை", "கட்டணம்", "எவ்வளவு", "ரூபாய்")):
                if service.get("price") and service["price"] != "Not provided":
                    ans = (
                        f"{service['name']} costs {service['price']}."
                        if not ta else
                        f"{service['name']} சேவையின் கட்டணம் {service['price']}."
                    )
                    return ans, "rule_based", "service", None
            details = [service["name"]]
            if service.get("price") and service["price"] != "Not provided":
                details.append(f"price: {service['price']}" if not ta else f"கட்டணம்: {service['price']}")
            if service.get("duration") and service["duration"] != "Not provided":
                details.append(f"duration: {service['duration']}" if not ta else f"நேரம்: {service['duration']}")
            if service.get("description"):
                details.append(service["description"])
            return (". ".join(details) + "."), "rule_based", "service", None

    # 5. Opening Hours
    if any(w in q for w in ("hour", "open", "close", "timing", "when", "days", "நேரம்", "திறந்திருக்", "மூடும்", "எப்போது")):
        if profile.get("hours"):
            ans = (
                f"Our business hours are {profile['hours']}."
                if not ta else
                f"எங்கள் வேலை நேரம்: {profile['hours']}."
            )
            return ans, "rule_based", "business", None

    # 6. Address / Location
    if any(w in q for w in ("address", "location", "where", "located", "place", "directions", "எங்கே", "முகவரி", "இடம்", "எங்கு")):
        if profile.get("address"):
            ans = (
                f"We are located at {profile['address']}."
                if not ta else
                f"எங்கள் முகவரி: {profile['address']}."
            )
            return ans, "rule_based", "business", None

    # 7. General services query
    if any(w in q for w in ("service", "services", "treatment", "treatments", "offer", "provide", "சேவை", "சேவைகள்", "சிகிச்சை")):
        if profile.get("services"):
            names = ", ".join(item["name"] for item in profile["services"])
            ans = (
                f"We currently provide these listed services: {names}."
                if not ta else
                f"நாங்கள் வழங்கும் சேவைகள்: {names}."
            )
            return ans, "rule_based", "business", None

    # 8. General doctors query
    if any(w in q for w in ("doctors", "staff", "dentists", "physicians", "மருத்துவர்கள்")):
        if profile.get("doctors"):
            names = ", ".join(f"{item['name']} ({item['specialty']})" if item.get('specialty') else item['name'] for item in profile["doctors"])
            ans = (
                f"Our listed doctors are: {names}."
                if not ta else
                f"எங்கள் மருத்துவர்கள்: {names}."
            )
            return ans, "rule_based", "doctor", None

    # 9. Fallback / Unconfirmed question
    return not_sure, "rule_based", "fallback", None


def answer_question(
    message: str,
    language: str,
    profile: dict[str, Any],
    history: list[ChatMessage] | None = None,
) -> tuple[str, str, str, str | None]:
    """Answer with local Ollama LLM if available; fall back to deterministic rules if offline."""
    llm_result = answer_question_llm(message, language, profile, history)
    if llm_result is not None:
        return llm_result
    return answer_question_rules(message, language, profile, history)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        with db_session() as conn:
            conn.execute("SELECT 1").fetchone()
        db_ok = True
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    ollama_info = check_ollama_status()
    engine = "llm" if (ollama_info["connected"] and ollama_info["model_available"]) else "rule_based"
    return {
        "status": "ok",
        "database": "connected",
        "ollama": ollama_info,
        "engine": engine,
    }


@app.get("/api/llm/status")
def llm_status() -> dict[str, Any]:
    return check_ollama_status()


@app.get("/api/business", response_model=BusinessProfile)
def read_business() -> dict[str, Any]:
    return get_profile()


@app.put("/api/business", response_model=BusinessProfile)
def update_business(profile: BusinessProfile) -> dict[str, Any]:
    return save_profile(profile)


def stream_chat_response(request: ChatRequest, profile: dict[str, Any]):
    t0 = time.perf_counter()
    ollama_ok = False
    try:
        check = httpx.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=1.5)
        ollama_ok = check.status_code == 200
    except Exception:
        ollama_ok = False

    if ollama_ok:
        system_prompt = build_system_prompt(profile, request.language)
        ollama_messages = [{"role": "system", "content": system_prompt}]
        if request.history:
            for item in request.history[-6:]:
                if item.content.strip():
                    ollama_messages.append({"role": item.role, "content": item.content.strip()})
        ollama_messages.append({"role": "user", "content": request.message.strip()})

        accumulated: list[str] = []
        try:
            with httpx.stream(
                "POST",
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "stream": True,
                    "messages": ollama_messages,
                    "options": {"temperature": 0.2, "num_predict": 250},
                },
                timeout=OLLAMA_TIMEOUT_SECONDS,
            ) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        content_piece = msg.get("content", "")
                        if content_piece:
                            clean_piece = re.sub(r"</?think>", "", content_piece)
                            if clean_piece:
                                accumulated.append(clean_piece)
                                yield json.dumps({"type": "token", "content": clean_piece}) + "\n"
                        if chunk.get("done", False):
                            break
                    except Exception:
                        continue

            if accumulated:
                full_text = "".join(accumulated).strip()
                if not re.search(r"^(Step \d|We are given|The user asks|Let me check)", full_text, flags=re.IGNORECASE):
                    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
                    yield json.dumps({
                        "type": "done",
                        "answer": full_text,
                        "engine": "llm",
                        "source": "llm",
                        "model": OLLAMA_MODEL,
                        "business_name": profile["name"],
                        "response_time_ms": duration_ms,
                    }) + "\n"
                    return
        except Exception:
            pass

    # Deterministic rule engine fallback
    answer, engine, source, model = answer_question_rules(
        request.message, request.language, profile, request.history
    )
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    yield json.dumps({"type": "token", "content": answer}) + "\n"
    yield json.dumps({
        "type": "done",
        "answer": answer,
        "engine": engine,
        "source": source,
        "model": model,
        "business_name": profile["name"],
        "response_time_ms": duration_ms,
    }) + "\n"


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    t0 = time.perf_counter()
    profile = get_profile()
    answer, engine, source, model = answer_question(
        request.message,
        request.language,
        profile,
        request.history,
    )
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    return ChatResponse(
        answer=answer,
        engine=engine,
        source=source,
        model=model,
        business_name=profile["name"],
        response_time_ms=duration_ms,
    )


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    profile = get_profile()
    return StreamingResponse(
        stream_chat_response(request, profile),
        media_type="application/x-ndjson",
    )
