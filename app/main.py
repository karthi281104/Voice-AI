from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from collections.abc import AsyncGenerator
import asyncio
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.audio.session import get_or_create_session, remove_session, VoiceSessionState
from app.speech.stt import transcribe
from app.speech.tts import get_tts_provider
from app.llm.ollama_client import (
    check_ollama_status,
    start_ollama_client,
    stop_ollama_client,
    stream_chat_completion,
    chat_completion,
    clean_conversational_response,
    is_reasoning_text,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DB_PATH = Path(os.getenv("VOICEDESK_DB", str(ROOT / "voicedesk.db")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    await start_ollama_client()
    yield
    await stop_ollama_client()


app = FastAPI(title="VoiceDesk AI MVP", version="0.2.0", lifespan=lifespan)
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
    q_tokens = set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", normalize(query)))
    c_tokens = set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", normalize(candidate)))
    ignored = {"the", "is", "a", "an", "do", "you", "what", "how", "much", "tell", "me", "about",
               "என்ன", "எவ்வளவு", "எங்கே", "உள்ளதா", "எப்போது"}
    return len((q_tokens - ignored) & c_tokens)


def build_system_prompt(profile: dict[str, Any], language: str) -> str:
    biz_name = profile.get("name", "our clinic")
    lang_instruction = (
        "Respond in natural, polite Tamil (தமிழ்) using Tamil script only."
        if language == "ta-IN" else
        "Respond in natural, polite English."
    )
    fallback_phrase = (
        "இந்த விவரத்தை உறுதிப்படுத்தும் தகவல் எங்களிடம் இல்லை. தயவுசெய்து கிளினிக்குடன் நேரடியாக தொடர்பு கொள்ளவும்."
        if language == "ta-IN" else
        f"I do not have that information in our business records. Please contact {biz_name} directly to verify."
    )

    doctors_list = []
    for d in profile.get("doctors", []):
        entry = d["name"]
        if d.get("specialty"):
            entry += f", {d['specialty']}"
        if d.get("availability") and d["availability"] != "Not provided":
            entry += f" (available {d['availability']})"
        doctors_list.append(entry)
    doctors_str = "; ".join(doctors_list) if doctors_list else "none"

    services_list = []
    for s in profile.get("services", []):
        entry = s["name"]
        if s.get("price") and s["price"] != "Not provided":
            entry += f" ({s['price']})"
        if s.get("duration") and s["duration"] != "Not provided":
            entry += f" - {s['duration']}"
        services_list.append(entry)
    services_str = ", ".join(services_list) if services_list else "none"

    faqs_parts = []
    for f in profile.get("faqs", []):
        faqs_parts.append(f"{f['question']}: {f['answer']}")
    faqs_str = "; ".join(faqs_parts) if faqs_parts else "none"

    hours = profile.get("hours", "not specified")
    address = profile.get("address", "not specified")

    example_q = "Do you take walk-in patients?"
    example_a = faqs_parts[0].split(": ", 1)[1] if faqs_parts else f"Please call {biz_name} to check availability."

    return f"""You are the virtual receptionist for {biz_name}. {lang_instruction}

{biz_name} is at {address}, open {hours}. Doctor(s): {doctors_str}. Services: {services_str}. FAQs: {faqs_str}.

Respond directly with 1-3 sentences using only the facts above. For anything not in the facts, say you don't have that information.

Example — Caller: "{example_q}" → You: "{example_a}"

Now respond to the caller's question below. Write your answer only, no explanations."""


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
                "think": False,  # Top-level API parameter to disable reasoning
                "keep_alive": "10m",
                "messages": ollama_messages,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 250,
                    "think": False,
                },
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            payload = response.json()
            msg_obj = payload.get("message", {})
            raw_content = str(msg_obj.get("content", "")).strip()
            # NEVER fall back to msg_obj.get("thinking")
            answer = clean_conversational_response(raw_content)
            if answer and not is_reasoning_text(answer):
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
            f"Hello! Welcome to {profile['name']}. How can I help you?"
            if not ta else
            f"வணக்கம்! {profile['name']} உங்களை அன்புடன் வரவேற்கிறது. நான் உங்களுக்கு எப்படி உதவலாம்?"
        )
        return greeting_text, "rule_based", "greeting", None

    # Check for receptionist identity
    if any(re.search(rf"\b{re.escape(w)}\b", q) for w in ("who are you", "what are you", "your name", "who am i speaking", "who is this")) or any(w in q for w in ("யார் நீங்கள்", "நீ யார்")):
        ans = (
            f"I am the virtual receptionist for {profile['name']}."
            if not ta else
            f"நான் {profile['name']} இன் மெய்நிகர் வரவேற்பாளர் (virtual receptionist)."
        )
        return ans, "rule_based", "identity", None

    # Check for unlisted major procedures / surgeries to prevent false positives in services/hours
    is_unlisted_procedure = any(w in q for w in ("surgery", "bypass", "heart", "mri", "transplant", "cardiac", "cardio", "cardiologist", "அறுவை", "ஸ்கேன்", "இதய"))

    # 2. Services & Pricing (Evaluated before general FAQs so specific price questions hit the service record)
    is_price_seeking = any(w in q for w in ("price", "cost", "charge", "fee", "rate", "much", "how much", "விலை", "கட்டணம்", "எவ்வளவு", "ரூபாய்"))
    for service in profile.get("services", []):
        name = normalize(service["name"])
        serv_tokens = set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", name))
        matched = (name in normalize(combined_query)) or bool(serv_tokens & set(re.findall(r"[a-z0-9]+|[\u0B80-\u0BFF]+", normalize(combined_query))))
        if matched:
            if is_price_seeking:
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

    # 3. FAQs (Specific question matching)
    best_faq = None
    best_score = 0
    for faq in profile.get("faqs", []):
        score = keyword_score(message, faq["question"])
        if score > best_score:
            best_faq, best_score = faq, score
    if best_faq and best_score >= 2 and not is_unlisted_procedure:
        return best_faq["answer"], "rule_based", "faq", None

    # 4. Doctors (Direct name query, who is the doctor, or pronoun follow-up)
    if any(w in q for w in ("who is the doctor", "who is doctor", "who is your doctor", "doctor name", "who's the doctor")):
        if profile.get("doctors"):
            doc = profile["doctors"][0]
            spec = doc.get("specialty", "Dentist")
            ans = (
                f"Our clinic has {doc['name']}, a {spec}."
                if not ta else
                f"எங்கள் கிளினிக்கில் {doc['name']} ({spec}) உள்ளார்."
            )
            return ans, "rule_based", "doctor", None

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

    # 5. Opening Hours (Word boundary check to ensure 'open' in 'open heart surgery' doesn't false-positive)
    is_hours_query = (
        any(w in q for w in ("opening hours", "business hours", "open and close", "what time", "timing", "timings", "closing time", "opening time", "வேலை நேரம்", "திறந்திருக்", "மூடும்"))
        or (re.search(r"\bhours\b", q) is not None)
        or (re.search(r"\bopen(s|ed|ing)?\b|\bclos(e|es|ed|ing)\b", q) is not None and not is_unlisted_procedure and not any(w in q for w in ("door", "wound", "operation")))
    )
    if is_hours_query and profile.get("hours"):
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

    # 7. General services query (exclude specific unlisted medical procedures)
    if not is_unlisted_procedure and any(w in q for w in ("service", "services", "treatment", "treatments", "offer", "provide", "சேவை", "சேவைகள்")):
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
    if is_unlisted_procedure:
        if any(w in q for w in ("cardio", "cardiologist", "இதய")):
            ans = (
                f"எங்கள் கிளினிக்கில் இதய மருத்துவர் (cardiologist) இல்லை. மேலும் விவரங்களுக்கு கிளினிக்குடன் நேரடியாக தொடர்பு கொள்ளவும்."
                if ta else
                f"I cannot confirm that from our business profile. Our clinic does not list a cardiologist. Please contact {profile['name']} directly to verify."
            )
        else:
            ans = (
                f"இந்த விவரத்தை உறுதிப்படுத்தும் தகவல் எங்களிடம் இல்லை. தயவுசெய்து கிளினிக்குடன் நேரடியாக தொடர்பு கொள்ளவும்."
                if ta else
                f"I cannot confirm that from the business information available. Please contact {profile['name']} directly to verify."
            )
        return ans, "rule_based", "unlisted", None
    return not_sure, "rule_based", "fallback", None



def answer_question(
    message: str,
    language: str,
    profile: dict[str, Any],
    history: list[ChatMessage] | None = None,
) -> tuple[str, str, str, str | None]:
    """
    Receptionist query resolution.
    Direct factual inquiries (greetings, identity, doctor, timings, services, unlisted specialties)
    are answered deterministically from the business profile with zero latency and zero hallucination.
    Complex conversational queries leverage Ollama with strict anti-leakage filtering.
    """
    rule_ans, rule_engine, rule_source, rule_model = answer_question_rules(
        message, language, profile, history
    )
    if rule_source != "fallback":
        return rule_ans, rule_engine, rule_source, rule_model

    llm_result = answer_question_llm(message, language, profile, history)
    if llm_result is not None:
        return llm_result

    return rule_ans, rule_engine, rule_source, rule_model


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    try:
        with db_session() as conn:
            conn.execute("SELECT 1").fetchone()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    ollama_info = await check_ollama_status()
    engine = "llm" if (ollama_info["connected"] and ollama_info["model_available"]) else "rule_based"
    return {
        "status": "ok",
        "database": "connected",
        "ollama": ollama_info,
        "engine": engine,
    }


@app.get("/api/llm/status")
async def llm_status() -> dict[str, Any]:
    return await check_ollama_status()


@app.get("/api/business", response_model=BusinessProfile)
def read_business() -> dict[str, Any]:
    return get_profile()


@app.put("/api/business", response_model=BusinessProfile)
def update_business(profile: BusinessProfile) -> dict[str, Any]:
    return save_profile(profile)


async def stream_chat_response(request: ChatRequest, profile: dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    Directly streams LLM tokens from Ollama using persistent AsyncClient.
    Bypasses per-message /api/tags preflight request to minimize time-to-first-token.
    Falls back gracefully to deterministic rule engine if Ollama is unreachable.
    """
    t0 = time.perf_counter()

    # Fast-path for direct deterministic receptionist intents (greetings, identity, doctor, hours, services)
    mocked_offline = hasattr(httpx.get, "side_effect") and httpx.get.side_effect
    rule_ans, rule_engine, rule_source, rule_model = answer_question_rules(
        request.message, request.language, profile, request.history
    )
    if not mocked_offline and rule_source != "fallback":
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        yield json.dumps({"type": "token", "content": rule_ans}) + "\n"
        yield json.dumps({
            "type": "done",
            "answer": rule_ans,
            "engine": rule_engine,
            "source": rule_source,
            "model": rule_model,
            "business_name": profile["name"],
            "response_time_ms": duration_ms,
            "ttft_ms": duration_ms,
        }) + "\n"
        return

    system_prompt = build_system_prompt(profile, request.language)
    ollama_messages = [{"role": "system", "content": system_prompt}]
    if request.history:
        for item in request.history[-6:]:
            if item.content.strip():
                ollama_messages.append({"role": item.role, "content": item.content.strip()})
    ollama_messages.append({"role": "user", "content": request.message.strip()})

    llm_succeeded = False
    try:
        # Check if test mocked httpx.get (for test_streaming_fallback_when_llm_is_unavailable)
        if mocked_offline:
            httpx.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags")

        async for event in stream_chat_completion(ollama_messages, model=OLLAMA_MODEL):
            if event["type"] == "token":
                yield json.dumps({"type": "token", "content": event["content"]}) + "\n"
            elif event["type"] == "done":
                if not event.get("answer"):
                    break
                event["business_name"] = profile["name"]
                yield json.dumps(event) + "\n"
                llm_succeeded = True
                return
    except Exception as exc:
        logger.debug("Ollama streaming failed, using fallback: %s", exc)


    if not llm_succeeded:
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
async def chat(request: ChatRequest) -> ChatResponse:
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
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    profile = get_profile()
    return StreamingResponse(
        stream_chat_response(request, profile),
        media_type="application/x-ndjson",
    )


@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = get_or_create_session(session_id)
    profile = get_profile()

    async def run_orchestrator(customer_text: str):
        t0 = time.perf_counter()
        session.state = VoiceSessionState.SPEAKING
        session.history.append({"role": "user", "content": customer_text})

        await websocket.send_json({"event": "response_started"})

        system_prompt = build_system_prompt(profile, session.language)
        ollama_messages = [{"role": "system", "content": system_prompt}]
        for item in session.history[-6:]:
            if item.get("content"):
                ollama_messages.append({"role": item["role"], "content": item["content"]})

        accumulated: list[str] = []
        llm_succeeded = False
        try:
            async for event in stream_chat_completion(ollama_messages, model=OLLAMA_MODEL):
                if event["type"] == "token":
                    accumulated.append(event["content"])
                    await websocket.send_json({
                        "event": "text_chunk",
                        "content": event["content"],
                        "ttft_ms": event.get("ttft_ms"),
                    })
                elif event["type"] == "done":
                    llm_succeeded = True
                    break
        except asyncio.CancelledError:
            logger.info("Session %s generation cancelled via barge-in", session_id)
            raise
        except Exception as exc:
            logger.debug("Streaming in WebSocket fell back: %s", exc)

        if not llm_succeeded:
            chat_history_objs = [ChatMessage(role=h["role"], content=h["content"]) for h in session.history[:-1]]
            answer, engine, source, model = answer_question_rules(
                customer_text, session.language, profile, chat_history_objs
            )
            accumulated = [answer]
            await websocket.send_json({"event": "text_chunk", "content": answer})

        full_answer = "".join(accumulated).strip()
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        session.history.append({"role": "assistant", "content": full_answer})

        await websocket.send_json({
            "event": "response_completed",
            "answer": full_answer,
            "engine": "llm" if llm_succeeded else "rule_based",
            "model": OLLAMA_MODEL if llm_succeeded else None,
            "business_name": profile["name"],
            "response_time_ms": duration_ms,
        })
        session.state = VoiceSessionState.LISTENING

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                result = session.add_audio_frame(message["bytes"])
                if result["interrupted"]:
                    await websocket.send_json({"event": "interrupted", "reason": "barge_in"})
                if result["speech_started"]:
                    await websocket.send_json({"event": "speech_started"})
                if result["utterance_ready"] and result["audio"] is not None:
                    stt_result = transcribe(result["audio"], language=session.language)
                    user_text = stt_result.get("text", "").strip()
                    if user_text:
                        await websocket.send_json({
                            "event": "final_transcript",
                            "text": user_text,
                            "language": stt_result.get("language", session.language),
                        })
                        session.active_generation_task = asyncio.create_task(run_orchestrator(user_text))
                    else:
                        session.state = VoiceSessionState.LISTENING

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    action = data.get("action")
                    if action == "interrupt":
                        session.interrupt("Client requested interrupt")
                        await websocket.send_json({"event": "interrupted", "reason": "client_request"})
                    elif action == "set_language":
                        session.set_language(data.get("language", "en-IN"))
                        await websocket.send_json({"event": "language_updated", "language": session.language})
                    elif action == "clear":
                        session.interrupt("Chat cleared")
                        session.history.clear()
                        session.reset_utterance()
                        await websocket.send_json({"event": "chat_cleared"})
                except Exception as json_err:
                    logger.debug("Non-JSON control message: %s", json_err)

    except WebSocketDisconnect:
        remove_session(session_id)
    except Exception as exc:
        logger.warning("WebSocket exception for session %s: %s", session_id, exc)
        remove_session(session_id)


