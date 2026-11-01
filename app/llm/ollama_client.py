from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

client: httpx.AsyncClient | None = None


class LLMFailureReason(str, Enum):
    """Structured classification of why the LLM path failed or was bypassed."""
    SUCCESS = "success"
    UNAVAILABLE = "ollama_unavailable"       # Connection refused / network error
    TIMEOUT = "ollama_timeout"               # Read/connect timeout
    MODEL_NOT_FOUND = "model_not_found"      # HTTP 404 or model not in /api/tags
    BAD_RESPONSE = "bad_response"            # Non-200 HTTP, malformed JSON
    FILTER_REJECTED = "filter_rejected"      # LLM responded but reasoning-filter wiped it
    EMPTY_ANSWER = "empty_answer"            # LLM returned empty content string


async def start_ollama_client() -> httpx.AsyncClient:
    """Initialize persistent AsyncClient with connection pooling and keep-alive."""
    global client
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL.rstrip("/"),
            timeout=httpx.Timeout(connect=3.0, read=OLLAMA_TIMEOUT_SECONDS, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
        logger.info("Initialized persistent Ollama AsyncClient for %s", OLLAMA_BASE_URL)
    return client


async def stop_ollama_client() -> None:
    """Gracefully close the persistent AsyncClient."""
    global client
    if client is not None:
        try:
            if not client.is_closed:
                await client.aclose()
        except Exception:
            pass
        finally:
            client = None
        logger.info("Closed persistent Ollama AsyncClient")


def get_ollama_client() -> httpx.AsyncClient:
    """Get active client or instantiate one if not started via lifespan."""
    global client
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL.rstrip("/"),
            timeout=httpx.Timeout(connect=3.0, read=OLLAMA_TIMEOUT_SECONDS, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
    return client


async def check_ollama_status() -> dict[str, Any]:
    """Check if Ollama server is reachable and if the configured model is ready."""
    http_client = get_ollama_client()
    try:
        response = await http_client.get("/api/tags", timeout=2.5)
        if response.status_code == 200:
            payload = response.json()
            models = [str(m.get("name", "")) for m in payload.get("models", [])]
            model_found = any(
                m == OLLAMA_MODEL
                or m.startswith(f"{OLLAMA_MODEL}:")
                or m.split(":")[0] == OLLAMA_MODEL.split(":")[0]
                for m in models
            )
            return {
                "connected": True,
                "base_url": OLLAMA_BASE_URL,
                "configured_model": OLLAMA_MODEL,
                "model_available": model_found,
                "available_models": models,
            }
        else:
            logger.warning("Ollama /api/tags returned HTTP %s", response.status_code)
    except httpx.ConnectError as exc:
        logger.warning("Ollama unreachable at %s: %s", OLLAMA_BASE_URL, exc)
        return {
            "connected": False,
            "base_url": OLLAMA_BASE_URL,
            "configured_model": OLLAMA_MODEL,
            "model_available": False,
            "error": f"Connection refused: {exc}",
            "failure_reason": LLMFailureReason.UNAVAILABLE,
        }
    except httpx.TimeoutException as exc:
        logger.warning("Ollama status check timed out: %s", exc)
        return {
            "connected": False,
            "base_url": OLLAMA_BASE_URL,
            "configured_model": OLLAMA_MODEL,
            "model_available": False,
            "error": f"Timeout: {exc}",
            "failure_reason": LLMFailureReason.TIMEOUT,
        }
    except Exception as exc:
        logger.warning("Ollama status check failed: %s", exc)
        return {
            "connected": False,
            "base_url": OLLAMA_BASE_URL,
            "configured_model": OLLAMA_MODEL,
            "model_available": False,
            "error": str(exc),
            "failure_reason": LLMFailureReason.BAD_RESPONSE,
        }
    return {
        "connected": False,
        "base_url": OLLAMA_BASE_URL,
        "configured_model": OLLAMA_MODEL,
        "model_available": False,
        "failure_reason": LLMFailureReason.BAD_RESPONSE,
    }


def is_reasoning_text(text: str) -> bool:
    """
    Check if the text contains leaked internal reasoning or meta-instructions.
    Only flags true internal reasoning/meta-commentary - NOT valid receptionist responses.
    """
    if not text:
        return True
    low = text.lower().strip()

    if low.startswith((
        "- keep", "- respond", "- the business", "- the doctors", "- doctors",
        "- services", "- hours", "- address", "- phone", "- clinic", "- faqs",
        "- q:", "- a:", "from the business", "since the business", "in the business",
        "wait, the business", "wait, the clinic", "option:", "options:",
        "hours: monday", "hours: 9", "let me interpret", "let me make sure",
        "doctors:", "services:", "business:",
        "the response needs", "the response should", "i'll state", "i will state",
        "about the clinic:", "your job:", "common questions:", "if a caller asks",
        "never explain", "reply only", "also, the faqs", "also, the clinic",
        "also, there are no", "therefore, we do not", "the faqs only",
        "also, i", "also, we", "i'll phrase it", "i recall", "the clinic details say",
        "the clinic's common questions", "clinic facts:", "from the clinic facts",
        "- the clinic is", "- open monday", "- open ", "- location:", "- doctor:",
        "background:",
    )):
        return True

    indicators = (
        "we are given", "let me check", "let me think", "let me interpret", "let me make sure",
        "the user asks", "the user said", "the user is asking", "the user just said",
        "the user asked", "okay, let me", "okay, the user",
        "the caller asks", "the caller said", "the caller is asking", "the caller just said",
        "the caller asked", "caller is asking",
        "step 1", "step 2", "steps:", "should be:", "rules:",
        "concise (1-3 sentences)", "i need to", "possible response",
        "translates to", "first, i need", "without a greeting", "double-checking",
        "we must respond", "we have to", "the answer:",
        "the business profile", "business profile:", "business profile json",
        "from the business profile", "since the business profile", "in the business profile",
        "as the virtual receptionist", "we are the virtual receptionist", "we are the receptionist",
        "there's no mention", "there is no mention", "i should respond", "the user has already",
        "so the answer is", "meaning we should", "meaning we", "the clinic's services listed",
        "but note:", "the rule says", "the rules say", "looking at the", "as per the",
        "so the answer should be", "so the response should be",
        "formulate a", "keep greetings to", "respond directly without", "we are to state",
        "we can state:", "we don't have multiple doctors", "let's re-read", "re-read:",
        "important: never reveal", "never reveal internal", "so, i will respond",
        "we have two options", "we have a few options", "wait, the business",
        "the example given is", "the example says", "doctors listed:", "services listed:",
        "so we can say", "so, we can say", "so we can write", "fits the example",
        "written as one string", "let's craft", "let's stick", "stick to that format",
        "we don't have any other", "however, the user", "but the user", "the answer should list",
        "so, the answer should", "so the answer should",
        "the response needs to be", "the response should be",
        "the user might be", "the user may be", "so i'll make", "so i will make",
        "i'll make it clear", "i will make it clear", "without any internal reasoning",
        "- doctors", "- services",
        "also, the faqs only cover", "also, there are no", "therefore, we do not have",
        "about the clinic:", "common questions:", "your job:",
        "wait, the clinic", "i recall the instruction", "i'll phrase it",
        "the clinic details say", "the clinic's common questions",
    )
    return any(ind in low for ind in indicators)


def clean_conversational_response(raw_text: str) -> str:
    """
    Robust cleaner that strips <think> tags, chain-of-thought preambles, and meta-commentary,
    leaving only the user-facing receptionist response. Returns empty string if only reasoning found.
    """
    if not raw_text:
        return ""
    text = raw_text.strip()

    # 1. If </think> present, extract text after the think block
    if "</think>" in text:
        after_think = text.rsplit("</think>", 1)[-1].strip()
        if after_think and not is_reasoning_text(after_think):
            return re.sub(r'^(?:Assistant|Receptionist|Response|Answer|Possible response|Final answer|Greeting):\s*', '', after_think, flags=re.IGNORECASE).strip('"\'  ')

    # 2. Check for explicit quoted response after transition phrase (inspect from latest to earliest)
    matches = list(re.finditer(
        r'(?:so\s+(?:my|the)\s+response\s+(?:should\s+be|is):?|therefore,?\s*(?:the\s*)?response\s*(?:is)?:?|response:?|receptionist:?|possible response:?)\s*["\u201c]([^"\u201d\n\r]{5,})["\u201d]',
        text,
        re.I
    ))
    for m in reversed(matches):
        cand = m.group(1).strip()
        if not is_reasoning_text(cand):
            return re.sub(r'^(?:Assistant|Receptionist|Response|Answer|Possible response|Final answer|Greeting):\s*', '', cand, flags=re.IGNORECASE).strip('"\'  ')

    # 3. Check if text without think tags is clean
    text_no_tags = re.sub(r"</?think>", "", text).strip()
    if not is_reasoning_text(text_no_tags):
        clean = re.sub(r'^(?:Assistant|Receptionist|Response|Answer|Possible response|Final answer|Greeting):\s*', '', text_no_tags, flags=re.IGNORECASE).strip('"\'  ')
        if clean:
            return clean

    # 4. Filter reasoning lines and extract clean concluding lines
    lines = [l.strip() for l in text_no_tags.splitlines() if l.strip()]
    candidate_lines = []
    for l in lines:
        cleaned_line = re.sub(r'^(?:Assistant|Receptionist|Response|Answer|Possible response|Final answer|Greeting):\s*', '', l, flags=re.IGNORECASE).strip('"\'  ')
        if not is_reasoning_text(cleaned_line) and not re.match(r"^\d+\.\s+", cleaned_line) and len(cleaned_line) > 4:
            candidate_lines.append(cleaned_line)

    if candidate_lines:
        clean = " ".join(candidate_lines).strip()
        if not is_reasoning_text(clean):
            return clean

    return ""


def clean_reasoning_tokens(raw_text: str) -> str:
    """Strip reasoning/think tags and leaked chain-of-thought artifacts."""
    return clean_conversational_response(raw_text)


async def stream_chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    options: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Stream tokens from Ollama /api/chat using persistent HTTP client.
    Yields token events then a done event with failure_reason field.
    """
    http_client = get_ollama_client()
    target_model = model or OLLAMA_MODEL

    default_options = {
        "temperature": 0.1,
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "600")),
        "think": False,
    }
    if options:
        default_options.update(options)

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": default_options,
    }

    t_start = time.perf_counter()
    t_first_token: float | None = None
    accumulated_user_tokens: list[str] = []
    in_think_block = False
    detected_mode = False
    reasoning_buffer = ""
    stream_started = False

    try:
        async with http_client.stream("POST", "/api/chat", json=payload) as response:
            if response.status_code == 404:
                logger.error("Ollama model not found: %s (HTTP 404)", target_model)
                yield {"type": "done", "answer": "", "engine": "llm", "source": "llm",
                       "model": target_model,
                       "response_time_ms": round((time.perf_counter() - t_start) * 1000, 1),
                       "ttft_ms": 0.0, "failure_reason": LLMFailureReason.MODEL_NOT_FOUND}
                return
            if response.status_code != 200:
                logger.error("Ollama HTTP %s for model %s", response.status_code, target_model)
                yield {"type": "done", "answer": "", "engine": "llm", "source": "llm",
                       "model": target_model,
                       "response_time_ms": round((time.perf_counter() - t_start) * 1000, 1),
                       "ttft_ms": 0.0, "failure_reason": LLMFailureReason.BAD_RESPONSE}
                return

            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Malformed JSON from Ollama stream: %s", exc)
                    continue

                msg = chunk.get("message", {})
                content = msg.get("content", "")
                thinking_piece = msg.get("thinking", "")
                if thinking_piece:
                    reasoning_buffer += thinking_piece

                if not content:
                    if chunk.get("done", False):
                        break
                    continue

                if "<think>" in content:
                    in_think_block = True
                    detected_mode = True
                    content = content.split("<think>", 1)[-1]

                if "</think>" in content:
                    in_think_block = False
                    detected_mode = True
                    reasoning_buffer = ""
                    content = content.split("</think>", 1)[-1]
                    stream_started = True

                if not detected_mode and content.strip():
                    low = content.lower().strip()
                    if is_reasoning_text(low) or any(low.startswith(p) for p in (
                        "we are", "the user", "let me", "okay,", "first,", "as the", "i need",
                        "looking at", "so the",
                    )):
                        in_think_block = True
                        detected_mode = True
                    elif len(low) >= 3:
                        in_think_block = False
                        detected_mode = True

                if in_think_block:
                    continue

                clean_piece = re.sub(r"</?think>", "", content)
                if not clean_piece:
                    continue

                if not stream_started:
                    reasoning_buffer += clean_piece
                    buf_low = reasoning_buffer.lower().strip()
                    starts_reasoning = is_reasoning_text(buf_low) or any(
                        buf_low.startswith(p)
                        for p in (
                            "we are given", "we are the receptionist", "the user said",
                            "the user asks", "let me check", "i need to", "possible response",
                            "first, i need", "steps:", "okay, the user",
                        )
                    )
                    if starts_reasoning:
                        resp_match = re.search(
                            r'(?:therefore,?\s*(?:the\s*)?response\s*(?:is)?:?|response:?|'
                            r'receptionist:?|possible response:?)\s*["\u201c]?([^"\u201d\n\r]+)',
                            reasoning_buffer, re.I
                        )
                        if resp_match:
                            clean_text = resp_match.group(1).strip()
                            if clean_text and not is_reasoning_text(clean_text):
                                stream_started = True
                                if t_first_token is None:
                                    t_first_token = time.perf_counter()
                                accumulated_user_tokens.append(clean_text)
                                yield {"type": "token", "content": clean_text, "is_first": True,
                                       "ttft_ms": round((t_first_token - t_start) * 1000, 1)}
                        elif len(reasoning_buffer) > 80:
                            cleaned = clean_conversational_response(reasoning_buffer)
                            if cleaned:
                                stream_started = True
                                reasoning_buffer = ""
                                if t_first_token is None:
                                    t_first_token = time.perf_counter()
                                accumulated_user_tokens.append(cleaned)
                                yield {"type": "token", "content": cleaned, "is_first": True,
                                       "ttft_ms": round((t_first_token - t_start) * 1000, 1)}
                            else:
                                logger.debug(
                                    "Buffer >80 chars is pure reasoning, no extractable answer: %.80s",
                                    reasoning_buffer
                                )
                        continue
                    else:
                        stream_started = True
                        clean_piece = reasoning_buffer
                        reasoning_buffer = ""

                if clean_piece:
                    if not accumulated_user_tokens:
                        clean_piece = clean_piece.lstrip("\n\r")
                    if clean_piece:
                        accumulated_user_tokens.append(clean_piece)
                        now = time.perf_counter()
                        is_first = t_first_token is None
                        if is_first:
                            t_first_token = now
                        yield {
                            "type": "token", "content": clean_piece, "is_first": is_first,
                            "ttft_ms": round((t_first_token - t_start) * 1000, 1) if is_first else None,
                        }

                if chunk.get("done", False):
                    break

    except httpx.ConnectError as exc:
        logger.error("Ollama connection refused (%s): %s", OLLAMA_BASE_URL, exc)
        yield {"type": "done", "answer": "", "engine": "llm", "source": "llm",
               "model": target_model,
               "response_time_ms": round((time.perf_counter() - t_start) * 1000, 1),
               "ttft_ms": 0.0, "failure_reason": LLMFailureReason.UNAVAILABLE}
        return
    except httpx.ReadTimeout as exc:
        logger.error("Ollama streaming read timeout after %.1fs: %s", OLLAMA_TIMEOUT_SECONDS, exc)
        yield {"type": "done", "answer": "", "engine": "llm", "source": "llm",
               "model": target_model,
               "response_time_ms": round((time.perf_counter() - t_start) * 1000, 1),
               "ttft_ms": 0.0, "failure_reason": LLMFailureReason.TIMEOUT}
        return
    except httpx.TimeoutException as exc:
        logger.error("Ollama streaming timeout: %s", exc)
        yield {"type": "done", "answer": "", "engine": "llm", "source": "llm",
               "model": target_model,
               "response_time_ms": round((time.perf_counter() - t_start) * 1000, 1),
               "ttft_ms": 0.0, "failure_reason": LLMFailureReason.TIMEOUT}
        return

    t_end = time.perf_counter()
    duration_ms = round((t_end - t_start) * 1000, 1)
    ttft_ms = round((t_first_token - t_start) * 1000, 1) if t_first_token else duration_ms

    full_text = clean_conversational_response("".join(accumulated_user_tokens))
    if not full_text and reasoning_buffer:
        full_text = clean_conversational_response(reasoning_buffer)
        if full_text:
            yield {"type": "token", "content": full_text, "is_first": True, "ttft_ms": duration_ms}

    failure_reason: LLMFailureReason | None = None
    if not full_text:
        if accumulated_user_tokens:
            failure_reason = LLMFailureReason.FILTER_REJECTED
            logger.warning(
                "Ollama response filtered (reasoning leakage) model=%s raw=%.120s",
                target_model, "".join(accumulated_user_tokens)[:120]
            )
        else:
            failure_reason = LLMFailureReason.EMPTY_ANSWER
            logger.warning("Ollama returned empty content for model %s", target_model)

    yield {
        "type": "done",
        "answer": full_text,
        "engine": "llm",
        "source": "llm",
        "model": target_model,
        "response_time_ms": duration_ms,
        "ttft_ms": ttft_ms,
        "failure_reason": failure_reason,
    }


async def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[str, float, LLMFailureReason | None] | tuple[None, float, LLMFailureReason]:
    """
    Non-streaming chat completion via persistent AsyncClient.
    Returns (answer, duration_ms, SUCCESS) on success or (None, duration_ms, reason) on failure.
    """
    http_client = get_ollama_client()
    target_model = model or OLLAMA_MODEL

    default_options = {
        "temperature": 0.1,
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "600")),
        "think": False,
    }
    if options:
        default_options.update(options)

    payload = {
        "model": target_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": default_options,
    }

    t_start = time.perf_counter()
    try:
        response = await http_client.post("/api/chat", json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)

        if response.status_code == 404:
            logger.error("Ollama model not found: %s (HTTP 404)", target_model)
            return None, duration_ms, LLMFailureReason.MODEL_NOT_FOUND

        if response.status_code != 200:
            logger.error("Ollama HTTP %s for model %s", response.status_code, target_model)
            return None, duration_ms, LLMFailureReason.BAD_RESPONSE

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            logger.error("Malformed JSON from Ollama: %s", exc)
            return None, duration_ms, LLMFailureReason.BAD_RESPONSE

        msg = data.get("message", {})
        raw_content = str(msg.get("content", "")).strip()
        if not raw_content:
            thinking = str(msg.get("thinking", "")).strip()
            if thinking:
                extracted = clean_conversational_response(thinking)
                if extracted:
                    raw_content = extracted
        if not raw_content:
            logger.warning("Ollama returned empty content for model %s", target_model)
            return None, duration_ms, LLMFailureReason.EMPTY_ANSWER

        answer = clean_conversational_response(raw_content)
        if not answer:
            logger.warning(
                "Ollama response filtered (reasoning leakage) model=%s raw=%.120s",
                target_model, raw_content[:120]
            )
            return None, duration_ms, LLMFailureReason.FILTER_REJECTED

        return answer, duration_ms, LLMFailureReason.SUCCESS

    except httpx.ConnectError as exc:
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        logger.error("Ollama connection refused (%s): %s", OLLAMA_BASE_URL, exc)
        return None, duration_ms, LLMFailureReason.UNAVAILABLE
    except httpx.TimeoutException as exc:
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        logger.error("Ollama timed out after %.1fs: %s", OLLAMA_TIMEOUT_SECONDS, exc)
        return None, duration_ms, LLMFailureReason.TIMEOUT
    except Exception as exc:
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        logger.error("Unexpected Ollama error: %s", exc, exc_info=True)
        return None, duration_ms, LLMFailureReason.BAD_RESPONSE
