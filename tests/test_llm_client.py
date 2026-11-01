from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.llm.ollama_client import (
    LLMFailureReason,
    check_ollama_status,
    chat_completion,
    stream_chat_completion,
    clean_conversational_response,
    is_reasoning_text,
    start_ollama_client,
    stop_ollama_client,
)


class TestOllamaClient(unittest.IsolatedAsyncioTestCase):
    async def test_client_lifecycle(self):
        """Verify client startup and shutdown."""
        import app.llm.ollama_client as oc
        oc.client = None
        await start_ollama_client()
        self.assertIsNotNone(oc.client)
        self.assertFalse(oc.client.is_closed)
        await stop_ollama_client()
        self.assertIsNone(oc.client)

    async def test_check_ollama_status_success(self):
        """Verify status check when Ollama is running and model exists."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen3:4b"}, {"name": "llama3:latest"}]
        }

        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_get_client.return_value = mock_client

            status = await check_ollama_status()
            self.assertTrue(status["connected"])
            self.assertTrue(status["model_available"])
            self.assertIn("qwen3:4b", status["available_models"])

    async def test_check_ollama_status_model_missing(self):
        """Verify status check when Ollama is running but configured model is not pulled."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "mistral:latest"}]}

        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_get_client.return_value = mock_client

            status = await check_ollama_status()
            self.assertTrue(status["connected"])
            self.assertFalse(status["model_available"])

    async def test_check_ollama_status_connect_error(self):
        """Verify status check when Ollama is offline."""
        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_get_client.return_value = mock_client

            status = await check_ollama_status()
            self.assertFalse(status["connected"])
            self.assertFalse(status["model_available"])
            self.assertEqual(status["failure_reason"], LLMFailureReason.UNAVAILABLE)

    async def test_check_ollama_status_timeout(self):
        """Verify status check handles timeout."""
        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ReadTimeout("Request timed out")
            mock_get_client.return_value = mock_client

            status = await check_ollama_status()
            self.assertFalse(status["connected"])
            self.assertEqual(status["failure_reason"], LLMFailureReason.TIMEOUT)

    async def test_chat_completion_success(self):
        """Verify successful non-streaming chat completion."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Our clinic opens at 9 AM Monday through Saturday."}
        }

        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            ans, dur, reason = await chat_completion([{"role": "user", "content": "when do you open?"}])
            self.assertEqual(reason, LLMFailureReason.SUCCESS)
            self.assertIn("9 AM", ans)

    async def test_chat_completion_with_thinking_tags_cleaned(self):
        """Verify that <think> tags are stripped and the clean answer is returned."""
        raw_llm_output = "<think>\nThe caller is asking about clinic hours.\nBusiness hours: 9 AM to 6 PM.\n</think>\nWe are open from 9 AM to 6 PM Monday through Saturday."
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": raw_llm_output}
        }

        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            ans, dur, reason = await chat_completion([{"role": "user", "content": "hours?"}])
            self.assertEqual(reason, LLMFailureReason.SUCCESS)
            self.assertNotIn("<think>", ans)
            self.assertNotIn("The caller is asking", ans)
            self.assertIn("9 AM to 6 PM", ans)

    async def test_chat_completion_reasoning_leakage_rejected(self):
        """Verify that pure chain-of-thought without an answer is rejected by the filter."""
        pure_reasoning = "Okay, let me check the Business Profile JSON. The user is asking who is the doctor. We are the virtual receptionist."
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": pure_reasoning}
        }

        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            ans, dur, reason = await chat_completion([{"role": "user", "content": "doctor?"}])
            self.assertIsNone(ans)
            self.assertEqual(reason, LLMFailureReason.FILTER_REJECTED)

    async def test_chat_completion_model_not_found(self):
        """Verify handling of HTTP 404 model not found."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            ans, dur, reason = await chat_completion([{"role": "user", "content": "hi"}])
            self.assertIsNone(ans)
            self.assertEqual(reason, LLMFailureReason.MODEL_NOT_FOUND)

    async def test_chat_completion_timeout(self):
        """Verify handling of timeout."""
        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ReadTimeout("Timeout")
            mock_get_client.return_value = mock_client

            ans, dur, reason = await chat_completion([{"role": "user", "content": "hi"}])
            self.assertIsNone(ans)
            self.assertEqual(reason, LLMFailureReason.TIMEOUT)

    async def test_chat_completion_connection_refused(self):
        """Verify handling of connection refused."""
        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_get_client.return_value = mock_client

            ans, dur, reason = await chat_completion([{"role": "user", "content": "hi"}])
            self.assertIsNone(ans)
            self.assertEqual(reason, LLMFailureReason.UNAVAILABLE)

    async def test_stream_chat_completion_connect_error(self):
        """Verify streaming handles connect error and emits done event with failure reason."""
        with patch("app.llm.ollama_client.get_ollama_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.stream.side_effect = httpx.ConnectError("Connection refused")
            mock_get_client.return_value = mock_client

            events = []
            async for ev in stream_chat_completion([{"role": "user", "content": "hi"}]):
                events.append(ev)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["type"], "done")
            self.assertEqual(events[0]["failure_reason"], LLMFailureReason.UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
