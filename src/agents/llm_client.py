"""
Shared async LLM client for all non-gatekeeper agents.

Supports multi-turn conversation via a messages list.
"""

from __future__ import annotations

import logging

import httpx

from config.settings import Settings, get_settings
from src.memory.key_store import effective_openrouter_auth_header, effective_openrouter_max_tokens, effective_openrouter_model

logger = logging.getLogger(__name__)


class LLMClient:
    """Generic async wrapper around OpenRouter chat-completions."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._http = httpx.AsyncClient(
            base_url=self._settings.openrouter_base_url,
            timeout=self._settings.openrouter_timeout,
            headers=self._settings.openrouter_headers,
        )

    def _auth_headers(self) -> dict[str, str]:
        """Return Authorization header, preferring key_store override over .env."""
        return {"Authorization": effective_openrouter_auth_header(self._settings.openrouter_api_key)}

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Send a messages list and return the assistant's text reply.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            model:       Override the default model.
            max_tokens:  Override the default max_tokens.
            temperature: Sampling temperature (lower = more deterministic).
            top_p:       Nucleus sampling threshold.
        """
        payload: dict = {
            "model":      model      or effective_openrouter_model(self._settings.openrouter_model),
            "max_tokens": max_tokens or effective_openrouter_max_tokens(self._settings.openrouter_max_tokens),
            "messages":   messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        logger.debug("LLMClient.chat → model=%s, messages_count=%d", payload["model"], len(messages))
        try:
            response = await self._http.post(
                "/chat/completions", json=payload, headers=self._auth_headers()
            )
            logger.debug("LLMClient HTTP status=%s", response.status_code)
            # Prefer to log the response text at DEBUG level for diagnosis
            try:
                logger.debug("LLMClient.raw_response_text=%s", response.text)
            except Exception:
                logger.debug("LLMClient.raw_response_text unavailable")

            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            # Log response text if available for debugging
            logger.exception("LLM HTTP request failed: %s", exc)
            raise

        # Log the parsed JSON at debug level to help diagnose malformed outputs
        logger.debug("LLMClient.raw_response_json=%r", data)

        try:
            message = data["choices"][0]["message"]
            content = message.get("content")
        except Exception as exc:
            logger.exception("Failed to extract content from LLM response: %s", exc)
            content = None
            message = {}

        if content is None:
            finish_reason = None
            try:
                finish_reason = data["choices"][0].get("finish_reason")
            except Exception:
                pass

            if finish_reason == "length":
                logger.warning(
                    "LLM hit max_tokens limit (finish_reason=length) before producing content. "
                    "Consider increasing OPENROUTER_MAX_TOKENS for reasoning models. "
                    "Attempting fallback to reasoning field."
                )
            else:
                logger.warning("LLM returned null content: %r", data)

            # Reasoning models (e.g. Qwen-thinking, DeepSeek-R1) put chain-of-thought
            # in the 'reasoning' field. If content is missing, try to surface the
            # reasoning text as a best-effort response.
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                logger.info("Falling back to reasoning field as response content.")
                return reasoning.strip()

            return ""

        return content.strip()

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        """
        Send a system prompt + messages list and return the assistant's text reply.

        Prepends the system_prompt as a {"role": "system"} message before the
        conversation history, then delegates to chat().

        Args:
            system_prompt: Instruction context for the LLM.
            messages:      List of {"role": ..., "content": ...} dicts (history + user).
            model:         Override the default model.
            max_tokens:    Override the default max_tokens.
            temperature:   Sampling temperature (lower = more deterministic).
            top_p:         Nucleus sampling threshold.
        """
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        return await self.chat(
            full_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
