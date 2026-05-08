"""
Shared async LLM client for all non-gatekeeper agents.

Supports multi-turn conversation via a messages list.
Supports two LLM providers: OpenRouter (default) and Ollama.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from ollama import AsyncClient as OllamaAsyncClient

from config.settings import Settings, get_settings
from src.memory.key_store import (
    PROVIDER_OLLAMA,
    PROVIDER_OPENROUTER,
    effective_ollama_auth_header,
    effective_ollama_host,
    effective_ollama_model,
    effective_openrouter_auth_header,
    effective_openrouter_max_tokens,
    effective_openrouter_model,
    get_active_provider,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client that supports OpenRouter and Ollama providers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._http = httpx.AsyncClient(
            base_url=self._settings.openrouter_base_url,
            timeout=self._settings.openrouter_timeout,
            headers=self._settings.openrouter_headers,
        )

    # ── Provider helpers ──────────────────────────────────────────────────────

    def _active_provider(self) -> str:
        return get_active_provider()

    def _auth_headers(self) -> dict[str, str]:
        """Return Authorization header for OpenRouter, preferring key_store override over .env."""
        return {"Authorization": effective_openrouter_auth_header(self._settings.openrouter_api_key)}

    # ── OpenRouter backend ────────────────────────────────────────────────────

    async def _chat_openrouter(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model":      model      or effective_openrouter_model(self._settings.openrouter_model),
            "max_tokens": max_tokens or effective_openrouter_max_tokens(self._settings.openrouter_max_tokens),
            "messages":   messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        logger.debug("LLMClient[openrouter].chat → model=%s, messages_count=%d", payload["model"], len(messages))
        try:
            response = await self._http.post(
                "/chat/completions", json=payload, headers=self._auth_headers()
            )
            logger.debug("LLMClient[openrouter] HTTP status=%s", response.status_code)
            if response.status_code == 429:
                logger.warning(
                    "LLMClient[openrouter] 429 rate-limited — retrying in 5s"
                )
                await asyncio.sleep(5)
                response = await self._http.post(
                    "/chat/completions", json=payload, headers=self._auth_headers()
                )
                logger.debug(
                    "LLMClient[openrouter] retry HTTP status=%s", response.status_code
                )
            try:
                logger.debug("LLMClient[openrouter].raw_response_text=%s", response.text)
            except Exception:
                logger.debug("LLMClient[openrouter].raw_response_text unavailable")

            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            logger.exception("LLM[openrouter] HTTP request failed: %s", exc)
            raise

        logger.debug("LLMClient[openrouter].raw_response_json=%r", data)

        if "error" in data:
            err = data["error"]
            logger.error(
                "LLM[openrouter] returned an API error (no choices): code=%s message=%s",
                err.get("code") if isinstance(err, dict) else None,
                err.get("message") if isinstance(err, dict) else err,
            )
            return ""

        try:
            message = data["choices"][0]["message"]
            content = message.get("content")
        except Exception as exc:
            logger.exception(
                "Failed to extract content from LLM[openrouter] response: %s – full response: %r",
                exc, data,
            )
            content = None
            message = {}

        if content is None:
            finish_reason = None
            try:
                finish_reason = data["choices"][0].get("finish_reason")
            except Exception as exc:  # noqa: BLE001
                logger.debug("LLMClient[openrouter]: could not extract finish_reason: %s", exc)

            if finish_reason == "length":
                # Token budget exhausted while the model was still reasoning.
                # The reasoning field contains incomplete internal monologue — NOT
                # a usable output.  Return empty so callers can surface the error
                # cleanly instead of trying to parse partial thoughts.
                logger.warning(
                    "LLM[openrouter] hit max_tokens limit (finish_reason=length). "
                    "Content is null and reasoning is incomplete — not falling back. "
                    "Increase OPENROUTER_MAX_TOKENS or switch to a non-reasoning model."
                )
                return ""

            # For other null-content cases, the reasoning field may legitimately
            # hold the final answer (some reasoning models route output there).
            logger.warning("LLM[openrouter] returned null content (finish_reason=%s): %r", finish_reason, data)
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                logger.info("Falling back to reasoning field as response content.")
                return reasoning.strip()

            return ""

        return content.strip()

    # ── Ollama backend ────────────────────────────────────────────────────────

    async def _chat_ollama(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = False,
        **_: object,
    ) -> str:
        host = effective_ollama_host(self._settings.ollama_host)
        auth_header = effective_ollama_auth_header(self._settings.ollama_api_key)
        headers: dict[str, str] = {}
        if auth_header:
            headers["Authorization"] = auth_header

        resolved_model = model or effective_ollama_model(self._settings.ollama_model)
        logger.debug("LLMClient[ollama].chat → model=%s, messages_count=%d", resolved_model, len(messages))

        client = OllamaAsyncClient(host=host, headers=headers)
        options: dict = {}
        if temperature is not None:
            options["temperature"] = temperature
        if top_p is not None:
            options["top_p"] = top_p

        options["num_ctx"] = 8192

        try:
            response = await client.chat(
                model=resolved_model,
                messages=messages,
                stream=False,
                **({"format": "json"} if json_mode else {}),
                options=options or None,
                keep_alive=-1,
            )
        except Exception as exc:
            logger.exception("LLM[ollama] request failed: %s", exc)
            raise

        try:
            content = response.message.content
        except Exception as exc:
            logger.exception("Failed to extract content from LLM[ollama] response: %s", exc)
            return ""

        logger.debug("LLMClient[ollama].raw_response=%r", response)
        return (content or "").strip()

    # ── Public API ────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Send a messages list and return the assistant's text reply.

        Routes to the active LLM provider (openrouter or ollama).

        Args:
            messages:    List of {"role": ..., "content": ...} dicts.
            model:       Override the default model.
            max_tokens:  Override the default max_tokens (OpenRouter only).
            temperature: Sampling temperature (lower = more deterministic).
            top_p:       Nucleus sampling threshold.
            json_mode:   When True, instructs the LLM to return valid JSON.
        """
        provider = self._active_provider()
        if provider == PROVIDER_OLLAMA:
            return await self._chat_ollama(
                messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                json_mode=json_mode,
            )
        # Default: openrouter
        return await self._chat_openrouter(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            json_mode=json_mode,
        )

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """
        Send a system prompt + messages list and return the assistant's text reply.

        Prepends the system_prompt as a {"role": "system"} message before the
        conversation history, then delegates to chat().

        Args:
            system_prompt: Instruction context for the LLM.
            messages:      List of {"role": ..., "content": ...} dicts (history + user).
            model:         Override the default model.
            max_tokens:    Override the default max_tokens (OpenRouter only).
            temperature:   Sampling temperature (lower = more deterministic).
            top_p:         Nucleus sampling threshold.
            json_mode:     When True, instructs the LLM to return valid JSON.
        """
        full_messages = [{"role": "system", "content": system_prompt}, *messages]
        return await self.chat(
            full_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            json_mode=json_mode,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
