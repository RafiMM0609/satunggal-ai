"""
CaptchaSolver – optional automatic CAPTCHA solving via 2captcha / CapMonster.

When ``CAPTCHA_API_KEY`` and (optionally) ``CAPTCHA_PROVIDER`` are set as
environment variables this module submits reCAPTCHA v2 or hCaptcha challenges
to the configured service and returns a token ready for JS injection.

If the env vars are absent, or the service call fails, every function returns
``None`` so callers can fall back gracefully to requesting manual assistance.

Supported providers (``CAPTCHA_PROVIDER`` env var):
  * ``2captcha``   – https://2captcha.com  (default)
  * ``capmonster`` – https://capmonster.cloud

Dependencies:
  * ``aiohttp`` for async HTTP calls (not in hard requirements; if absent the
    solver silently skips and returns None).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_PROVIDER       = os.getenv("CAPTCHA_PROVIDER", "2captcha").lower()
_API_KEY        = os.getenv("CAPTCHA_API_KEY", "")
_POLL_INTERVAL  = 5    # seconds between result polls
_MAX_POLLS      = 24   # up to 2 minutes total


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _post_form(session, url: str, data: dict) -> dict:  # type: ignore[type-arg]
    """POST *data* as form-encoded body and return parsed JSON (or raw dict)."""
    import json as _json

    async with session.post(url, data=data) as resp:
        text = await resp.text()
    try:
        return _json.loads(text)
    except Exception:
        return {"status": 0, "request": text}


async def _post_json(session, url: str, payload: dict) -> dict:  # type: ignore[type-arg]
    """POST *payload* as JSON body and return parsed JSON response."""
    async with session.post(url, json=payload) as resp:
        return await resp.json()


# ── Public API ────────────────────────────────────────────────────────────────

async def solve_recaptcha_v2(
    site_url: str,
    site_key: str,
) -> Optional[str]:
    """Submit a reCAPTCHA v2 challenge and return the ``g-recaptcha-response`` token.

    Returns ``None`` when:
      * No ``CAPTCHA_API_KEY`` is configured.
      * The ``aiohttp`` package is unavailable.
      * The solver service fails or times out.
    """
    if not _API_KEY:
        logger.debug("captcha_solver: no CAPTCHA_API_KEY set – skipping reCAPTCHA solve")
        return None

    try:
        import aiohttp
    except ImportError:
        logger.warning("captcha_solver: aiohttp not installed – cannot auto-solve reCAPTCHA")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            if _PROVIDER == "capmonster":
                return await _capmonster_solve(
                    session,
                    task_type="NoCaptchaTaskProxyless",
                    website_url=site_url,
                    website_key=site_key,
                    response_field="gRecaptchaResponse",
                )
            else:
                return await _2captcha_solve(
                    session,
                    method="userrecaptcha",
                    extra_params={"googlekey": site_key, "pageurl": site_url},
                )
    except Exception as exc:
        logger.warning("captcha_solver: reCAPTCHA solve error: %s", exc)
        return None


async def solve_hcaptcha(
    site_url: str,
    site_key: str,
) -> Optional[str]:
    """Submit an hCaptcha challenge and return the ``h-captcha-response`` token.

    Returns ``None`` on misconfiguration, missing dependency, or solver failure.
    """
    if not _API_KEY:
        logger.debug("captcha_solver: no CAPTCHA_API_KEY set – skipping hCaptcha solve")
        return None

    try:
        import aiohttp
    except ImportError:
        logger.warning("captcha_solver: aiohttp not installed – cannot auto-solve hCaptcha")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            if _PROVIDER == "capmonster":
                return await _capmonster_solve(
                    session,
                    task_type="HCaptchaTaskProxyless",
                    website_url=site_url,
                    website_key=site_key,
                    response_field="gRecaptchaResponse",
                )
            else:
                return await _2captcha_solve(
                    session,
                    method="hcaptcha",
                    extra_params={"sitekey": site_key, "pageurl": site_url},
                )
    except Exception as exc:
        logger.warning("captcha_solver: hCaptcha solve error: %s", exc)
        return None


# ── Provider implementations ──────────────────────────────────────────────────

async def _capmonster_solve(
    session,                 # type: ignore[type-arg]
    *,
    task_type: str,
    website_url: str,
    website_key: str,
    response_field: str,
) -> Optional[str]:
    """Submit a task to CapMonster Cloud and poll for the result."""
    payload = {
        "clientKey": _API_KEY,
        "task": {
            "type":       task_type,
            "websiteURL": website_url,
            "websiteKey": website_key,
        },
    }
    create_resp = await _post_json(session, "https://api.capmonster.cloud/createTask", payload)
    task_id = create_resp.get("taskId")
    if not task_id:
        logger.warning("captcha_solver: CapMonster createTask failed: %s", create_resp)
        return None

    for _ in range(_MAX_POLLS):
        await asyncio.sleep(_POLL_INTERVAL)
        result = await _post_json(
            session,
            "https://api.capmonster.cloud/getTaskResult",
            {"clientKey": _API_KEY, "taskId": task_id},
        )
        if result.get("status") == "ready":
            return result.get("solution", {}).get(response_field)

    logger.warning("captcha_solver: CapMonster polling timed out for task %s", task_id)
    return None


async def _2captcha_solve(
    session,                 # type: ignore[type-arg]
    *,
    method: str,
    extra_params: dict,
) -> Optional[str]:
    """Submit a task to 2captcha and poll for the result."""
    submit_data = {"key": _API_KEY, "method": method, "json": "1", **extra_params}
    resp = await _post_form(session, "https://2captcha.com/in.php", submit_data)
    if resp.get("status") != 1:
        logger.warning("captcha_solver: 2captcha in.php failed: %s", resp)
        return None

    captcha_id = resp.get("request")
    for _ in range(_MAX_POLLS):
        await asyncio.sleep(_POLL_INTERVAL)
        result = await _post_form(
            session,
            "https://2captcha.com/res.php",
            {"key": _API_KEY, "action": "get", "id": captcha_id, "json": "1"},
        )
        if result.get("status") == 1:
            return result.get("request")
        if result.get("request") not in ("CAPCHA_NOT_READY", "CAPTCHA_NOT_READY"):
            logger.warning("captcha_solver: 2captcha unexpected response: %s", result)
            break

    logger.warning("captcha_solver: 2captcha polling timed out for id=%s", captcha_id)
    return None
