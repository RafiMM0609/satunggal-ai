"""
REST API interface – FastAPI server as a second interface alongside Telegram.

Run standalone:
    uvicorn src.interfaces.rest_api:app --host 0.0.0.0 --port 8000

Or mount inside the Telegram webhook Starlette app (see interfaces/webhook.py).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from src.interfaces.auth import require_api_key
from src.orchestrator.main_loop import clear_session, process_message

app = FastAPI(
    title="AdvanceAI Agent API",
    description="REST interface to the multi-agent AI system.",
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str


class ChatResponse(BaseModel):
    session_id: str
    reply:      str


class ClearResponse(BaseModel):
    session_id: str
    status:     str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    _: None = Depends(require_api_key),
) -> ChatResponse:
    """
    Send a message and receive a reply.

    The session_id groups messages into a conversation.
    Use the same session_id across turns to preserve context.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    task = await process_message(req.session_id, req.message)
    return ChatResponse(session_id=req.session_id, reply=task.result or "")


@app.delete("/session/{session_id}", response_model=ClearResponse)
async def reset_session(
    session_id: str,
    _: None = Depends(require_api_key),
) -> ClearResponse:
    """Clear conversation history for a session."""
    await clear_session(session_id)
    return ClearResponse(session_id=session_id, status="cleared")
