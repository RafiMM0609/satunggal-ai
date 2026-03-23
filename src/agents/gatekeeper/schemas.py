"""Pydantic schemas shared across the system."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IntentCategory(str, Enum):
    """All possible intent classifications."""
    GENERAL_INQUIRY   = "general_inquiry"
    PRODUCT_QUESTION  = "product_question"
    COMPLAINT         = "complaint"
    ORDER_STATUS      = "order_status"
    TECHNICAL_SUPPORT = "technical_support"
    BILLING           = "billing"
    DATA_ANALYSIS      = "data_analysis"
    MANDAYS_PLANNING   = "mandays_planning"
    IMAGE_QUERY        = "image_query"
    RESEARCH           = "research"
    CONTENT_CREATION   = "content_creation"
    CODE_DEVELOPMENT    = "code_development"    # clone / edit / sandbox a repo
    CODE_INSPECTION    = "code_inspection"      # inspeksi repo, temukan akar masalah, beri rekomendasi (read-only)
    CODE_UNDERSTANDING = "code_understanding"   # tanya-jawab tentang isi repo: API, tech stack, model, dependency, dll.
    DOCUMENT_CREATION  = "document_creation"    # buat dokumen teknis PDF/Word dari repo atau topik
    SYSTEM_INFO        = "system_info"        # tanya info CPU, RAM, storage server
    LOG_VIEWER         = "log_viewer"         # lihat log bot untuk debugging
    QUIZ_GENERATION    = "quiz_generation"    # konversi PDF menjadi kuis interaktif HTML
    WEB_AUTOMATION     = "web_automation"     # autonomous browsing: buka URL, klik, isi form, screenshot
    UNKNOWN            = "unknown"


class IntentResult(BaseModel):
    """Classification result produced by GatekeeperAgent."""
    session_id:           str
    raw_text:             str              = Field(..., description="Normalised input text.")
    intent:               IntentCategory
    confidence:           float            = Field(..., ge=0.0, le=1.0)
    tools:                list[str]        = Field(default_factory=list, description="Ordered list of tool names to execute before calling the specialist agent.")
    model_used:           Optional[str]   = None
    metadata:             dict             = Field(default_factory=dict)
    needs_clarification:  bool             = Field(default=False, description="True when the intent is ambiguous and the bot should ask the user for more detail instead of proceeding.")
    clarification_question: Optional[str] = Field(default=None, description="Question to send back to the user when needs_clarification is True.")
