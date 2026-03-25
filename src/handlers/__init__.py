"""Handlers package — command & message handlers."""

from .command import (
    deploy,
    help_command,
    ping,
    reset,
    setapikey,
    setllmmodel,
    setmaxtokens,
    setollamamodel,
    setollamahost,
    setollamakey,
    setprovider,
    start,
)
from .message import echo_text, handle_docx_document, handle_pdf_document, handle_photo, unknown_message

__all__ = [
    "start",
    "help_command",
    "ping",
    "reset",
    "deploy",
    "setapikey",
    "setmaxtokens",
    "setllmmodel",
    "setprovider",
    "setollamakey",
    "setollamahost",
    "setollamamodel",
    "echo_text",
    "handle_docx_document",
    "handle_pdf_document",
    "handle_photo",
    "unknown_message",
]
