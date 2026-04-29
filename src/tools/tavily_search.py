"""
Tavily Search Tool – fetches up-to-date web results for research queries.

Usage (async-friendly wrapper around the sync TavilyClient):
    tool = TavilySearchTool()
    result = await tool.search("latest AI news 2026")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from tavily import TavilyClient

from config.settings import get_settings
from src.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from src.memory.state import AgentTask

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result returned by Tavily."""
    title:   str
    url:     str
    content: str
    score:   float = 0.0


@dataclass
class TavilySearchResponse:
    """Aggregated response from a Tavily search query."""
    query:           str
    results:         list[SearchResult] = field(default_factory=list)
    answer:          str | None         = None      # Tavily's AI-generated answer (if any)
    raw:             dict[str, Any]     = field(default_factory=dict)

    def as_context_text(self) -> str:
        """
        Format the search results as a context block to be injected into the
        LLM's system prompt.
        """
        lines: list[str] = [
            f"## Hasil Pencarian Web Terkini untuk: \"{self.query}\"",
            "",
        ]

        if self.answer:
            lines.append(f"**Ringkasan Cepat:** {self.answer}")
            lines.append("")

        for idx, r in enumerate(self.results, start=1):
            lines.append(f"### Sumber {idx}: {r.title}")
            lines.append(f"URL: {r.url}")
            lines.append(r.content.strip())
            lines.append("")

        return "\n".join(lines)


class TavilySearchTool(BaseTool):
    """
    Async-friendly wrapper around TavilyClient.

    The underlying SDK is synchronous, so searches are dispatched to a
    thread-pool executor so they don't block the event loop.
    Also implements BaseTool.run(task) so it can be called by the orchestrator.
    """

    name = "tavily_search"
    description = (
        "Search the web for up-to-date information using the Tavily search engine. "
        "Returns ranked results with titles, URLs, and content snippets. "
        "Use this tool when you need recent news, documentation, or factual data not in your training data."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string (plain-language question or keywords).",
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": "Search depth. 'advanced' uses more credits but returns richer context.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of search results to return.",
            },
        },
        "required": ["query"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "query":   {"type": "string", "description": "The original search query."},
            "answer":  {"type": "string", "description": "AI-generated answer from Tavily (may be null)."},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title":   {"type": "string"},
                        "url":     {"type": "string", "format": "uri"},
                        "content": {"type": "string"},
                        "score":   {"type": "number"},
                    },
                },
                "description": "Ranked list of web search results.",
            },
            "error": {"type": "string", "description": "Present only on failure."},
        },
    }

    def __init__(self) -> None:
        settings = get_settings()
        api_key  = settings.tavily_api_key
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is not set. "
                "Please add it to your .env file."
            )
        self._client       = TavilyClient(api_key)
        self._search_depth = settings.tavily_search_depth
        self._max_results  = settings.tavily_max_results
        self._time_range    = settings.tavily_time_range

    async def search(
        self,
        query: str,
        *,
        search_depth: str | None = None,
        max_results:  int  | None = None,
    ) -> TavilySearchResponse:
        """
        Run a Tavily web search and return structured results.

        Args:
            query:        The search query string.
            search_depth: Override instance default ("basic" or "advanced").
            max_results:  Override instance default number of results.

        Returns:
            TavilySearchResponse with parsed results and optional AI answer.
        """
        depth   = search_depth or self._search_depth
        n       = max_results  or self._max_results
        time    = self._time_range

        logger.info("TavilySearchTool.search query=%r depth=%s max=%d time=%s", query, depth, n, time)

        loop = asyncio.get_event_loop()
        try:
            raw: dict[str, Any] = await loop.run_in_executor(
                None,
                lambda: self._client.search(
                    query        = query,
                    search_depth = depth,
                    max_results  = n,
                    time_range   = time,
                    include_answer = True,
                ),
            )
        except Exception as exc:
            logger.exception("TavilySearchTool request failed: %s", exc)
            raise

        logger.debug("TavilySearchTool raw response keys: %s", list(raw.keys()))

        results: list[SearchResult] = [
            SearchResult(
                title   = item.get("title", ""),
                url     = item.get("url", ""),
                content = item.get("content", ""),
                score   = float(item.get("score", 0.0)),
            )
            for item in raw.get("results", [])
        ]

        return TavilySearchResponse(
            query   = query,
            results = results,
            answer  = raw.get("answer") or None,
            raw     = raw,
        )

    async def run(self, task: "AgentTask") -> dict[str, Any]:
        """BaseTool interface: search using task.user_input and return context text.

        Returns:
            dict with keys:
              - context_text (str) : formatted search context for LLM injection
              - results_count (int): number of results found
              - error (str)        : present only on failure
        """
        try:
            response = await self.search(task.user_input)
            if not response.results:
                logger.info("TavilySearchTool: no results for query=%r", task.user_input)
                return {"context_text": "", "results_count": 0}
            context_text = response.as_context_text()
            logger.info(
                "TavilySearchTool.run: session=%s results=%d",
                task.session_id, len(response.results),
            )
            return {"context_text": context_text, "results_count": len(response.results)}
        except Exception as exc:
            logger.warning("TavilySearchTool.run failed (non-fatal): %s", exc)
            return {"context_text": "", "results_count": 0, "error": str(exc)}
