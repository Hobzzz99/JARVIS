"""JARVIS MCP server.

Exposes the retrieval and memory tools over the Model Context Protocol via
stdio, so any MCP-compatible client (Claude Desktop, Claude Code, a custom
host) can drive the same capabilities the internal agents use.

Run it with::

    python -m jarvis_mcp.server
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from config import get_logger
from jarvis_mcp.tools.arxiv_tool import fetch_ai_papers
from jarvis_mcp.tools.memory_tool import (
    get_recent_briefings,
    get_workflow_logs,
    load_preferences,
    save_briefing,
    save_preferences,
)
from jarvis_mcp.tools.news_tool import fetch_ai_news

logger = get_logger("jarvis.mcp.server")

app = Server("jarvis-mcp")

# Single source of truth: name -> (callable, MCP schema).
TOOL_REGISTRY: dict[str, tuple[Any, types.Tool]] = {
    "fetch_ai_news": (
        fetch_ai_news,
        types.Tool(
            name="fetch_ai_news",
            description="Fetch the latest English-language AI news articles from NewsAPI.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search expression.",
                        "default": "artificial intelligence",
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Maximum number of articles to return.",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
            },
        ),
    ),
    "fetch_ai_papers": (
        fetch_ai_papers,
        types.Tool(
            name="fetch_ai_papers",
            description="Fetch the most recent AI research preprints from arXiv (no API key required).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "arXiv search expression.",
                        "default": "LLM agents",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of papers to return.",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
            },
        ),
    ),
    "load_preferences": (
        load_preferences,
        types.Tool(
            name="load_preferences",
            description="Load operator preferences (name, interests, favourite sources) from memory.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ),
    "save_preferences": (
        save_preferences,
        types.Tool(
            name="save_preferences",
            description="Persist operator preferences to memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "preferences": {
                        "type": "object",
                        "description": "Full preferences document to store.",
                    }
                },
                "required": ["preferences"],
            },
        ),
    ),
    "save_briefing": (
        save_briefing,
        types.Tool(
            name="save_briefing",
            description="Append a briefing to the persistent archive.",
            inputSchema={
                "type": "object",
                "properties": {
                    "briefing": {"type": "string", "description": "Briefing body text."},
                    "briefing_date": {
                        "type": "string",
                        "description": "ISO date; defaults to today.",
                    },
                },
                "required": ["briefing"],
            },
        ),
    ),
    "get_recent_briefings": (
        get_recent_briefings,
        types.Tool(
            name="get_recent_briefings",
            description="Return the last N briefings so the planner can avoid repeating topics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "How many to return.", "default": 3}
                },
            },
        ),
    ),
    "get_workflow_logs": (
        get_workflow_logs,
        types.Tool(
            name="get_workflow_logs",
            description="Return the last N workflow log entries for observability.",
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "How many to return.", "default": 50}
                },
            },
        ),
    ),
}


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Advertise every registered tool to the MCP client."""
    return [schema for _, schema in TOOL_REGISTRY.values()]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Dispatch an MCP tool call and return its result as JSON text content."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown tool: {name}")

    handler, _ = entry
    logger.info("MCP tool call: %s(%s)", name, ", ".join(arguments or {}))
    # Tools are synchronous and some are I/O bound — keep the event loop free.
    result = await asyncio.to_thread(handler, **(arguments or {}))
    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def main() -> None:
    """Serve the MCP protocol over stdio until the client disconnects."""
    logger.info("Starting JARVIS MCP server with %d tools", len(TOOL_REGISTRY))
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
