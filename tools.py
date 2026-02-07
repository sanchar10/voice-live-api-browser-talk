"""
tools.py — Tool definitions for the Voice Live agent.

Two kinds of tools live here:
  1. **Local FunctionTools** — schemas + handler functions that run in this process
  2. **Remote MCP servers** — loaded from mcp_servers.json; Azure's service calls
     them directly over HTTPS (your code never touches the MCP request/response)

Add / remove tools here — the rest of the code picks them up automatically.
"""

import json
import logging
import random
from pathlib import Path

from azure.ai.voicelive.models import FunctionTool, MCPServer

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Local tool handlers — each receives the raw arguments dict → returns JSON
# ═══════════════════════════════════════════════════════════════════════════

def get_stock_price(arguments: dict) -> str:
    """Return a dummy stock price for the requested symbol."""
    symbol = arguments.get("symbol", "UNKNOWN").upper()

    # In a real app you'd call a market-data API here
    fake_prices = {
        "MSFT":  {"price": 425.30, "change": "+1.2%"},
        "AAPL":  {"price": 198.50, "change": "-0.4%"},
        "GOOGL": {"price": 178.20, "change": "+0.8%"},
        "AMZN":  {"price": 186.90, "change": "+1.5%"},
    }
    data = fake_prices.get(symbol, {"price": 100.00, "change": "0.0%"})
    return json.dumps({"symbol": symbol, **data})


def tell_joke(arguments: dict) -> str:
    """Return a random joke, optionally filtered by topic."""
    topic = arguments.get("topic", "").lower()

    jokes = [
        {"setup": "Why do programmers prefer dark mode?", "punchline": "Because light attracts bugs."},
        {"setup": "Why was the JavaScript developer sad?", "punchline": "Because he didn't Node how to Express himself."},
        {"setup": "What's a computer's favorite snack?", "punchline": "Microchips."},
        {"setup": "Why did the developer go broke?", "punchline": "Because he used up all his cache."},
        {"setup": "What do you call a bear with no teeth?", "punchline": "A gummy bear."},
        {"setup": "Why don't scientists trust atoms?", "punchline": "Because they make up everything."},
        {"setup": "What did the ocean say to the shore?", "punchline": "Nothing, it just waved."},
        {"setup": "Why did the scarecrow win an award?", "punchline": "Because he was outstanding in his field."},
    ]

    if topic:
        filtered = [j for j in jokes if topic in j["setup"].lower() or topic in j["punchline"].lower()]
        joke = random.choice(filtered) if filtered else random.choice(jokes)
    else:
        joke = random.choice(jokes)

    return json.dumps(joke)


# ═══════════════════════════════════════════════════════════════════════════
# Local FunctionTool schemas (sent to the model so it knows what's available)
# ═══════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS: list[FunctionTool] = [
    FunctionTool(
        name="get_stock_price",
        description="Get the current stock price for a given ticker symbol",
        parameters={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. MSFT, AAPL, GOOGL",
                },
            },
            "required": ["symbol"],
        },
    ),
    FunctionTool(
        name="tell_joke",
        description="Tell a funny joke. Optionally specify a topic.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Optional topic for the joke, e.g. programming, animals, science",
                },
            },
            "required": [],
        },
    ),
]

# Map of function-name → handler callable (used by voice_agent to dispatch)
TOOL_HANDLERS: dict[str, callable] = {
    "get_stock_price": get_stock_price,
    "tell_joke": tell_joke,
}


# ═══════════════════════════════════════════════════════════════════════════
# Remote MCP servers — loaded from mcp_servers.json
# ═══════════════════════════════════════════════════════════════════════════

def load_mcp_servers(path: str = "mcp_servers.json") -> list[MCPServer]:
    """Read MCP server definitions from a JSON file.

    Returns an empty list if the file doesn't exist (graceful fallback).
    Each entry becomes an MCPServer object that Azure Voice Live connects to
    directly over HTTPS — your code never proxies the MCP traffic.
    """
    config_path = Path(path)
    if not config_path.exists():
        logger.info("No %s found — MCP servers disabled", path)
        return []

    try:
        entries = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return []

    servers: list[MCPServer] = []
    for entry in entries:
        kwargs = {
            "server_label": entry["server_label"],
            "server_url": entry["server_url"],
            "require_approval": entry.get("require_approval", "never"),
        }
        if "allowed_tools" in entry:
            kwargs["allowed_tools"] = entry["allowed_tools"]
        servers.append(MCPServer(**kwargs))
        logger.info("MCP server loaded: %s → %s", entry["server_label"], entry["server_url"])

    return servers


# Load at import time — voice_agent.py imports this list directly
MCP_SERVERS: list[MCPServer] = load_mcp_servers()
