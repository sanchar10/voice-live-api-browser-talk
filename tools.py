"""
tools.py — Dummy tool definitions for the Voice Live agent.

Each tool is defined with:
  1. A FunctionTool schema (tells the model *when* and *how* to call it)
  2. A handler function  (executes when the model invokes the tool)

Add / remove tools here — the rest of the code picks them up automatically.
"""

import json
from azure.ai.voicelive.models import FunctionTool

# ---------------------------------------------------------------------------
# Tool handlers — each receives the raw arguments dict and returns a JSON str
# ---------------------------------------------------------------------------

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


def get_weather(arguments: dict) -> str:
    """Return dummy weather data for the requested city."""
    city = arguments.get("city", "Unknown")

    # In a real app you'd call a weather API here
    return json.dumps({
        "city": city,
        "temperature": "62°F / 17°C",
        "condition": "Partly cloudy",
        "humidity": "58%",
    })

# ---------------------------------------------------------------------------
# Schema definitions — these are sent to the model so it knows what's available
# ---------------------------------------------------------------------------

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
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. Seattle, New York, London",
                },
            },
            "required": ["city"],
        },
    ),
]

# Map of function-name → handler callable (used by voice_agent to dispatch)
TOOL_HANDLERS: dict[str, callable] = {
    "get_stock_price": get_stock_price,
    "get_weather": get_weather,
}
