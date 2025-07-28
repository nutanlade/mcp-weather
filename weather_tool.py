from typing import Any
import httpx, asyncio, json
from mcp.server.fastmcp import FastMCP
from fastapi import FastAPI

from fastapi.responses import HTMLResponse
from fastapi import Request, Form
from fastapi.templating import Jinja2Templates
from httpx import ReadTimeout, RequestError

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OLLAMA_API = "http://localhost:11434/api/generate"


# NWS Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

# ---- tune these to your needs ----
OLLAMA_CHAT_URL = "http://localhost:11434/v1/chat/completions"
FASTMCP_TOOL_URL = "http://localhost:8000"  # wherever your FastMCP server is running
MAX_RETRIES = 3
BASE_BACKOFF_SECS = 1.5

app = FastAPI(
    title="Weather Agent",
    description="MCP-powered weather agent with NWS data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

#mcp = FastMCP(app=app)  # Main MCP server object
#mcp = FastMCP(app=app, tools_module="weather_tool")  # or just __name__ if it's same file

mcp = FastMCP(app=app, tools_module=__name__)


TOOLS = [
  {
    "name": "get_alerts",
    "description": "Fetch active weather alerts for a US state",
    "parameters": {
      "type": "object",
      "properties": {"state": {"type": "string"}},
      "required": ["state"]
    }
  },
  {
    "name": "get_forecast",
    "description": "Get 5-period forecast from coordinates",
    "parameters": {
      "type": "object",
      "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"}
      },
      "required": ["latitude", "longitude"]
    }
  }
]

SYSTEM_PROMPT = f"""
You have the following tools at your disposal:
{json.dumps(TOOLS, indent=2)}

When the user asks for weather alerts or forecast, call the appropriate tool.
"""


async def make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

@app.get("/")
async def root():
    return {"message": "Weather MCP agent is running. Visit /docs to explore the tools."}


@mcp.tool()
async def get_alerts(state: str) -> str:
    logger.info("Received alerts result: %s", state)
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data or not data["features"]:
        return "No active alerts for this state."

    return "\n---\n".join([format_alert(f) for f in data["features"]])

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    logger.info("Received forecast result: %s %s", latitude, longitude)
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Could not fetch forecast data."

    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Could not fetch forecast details."

    periods = forecast_data["properties"]["periods"]
    return "\n---\n".join([
        f"{p['name']}:\nTemp: {p['temperature']}°{p['temperatureUnit']}\nWind: {p['windSpeed']} {p['windDirection']}\n{p['detailedForecast']}"
        for p in periods[:5]
    ])


async def call_ollama(prompt: str, model: str = "llama3") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_API, json=payload)
            result = response.json()["response"]
    except Exception as e:
        result = f"Error: {str(e)}"

    return f"<div class='message'><strong>You:</strong> {prompt}<br/><strong>AI:</strong> {result}</div>"

@mcp.tool()
async def ask_llm(prompt: str) -> str:
    """Ask a local LLM like LLaMA3 for a general-purpose response."""
    try:
        return await call_ollama(prompt)
    except Exception as e:
        return f"Failed to get response from local LLM: {e}"


templates = Jinja2Templates(directory="templates")

@app.get("/chat", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# @app.post("/chat", response_class=HTMLResponse)
# async def chat(prompt: str = Form(...)):
#     reply = await call_ollama(prompt)
#     return f"<div class='message'><b>You:</b> {prompt}<br/><b>LLM:</b> {reply}</div>"

# @app.post("/chat", response_class=HTMLResponse)
# async def chat(request: Request, prompt: str = Form(...)):
#     print(f"🔧 Tool called: chat({prompt})")
#     #reply = await mcp.agent.chat(prompt)  # MCP will decide if it needs to call any tool
#     agent = mcp.get_agent()
#     reply = await agent.chat(prompt)

#     return templates.TemplateResponse("chat.html", {
#         "request": request,
#         "response": f"<div class='message'><b>You:</b> {prompt}<br/><b>LLM:</b> {reply}</div>"
#     })

@app.post("/chat", response_class=HTMLResponse)
async def chat(prompt: str = Form(...)):
    reply = await agent_chat(prompt)
    return f"<div class='message'><b>You:</b> {prompt}<br/><b>LLM:</b> {reply}</div>"

# A helper to create a rich timeout object
def _timeout(total: float = 60.0) -> httpx.Timeout:
    # You can tune each phase if you want
    return httpx.Timeout(timeout=total, connect=10.0, read=45.0, write=10.0, pool=10.0)


async def _post_with_retries(client: httpx.AsyncClient, url: str, json_payload: dict,
                             *, max_retries: int = MAX_RETRIES, base_backoff: float = BASE_BACKOFF_SECS):
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await client.post(url, json=json_payload)
        except ReadTimeout as e:
            last_exc = e
            if attempt == max_retries:
                break
            await asyncio.sleep(base_backoff * (2 ** (attempt - 1)))
        except RequestError as e:
            # Covers DNS errors, connection refused, etc.
            last_exc = e
            if attempt == max_retries:
                break
            await asyncio.sleep(base_backoff * (2 ** (attempt - 1)))
    # Exhausted retries
    raise last_exc if last_exc else RuntimeError("Unknown error without exception")


def format_forecast(json_result: str) -> str:
    try:
        logger.info("Received format_forecast result: %s", json_result)
        data = json.loads(json_result)
        days = data.get("data", [])
        if not days:
            return "No forecast data available."

        lines = ["📅 5-Day Forecast:"]
        for day in days:
            date = day.get("date", "")[:10]
            temp = day.get("temp", "N/A")
            cond = day.get("conditions", "")
            lines.append(f"• {date}: {temp}°F, {cond}")
        return "\n".join(lines)
    except Exception:
        return "Sorry, couldn't format the forecast properly."


async def agent_chat(user_prompt: str, model: str = "llama3") -> str:
    """
    Send a chat message to your local LLM (Ollama / LM Studio OpenAI-compatible)
    and let it *suggest* tool calls. If it asks to call a tool, we execute it
    via FastMCP HTTP and then send the tool result back to the LLM.
    """
    try:
        print(f"🔧 Tool called: chat({user_prompt})")
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            # 1) Ask the model
            req = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                # If your local server supports function/tool calling OpenAI style:
                "functions": TOOLS,
                "function_call": "auto",
                # For LM Studio you might instead need:
                # "tools": [...],  "tool_choice": "auto"
            }

            resp = await _post_with_retries(client, OLLAMA_CHAT_URL, req)
            data = resp.json()

            msg = data["choices"][0]["message"]

            # 2) If it wants to call a tool, do it
            if "function_call" in msg:
                fn = msg["function_call"]["name"]
                args = json.loads(msg["function_call"]["arguments"] or "{}")

                # Execute the tool via your MCP HTTP surface.
                # Adjust path/schema to match how FastMCP exposes them (if at all).
                # If you don't have an HTTP surface, import and call the python function directly instead.
                tool_resp = await _post_with_retries(
                    client,
                    f"{FASTMCP_TOOL_URL}/tools/{fn}",
                    args
                )

                if fn == "get_forecast":
                    tool_output = format_forecast(tool_resp.text)
                else:
                    tool_output = tool_resp.json().get("output", str(tool_resp.text))

                # 3) Send tool result back to the model to produce final answer
                followup_req = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": msg.get("content") or "", "function_call": msg["function_call"]},
                        #{"role": "function", "name": fn, "content": tool_output},
                        {"role": "function", "name": fn, "content": f"Here is the weather forecast result:\n{tool_output}\nPlease summarize it for the user in a friendly tone."}

                    ],
                }
                followup = await _post_with_retries(client, OLLAMA_CHAT_URL, followup_req)
                followup_data = followup.json()
                return followup_data["choices"][0]["message"]["content"]

            # No tool call, just return model content
            return msg.get("content", "").strip()

    except ReadTimeout:
        return "The local LLM did not respond in time (timeout). Please try again."
    except RequestError as e:
        return f"Failed to reach the local LLM API: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
