```markdown
# 🌤️ Weather Agent with Local LLM using FastMCP & Ollama

A beginner-friendly AI Agent project powered by [FastMCP](https://gofastmcp.com/), [Ollama](https://ollama.com/), and the National Weather Service (NWS) API. This agent uses a local LLaMA3 model to understand user queries and trigger tools like weather forecast and alerts using natural language.

## 🚀 Features

- 🤖 Local LLM integration via Ollama (e.g., LLaMA3)
- 🌦️ Fetches real-time weather forecasts using the NWS API
- ⚠️ Active weather alerts by U.S. state
- 💬 Chat interface with HTMX (or use FastAPI endpoints)
- 🧠 MCP agent handles tool routing from user input
- 🔧 Extensible tool system via `@mcp.tool()`

---

## 🧰 Technologies Used

- [FastAPI](https://fastapi.tiangolo.com/)
- [FastMCP](https://gofastmcp.com/)
- [Ollama](https://ollama.com/)
- [httpx](https://www.python-httpx.org/)
- [Jinja2](https://palletsprojects.com/p/jinja/)
- National Weather Service API (`api.weather.gov`)

---

## 📁 Project Structure

```

weather\_agent/
├── templates/
│   └── chat.html           # Simple HTMX-based UI
├── weather\_tool.py         # Main FastAPI + MCP logic
├── README.md               # You're here!
└── requirements.txt

````

---

## ⚙️ Setup Instructions

### 1. 🐍 Clone and Install

```bash
git clone https://github.com/nutanlade/mcp-weather.git
cd weather-agent-llm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### 2. 🦙 Start Ollama with LLaMA3

Make sure you have Ollama installed from [ollama.com](https://ollama.com).

```bash
ollama run llama3
```

> You can change the model in the script if you use something else.

### 3. 🚀 Run the FastAPI + MCP Server

```bash
uvicorn weather_tool:app --reload
```

Visit `http://localhost:8000/chat` for chat UI, or explore `/docs` for Swagger UI.

---

## 💬 How It Works

1. User submits a query like **"What’s the forecast for CA?"**
2. Local LLaMA3 model (via Ollama) parses intent using MCP
3. MCP agent triggers the right tool (e.g., `get_forecast`)
4. Tool fetches forecast using NWS API and responds back

![Architecture Diagram](https://github.com/nutanlade/mcp-weather/assets/architecture.png)

> ⚠️ Replace with actual image or use the provided architecture diagram from this repo.

---

## 🛠 Tools Available

| Tool                     | Description                                  |
| ------------------------ | -------------------------------------------- |
| `get_forecast(lat, lon)` | Returns 5-day forecast for given coordinates |
| `get_alerts(state_code)` | Returns active alerts for the U.S. state     |
| `ask_llm(prompt)`        | General-purpose chat with local LLaMA        |

---

## 🧪 Example Prompts

* "What's the weather like in San Francisco?"
* "Are there any alerts in NY?"
* "What will it be like in 90210 tomorrow?"
* "Explain how this agent works."

---

## 🧠 Want to Learn More?

* 👉 [FastMCP GitHub](https://github.com/lowink/fastmcp)
* 👉 [Ollama Documentation](https://ollama.com/library)
* 👉 [NWS API Docs](https://www.weather.gov/documentation/services-web-api)

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## 🙋‍♀️ Author

Built with ❤️ by Nutan Kumar Lade (https://www.linkedin.com/in/nutankumarlade/)

Feel free to connect or contribute!

```
