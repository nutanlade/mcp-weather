# main.py
from weather_tool import mcp

if __name__ == "__main__":
    mcp.run(transport="streamable-http")  # MCP entry point
