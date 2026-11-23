"""Expose the running scene server's tool catalog to an MCP client over stdio."""
import argparse
import asyncio
import json
import os

import httpx
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


async def serve(base_url):
    server = Server("scene-builder")
    async with httpx.AsyncClient(base_url=base_url.rstrip("/") + "/", timeout=180) as client:
        async def data(method, path, **kwargs):
            response = await client.request(method, path, **kwargs)
            if not response.is_success:
                try:
                    detail = response.json().get("detail", response.reason_phrase)
                except ValueError:
                    detail = response.reason_phrase
                raise ValueError(f"Scene API HTTP {response.status_code}: {detail}")
            return response.json()["data"]

        @server.list_tools()
        async def list_tools():
            catalog = await data("GET", "api/builder/tools")
            functions = (item["function"] for item in catalog["functions"])
            return [types.Tool(name=f["name"], description=f["description"], inputSchema=f["parameters"])
                    for f in functions]

        @server.call_tool()
        async def call_tool(name, arguments):
            result = await data("POST", "api/builder/tools/call", json={"name": name, "arguments": arguments})
            return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.environ.get("SCENE_URL", "http://127.0.0.1:8090"))
    asyncio.run(serve(parser.parse_args().server))
