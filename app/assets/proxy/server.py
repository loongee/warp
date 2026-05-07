"""Warp Local Proxy — minimal server for local DeepSeek AI conversations.

Usage:
    DEEPSEEK_API_KEY=sk-xxx uvicorn server:app --port 8765

Then start Warp with:
    WARP_SERVER_ROOT_URL=http://localhost:8765 ./target/debug/warp-oss
"""
import sys
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import SERVER_PORT
from handlers.graphql import router as graphql_router
from handlers.multi_agent import router as multi_agent_router

app = FastAPI(title="Warp Local Proxy")

# Mount route handlers.
app.include_router(graphql_router)
app.include_router(multi_agent_router)


# --- Catch-all endpoints for auth and other Warp client requests ---

@app.post("/api/v1/auth/{path:path}")
@app.get("/api/v1/auth/{path:path}")
async def auth_mock(path: str, request: Request):
    """Mock all auth endpoints — return success."""
    return JSONResponse({"status": "ok"})


@app.get("/api/v1/{path:path}")
async def api_get_fallback(path: str, request: Request):
    """Catch-all for GET /api/v1/* — return empty JSON."""
    print(f"[api] Unhandled GET /api/v1/{path}")
    return JSONResponse({})


@app.post("/api/v1/{path:path}")
async def api_post_fallback(path: str, request: Request):
    """Catch-all for POST /api/v1/* — return empty JSON."""
    print(f"[api] Unhandled POST /api/v1/{path}")
    return JSONResponse({})


@app.post("/ai/{path:path}")
async def ai_fallback(path: str, request: Request):
    """Catch-all for other /ai/* endpoints (e.g. passive-suggestions)."""
    if path != "multi-agent":
        print(f"[ai] Unhandled POST /ai/{path}")
    return JSONResponse({})


# --- Health check ---

@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Shutdown endpoint ---

@app.post("/shutdown")
async def shutdown():
    """Gracefully shut down the proxy server."""
    import asyncio
    import os
    import signal

    asyncio.get_event_loop().call_later(0.5, os.kill, os.getpid(), signal.SIGTERM)
    return {"status": "shutting_down"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
