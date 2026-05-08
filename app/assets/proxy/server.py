"""Warp Local Proxy — minimal server for local DeepSeek AI conversations.

Usage:
    DEEPSEEK_API_KEY=sk-xxx uvicorn server:app --port 8765

Then start Warp with:
    WARP_SERVER_ROOT_URL=http://localhost:8765 ./target/debug/warp-oss
"""
import sys
import os
import signal
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import SERVER_PORT
from handlers.graphql import router as graphql_router
from handlers.multi_agent import router as multi_agent_router

app = FastAPI(title="Warp Local Proxy")

# Mount route handlers.
app.include_router(graphql_router)
app.include_router(multi_agent_router)


# --- Parent-liveness monitor via pipe ---

def _monitor_parent_pipe():
    """Monitor the pipe FD inherited from the parent process.

    The parent (Warp) creates a pipe and passes the read-end FD number via
    WARP_PARENT_ALIVE_FD. The parent keeps the write-end open. When the
    parent exits (even via SIGKILL), the OS closes the write-end and we
    see EOF on the read-end, at which point we self-terminate.
    """
    fd_str = os.environ.get("WARP_PARENT_ALIVE_FD")
    if not fd_str:
        print("[server] WARP_PARENT_ALIVE_FD not set, parent-liveness monitor disabled.")
        return

    try:
        fd = int(fd_str)
    except ValueError:
        print(f"[server] Invalid WARP_PARENT_ALIVE_FD value: {fd_str}")
        return

    print(f"[server] Monitoring parent liveness via fd {fd}...")

    try:
        # This blocks until the write-end is closed (parent exits) → returns b''
        # or until any data is written (shouldn't happen) → also means exit.
        data = os.read(fd, 1)
        # If we get here, the pipe was closed (EOF) or unexpected data.
        print(f"[server] Parent pipe closed (read returned {repr(data)}), shutting down...")
    except OSError as e:
        print(f"[server] Parent pipe error: {e}, shutting down...")

    # Self-terminate. Use SIGTERM for graceful uvicorn shutdown.
    os.kill(os.getpid(), signal.SIGTERM)


# Start the monitor thread on import (when uvicorn loads the app).
_parent_monitor_thread = threading.Thread(
    target=_monitor_parent_pipe,
    daemon=True,
    name="parent-liveness-monitor",
)
_parent_monitor_thread.start()


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
