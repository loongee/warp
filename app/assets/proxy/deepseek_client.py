"""DeepSeek streaming client wrapper with tool/function calling support."""
import json
import logging
import httpx
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

logger = logging.getLogger("warp-proxy.deepseek")

# Tools definition for DeepSeek (OpenAI-compatible function calling format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a shell command on the user's machine. Use this to execute commands, install packages, run scripts, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "is_read_only": {
                        "type": "boolean",
                        "description": "Whether the command only reads data (no side effects)",
                        "default": False,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "description": "Read the contents of one or more files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Absolute path to the file"},
                            },
                            "required": ["path"],
                        },
                        "description": "List of files to read",
                    },
                },
                "required": ["files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_file_diffs",
            "description": "Apply diffs/edits to files. Use this to create or modify files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief description of the changes",
                    },
                    "diffs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path"},
                                "diff": {"type": "string", "description": "Unified diff content"},
                            },
                            "required": ["path", "diff"],
                        },
                        "description": "List of file diffs to apply",
                    },
                },
                "required": ["summary", "diffs"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an AI assistant integrated into the Warp terminal. You help users with coding, shell commands, and system tasks.

When the user asks you to run a command, use the `run_shell_command` tool to execute it. The user will see the command and can approve or reject it before execution.

When you need to read files, use the `read_files` tool.

When you need to create or edit files, use the `apply_file_diffs` tool.

Be concise and helpful. When suggesting commands, always use the tool so the user gets the interactive run/reject UI."""


# In-memory cache for reasoning_content per conversation.
# Key: conversation_id, Value: list of reasoning strings (one per assistant tool_calls turn)
_reasoning_cache: dict[str, list[str]] = {}


def get_all_cached_reasoning(conversation_id: str) -> list[str]:
    """Get all cached reasoning_content entries for a conversation."""
    return _reasoning_cache.get(conversation_id, [])


def cache_reasoning(conversation_id: str, reasoning: str):
    """Append reasoning_content for a conversation turn."""
    if conversation_id not in _reasoning_cache or not isinstance(_reasoning_cache[conversation_id], list):
        _reasoning_cache[conversation_id] = []
    _reasoning_cache[conversation_id].append(reasoning)
    _reasoning_cache[conversation_id] = reasoning


async def stream_chat(messages: list[dict], model: str | None = None, tools_enabled: bool = True, conversation_id: str = "", api_key: str | None = None):
    """Stream chat completions from DeepSeek API with tool calling support.

    Yields tuples of (event_type, data):
      - ("content", str): text content chunk
      - ("reasoning", str): reasoning content chunk (chain-of-thought)
      - ("tool_call_start", {"id": str, "name": str}): start of a tool call
      - ("tool_call_args", str): argument chunk for current tool call
      - ("tool_call_end", dict): end of tool call with full data
      - ("done", None): stream complete
    """
    # Use API key from request, fall back to env var.
    effective_api_key = api_key or DEEPSEEK_API_KEY
    if not effective_api_key:
        raise ValueError("No DeepSeek API key provided (neither from client nor DEEPSEEK_API_KEY env var)")

    model = model or DEEPSEEK_MODEL
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"

    # Prepend system prompt
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    payload = {
        "model": model,
        "messages": full_messages,
        "stream": True,
    }

    if tools_enabled:
        payload["tools"] = TOOLS

    headers = {
        "Authorization": f"Bearer {effective_api_key}",
        "Content-Type": "application/json",
    }

    # Track tool call state during streaming
    current_tool_calls = {}  # index -> {id, name, arguments}
    collected_reasoning = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error("Error %d: %s", resp.status_code, body.decode())
                logger.error("Request payload messages: %s", json.dumps(full_messages, ensure_ascii=False, default=str)[:2000])
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    # Always cache reasoning for this conversation (even if empty)
                    # to keep index in sync with assistant message count.
                    if conversation_id:
                        cache_reasoning(conversation_id, collected_reasoning)
                    # Flush any pending tool calls
                    for idx in sorted(current_tool_calls.keys()):
                        tc = current_tool_calls[idx]
                        yield ("tool_call_end", tc)
                    yield ("done", None)
                    return

                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                # Handle reasoning_content (DeepSeek thinking mode)
                reasoning_content = delta.get("reasoning_content")
                if reasoning_content:
                    collected_reasoning += reasoning_content
                    yield ("reasoning", reasoning_content)

                # Handle text content
                content = delta.get("content")
                if content:
                    yield ("content", content)

                # Handle tool calls
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc.get("id", ""),
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": "",
                            }
                            # Emit start event once we have the name
                            if current_tool_calls[idx]["name"]:
                                yield ("tool_call_start", current_tool_calls[idx])
                        else:
                            # Update id/name if they come in later chunks
                            if tc.get("id"):
                                current_tool_calls[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                current_tool_calls[idx]["name"] = tc["function"]["name"]
                                yield ("tool_call_start", current_tool_calls[idx])

                        # Accumulate arguments
                        args_chunk = tc.get("function", {}).get("arguments", "")
                        if args_chunk:
                            current_tool_calls[idx]["arguments"] += args_chunk
                            yield ("tool_call_args", args_chunk)

                # Check for finish_reason
                finish_reason = choices[0].get("finish_reason")
                if finish_reason == "tool_calls":
                    # Always cache reasoning (even if empty) to keep index in sync
                    if conversation_id:
                        cache_reasoning(conversation_id, collected_reasoning)
                    for idx in sorted(current_tool_calls.keys()):
                        tc = current_tool_calls[idx]
                        yield ("tool_call_end", tc)
                    yield ("done", None)
                    return
