"""POST /ai/multi-agent handler — the core AI conversation endpoint.

Receives a protobuf Request, calls DeepSeek, and streams back
ResponseEvent messages as SSE (base64url-encoded protobuf).
"""
import base64
import json
import logging
import sys
import uuid
import os

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

# Add the local proto/ directory (copied from warp-proto-apis) to the path.
_PROTO_DIR = os.path.join(os.path.dirname(__file__), "..", "proto")
if os.path.abspath(_PROTO_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROTO_DIR))

import request_pb2  # noqa: E402
import response_pb2  # noqa: E402
import task_pb2  # noqa: E402
from google.protobuf import field_mask_pb2  # noqa: E402

from deepseek_client import stream_chat, get_all_cached_reasoning, cache_reasoning  # noqa: E402

logger = logging.getLogger("warp-proxy.multi_agent")

router = APIRouter()


def _b64url_encode(data: bytes) -> str:
    """Encode bytes to base64url (URL-safe, WITH padding) as Warp expects."""
    return base64.urlsafe_b64encode(data).decode("ascii")


def _make_event(proto_msg) -> dict:
    """Wrap a serialized protobuf in an SSE event dict."""
    encoded = _b64url_encode(proto_msg.SerializeToString())
    return {"data": f'"{encoded}"'}


def _extract_user_query(req: request_pb2.Request) -> str:
    """Extract the user's text query from the protobuf Request."""
    inp = req.input
    if not inp or not inp.HasField("type"):
        return ""

    field_name = inp.WhichOneof("type")

    # New-style: user_inputs (array of inputs)
    if field_name == "user_inputs":
        for user_input in inp.user_inputs.inputs:
            if user_input.HasField("user_query"):
                return user_input.user_query.query
        return ""

    # Legacy: direct user_query
    if field_name == "user_query":
        return inp.user_query.query

    return ""


def _extract_referenced_attachments(req: request_pb2.Request) -> str:
    """Extract referenced_attachments from the UserQuery and format as context string.

    When a user uses 'Attach as agent context', Warp stores those attachments in
    user_query.referenced_attachments (a map<string, Attachment>). This function
    extracts them and returns a formatted context string.
    """
    inp = req.input
    if not inp or not inp.HasField("type"):
        return ""

    attachments_map = None
    field_name = inp.WhichOneof("type")

    if field_name == "user_inputs":
        for user_input in inp.user_inputs.inputs:
            if user_input.HasField("user_query"):
                attachments_map = user_input.user_query.referenced_attachments
                break
    elif field_name == "user_query":
        attachments_map = inp.user_query.referenced_attachments

    if not attachments_map:
        return ""

    context_parts = []
    for key, attachment in attachments_map.items():
        value_field = attachment.WhichOneof("value")
        if not value_field:
            continue

        if value_field == "plain_text":
            context_parts.append(f"[Attached text - {key}]\n{attachment.plain_text}")
        elif value_field == "executed_shell_command":
            cmd = attachment.executed_shell_command
            context_parts.append(
                f"[Attached command output - {key}]\n"
                f"$ {cmd.command}\n"
                f"Exit code: {cmd.exit_code}\n"
                f"Output:\n{cmd.output}"
            )
        elif value_field == "running_shell_command":
            cmd = attachment.running_shell_command
            snapshot_output = cmd.snapshot.output if cmd.HasField("snapshot") else ""
            context_parts.append(
                f"[Attached running command - {key}]\n"
                f"$ {cmd.command}\n"
                f"Output so far:\n{snapshot_output}"
            )
        elif value_field == "document_content":
            doc = attachment.document_content
            # DocumentContent has various fields; try to extract text
            context_parts.append(f"[Attached document - {key}]\n{doc}")
        elif value_field == "file_path_reference":
            ref = attachment.file_path_reference
            context_parts.append(f"[Attached file reference - {key}]\nFile: {ref.file_path}")
        elif value_field == "diff_set":
            ds = attachment.diff_set
            hunks_text = []
            for hunk in ds.hunks:
                hunks_text.append(f"  {hunk.file_path}:\n{hunk.diff_content}")
            context_parts.append(
                f"[Attached diff - {key}]\n" + "\n".join(hunks_text)
            )
        elif value_field == "drive_object":
            obj = attachment.drive_object
            payload_field = obj.WhichOneof("object_payload")
            if payload_field == "workflow":
                context_parts.append(
                    f"[Attached workflow - {key}]\n"
                    f"Name: {obj.workflow.name}\n"
                    f"Description: {obj.workflow.description}\n"
                    f"Command: {obj.workflow.command}"
                )
            elif payload_field == "notebook":
                context_parts.append(
                    f"[Attached notebook - {key}]\n"
                    f"Title: {obj.notebook.title}\n"
                    f"Content:\n{obj.notebook.content}"
                )
            elif payload_field == "generic_string_object":
                context_parts.append(
                    f"[Attached object - {key}]\n{obj.generic_string_object.payload}"
                )
        else:
            context_parts.append(f"[Attached context - {key}] (type: {value_field})")

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)


def _extract_input_context(req: request_pb2.Request) -> str:
    """Extract InputContext (files, selected text, images, commands) from req.input.context.

    This contains contextual information provided as agent context via
    "Attach as agent context". The main sources are:
    - executed_shell_commands: command blocks attached as context
    - selected_text: text selections attached as context
    - files: file contents attached as context
    - images: image data attached as context
    """
    inp = req.input
    if not inp:
        return ""

    # InputContext is always present (not part of oneof), accessed via inp.context
    ctx = inp.context
    if not ctx:
        return ""

    context_parts = []

    # Executed shell commands (this is where "Attach as agent context" puts command blocks)
    for cmd in ctx.executed_shell_commands:
        cmd_text = f"[Attached command output]\n$ {cmd.command}"
        if cmd.exit_code != 0:
            cmd_text += f"\nExit code: {cmd.exit_code}"
        if cmd.output:
            cmd_text += f"\nOutput:\n{cmd.output}"
        context_parts.append(cmd_text)

    # Selected text blocks
    for selected in ctx.selected_text:
        if selected.text:
            context_parts.append(f"[Selected text]\n{selected.text}")

    # Attached files (FileContent)
    for file_entry in ctx.files:
        if file_entry.content:
            fc = file_entry.content
            # FileContent has path and content fields
            path = fc.path if hasattr(fc, 'path') and fc.path else "unknown"
            content = fc.content if hasattr(fc, 'content') and fc.content else ""
            if content:
                context_parts.append(f"[Attached file: {path}]\n{content}")

    # Images (binary data - we note their presence but don't pass raw bytes to text model)
    for img in ctx.images:
        mime = img.mime_type if img.mime_type else "image/*"
        context_parts.append(f"[Attached image ({mime}) - binary data not shown]")

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)


def _extract_tool_call_results(req: request_pb2.Request) -> list[dict]:
    """Extract tool call results from the request (user approved/rejected a command)."""
    results = []
    inp = req.input
    if not inp or not inp.HasField("type"):
        return results

    field_name = inp.WhichOneof("type")

    if field_name == "user_inputs":
        for user_input in inp.user_inputs.inputs:
            if user_input.HasField("tool_call_result"):
                tcr = user_input.tool_call_result
                result_data = {"tool_call_id": tcr.tool_call_id}

                # Extract the result based on the type
                result_field = tcr.WhichOneof("result")
                if result_field == "run_shell_command":
                    rsc = tcr.run_shell_command
                    if rsc.HasField("command_finished"):
                        result_data["content"] = (
                            f"Command: {rsc.command}\n"
                            f"Exit code: {rsc.command_finished.exit_code}\n"
                            f"Output: {rsc.command_finished.output}"
                        )
                    elif rsc.HasField("permission_denied"):
                        result_data["content"] = f"Command '{rsc.command}' was rejected by user."
                    elif rsc.output:
                        result_data["content"] = f"Command: {rsc.command}\nOutput: {rsc.output}"
                    else:
                        result_data["content"] = f"Command: {rsc.command}\nExit code: {rsc.exit_code}"
                elif result_field == "read_files":
                    # Aggregate file contents
                    parts = []
                    for f in tcr.read_files.files:
                        parts.append(f"File: {f.path}\n{f.content}")
                    result_data["content"] = "\n---\n".join(parts) if parts else "Files read."
                else:
                    result_data["content"] = "Tool call completed."

                results.append(result_data)

    # Legacy: direct tool_call_result
    elif field_name == "tool_call_result":
        tcr = inp.tool_call_result
        result_data = {
            "tool_call_id": tcr.tool_call_id,
            "content": "Tool call completed.",
        }
        results.append(result_data)

    return results


def _extract_conversation_history(req: request_pb2.Request) -> list[dict]:
    """Rebuild conversation history from task_context.tasks into OpenAI messages format.

    DeepSeek requires that every assistant message with tool_calls is followed by
    a tool message for each tool_call_id. We collect tool_call and tool_call_result
    pairs and only synthesize missing results if truly absent.
    """
    messages = []
    if not req.task_context or not req.task_context.tasks:
        return messages

    for task in req.task_context.tasks:
        # First pass: collect all messages in order, tracking tool calls and results
        raw_msgs = []
        for msg in task.messages:
            field = msg.WhichOneof("message")
            if field == "user_query":
                raw_msgs.append(("user_query", msg))
            elif field == "agent_output":
                raw_msgs.append(("agent_output", msg))
            elif field == "tool_call":
                raw_msgs.append(("tool_call", msg))
            elif field == "tool_call_result":
                raw_msgs.append(("tool_call_result", msg))

        # Second pass: build OpenAI messages, grouping tool_calls and ensuring results follow
        i = 0
        while i < len(raw_msgs):
            msg_type, msg = raw_msgs[i]

            if msg_type == "user_query":
                # Include referenced_attachments from history messages
                uq = msg.user_query
                query_text = uq.query
                if uq.referenced_attachments:
                    att_parts = []
                    for key, attachment in uq.referenced_attachments.items():
                        value_field = attachment.WhichOneof("value")
                        if value_field == "plain_text":
                            att_parts.append(f"[Attached: {key}]\n{attachment.plain_text}")
                        elif value_field == "executed_shell_command":
                            cmd = attachment.executed_shell_command
                            att_parts.append(
                                f"[Attached command: {key}]\n$ {cmd.command}\n"
                                f"Output:\n{cmd.output}"
                            )
                        elif value_field == "file_path_reference":
                            att_parts.append(f"[Attached file: {attachment.file_path_reference.file_path}]")
                        elif value_field == "document_content":
                            att_parts.append(f"[Attached document: {key}]\n{attachment.document_content}")
                        elif value_field == "diff_set":
                            hunks = [f"  {h.file_path}:\n{h.diff_content}" for h in attachment.diff_set.hunks]
                            att_parts.append(f"[Attached diff: {key}]\n" + "\n".join(hunks))
                        elif value_field:
                            att_parts.append(f"[Attached: {key}] (type: {value_field})")
                    if att_parts:
                        query_text = (
                            "<attached_context>\n"
                            + "\n\n".join(att_parts)
                            + "\n</attached_context>\n\n"
                            + query_text
                        )
                messages.append({"role": "user", "content": query_text})
                i += 1

            elif msg_type == "agent_output":
                if msg.agent_output.text:
                    messages.append({"role": "assistant", "content": msg.agent_output.text})
                i += 1

            elif msg_type == "tool_call":
                # Collect consecutive tool_calls into one assistant message
                tool_calls = []
                tool_call_ids = []
                while i < len(raw_msgs) and raw_msgs[i][0] == "tool_call":
                    tc = raw_msgs[i][1].tool_call
                    tool_field = tc.WhichOneof("tool")
                    tool_call_entry = None

                    if tool_field == "run_shell_command":
                        args = json.dumps({"command": tc.run_shell_command.command, "is_read_only": tc.run_shell_command.is_read_only})
                        tool_call_entry = {
                            "id": tc.tool_call_id,
                            "type": "function",
                            "function": {"name": "run_shell_command", "arguments": args},
                        }
                    elif tool_field == "read_files":
                        files = [{"name": f.name} for f in tc.read_files.files]
                        args = json.dumps({"files": files})
                        tool_call_entry = {
                            "id": tc.tool_call_id,
                            "type": "function",
                            "function": {"name": "read_files", "arguments": args},
                        }

                    if tool_call_entry:
                        tool_calls.append(tool_call_entry)
                        tool_call_ids.append(tc.tool_call_id)
                    i += 1

                # Emit assistant message with tool_calls
                if tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    })

                # Now collect the corresponding tool_call_results
                answered_ids = set()
                while i < len(raw_msgs) and raw_msgs[i][0] == "tool_call_result":
                    tcr = raw_msgs[i][1].tool_call_result
                    content = "Tool executed."
                    result_field = tcr.WhichOneof("result") if hasattr(tcr, 'WhichOneof') else None
                    if result_field == "run_shell_command":
                        rsc = tcr.run_shell_command
                        if hasattr(rsc, 'output') and rsc.output:
                            content = rsc.output
                        elif hasattr(rsc, 'command_finished') and rsc.HasField("command_finished"):
                            content = f"Exit code: {rsc.command_finished.exit_code}\nOutput: {rsc.command_finished.output}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tcr.tool_call_id,
                        "content": content,
                    })
                    answered_ids.add(tcr.tool_call_id)
                    i += 1

                # Do NOT synthesize missing tool results here.
                # If results are missing from history, they will come as the current request's input.

            elif msg_type == "tool_call_result":
                # Orphaned tool_call_result (shouldn't happen, but handle gracefully)
                # Skip it to avoid duplicate tool messages
                i += 1

            else:
                i += 1

    return messages


@router.post("/ai/multi-agent")
async def multi_agent_handler(request: Request):
    """Handle the core AI conversation endpoint."""
    body = await request.body()

    # 1. Parse protobuf request.
    req = request_pb2.Request()
    req.ParseFromString(body)

    # Extract API key from request settings (sent by Warp client).
    # The DeepSeek key is passed via the 'openai' field (OpenAI-compatible).
    api_key = req.settings.api_keys.openai if req.settings.api_keys.openai else None

    # Debug: show what keys the client sent
    ak = req.settings.api_keys
    logger.info("API keys received - openai: %s, anthropic: %s, google: %s, open_router: %s",
                '***' + ak.openai[-4:] if ak.openai else 'empty',
                '***' + ak.anthropic[-4:] if ak.anthropic else 'empty',
                '***' + ak.google[-4:] if ak.google else 'empty',
                '***' + ak.open_router[-4:] if ak.open_router else 'empty')

    user_query = _extract_user_query(req)
    tool_call_results = _extract_tool_call_results(req)

    # Extract attached context (from "Attach as agent context")
    referenced_attachments_text = _extract_referenced_attachments(req)
    input_context_text = _extract_input_context(req)

    if user_query:
        logger.info("User query: %s...", user_query[:100])
    if referenced_attachments_text:
        logger.info("Referenced attachments found (%d chars)", len(referenced_attachments_text))
    if input_context_text:
        logger.info("Input context found (%d chars)", len(input_context_text))
    if tool_call_results:
        logger.info("Tool call results: %d result(s)", len(tool_call_results))

    # 2. Build conversation history.
    history = _extract_conversation_history(req)

    # Add current input - prepend context to user query if available
    if user_query:
        # Build user message with context included
        context_prefix = ""
        if referenced_attachments_text or input_context_text:
            context_sections = []
            if input_context_text:
                context_sections.append(input_context_text)
            if referenced_attachments_text:
                context_sections.append(referenced_attachments_text)
            context_prefix = (
                "<attached_context>\n"
                + "\n\n".join(context_sections)
                + "\n</attached_context>\n\n"
            )
        history.append({"role": "user", "content": context_prefix + user_query})
    elif tool_call_results:
        for tcr in tool_call_results:
            history.append({
                "role": "tool",
                "tool_call_id": tcr["tool_call_id"],
                "content": tcr["content"],
            })

    # 3. Generate IDs.
    conversation_id = req.metadata.conversation_id if req.metadata.conversation_id else str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    task_id = "task-1"
    user_msg_id = str(uuid.uuid4())
    agent_msg_id = str(uuid.uuid4())
    is_new_conversation = not req.metadata.conversation_id

    # Inject reasoning_content into ALL assistant messages.
    # DeepSeek thinking mode requires reasoning_content to be passed back.
    # Use cached values where available, otherwise use empty string.
    cached_reasonings = get_all_cached_reasoning(conversation_id)
    reasoning_idx = 0
    for msg in history:
        if msg.get("role") == "assistant":
            if "reasoning_content" not in msg:
                if reasoning_idx < len(cached_reasonings):
                    msg["reasoning_content"] = cached_reasonings[reasoning_idx]
                else:
                    msg["reasoning_content"] = ""
            reasoning_idx += 1

    # Final cleanup: ensure every assistant message with tool_calls is followed by
    # tool messages. If not, strip tool_calls and convert to a plain assistant message.
    cleaned_history = []
    for i, msg in enumerate(history):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Check if next message(s) are tool messages
            next_idx = i + 1
            has_tool_response = (
                next_idx < len(history) and history[next_idx].get("role") == "tool"
            )
            if not has_tool_response:
                # Strip tool_calls — the result was already incorporated in a later response
                cleaned_msg = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "reasoning_content": msg.get("reasoning_content", ""),
                }
                cleaned_history.append(cleaned_msg)
            else:
                cleaned_history.append(msg)
        else:
            cleaned_history.append(msg)
    history = cleaned_history

    # Debug: print history roles
    logger.info("History (%d messages): %s", len(history), [m['role'] for m in history])

    async def event_generator():
        # Event 1: StreamInit
        init_event = response_pb2.ResponseEvent()
        init_event.init.conversation_id = conversation_id
        init_event.init.request_id = request_id
        yield _make_event(init_event)

        # Event 2: CreateTask (only for new conversations)
        if is_new_conversation:
            create_task_event = response_pb2.ResponseEvent()
            action = create_task_event.client_actions.actions.add()
            action.create_task.task.id = task_id
            action.create_task.task.description = user_query[:100] if user_query else "Task"
            yield _make_event(create_task_event)

        # Event 3: Add user query message (only if there's a user query)
        if user_query:
            user_msg_event = response_pb2.ResponseEvent()
            action = user_msg_event.client_actions.actions.add()
            action.add_messages_to_task.task_id = task_id
            user_msg = action.add_messages_to_task.messages.add()
            user_msg.id = user_msg_id
            user_msg.task_id = task_id
            user_msg.request_id = request_id
            user_msg.user_query.query = user_query
            yield _make_event(user_msg_event)

        # Event 4: Add empty agent output message (will be appended to)
        agent_msg_event = response_pb2.ResponseEvent()
        action = agent_msg_event.client_actions.actions.add()
        action.add_messages_to_task.task_id = task_id
        agent_msg = action.add_messages_to_task.messages.add()
        agent_msg.id = agent_msg_id
        agent_msg.task_id = task_id
        agent_msg.request_id = request_id
        agent_msg.agent_output.text = ""
        yield _make_event(agent_msg_event)

        # Events 5..N: Stream DeepSeek response
        try:
            collected_text = ""
            current_tool_call = None

            async for event_type, data in stream_chat(history, conversation_id=conversation_id, api_key=api_key):
                if event_type == "content":
                    # Stream text chunk
                    collected_text += data
                    append_event = response_pb2.ResponseEvent()
                    action = append_event.client_actions.actions.add()
                    action.append_to_message_content.task_id = task_id
                    action.append_to_message_content.message.id = agent_msg_id
                    action.append_to_message_content.message.agent_output.text = data
                    action.append_to_message_content.mask.CopyFrom(
                        field_mask_pb2.FieldMask(paths=["agent_output.text"])
                    )
                    yield _make_event(append_event)

                elif event_type == "reasoning":
                    pass  # Reasoning is cached in deepseek_client, not shown to user

                elif event_type == "tool_call_start":
                    current_tool_call = data
                    logger.info("Tool call: %s", data['name'])

                elif event_type == "tool_call_args":
                    pass  # Arguments accumulated in deepseek_client

                elif event_type == "tool_call_end":
                    # Emit a ToolCall message to Warp
                    tc_data = data
                    tool_name = tc_data["name"]
                    tool_call_id = tc_data["id"] or str(uuid.uuid4())

                    try:
                        args = json.loads(tc_data["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    tc_msg_id = str(uuid.uuid4())

                    # Add ToolCall message to task
                    tc_event = response_pb2.ResponseEvent()
                    action = tc_event.client_actions.actions.add()
                    action.add_messages_to_task.task_id = task_id
                    tc_msg = action.add_messages_to_task.messages.add()
                    tc_msg.id = tc_msg_id
                    tc_msg.task_id = task_id
                    tc_msg.request_id = request_id
                    tc_msg.tool_call.tool_call_id = tool_call_id

                    if tool_name == "run_shell_command":
                        tc_msg.tool_call.run_shell_command.command = args.get("command", "")
                        tc_msg.tool_call.run_shell_command.is_read_only = args.get("is_read_only", False)
                    elif tool_name == "read_files":
                        files = args.get("files", [])
                        for f in files:
                            file_entry = tc_msg.tool_call.read_files.files.add()
                            file_entry.name = f.get("path", "")
                    elif tool_name == "apply_file_diffs":
                        tc_msg.tool_call.apply_file_diffs.summary = args.get("summary", "")
                        # Note: diffs structure may need additional mapping

                    yield _make_event(tc_event)
                    current_tool_call = None

                elif event_type == "done":
                    break

        except Exception as e:
            logger.exception("DeepSeek error: %s", e)
            error_event = response_pb2.ResponseEvent()
            error_event.finished.internal_error.message = str(e)
            yield _make_event(error_event)
            return

        # Final event: StreamFinished (Done)
        finished_event = response_pb2.ResponseEvent()
        finished_event.finished.done.CopyFrom(
            response_pb2.ResponseEvent.StreamFinished.Done()
        )
        yield _make_event(finished_event)

    return EventSourceResponse(event_generator())
