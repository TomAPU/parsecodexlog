#!/usr/bin/env python3
"""
Lightweight parser for Codex and Claude CLI JSONL logs.

Usage:
    python3 parsecodexlog.py path/to/log.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


CODEX_RECORD_TYPES = {"turn_context", "session_meta", "event_msg", "response_item"}
CLAUDE_RECORD_TYPES = {
    "assistant",
    "last-prompt",
    "progress",
    "queue-operation",
    "system",
    "user",
}

IGNORED_RECORD_TYPES: set[str] = set()
IGNORED_EVENT_SUBTYPES: set[str] = set()
IGNORED_RESPONSE_SUBTYPES = {"reasoning"}
IGNORED_CLAUDE_RECORD_TYPES = {"last-prompt", "queue-operation"}
IGNORED_CLAUDE_CONTENT_TYPES = {"thinking"}


@dataclass
class ParsedMessage:
    timestamp: str
    type: str
    role: Optional[str]
    content: Optional[str]
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    source_line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "type": self.type,
            "role": self.role,
            "content": self.content,
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "output": self.output,
            "metadata": self.metadata,
            "source_line": self.source_line,
        }


def parse_codex_log(path: Path) -> List[ParsedMessage]:
    """
    Parse a Codex or Claude JSONL log into a list of ParsedMessage objects.

    - Preserves turn context blocks, token usage metrics, and ignores only encrypted reasoning notes.
    - Auto-detects Claude-style top-level assistant/user/tool-result records and
      normalizes them to the same ParsedMessage ABI used for Codex logs.
    - Correlates function calls with their outputs via call_id.
    """
    messages: List[ParsedMessage] = []
    pending_calls: Dict[str, ParsedMessage] = {}
    parser_state: Dict[str, Any] = {"format": None, "claude_session_meta_emitted": False}

    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            _consume_record(record, lineno, messages, pending_calls, parser_state)

    return messages


def extract_vm_compile_c_and_upload_flags(
    messages: Iterable[ParsedMessage],
    *,
    default_flags: Optional[List[str]] = None,
) -> List[List[str]]:
    """Return deduplicated gcc flag lists used by vm_compile_c_and_upload calls.

    Notes:
    - Many logs omit `flags`, which means the tool uses its default (usually ["-static"]).
    - We normalize missing/empty flags to `default_flags` when provided.
    """
    if default_flags is None:
        default_flags = ["-static"]

    seen: set[tuple[str, ...]] = set()
    out: List[List[str]] = []

    for msg in messages:
        if msg.type != "function_call":
            continue
        if not msg.name:
            continue
        if msg.name != "mcp__kernelmcp__vm_compile_c_and_upload":
            continue

        args = msg.arguments
        flags: Optional[List[str]] = None
        if isinstance(args, dict):
            raw_flags = args.get("flags")
            if isinstance(raw_flags, list) and all(isinstance(x, str) for x in raw_flags):
                flags = [x for x in raw_flags if x]

        if not flags:
            flags = list(default_flags)

        key = tuple(flags)
        if key in seen:
            continue
        seen.add(key)
        out.append(flags)

    # stable ordering for reproducibility
    out.sort(key=lambda xs: (len(xs), xs))
    return out


def _consume_record(
    record: Dict[str, Any],
    lineno: int,
    messages: List[ParsedMessage],
    pending_calls: Dict[str, ParsedMessage],
    parser_state: Dict[str, Any],
) -> None:
    rectype = record.get("type")
    if rectype in IGNORED_RECORD_TYPES:
        return

    if rectype in CODEX_RECORD_TYPES:
        parser_state["format"] = "codex"
    elif rectype in CLAUDE_RECORD_TYPES:
        parser_state["format"] = "claude"

    timestamp = str(record.get("timestamp", "unknown"))
    payload = record.get("payload") or {}

    if rectype in CLAUDE_RECORD_TYPES:
        if (
            not parser_state.get("claude_session_meta_emitted")
            and rectype not in IGNORED_CLAUDE_RECORD_TYPES
        ):
            _emit_claude_session_meta(record, timestamp, lineno, messages)
            parser_state["claude_session_meta_emitted"] = True
        _consume_claude_record(record, lineno, messages, pending_calls)
        return

    if rectype == "turn_context":
        messages.append(
            ParsedMessage(
                timestamp=timestamp,
                type="turn_context",
                role="system",
                content=_format_json(payload),
                metadata=payload,
                source_line=lineno,
            )
        )
        return

    if rectype == "session_meta":
        messages.append(
            ParsedMessage(
                timestamp=timestamp,
                type="session_meta",
                role="system",
                content=_format_json(payload),
                metadata=payload,
                source_line=lineno,
            )
        )
        return

    if rectype == "event_msg":
        _handle_event_msg(payload, timestamp, lineno, messages)
        return

    if rectype == "response_item":
        _handle_response_item(payload, timestamp, lineno, messages, pending_calls)
        return

    # Unknown record type, keep a minimal placeholder.
    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type=rectype or "unknown",
            role=None,
            content=_format_json(payload),
            metadata=payload,
            source_line=lineno,
        )
    )


def _consume_claude_record(
    record: Dict[str, Any],
    lineno: int,
    messages: List[ParsedMessage],
    pending_calls: Dict[str, ParsedMessage],
) -> None:
    rectype = record.get("type")
    timestamp = str(record.get("timestamp", "unknown"))

    if rectype in IGNORED_CLAUDE_RECORD_TYPES:
        return

    if rectype == "assistant":
        _handle_claude_assistant_record(record, timestamp, lineno, messages, pending_calls)
        return

    if rectype == "user":
        _handle_claude_user_record(record, timestamp, lineno, messages, pending_calls)
        return

    if rectype == "system":
        _handle_claude_system_record(record, timestamp, lineno, messages)
        return

    if rectype == "progress":
        _handle_claude_progress_record(record, timestamp, lineno, messages)
        return

    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type=rectype or "unknown",
            role=None,
            content=_format_json(record),
            metadata=record,
            source_line=lineno,
        )
    )


def _emit_claude_session_meta(
    record: Dict[str, Any],
    timestamp: str,
    lineno: int,
    messages: List[ParsedMessage],
) -> None:
    metadata = {
        "id": record.get("sessionId"),
        "timestamp": timestamp,
        "cwd": record.get("cwd"),
        "cli_version": record.get("version"),
        "git_branch": record.get("gitBranch"),
        "user_type": record.get("userType"),
        "permission_mode": record.get("permissionMode"),
        "source": "claude",
    }
    message_payload = {k: v for k, v in metadata.items() if v is not None}
    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type="session_meta",
            role="system",
            content=_format_json(message_payload),
            metadata=message_payload,
            source_line=lineno,
        )
    )


def _handle_claude_assistant_record(
    record: Dict[str, Any],
    timestamp: str,
    lineno: int,
    messages: List[ParsedMessage],
    pending_calls: Dict[str, ParsedMessage],
) -> None:
    payload = record.get("message") or {}
    role = payload.get("role") or "assistant"
    text_chunks: List[str] = []

    for item in _iter_claude_content_items(payload.get("content")):
        kind = item.get("type")
        if kind in IGNORED_CLAUDE_CONTENT_TYPES:
            continue
        if kind == "text":
            text = _coerce_to_text(item.get("text"))
            if text:
                text_chunks.append(text)
            continue

        _flush_message_chunks(text_chunks, timestamp, role, lineno, messages)

        if kind == "tool_use":
            call_id = _coerce_to_text(item.get("id"))
            msg = ParsedMessage(
                timestamp=timestamp,
                type="function_call",
                role="assistant",
                name=_coerce_to_text(item.get("name")),
                call_id=call_id,
                arguments=_maybe_parse_json(item.get("input")),
                content=None,
                source_line=lineno,
            )
            messages.append(msg)
            if call_id:
                pending_calls[call_id] = msg
            continue

        messages.append(
            ParsedMessage(
                timestamp=timestamp,
                type=f"response:{kind or 'unknown'}",
                role=role,
                content=_format_json(item),
                metadata=item,
                source_line=lineno,
            )
        )

    _flush_message_chunks(text_chunks, timestamp, role, lineno, messages)


def _handle_claude_user_record(
    record: Dict[str, Any],
    timestamp: str,
    lineno: int,
    messages: List[ParsedMessage],
    pending_calls: Dict[str, ParsedMessage],
) -> None:
    payload = record.get("message") or {}
    role = payload.get("role") or "user"
    content = payload.get("content")

    if isinstance(content, str):
        messages.append(
            ParsedMessage(
                timestamp=timestamp,
                type="message",
                role=role,
                content=content,
                source_line=lineno,
            )
        )
        return

    text_chunks: List[str] = []
    for item in _iter_claude_content_items(content):
        kind = item.get("type")
        if kind == "tool_result":
            _flush_message_chunks(text_chunks, timestamp, role, lineno, messages)
            call_id = _coerce_to_text(item.get("tool_use_id"))
            output_value = _extract_claude_tool_result_output(record, item)
            _attach_function_output(
                timestamp=timestamp,
                lineno=lineno,
                call_id=call_id,
                output_value=output_value,
                messages=messages,
                pending_calls=pending_calls,
            )
            continue

        text = _coerce_to_text(item.get("text"))
        if text:
            text_chunks.append(text)
            continue

        text_chunks.append(_format_json(item))

    _flush_message_chunks(text_chunks, timestamp, role, lineno, messages)


def _handle_claude_system_record(
    record: Dict[str, Any],
    timestamp: str,
    lineno: int,
    messages: List[ParsedMessage],
) -> None:
    subtype = _coerce_to_text(record.get("subtype")) or "system"
    content = _coerce_to_text(record.get("content"))
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"content", "type", "timestamp"}
    }
    if content is None:
        content = _format_json(metadata)
    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type=f"event:{subtype}",
            role="system",
            content=content,
            metadata=metadata or None,
            source_line=lineno,
        )
    )


def _handle_claude_progress_record(
    record: Dict[str, Any],
    timestamp: str,
    lineno: int,
    messages: List[ParsedMessage],
) -> None:
    data = record.get("data")
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"type", "timestamp"}
    }
    subtype = None
    if isinstance(data, dict):
        subtype = _coerce_to_text(data.get("type"))
    content = _format_json(data if data is not None else metadata)
    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type=f"event:{subtype or 'progress'}",
            role="system",
            content=content,
            metadata=metadata or None,
            source_line=lineno,
        )
    )


def _handle_event_msg(
    payload: Dict[str, Any],
    timestamp: str,
    lineno: int,
    messages: List[ParsedMessage],
) -> None:
    subtype = payload.get("type")
    if subtype in IGNORED_EVENT_SUBTYPES:
        return

    role = None
    if subtype == "user_message":
        role = "user"
    elif subtype == "agent_message":
        role = "assistant"

    metadata: Optional[Dict[str, Any]] = None
    if subtype == "token_count":
        metadata = payload
        content = _summarize_token_count(payload)
    else:
        content = _coerce_to_text(payload.get("message"))
        if content is None and payload:
            metadata = payload
            content = _format_json(payload)

    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type=f"event:{subtype or 'unknown'}",
            role=role,
            content=content,
            metadata=metadata,
            source_line=lineno,
        )
    )


def _handle_response_item(
    payload: Dict[str, Any],
    timestamp: str,
    lineno: int,
    messages: List[ParsedMessage],
    pending_calls: Dict[str, ParsedMessage],
) -> None:
    subtype = payload.get("type")
    if subtype in IGNORED_RESPONSE_SUBTYPES:
        return

    if subtype == "message":
        content = _extract_text_chunks(payload.get("content") or [])
        messages.append(
            ParsedMessage(
                timestamp=timestamp,
                type="message",
                role=payload.get("role"),
                content="\n\n".join(content) if content else None,
                source_line=lineno,
            )
        )
        return

    if subtype == "function_call":
        call_id = payload.get("call_id")
        msg = ParsedMessage(
            timestamp=timestamp,
            type="function_call",
            role="assistant",
            name=payload.get("name"),
            call_id=call_id,
            arguments=_maybe_parse_json(payload.get("arguments")),
            content=None,
            source_line=lineno,
        )
        messages.append(msg)
        if call_id:
            pending_calls[call_id] = msg
        return

    if subtype == "function_call_output":
        call_id = payload.get("call_id")
        output_value = _maybe_parse_json(payload.get("output"))
        _attach_function_output(
            timestamp=timestamp,
            lineno=lineno,
            call_id=call_id,
            output_value=output_value,
            messages=messages,
            pending_calls=pending_calls,
        )
        return

    # Keep any other response items in a generic form.
    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type=f"response:{subtype or 'unknown'}",
            role=payload.get("role"),
            content=_format_json(payload),
            metadata=payload,
            source_line=lineno,
        )
    )


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)) or value is None:
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _attach_function_output(
    *,
    timestamp: str,
    lineno: int,
    call_id: Optional[str],
    output_value: Any,
    messages: List[ParsedMessage],
    pending_calls: Dict[str, ParsedMessage],
) -> None:
    existing = pending_calls.get(call_id or "")
    if existing:
        existing.output = output_value
        return

    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type="function_call",
            role="assistant",
            call_id=call_id,
            output=output_value,
            content=None,
            source_line=lineno,
        )
    )


def _extract_claude_tool_result_output(record: Dict[str, Any], item: Dict[str, Any]) -> Any:
    mcp_meta = record.get("mcpMeta")
    if isinstance(mcp_meta, dict) and "structuredContent" in mcp_meta:
        return mcp_meta.get("structuredContent")

    if "toolUseResult" in record:
        return _maybe_parse_json(record.get("toolUseResult"))

    return _maybe_parse_json(item.get("content"))


def _iter_claude_content_items(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
            else:
                yield {"type": "text", "text": _coerce_to_text(item)}
        return

    if value is None:
        return

    yield {"type": "text", "text": _coerce_to_text(value)}


def _flush_message_chunks(
    text_chunks: List[str],
    timestamp: str,
    role: Optional[str],
    lineno: int,
    messages: List[ParsedMessage],
) -> None:
    if not text_chunks:
        return
    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type="message",
            role=role,
            content="\n\n".join(text_chunks),
            source_line=lineno,
        )
    )
    text_chunks.clear()


def _extract_text_chunks(content_items: Iterable[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for item in content_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"input_text", "output_text"}:
            text = item.get("text")
            if text:
                texts.append(str(text))
    return texts


def _coerce_to_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _format_json(value: Any) -> str:
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, indent=2)


def _summarize_token_count(payload: Dict[str, Any]) -> str:
    info = payload.get("info") or {}
    total_usage = info.get("total_token_usage") or {}
    last_usage = info.get("last_token_usage") or {}

    segments: List[str] = []
    if total_usage:
        segments.append(f"total({_format_usage(total_usage)})")
    if last_usage:
        segments.append(f"last({_format_usage(last_usage)})")
    ctx_window = info.get("model_context_window")
    if ctx_window is not None:
        segments.append(f"context_window={ctx_window}")

    if segments:
        return "; ".join(segments)
    return _format_json(payload)


def _format_usage(usage: Dict[str, Any]) -> str:
    labels = {
        "input_tokens": "in",
        "cached_input_tokens": "cached",
        "output_tokens": "out",
        "reasoning_output_tokens": "reasoning",
        "total_tokens": "total",
    }
    parts: List[str] = []
    for key, label in labels.items():
        value = usage.get(key)
        if value is not None:
            parts.append(f"{label}={value}")
    return ", ".join(parts) if parts else "n/a"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a Codex or Claude JSONL log into a list of messages."
    )
    parser.add_argument(
        "logfile",
        type=Path,
        help="Path to the JSONL conversation log.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    messages = parse_codex_log(args.logfile)
    print(json.dumps([m.to_dict() for m in messages], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
