#!/usr/bin/env python3
"""
Lightweight parser for Codex CLI JSONL logs.

Usage:
    python3 parsecodexlog.py path/to/log.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


IGNORED_RECORD_TYPES = {"turn_context"}
IGNORED_EVENT_SUBTYPES = {"token_count"}
IGNORED_RESPONSE_SUBTYPES = {"reasoning"}


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
            "source_line": self.source_line,
        }


def parse_codex_log(path: Path) -> List[ParsedMessage]:
    """
    Parse a Codex JSONL log into a list of ParsedMessage objects.

    - Ignores turn context blocks, token usage metrics, and encrypted reasoning notes.
    - Correlates function calls with their outputs via call_id.
    """
    messages: List[ParsedMessage] = []
    pending_calls: Dict[str, ParsedMessage] = {}

    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            _consume_record(record, lineno, messages, pending_calls)

    return messages


def _consume_record(
    record: Dict[str, Any],
    lineno: int,
    messages: List[ParsedMessage],
    pending_calls: Dict[str, ParsedMessage],
) -> None:
    rectype = record.get("type")
    if rectype in IGNORED_RECORD_TYPES:
        return

    timestamp = str(record.get("timestamp", "unknown"))
    payload = record.get("payload") or {}

    if rectype == "session_meta":
        messages.append(
            ParsedMessage(
                timestamp=timestamp,
                type="session_meta",
                role="system",
                content=json.dumps(payload, ensure_ascii=False, indent=2),
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
            content=json.dumps(payload, ensure_ascii=False, indent=2),
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

    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type=f"event:{subtype or 'unknown'}",
            role=role,
            content=_coerce_to_text(payload.get("message")),
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
        return

    # Keep any other response items in a generic form.
    messages.append(
        ParsedMessage(
            timestamp=timestamp,
            type=f"response:{subtype or 'unknown'}",
            role=payload.get("role"),
            content=json.dumps(payload, ensure_ascii=False, indent=2),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a Codex JSONL log into a list of messages."
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
