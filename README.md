# parsecodexlog

Lightweight, self-contained parser for Codex CLI JSONL logs. It turns raw log lines into a structured, easy-to-consume list of message objects you can use in scripts or downstream tools—no web UI required.

## What it does
- Reads Codex JSONL logs and emits a list of normalized `ParsedMessage` objects.
- Correlates `function_call` entries with their matching `function_call_output` by `call_id`, merging them into a single message.
- Preserves turn context blocks and token usage (`token_count`) events so you can track sandbox/model shifts and billing data, while still skipping encrypted reasoning notes.
- Safely attempts JSON parsing of function arguments and outputs when they are stringified JSON.

## Usage
CLI:
- `python3 parsecodexlog.py sample.jsonl > parsed.json`
- `python3 analyze_fail.py logs/` — recursively scans JSONL logs, prints files/functions with `tool call error:` outputs, and summarizes failure counts per function.

Library:
```python
from pathlib import Path
from parsecodexlog import parse_codex_log

messages = parse_codex_log(Path("sample.jsonl"))
for msg in messages:
    print(msg.type, msg.role, msg.call_id, msg.content)
```

## High-level design
- **Data model:** `ParsedMessage` dataclass with fields `timestamp`, `type`, `role`, `content`, `call_id`, `name`, `arguments`, `output`, `metadata`, `source_line`.
- **Filtering:** Only ignores response items of subtype `reasoning` (encrypted). Turn context records and `token_count` events are parsed into readable summaries with their raw payload attached to `metadata`.
- **Message parsing:** Collects `message` text from `input_text`/`output_text` chunks; treats other subtypes generically.
- **Function call correlation:** Stores `function_call` messages keyed by `call_id`; when a `function_call_output` with the same `call_id` arrives, the output is attached to the existing message. Orphan outputs become standalone messages.
- **JSON coercion:** Arguments/outputs are parsed as JSON when possible; otherwise kept as raw values/strings.

## Provenance and testing
This project (code and documentation) was generated with the assistance of an AI model and then exercised on sample Codex logs (e.g., `sample.jsonl`) to validate that parsing and function-call correlation work as described.

## Notes and limitations
- Only ignores the commonly noisy meta records listed above; adjust filters in `parsecodexlog.py` if you need more or fewer event types.
- Arguments/outputs that are not valid JSON remain as strings or raw values; down-stream code should handle both.
