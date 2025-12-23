#!/usr/bin/env python3
"""
Scan Codex JSONL logs for failed function tool calls.

For each JSONL file under the provided folder, this script parses the log with
parsecodexlog.parse_codex_log, flags function_call entries whose outputs start
with "tool call error:", prints the file path and failing function names, and
summarizes failure counts per function.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, List

from parsecodexlog import ParsedMessage, parse_codex_log


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Codex CLI JSONL logs for tool call errors."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder that contains JSONL logs (searched recursively).",
    )
    return parser.parse_args()


def _iter_jsonl_files(root: Path) -> Iterator[Path]:
    if root.is_file() and root.suffix.lower() == ".jsonl":
        yield root.resolve()
        return
    for path in sorted(root.rglob("*.jsonl")):
        if path.is_file():
            yield path.resolve()


def _failed_calls(messages: Iterable[ParsedMessage]) -> List[ParsedMessage]:
    failures: List[ParsedMessage] = []
    for message in messages:
        if message.type != "function_call":
            continue
        output = message.output
        if isinstance(output, str) and output.lstrip().startswith("tool call error:"):
            failures.append(message)
    return failures


def main() -> None:
    args = _parse_args()
    root = args.folder

    if not root.exists():
        print(f"Folder does not exist: {root}", file=sys.stderr)
        raise SystemExit(1)

    fail_counts: Counter[str] = Counter()
    files_with_failures = 0

    for jsonl_path in _iter_jsonl_files(root):
        try:
            messages = parse_codex_log(jsonl_path)
        except Exception as exc:  # noqa: BLE001 - surface parsing problems
            print(f"Failed to parse {jsonl_path}: {exc}", file=sys.stderr)
            continue

        failures = _failed_calls(messages)
        if not failures:
            continue

        files_with_failures += 1
        print(jsonl_path)
        for message in failures:
            name = message.name or "<unknown>"
            fail_counts[name] += 1
            print(f"  {name}")

    if files_with_failures == 0:
        print("No failed function calls found.")
        return

    print("\nFailure counts by function:")
    for name, count in sorted(fail_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{count:4}  {name}")


if __name__ == "__main__":
    main()
