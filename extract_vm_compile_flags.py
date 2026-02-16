#!/usr/bin/env python3
"""Extract and deduplicate vm_compile_c_and_upload gcc flags from Codex JSONL logs.

Usage:
  python3 extract_vm_compile_flags.py path/to/rollout.jsonl
  python3 extract_vm_compile_flags.py /path/to/root_dir_with_jsonls
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List

from parsecodexlog import extract_vm_compile_c_and_upload_flags, parse_codex_log


def _iter_jsonl_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*.jsonl")):
        yield p


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplicate vm_compile_c_and_upload flags from Codex JSONL logs.")
    p.add_argument("path", type=Path, help="A JSONL file or a directory to scan recursively")
    p.add_argument(
        "--default",
        default="-static",
        help='Default flags to assume when `flags` is missing (comma-separated). Default: "-static"',
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    default_flags = [x for x in (args.default or "").split(",") if x]

    all_unique: set[tuple[str, ...]] = set()
    total_calls = 0
    explicit_flags_calls = 0

    per_file: List[dict] = []

    for f in _iter_jsonl_files(args.path):
        messages = parse_codex_log(f)
        # Count calls in this file, and whether flags were explicit.
        for m in messages:
            if m.type != "function_call" or m.name != "mcp__kernelmcp__vm_compile_c_and_upload":
                continue
            total_calls += 1
            if isinstance(m.arguments, dict) and isinstance(m.arguments.get("flags"), list):
                explicit_flags_calls += 1

        flags_sets = extract_vm_compile_c_and_upload_flags(messages, default_flags=default_flags)
        for flags in flags_sets:
            all_unique.add(tuple(flags))

        per_file.append(
            {
                "file": str(f),
                "unique_flag_sets": flags_sets,
                "unique_count": len(flags_sets),
            }
        )

    unique_sorted = [list(t) for t in sorted(all_unique, key=lambda xs: (len(xs), list(xs)))]

    # Intermediate values for sanity-checking (especially when results are unexpectedly 0).
    result = {
        "scanned": str(args.path),
        "files": len(per_file),
        "total_vm_compile_calls": total_calls,
        "calls_with_explicit_flags_field": explicit_flags_calls,
        "default_flags_assumed_when_missing": default_flags,
        "unique_flag_sets_overall": unique_sorted,
        "unique_flag_sets_overall_count": len(unique_sorted),
        "per_file": per_file,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

