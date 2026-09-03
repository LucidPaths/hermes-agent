#!/usr/bin/env python3
"""Evaluate GitHub Actions ``needs`` results for the required-check gate."""

import json
import os
import sys
from typing import Any


PASSING_RESULTS = {"success", "skipped"}


def _error(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def _job_label(name: str) -> str:
    """Render untrusted job names without control characters in log output."""
    return json.dumps(name, ensure_ascii=True)


def _load_needs() -> dict[str, Any] | None:
    try:
        needs = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        _error(f"needs is not valid JSON: {exc.msg}")
        return None
    if not isinstance(needs, dict) or not needs:
        _error("needs must be a non-empty object")
        return None
    for name, info in needs.items():
        if not isinstance(name, str) or not name:
            _error("needs job keys must be non-empty strings")
            return None
        if not isinstance(info, dict):
            _error(f"needs job {_job_label(name)} must be an object")
            return None
        if "result" not in info:
            _error(f"needs job {_job_label(name)} is missing result")
            return None
        result = info["result"]
        if not isinstance(result, str):
            _error(
                f"needs job {_job_label(name)} has malformed result; "
                "expected a string"
            )
            return None
    return needs


def main() -> int:
    needs = _load_needs()
    if needs is None:
        return 1

    # Compact JSON is deliberately one line for both GITHUB_OUTPUT and callers.
    compact = {name: info["result"] for name, info in needs.items()}
    needs_json = json.dumps(compact, ensure_ascii=True, separators=(",", ":"))
    print(f"needs-json={needs_json}")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"needs-json={needs_json}\n")

    failed = [name for name, info in needs.items() if info["result"] not in PASSING_RESULTS]
    for name, info in sorted(needs.items()):
        icon = "✅" if info["result"] in PASSING_RESULTS else "❌"
        print(f"{icon} {_job_label(name)}: {_job_label(info['result'])}")
    if failed:
        _error(
            f"needs job(s) did not pass: "
            + ", ".join(_job_label(name) for name in sorted(failed))
        )
        return 1
    print("All checks passed (or were skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
