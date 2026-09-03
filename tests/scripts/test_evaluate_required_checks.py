"""Executable tests for the required-check aggregate policy."""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "evaluate_required_checks.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yaml"


def run_evaluator(payload):
    output_file = Path(os.environ.get("TMPDIR", "/tmp")) / "evaluate-required-checks-test-output"
    output_file.unlink(missing_ok=True)
    env = {**os.environ, "GITHUB_OUTPUT": str(output_file)}
    process = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    output = output_file.read_text() if output_file.exists() else ""
    output_file.unlink(missing_ok=True)
    return process, output


def test_success_and_skipped_pass():
    process, output = run_evaluator(
        {"tests": {"result": "success"}, "docs": {"result": "skipped"}}
    )

    assert process.returncode == 0
    assert "All checks passed (or were skipped)" in process.stdout
    assert 'needs-json={"tests":"success","docs":"skipped"}' in output
    assert output.count("\n") == 1


def test_empty_needs_fails_closed_without_traceback():
    process, output = run_evaluator({})

    assert process.returncode == 1
    assert "ERROR: needs must be a non-empty object" in process.stderr
    assert "Traceback" not in process.stderr
    assert output == ""


def test_malformed_top_level_fails_closed_without_traceback():
    process, _ = run_evaluator("[]")

    assert process.returncode == 1
    assert "ERROR: needs must be a non-empty object" in process.stderr
    assert "Traceback" not in process.stderr


def test_missing_null_unknown_and_malformed_results_fail_closed():
    payloads = [
        {"job": {}},
        {"job": {"result": None}},
        {"job": {"result": "unknown"}},
        {"job": {"result": 1}},
        {"job": "success"},
    ]
    for payload in payloads:
        process, _ = run_evaluator(payload)
        assert process.returncode == 1, payload
        assert process.stderr.startswith("ERROR: needs job")
        assert "Traceback" not in process.stderr


def test_workflow_command_injection_is_not_emitted_for_invalid_input():
    payload = {
        "job\n::error file=x%0A": {"result": "failure\n::error::injected%"},
    }
    process, output = run_evaluator(payload)

    assert process.returncode == 1
    assert not any(line.startswith("::error::") for line in process.stdout.splitlines())
    assert not any(line.startswith("::error::") for line in process.stderr.splitlines())
    assert output.endswith("\n") and output.count("\n") == 1
    assert json.loads(output.strip().removeprefix("needs-json=")) == {
        "job\n::error file=x%0A": "failure\n::error::injected%"
    }


def test_output_json_escapes_newlines_and_percent_signs():
    process, output = run_evaluator({"job\n%": {"result": "success"}})

    assert process.returncode == 0
    assert "::error::" not in process.stdout
    assert output.endswith("\n") and output.count("\n") == 1
    assert json.loads(output.strip().removeprefix("needs-json=")) == {
        "job\n%": "success"
    }


def test_workflow_security_and_gate_configuration():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    jobs = workflow["jobs"]
    aggregate = jobs["all-checks-pass"]

    checkout = next(step for step in aggregate["steps"] if "actions/checkout@" in step["uses"])
    assert checkout["uses"] == (
        "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
    )
    assert checkout["with"]["persist-credentials"] is False
    assert aggregate["permissions"] == {"contents": "read"}
    assert "infographic-check" in aggregate["needs"]
    assert aggregate["needs"].index("infographic-check") < aggregate["needs"].index(
        "profile-artifact-check"
    )
