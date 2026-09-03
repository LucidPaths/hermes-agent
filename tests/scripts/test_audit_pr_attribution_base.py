"""Regression tests for PR-base attribution selection and fail-closed behavior."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

import audit_pr_attribution as audit  # noqa: E402


WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "contributor-check.yml"


def test_new_emails_uses_reviewed_base_ref(monkeypatch):
    calls = []

    def fake_run(*args, check=True):
        calls.append((args, check))
        if args[:3] == ("git", "merge-base", "origin/evo-production"):
            return "base-sha"
        if args[:2] == ("git", "log"):
            return "contributor@example.com"
        raise AssertionError(args)

    monkeypatch.setattr(audit, "run", fake_run)

    assert audit.new_emails("evo-production") == ["contributor@example.com"]
    assert calls[:2] == [
        (("git", "merge-base", "origin/evo-production", "HEAD"), True),
        (("git", "log", "base-sha..HEAD", "--format=%ae", "--no-merges"), True),
    ]


def test_new_emails_fails_closed_when_git_log_fails(monkeypatch):
    def fake_run(*args, check=True):
        if args[:3] == ("git", "merge-base", "origin/main"):
            return "base-sha"
        if args[:2] == ("git", "log"):
            if check:
                raise RuntimeError("git log failed: fatal: bad object")
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(audit, "run", fake_run)

    with pytest.raises(RuntimeError, match="git log failed"):
        audit.new_emails("main")


def test_default_base_ref_prefers_explicit_environment_over_remote_head(monkeypatch):
    monkeypatch.setenv("GITHUB_BASE_REF", "release")
    monkeypatch.setattr(audit, "run", lambda *args, **kwargs: "origin/main")

    assert audit._default_base_ref() == "release"


def test_explicit_base_ref_precedes_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_BASE_REF", "release")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        if args[:3] == ("git", "merge-base", "origin/hotfix"):
            return "base-sha"
        if args[:2] == ("git", "log"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(audit, "run", fake_run)

    assert audit.new_emails("hotfix") == []
    assert calls[0] == ("git", "merge-base", "origin/hotfix", "HEAD")


def test_default_base_ref_uses_a_valid_origin_head_target(monkeypatch):
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        if args == ("git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"):
            return "origin/release"
        if args == ("git", "rev-parse", "--verify", "refs/remotes/origin/release^{commit}"):
            return "release-sha"
        raise AssertionError(args)

    monkeypatch.setattr(audit, "run", fake_run)

    assert audit._default_base_ref() == "release"
    assert calls == [
        (("git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"), {}),
        (("git", "rev-parse", "--verify", "refs/remotes/origin/release^{commit}"), {}),
    ]


def test_default_base_ref_rejects_dangling_origin_head_target(monkeypatch):
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    def fake_run(*args, **kwargs):
        if args == ("git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"):
            return "origin/release"
        assert args == ("git", "rev-parse", "--verify", "refs/remotes/origin/release^{commit}")
        raise RuntimeError("git rev-parse failed: fatal: bad object")

    monkeypatch.setattr(audit, "run", fake_run)

    with pytest.raises(audit.BaseRefError, match=r"--base-ref.*GITHUB_BASE_REF"):
        audit._default_base_ref()


def test_new_emails_wraps_merge_base_failure_as_actionable_base_error(monkeypatch):
    def fake_run(*args, **kwargs):
        assert args == ("git", "merge-base", "origin/release", "HEAD")
        raise RuntimeError("git merge-base failed: no common ancestor")

    monkeypatch.setattr(audit, "run", fake_run)

    with pytest.raises(audit.BaseRefError, match=r"--base-ref.*GITHUB_BASE_REF") as exc_info:
        audit.new_emails("release")
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "no common ancestor" in str(exc_info.value.__cause__)


def test_precedence_is_explicit_then_environment_then_origin_head(monkeypatch):
    monkeypatch.setenv("GITHUB_BASE_REF", "release")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        if args[:3] == ("git", "merge-base", "origin/hotfix"):
            return "base-sha"
        if args[:2] == ("git", "log"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(audit, "run", fake_run)
    assert audit.new_emails("hotfix") == []
    assert calls[0] == ("git", "merge-base", "origin/hotfix", "HEAD")

    calls.clear()
    def environment_run(*args, **kwargs):
        calls.append(args)
        if args[:3] == ("git", "merge-base", "origin/release"):
            return "env-base"
        if args[:2] == ("git", "log"):
            return ""
        raise AssertionError(args)
    monkeypatch.setattr(audit, "run", environment_run)
    assert audit.new_emails() == []
    assert calls[0] == ("git", "merge-base", "origin/release", "HEAD")

    monkeypatch.delenv("GITHUB_BASE_REF")
    calls.clear()
    def origin_head_run(*args, **kwargs):
        calls.append(args)
        if args == ("git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"):
            return "origin/topic"
        if args == ("git", "rev-parse", "--verify", "refs/remotes/origin/topic^{commit}"):
            return "topic-sha"
        raise AssertionError(args)
    monkeypatch.setattr(audit, "run", origin_head_run)
    assert audit._default_base_ref() == "topic"


def test_default_base_ref_fails_closed_when_remote_head_is_missing(monkeypatch):
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    def fake_run(*args, **kwargs):
        assert args == ("git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
        return ""

    monkeypatch.setattr(audit, "run", fake_run)

    with pytest.raises(audit.BaseRefError, match="origin/HEAD"):
        audit._default_base_ref()


def test_cli_reports_actionable_missing_default_metadata(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    monkeypatch.setattr(
        audit,
        "new_emails",
        lambda base_ref=None: (_ for _ in ()).throw(
            audit.BaseRefError("origin/HEAD missing; use --base-ref or GITHUB_BASE_REF")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["audit_pr_attribution.py"])

    with pytest.raises(SystemExit) as exc:
        audit.main()

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "origin/HEAD" in error
    assert "--base-ref" in error


def test_workflow_keeps_pr_base_default_branch_expression_and_strict_shell():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "BASE_REF: ${{ github.event.pull_request.base.ref || github.event.repository.default_branch }}" in workflow
    assert "set -euo pipefail" in workflow
    assert 'git merge-base "origin/${BASE_REF}" HEAD' in workflow
    assert 'git log "${MERGE_BASE}..HEAD"' in workflow
