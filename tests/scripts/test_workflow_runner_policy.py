"""Static policy tests for fork-safe GitHub Actions runner selection."""

from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
LARGER_RUNNER_PATTERN = re.compile(
    r"\b(?:ubuntu|windows|macos)-latest-\d+-(?:arm-)?core\b"
)
OWNER_CONDITIONED_RUNNER = re.compile(
    r"github\.repository_owner\s*==\s*'NousResearch'\s*&&\s*"
    r"'[^']+'\s*\|\|\s*'(?:ubuntu|windows|macos)-latest'"
)
JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
RUNNER_ASSIGNMENT = re.compile(r"^\s+(?:-\s+)?(runs-on|runner):\s*(.*?)\s*$")

EXPECTED_DOCKER_ASSIGNMENTS = Counter(
    {
        ("build", "runner", "ubuntu-latest-32-core"): 1,
        ("build", "runner", "ubuntu-latest-32-arm-core"): 1,
        ("publish", "runner", "ubuntu-latest-32-core"): 1,
        ("publish", "runner", "ubuntu-latest-32-arm-core"): 1,
    }
)


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _job_blocks(workflow: str) -> list[tuple[str, list[str]]]:
    """Return direct jobs and their bounded source lines."""
    lines = workflow.splitlines()
    jobs_start = next(i for i, line in enumerate(lines) if line == "jobs:")
    headers = [
        (i, match.group(1))
        for i, line in enumerate(lines[jobs_start + 1 :], jobs_start + 1)
        if (match := JOB_HEADER.match(line))
    ]
    return [
        (name, lines[start:end])
        for (start, name), (end, _) in zip(headers, [*headers[1:], (len(lines), "")])
    ]


def _larger_runner_assignments(
    workflow_name: str, workflow: str
) -> list[tuple[str, int, str, str, str, str]]:
    """Find every same-line runs-on/matrix-runner assignment with a larger label."""
    assignments = []
    for job, lines in _job_blocks(workflow):
        for line_number, line in enumerate(lines, 1):
            match = RUNNER_ASSIGNMENT.match(line)
            if not match:
                continue
            field, value = match.groups()
            for label in LARGER_RUNNER_PATTERN.findall(value):
                assignments.append((workflow_name, line_number, job, field, label, value))
    return assignments


def _is_owner_conditioned(value: str, label: str) -> bool:
    return label in value and bool(OWNER_CONDITIONED_RUNNER.search(value))


def _policy_violations(workflows: dict[str, str]) -> list[str]:
    """Return all larger-label assignments that violate the fork policy."""
    violations = []
    docker_assignments = Counter()
    for workflow_name, workflow in workflows.items():
        for source_name, line_number, job, field, label, value in _larger_runner_assignments(
            workflow_name, workflow
        ):
            if workflow_name == "docker.yml":
                docker_assignments[(job, field, label)] += 1
            elif not _is_owner_conditioned(value, label):
                violations.append(
                    f"{source_name}:{line_number} {job}.{field}={label} is not owner-conditioned"
                )

    for assignment, count in (docker_assignments - EXPECTED_DOCKER_ASSIGNMENTS).items():
        violations.append(f"docker.yml {assignment} occurs {count} time(s) unexpectedly")
    for assignment, count in (EXPECTED_DOCKER_ASSIGNMENTS - docker_assignments).items():
        violations.append(f"docker.yml is missing {count} occurrence(s) of {assignment}")
    return violations


def test_every_larger_runner_assignment_is_fork_safe_or_an_exact_docker_exception():
    workflows = {
        path.name: path.read_text(encoding="utf-8")
        for path in WORKFLOWS.iterdir()
        if path.suffix in {".yml", ".yaml"}
    }
    assert not _policy_violations(workflows)


def test_unexpected_workflow_static_larger_runner_is_rejected():
    workflow = "jobs:\n  rogue:\n    runs-on: ubuntu-latest-64-core\n"
    assert _policy_violations({"rogue.yml": workflow})


def test_additional_docker_larger_runner_is_rejected():
    workflow = (
        "jobs:\n"
        "  build:\n"
        "    if: github.repository == 'NousResearch/hermes-agent'\n"
        "    strategy:\n"
        "      matrix:\n"
        "        include:\n"
        "          - runner: ubuntu-latest-32-core\n"
        "          - runner: ubuntu-latest-64-core\n"
    )
    violations = _policy_violations({"docker.yml": workflow})
    assert any("64-core" in violation for violation in violations)


def test_docker_exceptions_are_exactly_gated_build_and_publish_assignments():
    docker = _workflow("docker.yml")
    jobs = dict(_job_blocks(docker))
    build = "\n".join(jobs["build"])
    publish = "\n".join(jobs["publish"])
    assert "needs: [detect]" in build
    assert (
        "if: github.repository == 'NousResearch/hermes-agent' && "
        "needs.detect.outputs.build == 'true'"
    ) in build
    assert (
        "if: github.repository == 'NousResearch/hermes-agent' && "
        "(github.event_name == 'push' && github.ref == 'refs/heads/main' || "
        "github.event_name == 'release')"
    ) in publish
    assert Counter(
        (job, field, label)
        for _, _, job, field, label, _ in _larger_runner_assignments("docker.yml", docker)
    ) == EXPECTED_DOCKER_ASSIGNMENTS
