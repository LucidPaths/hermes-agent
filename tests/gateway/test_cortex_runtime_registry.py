from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gateway.cortex_runtime_registry import (
    GatewayCortexRegistryKey,
    GatewayCortexRuntimeRegistry,
)


def test_registry_authority_key_is_read_only(tmp_path: Path) -> None:
    registry = GatewayCortexRuntimeRegistry(
        hermes_home=tmp_path.resolve(),
        profile_identity="default",
    )
    replacement = GatewayCortexRegistryKey(
        hermes_home=(tmp_path / "other").resolve(),
        profile_identity="other",
        state_root=(tmp_path / "other/memory/hermes_cortex").resolve(),
    )
    with pytest.raises(AttributeError):
        registry.key = replacement


def test_gateway_factory_is_off_for_every_non_cortex_provider() -> None:
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    borrowed: list[str] = []
    runner = object.__new__(GatewayRunner)
    runner._active_profile_name = lambda: "default"
    runner._cortex_runtime_registry = SimpleNamespace(
        borrow=lambda name="hermes_cortex": borrowed.append(name) or object(),
        key=SimpleNamespace(
            hermes_home=gateway_run._gateway_config_home().resolve(),
            profile_identity="default",
        ),
    )
    assert runner._memory_provider_factory_for({}) is None
    assert runner._memory_provider_factory_for({"memory": {"provider": "honcho"}}) is None
    factory = runner._memory_provider_factory_for(
        {"memory": {"provider": "hermes_cortex"}}
    )
    assert factory is not None
    factory("hermes_cortex")
    assert borrowed == ["hermes_cortex"]


def test_gateway_factory_refuses_foreign_profile_identity(tmp_path: Path) -> None:
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._active_profile_name = lambda: "secondary"
    runner._cortex_runtime_registry = SimpleNamespace(
        borrow=lambda _name="hermes_cortex": object(),
        key=SimpleNamespace(
            hermes_home=tmp_path.resolve(),
            profile_identity="primary",
        ),
    )
    with pytest.raises(RuntimeError, match="cortex-registry-identity-mismatch"):
        runner._memory_provider_factory_for(
            {"memory": {"provider": "hermes_cortex"}}
        )


def test_configured_cortex_startup_failure_is_fail_closed() -> None:
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    class _FailingRegistry:
        key = SimpleNamespace(
            hermes_home=gateway_run._gateway_config_home().resolve(),
            profile_identity="default",
        )

        def borrow(self, _name: str = "hermes_cortex") -> object:
            return object()

        def ensure_started(self) -> None:
            raise RuntimeError("lease-held")

    statuses: list[tuple[str, str]] = []
    runner = object.__new__(GatewayRunner)
    runner._active_profile_name = lambda: "default"
    runner._cortex_runtime_registry = _FailingRegistry()
    runner._exit_with_failure = False
    runner._exit_code = None
    runner._exit_reason = None
    runner._update_runtime_status = lambda state, reason=None: statuses.append(
        (state, reason)
    )

    assert not runner._ensure_cortex_runtime_for_config(
        {"memory": {"provider": "hermes_cortex"}}
    )
    assert runner._exit_with_failure is True
    assert runner._exit_code == 1
    assert runner._exit_reason == "Cortex runtime startup failed"
    assert statuses == [("startup_failed", "Cortex runtime startup failed")]


def test_non_cortex_startup_never_touches_registry() -> None:
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._cortex_runtime_registry = SimpleNamespace(
        ensure_started=lambda: pytest.fail("registry must remain dormant")
    )
    assert runner._ensure_cortex_runtime_for_config({})
    assert runner._ensure_cortex_runtime_for_config(
        {"memory": {"provider": "honcho"}}
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.owner_thread_id = 123
        self.runtime_key = SimpleNamespace(generation=0)

    def startup_readiness(self) -> SimpleNamespace:
        return SimpleNamespace(state="ready", reason="")

    def shutdown(self, deadline_s: float) -> bool:
        assert deadline_s >= 0
        self.shutdown_calls += 1
        self.owner_thread_id = None
        return True


class _FakeProvider:
    def __init__(self, *, runtime: _FakeRuntime | None = None) -> None:
        self.runtime_arg = runtime


def test_failed_startup_is_retained_without_retry_and_closed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeRuntime()
    runtime.startup_readiness = lambda: SimpleNamespace(
        state="failed", reason="writer-lease-unavailable"
    )
    load_calls = 0
    factory_calls = 0

    def load_provider(_name: str) -> _FakeProvider:
        nonlocal load_calls
        load_calls += 1
        return _FakeProvider()

    def create_runtime(**_kwargs: Any) -> _FakeRuntime:
        nonlocal factory_calls
        factory_calls += 1
        return runtime

    monkeypatch.setattr(
        sys.modules[_FakeProvider.__module__], "create_runtime", create_runtime, raising=False
    )
    registry = GatewayCortexRuntimeRegistry(
        hermes_home=tmp_path.resolve(),
        profile_identity="default",
        provider_loader=load_provider,
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cortex-runtime-startup-failed"):
            registry.borrow()
    assert load_calls == 1
    assert factory_calls == 1
    assert registry.shutdown(2.0)
    assert runtime.shutdown_calls == 1
    assert registry.shutdown(2.0)
    assert runtime.shutdown_calls == 1


def test_concurrent_borrows_share_one_runtime_and_return_distinct_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeRuntime()
    load_calls = 0
    factory_calls = 0

    def load_provider(name: str) -> _FakeProvider:
        nonlocal load_calls
        assert name == "hermes_cortex"
        load_calls += 1
        return _FakeProvider()

    def create_runtime(*, hermes_home: str, launch_kwargs: dict[str, Any]) -> _FakeRuntime:
        nonlocal factory_calls
        assert Path(hermes_home) == tmp_path.resolve()
        assert launch_kwargs == {
            "agent_identity": "default",
            "agent_workspace": "hermes",
        }
        factory_calls += 1
        return runtime

    monkeypatch.setattr(
        sys.modules[_FakeProvider.__module__], "create_runtime", create_runtime, raising=False
    )
    registry = GatewayCortexRuntimeRegistry(
        hermes_home=tmp_path.resolve(),
        profile_identity="default",
        provider_loader=load_provider,
    )
    providers: list[_FakeProvider] = []
    workers = [threading.Thread(target=lambda: providers.append(registry.borrow())) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(2.0)
        assert not worker.is_alive()

    assert load_calls == 1
    assert factory_calls == 1
    assert len(providers) == 8
    assert len({id(provider) for provider in providers}) == 8
    assert all(provider.runtime_arg is runtime for provider in providers)
    assert registry.owner_thread_id == 123
    assert registry.shutdown(2.0)
    assert runtime.shutdown_calls == 1
    assert registry.shutdown(2.0)
    assert runtime.shutdown_calls == 1
    with pytest.raises(RuntimeError, match="cortex-registry-draining"):
        registry.borrow()
