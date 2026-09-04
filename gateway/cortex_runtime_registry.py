"""Gateway-owned Cortex runtime registry with binding-local provider leases."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GatewayCortexRegistryKey:
    hermes_home: Path
    profile_identity: str
    state_root: Path


class GatewayCortexRuntimeRegistry:
    """Own exactly one Cortex runtime for one messaging GatewayRunner."""

    def __init__(
        self,
        *,
        hermes_home: Path,
        profile_identity: str,
        provider_loader: Callable[[str], Any] | None = None,
    ) -> None:
        home = Path(hermes_home).expanduser()
        if not home.is_absolute() or not profile_identity.strip():
            raise ValueError("invalid Cortex gateway registry identity")
        home = home.resolve()
        self._key = GatewayCortexRegistryKey(
            hermes_home=home,
            profile_identity=profile_identity,
            state_root=(home / "memory" / "hermes_cortex").resolve(),
        )
        self._provider_loader = provider_loader
        self._provider_type: type[Any] | None = None
        self._runtime: Any = None
        self._startup_error: Exception | None = None
        self._draining = False
        self._closed = False
        self._lock = threading.Lock()

    @property
    def key(self) -> GatewayCortexRegistryKey:
        """Immutable construction-bound runtime authority identity."""
        return self._key

    def _load_provider(self) -> Any:
        loader = self._provider_loader
        if loader is None:
            from plugins.memory import load_memory_provider

            loader = load_memory_provider
        provider = loader("hermes_cortex")
        if provider is None:
            raise RuntimeError("cortex-provider-unavailable")
        return provider

    def _ensure_runtime_locked(self) -> None:
        if self._startup_error is not None:
            raise RuntimeError("cortex-runtime-startup-failed") from self._startup_error
        if self._runtime is not None:
            return
        try:
            prototype = self._load_provider()
            module = sys.modules.get(type(prototype).__module__)
            factory = getattr(module, "create_runtime", None) if module is not None else None
            if not callable(factory):
                raise RuntimeError("cortex-runtime-factory-unavailable")
            runtime = factory(
                hermes_home=str(self.key.hermes_home),
                launch_kwargs={
                    "agent_identity": self.key.profile_identity,
                    "agent_workspace": "hermes",
                },
            )
            self._runtime = runtime
            readiness = runtime.startup_readiness()
            state = getattr(readiness, "state", "")
            if getattr(state, "value", state) != "ready":
                reason = str(getattr(readiness, "reason", "") or "cortex-runtime-not-ready")
                raise RuntimeError(reason)
            self._provider_type = type(prototype)
            self._runtime = runtime
        except Exception as exc:
            self._startup_error = exc
            raise RuntimeError("cortex-runtime-startup-failed") from exc

    def ensure_started(self) -> None:
        """Create and validate the sole runtime without borrowing a provider."""
        with self._lock:
            if self._draining or self._closed:
                raise RuntimeError("cortex-registry-draining")
            self._ensure_runtime_locked()

    def borrow(self, provider_name: str = "hermes_cortex") -> Any:
        """Return a fresh provider bound to this registry's explicit owner."""
        with self._lock:
            if provider_name != "hermes_cortex":
                raise RuntimeError("cortex-provider-mismatch")
            if self._draining or self._closed:
                raise RuntimeError("cortex-registry-draining")
            self._ensure_runtime_locked()
            assert self._provider_type is not None
            return self._provider_type(runtime=self._runtime)

    @property
    def owner_thread_id(self) -> int | None:
        with self._lock:
            if self._runtime is None:
                return None
            return self._runtime.owner_thread_id

    def shutdown(self, deadline_s: float) -> bool:
        """Reject new borrows and close the owner once all bindings/jobs drain."""
        with self._lock:
            self._draining = True
            if self._closed:
                return True
            if self._runtime is None:
                self._closed = True
                return True
            closed = bool(self._runtime.shutdown(deadline_s))
            if closed:
                self._closed = True
            return closed
