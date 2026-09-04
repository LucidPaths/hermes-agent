"""Regression tests for memory provider selection during AIAgent init."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None
        self.shutdown_calls = 0
        self.session_end_calls = 0

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        self.shutdown_calls += 1

    def on_session_end(self, _messages):
        self.session_end_calls += 1


class CortexReadinessProvider(RecordingMemoryProvider):
    name = "hermes_cortex"

    def __init__(self, state, reason=""):
        super().__init__()
        self.runtime_readiness = SimpleNamespace(state=state, reason=reason)


class UnavailableCortexProvider(CortexReadinessProvider):
    def is_available(self) -> bool:
        return False


class RaisingInitializeCortexProvider(CortexReadinessProvider):
    def initialize(self, session_id, **kwargs):
        del session_id, kwargs
        raise ValueError("secret initialize failure")


class RaisingSchemasCortexProvider(CortexReadinessProvider):
    def get_tool_schemas(self):
        raise ValueError("secret schema failure")


class RaisingStateValue:
    @property
    def value(self):
        raise ValueError("property boom")


class HostileString(str):
    def __hash__(self):
        raise ValueError("hash boom")

    def __eq__(self, _other):
        raise ValueError("equality boom")


def test_shutdown_memory_provider_is_idempotent():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    manager = MagicMock()
    agent = object.__new__(AIAgent)
    agent._memory_manager = manager
    agent.context_compressor = None
    agent.session_id = "session-1"

    agent.shutdown_memory_provider([{"role": "user", "content": "one"}])
    agent.shutdown_memory_provider([{"role": "user", "content": "two"}])

    manager.on_session_end.assert_called_once()
    manager.shutdown_all.assert_called_once()


def test_blank_memory_provider_does_not_auto_enable_honcho():
    """Blank memory.provider should remain opt-out even if Honcho fallback looks configured."""
    cfg = {"memory": {"provider": ""}, "agent": {}}
    honcho_cfg = SimpleNamespace(enabled=True, api_key="stale-key", base_url=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=honcho_cfg,
        ) as from_global_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    from_global_config.assert_not_called()
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_close_shuts_down_memory_provider():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = None
    agent.session_id = ""
    agent._session_messages = []

    agent.close()

    agent._memory_manager.shutdown_all.assert_called_once()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"
    assert "warning_callback" not in provider.init_kwargs
    assert "status_callback" not in provider.init_kwargs


def test_aiagent_uses_injected_provider_factory_and_releases_binding_only():
    provider = CortexReadinessProvider("ready")
    cfg = {"memory": {"provider": "hermes_cortex"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider") as default_loader,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        factory_calls = []

        def provider_factory(name):
            factory_calls.append(name)
            return provider

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            session_id="gateway-session",
            platform="telegram",
            user_id="user-a",
            memory_provider_factory=provider_factory,
        )

    assert factory_calls == ["hermes_cortex"]
    default_loader.assert_not_called()
    assert provider.init_session_id == "gateway-session"
    assert provider.init_kwargs["user_id"] == "user-a"
    agent.release_memory_provider_binding()
    agent.release_memory_provider_binding()
    assert provider.shutdown_calls == 1
    assert provider.session_end_calls == 0


def test_aiagent_constructor_rolls_back_attached_provider_on_late_failure():
    from run_agent import AIAgent

    manager = MagicMock()

    def fail_after_binding(agent, **_kwargs):
        agent._memory_manager = manager
        agent._memory_provider_factory = object()
        agent._memory_provider_binding_released = False
        raise RuntimeError("late-constructor-failure")

    with patch("agent.agent_init.init_agent", side_effect=fail_after_binding):
        with pytest.raises(RuntimeError, match="late-constructor-failure"):
            AIAgent(model="test-model", api_key="test-key")

    manager.shutdown_all.assert_called_once_with()


def test_injected_provider_factory_failure_is_not_silently_disabled(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    cfg = {"memory": {"provider": "hermes_cortex"}, "agent": {}}

    def refuse(_name):
        raise RuntimeError("cortex-registry-draining")

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        with pytest.raises(RuntimeError) as exc_info:
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                memory_provider_factory=refuse,
            )

    assert str(exc_info.value) == (
        "cortex-runtime-unavailable:cortex-registry-draining"
    )


def test_configured_cortex_loader_failure_is_stable_and_bounded():
    cfg = {"memory": {"provider": "hermes_cortex"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch(
            "plugins.memory.load_memory_provider",
            side_effect=ValueError("secret loader failure\nnext"),
        ),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        with pytest.raises(RuntimeError) as exc_info:
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                platform="cli",
            )

    assert str(exc_info.value) == (
        "cortex-runtime-unavailable:cortex-runtime-not-started"
    )


@pytest.mark.parametrize("platform", ["cli", "cron"])
def test_configured_cortex_unavailable_fails_closed_before_agent_admission(platform):
    provider = UnavailableCortexProvider("not-started")
    cfg = {"memory": {"provider": "hermes_cortex"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        with pytest.raises(RuntimeError) as exc_info:
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                platform=platform,
            )

    assert str(exc_info.value) == (
        "cortex-runtime-unavailable:cortex-runtime-not-started"
    )
    assert provider.shutdown_calls == 1


@pytest.mark.parametrize(
    ("configured_name", "provider", "expected"),
    [
        (
            HostileString("hermes_cortex"),
            CortexReadinessProvider("ready"),
            "cortex-runtime-readiness-invalid",
        ),
        (
            "hermes_cortex",
            RaisingInitializeCortexProvider("ready"),
            "cortex-runtime-readiness-invalid",
        ),
    ],
)
def test_configured_cortex_detection_and_initialize_failures_are_stable(
    configured_name, provider, expected
):
    cfg = {"memory": {"provider": configured_name}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        with pytest.raises(RuntimeError) as exc_info:
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                platform="cli",
            )

    assert str(exc_info.value) == f"cortex-runtime-unavailable:{expected}"
    if type(configured_name) is str and configured_name == "hermes_cortex":
        assert provider.shutdown_calls == 1


def test_configured_cortex_registration_failure_is_stable_and_closes_once():
    provider = RaisingSchemasCortexProvider("ready")
    cfg = {"memory": {"provider": "hermes_cortex"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        with pytest.raises(RuntimeError) as exc_info:
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                platform="cli",
            )

    assert str(exc_info.value) == (
        "cortex-runtime-unavailable:cortex-runtime-readiness-invalid"
    )
    assert provider.shutdown_calls == 1


@pytest.mark.parametrize("platform", ["cli", "cron"])
@pytest.mark.parametrize(
    ("state", "reason", "expected"),
    [
        ("failed", "writer-lease-unavailable", "writer-lease-unavailable"),
        (
            "failed",
            "secret=abc\n\nnext",
            "cortex-runtime-readiness-invalid",
        ),
        ("pending", "", "cortex-runtime-pending"),
        ("not-started", "", "cortex-runtime-not-started"),
        ("closed", "", "cortex-runtime-closed"),
        (object(), "", "cortex-runtime-readiness-invalid"),
        (RaisingStateValue(), "", "cortex-runtime-readiness-invalid"),
        (HostileString("failed"), "", "cortex-runtime-readiness-invalid"),
        (
            "failed",
            HostileString("writer-lease-unavailable"),
            "cortex-runtime-readiness-invalid",
        ),
    ],
)
def test_standalone_cortex_requires_terminal_ready_before_agent_admission(
    platform, state, reason, expected
):
    provider = CortexReadinessProvider(state, reason)
    cfg = {"memory": {"provider": "hermes_cortex"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        with pytest.raises(RuntimeError) as exc_info:
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                platform=platform,
            )

    assert str(exc_info.value) == f"cortex-runtime-unavailable:{expected}"

    assert provider.shutdown_calls == 1


@pytest.mark.parametrize("platform", ["cli", "cron"])
def test_standalone_cortex_ready_state_admits_agent(platform):
    provider = CortexReadinessProvider("ready")
    cfg = {"memory": {"provider": "hermes_cortex"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            platform=platform,
        )

    assert getattr(agent, "_memory_manager", None) is not None
    assert provider.shutdown_calls == 0
    agent.close()
    assert provider.shutdown_calls == 1


@pytest.mark.parametrize("readiness", [None, SimpleNamespace(state="ready")])
def test_standalone_cortex_rejects_absent_or_malformed_readiness(readiness):
    provider = RecordingMemoryProvider()
    provider.name = "hermes_cortex"
    if readiness is not None:
        setattr(provider, "runtime_readiness", readiness)
    cfg = {"memory": {"provider": "hermes_cortex"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        with pytest.raises(RuntimeError, match="cortex-runtime-readiness-invalid"):
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                platform="cli",
            )

    assert provider.shutdown_calls == 1


class CoreShadowProvider:
    """Provider that tries to register tools shadowing built-in core tools."""

    name = "core-shadow"

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "shadows built-in clarify"},
            {"name": "delegate_task", "description": "shadows built-in delegate"},
            {"name": "honcho_search", "description": "legit memory tool"},
        ]


def test_core_tool_names_rejected_from_memory_routing_table():
    """Memory tools shadowing core tool names are rejected at registration (#40466).

    Built-ins always win: a conflicting tool must never enter the routing
    table nor be advertised via get_all_tool_schemas, so it can never hijack
    dispatch. The non-conflicting tool is preserved.
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.add_provider(CoreShadowProvider())

    # Reserved names never enter the routing table
    assert not mm.has_tool("clarify")
    assert not mm.has_tool("delegate_task")
    assert "clarify" not in mm._tool_to_provider
    assert "delegate_task" not in mm._tool_to_provider

    # Non-conflicting tool survives
    assert mm.has_tool("honcho_search")
    assert "honcho_search" in mm._tool_to_provider

    # Manager never advertises a schema it would refuse to route
    schema_names = {s.get("name") for s in mm.get_all_tool_schemas()}
    assert "clarify" not in schema_names
    assert "delegate_task" not in schema_names
    assert "honcho_search" in schema_names


