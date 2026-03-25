from __future__ import annotations

import sys
import types
from pathlib import Path

from terrain_splitter.exact_bridge import LambdaExactRuntimeBridge, LocalExactRuntimeSidecarBridge


def test_lambda_exact_runtime_bridge_uses_configured_read_timeout(monkeypatch) -> None:
    monkeypatch.setenv("TERRAIN_SPLITTER_LAMBDA_INVOKE_READ_TIMEOUT_SEC", "300")
    monkeypatch.setenv("TERRAIN_SPLITTER_EXACT_CANDIDATE_MAX_INFLIGHT", "3")

    captured: dict[str, object] = {}

    class FakeConfig:
        def __init__(self, *, read_timeout: int) -> None:
            self.read_timeout = read_timeout

    def fake_client(name: str, *, config: FakeConfig) -> object:
        captured["name"] = name
        captured["config"] = config
        return object()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = fake_client  # type: ignore[attr-defined]
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = FakeConfig  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_botocore_config)

    LambdaExactRuntimeBridge("terrain-splitter-terrain-splitter-exact")

    assert captured["name"] == "lambda"
    assert isinstance(captured["config"], FakeConfig)
    assert captured["config"].read_timeout == 300


def test_lambda_exact_runtime_bridge_reports_fanout_settings(monkeypatch) -> None:
    monkeypatch.setenv("TERRAIN_SPLITTER_EXACT_CANDIDATE_MAX_INFLIGHT", "7")

    class FakeConfig:
        def __init__(self, *, read_timeout: int) -> None:
            self.read_timeout = read_timeout

    def fake_client(_name: str, *, config: FakeConfig) -> object:
        return object()

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = fake_client  # type: ignore[attr-defined]
    fake_botocore_config = types.ModuleType("botocore.config")
    fake_botocore_config.Config = FakeConfig  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_botocore_config)

    bridge = LambdaExactRuntimeBridge("terrain-splitter-terrain-splitter-exact")

    assert bridge.supports_candidate_fanout() is True
    assert bridge.candidate_max_inflight() == 7


def test_local_exact_runtime_sidecar_bridge_reports_fanout_settings(monkeypatch) -> None:
    monkeypatch.setenv("TERRAIN_SPLITTER_EXACT_CANDIDATE_MAX_INFLIGHT", "7")
    bridge = LocalExactRuntimeSidecarBridge(Path("/tmp/repo"))
    try:
        assert bridge.supports_candidate_fanout() is True
        assert bridge.candidate_max_inflight() == 7
    finally:
        bridge.close()


def test_local_exact_runtime_sidecar_bridge_closes_batch_states() -> None:
    bridge = LocalExactRuntimeSidecarBridge(Path("/tmp/repo"))
    closed: list[object] = []
    batch_handle = bridge.begin_candidate_batch()
    state = bridge._get_state(batch_handle=batch_handle)
    sentinel_proc = object()
    state.proc = sentinel_proc  # type: ignore[assignment]

    def fake_close_state(candidate_state) -> None:
        closed.append(candidate_state.proc)
        candidate_state.proc = None

    bridge._close_state = fake_close_state  # type: ignore[method-assign]

    try:
        bridge.end_candidate_batch(batch_handle)
        assert batch_handle not in bridge._batch_states
        assert closed == [sentinel_proc]
    finally:
        bridge.close()
