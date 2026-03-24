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


def test_local_exact_runtime_sidecar_bridge_keeps_serial_candidate_behavior() -> None:
    bridge = LocalExactRuntimeSidecarBridge(Path("/tmp/repo"))
    try:
        assert bridge.supports_candidate_fanout() is False
        assert bridge.candidate_max_inflight() == 1
    finally:
        bridge.close()
