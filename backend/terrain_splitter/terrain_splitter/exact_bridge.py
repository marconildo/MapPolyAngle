from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger("uvicorn.error")


def _resolve_lambda_invoke_read_timeout_sec() -> int:
    raw = os.environ.get("TERRAIN_SPLITTER_LAMBDA_INVOKE_READ_TIMEOUT_SEC")
    if raw is None or raw.strip() == "":
        return 300
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "[terrain-split-backend] invalid TERRAIN_SPLITTER_LAMBDA_INVOKE_READ_TIMEOUT_SEC=%r; falling back to 300s",
            raw,
        )
        return 300


def _resolve_exact_candidate_max_inflight() -> int:
    raw = os.environ.get("TERRAIN_SPLITTER_EXACT_CANDIDATE_MAX_INFLIGHT")
    if raw is None or raw.strip() == "":
        return 8
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "[terrain-split-backend] invalid TERRAIN_SPLITTER_EXACT_CANDIDATE_MAX_INFLIGHT=%r; falling back to 8",
            raw,
        )
        return 8


class ExactRuntimeBridge:
    def supports_candidate_fanout(self) -> bool:
        return False

    def candidate_max_inflight(self) -> int:
        return 1

    def begin_candidate_batch(self) -> Any | None:
        return None

    def end_candidate_batch(self, batch_handle: Any | None) -> None:
        return None

    def optimize_bearing(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def evaluate_solution(self, request: dict[str, Any], *, batch_handle: Any | None = None) -> dict[str, Any]:
        raise NotImplementedError

    def rerank_solutions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


@dataclass
class _LocalSidecarState:
    proc: subprocess.Popen[str] | None = None
    stderr_thread: threading.Thread | None = None
    counter: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class LocalExactRuntimeSidecarBridge(ExactRuntimeBridge):
    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._candidate_max_inflight = _resolve_exact_candidate_max_inflight()
        self._states_lock = threading.Lock()
        self._default_state = _LocalSidecarState()
        self._batch_states: dict[object, dict[int, _LocalSidecarState]] = {}
        atexit.register(self.close)

    def supports_candidate_fanout(self) -> bool:
        return True

    def candidate_max_inflight(self) -> int:
        return self._candidate_max_inflight

    def begin_candidate_batch(self) -> object:
        batch_handle = object()
        with self._states_lock:
            self._batch_states[batch_handle] = {}
        return batch_handle

    def end_candidate_batch(self, batch_handle: Any | None) -> None:
        if batch_handle is None:
            return
        with self._states_lock:
            states = list(self._batch_states.pop(batch_handle, {}).values())
        for state in states:
            self._close_state(state)

    def _get_state(self, *, batch_handle: Any | None = None) -> _LocalSidecarState:
        if batch_handle is None:
            return self._default_state
        thread_id = threading.get_ident()
        with self._states_lock:
            batch_states = self._batch_states.setdefault(batch_handle, {})
            state = batch_states.get(thread_id)
            if state is None:
                state = _LocalSidecarState()
                batch_states[thread_id] = state
        return state

    def _spawn(self, state: _LocalSidecarState) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env.setdefault("EXACT_RUNTIME_PROVIDER_MODE", "local-http")
        env.setdefault("EXACT_RUNTIME_INTERNAL_BASE_URL", os.environ.get("TERRAIN_SPLITTER_INTERNAL_BASE_URL", "http://127.0.0.1:8090"))
        env.setdefault("NODE_NO_WARNINGS", "1")
        tsx_bin = self._repo_root / "node_modules" / ".bin" / "tsx"
        if tsx_bin.exists():
            command = [str(tsx_bin), "src/overlap/exact-runtime/sidecar.ts"]
        else:
            command = ["npx", "--yes", "tsx", "src/overlap/exact-runtime/sidecar.ts"]
        proc = subprocess.Popen(
            command,
            cwd=self._repo_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if proc.stderr is not None:
            state.stderr_thread = threading.Thread(target=self._forward_stderr, args=(proc.stderr,), daemon=True)
            state.stderr_thread.start()
        return proc

    def _forward_stderr(self, pipe: TextIO) -> None:
        try:
            for line in pipe:
                message = line.rstrip()
                if message:
                    logger.error("%s", message)
        except Exception as exc:  # noqa: BLE001
            logger.error("[exact-runtime-sidecar] stderr pump failed error=%s", exc)

    def _close_state(self, state: _LocalSidecarState) -> None:
        proc, state.proc = state.proc, None
        state.stderr_thread = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass

    def _ensure_proc(self, state: _LocalSidecarState) -> subprocess.Popen[str]:
        if state.proc is None or state.proc.poll() is not None:
            self._close_state(state)
            state.proc = self._spawn(state)
        return state.proc

    def _request(self, payload: dict[str, Any], *, batch_handle: Any | None = None) -> dict[str, Any]:
        state = self._get_state(batch_handle=batch_handle)
        with state.lock:
            proc = self._ensure_proc(state)
            state.counter += 1
            request_id = f"exact-{threading.get_ident()}-{state.counter}"
            envelope = {"id": request_id, "request": payload}
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("Exact runtime sidecar pipes are not available.")
            proc.stdin.write(json.dumps(envelope) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                stderr = ""
                if proc.stderr is not None:
                    try:
                        stderr = proc.stderr.read()
                    except Exception:  # noqa: BLE001
                        stderr = ""
                raise RuntimeError(f"Exact runtime sidecar exited unexpectedly. {stderr}".strip())
            response = json.loads(line)
            if response.get("id") != request_id:
                raise RuntimeError("Exact runtime sidecar returned an unexpected response id.")
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "Exact runtime sidecar request failed.")
            return response["response"]

    def optimize_bearing(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request({"operation": "optimize-bearing", **request})

    def evaluate_solution(self, request: dict[str, Any], *, batch_handle: Any | None = None) -> dict[str, Any]:
        return self._request({"operation": "evaluate-solution", **request}, batch_handle=batch_handle)

    def rerank_solutions(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request({"operation": "rerank-solutions", **request})

    def close(self) -> None:
        with self._states_lock:
            batch_states = list(self._batch_states.values())
            self._batch_states = {}
        self._close_state(self._default_state)
        for batch in batch_states:
            for state in batch.values():
                self._close_state(state)


class LambdaExactRuntimeBridge(ExactRuntimeBridge):
    def __init__(self, function_name: str) -> None:
        import boto3
        from botocore.config import Config

        self._function_name = function_name
        self._candidate_max_inflight = _resolve_exact_candidate_max_inflight()
        self._client = boto3.client(
            "lambda",
            config=Config(read_timeout=_resolve_lambda_invoke_read_timeout_sec()),
        )

    def supports_candidate_fanout(self) -> bool:
        return True

    def candidate_max_inflight(self) -> int:
        return self._candidate_max_inflight

    def _invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.invoke(
            FunctionName=self._function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        raw_payload = response.get("Payload")
        text = raw_payload.read().decode("utf-8") if raw_payload is not None else ""
        if not text:
            raise RuntimeError("Exact runtime Lambda returned an empty payload.")
        data = json.loads(text)
        if isinstance(data, dict) and data.get("errorMessage"):
            raise RuntimeError(str(data["errorMessage"]))
        return data

    def optimize_bearing(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._invoke({"operation": "optimize-bearing", **request})

    def evaluate_solution(self, request: dict[str, Any], *, batch_handle: Any | None = None) -> dict[str, Any]:
        return self._invoke({"operation": "evaluate-solution", **request})

    def rerank_solutions(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._invoke({"operation": "rerank-solutions", **request})


def create_exact_runtime_bridge() -> ExactRuntimeBridge | None:
    if os.environ.get("TERRAIN_SPLITTER_DISABLE_EXACT_POSTPROCESS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    exact_lambda_function_name = os.environ.get("TERRAIN_SPLITTER_EXACT_LAMBDA_FUNCTION_NAME")
    if exact_lambda_function_name:
        return LambdaExactRuntimeBridge(exact_lambda_function_name)
    repo_root = Path(__file__).resolve().parents[3]
    return LocalExactRuntimeSidecarBridge(repo_root)
