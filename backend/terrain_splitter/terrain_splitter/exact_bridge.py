from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger("uvicorn.error")


class ExactRuntimeBridge:
    def optimize_bearing(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def rerank_solutions(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class LocalExactRuntimeSidecarBridge(ExactRuntimeBridge):
    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._lock = threading.Lock()
        self._counter = 0
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_thread: threading.Thread | None = None
        atexit.register(self.close)

    def _spawn(self) -> subprocess.Popen[str]:
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
            self._stderr_thread = threading.Thread(target=self._forward_stderr, args=(proc.stderr,), daemon=True)
            self._stderr_thread.start()
        return proc

    def _forward_stderr(self, pipe: TextIO) -> None:
        try:
            for line in pipe:
                message = line.rstrip()
                if message:
                    logger.error("%s", message)
        except Exception as exc:  # noqa: BLE001
            logger.error("[exact-runtime-sidecar] stderr pump failed error=%s", exc)

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self.close()
            self._proc = self._spawn()
        return self._proc

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            proc = self._ensure_proc()
            self._counter += 1
            request_id = f"exact-{self._counter}"
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

    def rerank_solutions(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._request({"operation": "rerank-solutions", **request})

    def close(self) -> None:
        proc, self._proc = self._proc, None
        self._stderr_thread = None
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


class LambdaExactRuntimeBridge(ExactRuntimeBridge):
    def __init__(self, function_name: str) -> None:
        import boto3

        self._function_name = function_name
        self._client = boto3.client("lambda")

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
