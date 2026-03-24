from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, status

from .dsm_store import DsmDatasetStore, S3BackedDsmDatasetStore, derive_descriptor_from_path
from .schemas import DsmSourceDescriptorModel

DEFAULT_UPLOAD_TTL_SEC = 3600
STREAM_CHUNK_SIZE = 1024 * 1024


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _upload_ttl_sec() -> int:
    raw = os.environ.get("TERRAIN_SPLITTER_DSM_UPLOAD_TTL_SEC")
    if raw is None or raw.strip() == "":
        return DEFAULT_UPLOAD_TTL_SEC
    try:
        return max(60, int(raw))
    except ValueError:
        return DEFAULT_UPLOAD_TTL_SEC


def _uses_s3_upload_flow(dataset_store: DsmDatasetStore) -> bool:
    return bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME")) and isinstance(dataset_store, S3BackedDsmDatasetStore)


def uses_presigned_dsm_upload_flow(dataset_store: DsmDatasetStore) -> bool:
    return _uses_s3_upload_flow(dataset_store)


def compute_file_sha256(path: Path) -> str:
    digest, _ = compute_file_sha256_and_size(path)
    return digest


def compute_file_sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(STREAM_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total_size += len(chunk)
    return digest.hexdigest(), total_size


def compute_file_size_bytes(path: Path) -> int:
    return int(path.stat().st_size)


@dataclass(slots=True)
class DsmUploadSession:
    uploadId: str
    sha256: str
    fileSizeBytes: int
    originalName: str
    contentType: str | None
    createdAtIso: str
    expiresAtIso: str
    stagedFilePath: str | None = None
    stagedObjectKey: str | None = None

    @property
    def expires_at(self) -> datetime:
        return _parse_iso8601(self.expiresAtIso)

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.expires_at <= (now or _utc_now())


@dataclass(slots=True)
class DsmPreparedUpload:
    status: str
    descriptor: DsmSourceDescriptorModel | None = None
    reused_existing: bool = False
    upload_id: str | None = None
    upload_target_url: str | None = None
    upload_target_headers: dict[str, str] | None = None
    expires_at_iso: str | None = None


def _upload_expires_at_iso() -> str:
    return (_utc_now() + timedelta(seconds=_upload_ttl_sec())).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _session_dir(staging_dir: Path, upload_id: str) -> Path:
    path = staging_dir / upload_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_manifest_path(staging_dir: Path, upload_id: str) -> Path:
    return _session_dir(staging_dir, upload_id) / "session.json"


def _session_payload_path(staging_dir: Path, upload_id: str, original_name: str) -> Path:
    suffix = Path(original_name).suffix or ".bin"
    return _session_dir(staging_dir, upload_id) / f"payload{suffix}"


def _write_local_session(staging_dir: Path, session: DsmUploadSession) -> None:
    manifest_path = _session_manifest_path(staging_dir, session.uploadId)
    manifest_path.write_text(json.dumps(asdict(session), indent=2, sort_keys=True))


def _read_local_session(staging_dir: Path, upload_id: str) -> DsmUploadSession:
    manifest_path = _session_manifest_path(staging_dir, upload_id)
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="DSM upload session was not found.")
    payload = json.loads(manifest_path.read_text())
    return DsmUploadSession(**payload)


def _delete_local_session(staging_dir: Path, upload_id: str) -> None:
    shutil.rmtree(staging_dir / upload_id, ignore_errors=True)


def _cleanup_expired_local_sessions(staging_dir: Path, *, preserve_upload_id: str | None = None) -> None:
    if not staging_dir.exists():
        return
    now = _utc_now()
    for entry in staging_dir.iterdir():
        if not entry.is_dir():
            continue
        if preserve_upload_id and entry.name == preserve_upload_id:
            continue
        manifest_path = entry / "session.json"
        if not manifest_path.exists():
            shutil.rmtree(entry, ignore_errors=True)
            continue
        try:
            session = DsmUploadSession(**json.loads(manifest_path.read_text()))
        except Exception:
            shutil.rmtree(entry, ignore_errors=True)
            continue
        if session.is_expired(now):
            shutil.rmtree(entry, ignore_errors=True)


def _s3_upload_prefix(dataset_store: S3BackedDsmDatasetStore) -> str:
    prefix = dataset_store.prefix.strip("/")
    return f"{prefix}/uploads" if prefix else "uploads"


def _s3_session_manifest_key(dataset_store: S3BackedDsmDatasetStore, upload_id: str) -> str:
    return f"{_s3_upload_prefix(dataset_store)}/{upload_id}/session.json"


def _s3_session_payload_key(dataset_store: S3BackedDsmDatasetStore, upload_id: str, original_name: str) -> str:
    suffix = Path(original_name).suffix or ".bin"
    return f"{_s3_upload_prefix(dataset_store)}/{upload_id}/payload{suffix}"


def _write_s3_session(dataset_store: S3BackedDsmDatasetStore, session: DsmUploadSession) -> None:
    client = dataset_store._s3_client()
    client.put_object(
        Bucket=dataset_store.bucket,
        Key=_s3_session_manifest_key(dataset_store, session.uploadId),
        Body=json.dumps(asdict(session), indent=2, sort_keys=True).encode("utf-8"),
        ContentType="application/json",
    )


def _read_s3_session(dataset_store: S3BackedDsmDatasetStore, upload_id: str) -> DsmUploadSession:
    client = dataset_store._s3_client()
    try:
        payload = client.get_object(Bucket=dataset_store.bucket, Key=_s3_session_manifest_key(dataset_store, upload_id))["Body"].read()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=404, detail="DSM upload session was not found.") from exc
    return DsmUploadSession(**json.loads(payload.decode("utf-8")))


def _delete_s3_session(dataset_store: S3BackedDsmDatasetStore, session: DsmUploadSession) -> None:
    client = dataset_store._s3_client()
    for key in filter(None, [session.stagedObjectKey, _s3_session_manifest_key(dataset_store, session.uploadId)]):
        try:
            client.delete_object(Bucket=dataset_store.bucket, Key=key)
        except Exception:
            pass


def prepare_dsm_upload(
    *,
    dataset_store: DsmDatasetStore,
    staging_dir: Path,
    base_url: str,
    sha256: str,
    file_size_bytes: int,
    original_name: str,
    content_type: str | None,
) -> DsmPreparedUpload:
    existing_descriptor = dataset_store.get_dataset_descriptor(sha256)
    if existing_descriptor is not None:
        return DsmPreparedUpload(
            status="existing",
            descriptor=existing_descriptor,
            reused_existing=True,
        )

    upload_id = uuid.uuid4().hex
    expires_at_iso = _upload_expires_at_iso()
    normalized_content_type = content_type.strip() if isinstance(content_type, str) and content_type.strip() else None

    if _uses_s3_upload_flow(dataset_store):
        s3_store = dataset_store
        assert isinstance(s3_store, S3BackedDsmDatasetStore)
        staged_object_key = _s3_session_payload_key(s3_store, upload_id, original_name)
        session = DsmUploadSession(
            uploadId=upload_id,
            sha256=sha256,
            fileSizeBytes=file_size_bytes,
            originalName=original_name,
            contentType=normalized_content_type,
            createdAtIso=_utc_now_iso(),
            expiresAtIso=expires_at_iso,
            stagedObjectKey=staged_object_key,
        )
        _write_s3_session(s3_store, session)
        params: dict[str, Any] = {
            "Bucket": s3_store.bucket,
            "Key": staged_object_key,
        }
        headers: dict[str, str] = {}
        if normalized_content_type:
            params["ContentType"] = normalized_content_type
            headers["Content-Type"] = normalized_content_type
        upload_url = s3_store._s3_client().generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=_upload_ttl_sec(),
            HttpMethod="PUT",
        )
        return DsmPreparedUpload(
            status="upload-required",
            upload_id=upload_id,
            upload_target_url=upload_url,
            upload_target_headers=headers,
            expires_at_iso=expires_at_iso,
        )

    _cleanup_expired_local_sessions(staging_dir)
    staged_file_path = _session_payload_path(staging_dir, upload_id, original_name)
    session = DsmUploadSession(
        uploadId=upload_id,
        sha256=sha256,
        fileSizeBytes=file_size_bytes,
        originalName=original_name,
        contentType=normalized_content_type,
        createdAtIso=_utc_now_iso(),
        expiresAtIso=expires_at_iso,
        stagedFilePath=str(staged_file_path),
    )
    _write_local_session(staging_dir, session)
    upload_url = f"{base_url.rstrip('/')}/v1/dsm/upload-sessions/{upload_id}"
    headers: dict[str, str] = {}
    if normalized_content_type:
        headers["Content-Type"] = normalized_content_type
    return DsmPreparedUpload(
        status="upload-required",
        upload_id=upload_id,
        upload_target_url=upload_url,
        upload_target_headers=headers,
        expires_at_iso=expires_at_iso,
    )


async def store_local_upload_payload(*, staging_dir: Path, upload_id: str, request: Request) -> None:
    _cleanup_expired_local_sessions(staging_dir, preserve_upload_id=upload_id)
    session = _read_local_session(staging_dir, upload_id)
    if session.is_expired():
        _delete_local_session(staging_dir, upload_id)
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="DSM upload session has expired.")
    if not session.stagedFilePath:
        raise HTTPException(status_code=400, detail="DSM upload session does not accept local payload uploads.")

    staged_file_path = Path(session.stagedFilePath)
    if staged_file_path.exists() and staged_file_path.stat().st_size > 0:
        raise HTTPException(status_code=409, detail="DSM upload payload already exists for this session.")

    staged_file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = staged_file_path.with_suffix(staged_file_path.suffix + ".part")
    try:
        with temp_path.open("wb") as handle:
            async for chunk in request.stream():
                if not chunk:
                    continue
                handle.write(chunk)
        temp_path.replace(staged_file_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _download_s3_staged_object_to_file(
    dataset_store: S3BackedDsmDatasetStore,
    session: DsmUploadSession,
    staging_dir: Path,
) -> tuple[Path, str, int]:
    if not session.stagedObjectKey:
        raise HTTPException(status_code=400, detail="DSM upload session is missing the staged S3 object key.")
    client = dataset_store._s3_client()
    try:
        body = client.get_object(Bucket=dataset_store.bucket, Key=session.stagedObjectKey)["Body"]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Staged DSM upload payload was not found.") from exc

    suffix = Path(session.originalName).suffix or ".bin"
    temp_dir = _session_dir(staging_dir, session.uploadId)
    temp_path = temp_dir / f"download{suffix}"
    digest = hashlib.sha256()
    total_size = 0
    with temp_path.open("wb") as handle:
        while True:
            chunk = body.read(STREAM_CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            total_size += len(chunk)
    return temp_path, digest.hexdigest(), total_size


def finalize_dsm_upload(
    *,
    dataset_store: DsmDatasetStore,
    staging_dir: Path,
    upload_id: str,
) -> tuple[DsmSourceDescriptorModel, bool]:
    if _uses_s3_upload_flow(dataset_store):
        s3_store = dataset_store
        assert isinstance(s3_store, S3BackedDsmDatasetStore)
        session = _read_s3_session(s3_store, upload_id)
        
        def cleanup_session() -> None:
            _delete_s3_session(s3_store, session)
            _delete_local_session(staging_dir, upload_id)

        if session.is_expired():
            cleanup_session()
            raise HTTPException(status_code=status.HTTP_410_GONE, detail="DSM upload session has expired.")
        try:
            staged_file_path, actual_sha256, actual_size = _download_s3_staged_object_to_file(s3_store, session, staging_dir)
        except Exception:
            cleanup_session()
            raise
    else:
        _cleanup_expired_local_sessions(staging_dir, preserve_upload_id=upload_id)
        session = _read_local_session(staging_dir, upload_id)
        
        def cleanup_session() -> None:
            _delete_local_session(staging_dir, upload_id)

        if session.is_expired():
            cleanup_session()
            raise HTTPException(status_code=status.HTTP_410_GONE, detail="DSM upload session has expired.")
        if not session.stagedFilePath:
            cleanup_session()
            raise HTTPException(status_code=400, detail="DSM upload session is missing the local staged file path.")
        staged_file_path = Path(session.stagedFilePath)
        if not staged_file_path.exists():
            cleanup_session()
            raise HTTPException(status_code=400, detail="Staged DSM upload payload was not found.")
        actual_sha256, actual_size = compute_file_sha256_and_size(staged_file_path)

    try:
        if actual_sha256 != session.sha256:
            raise HTTPException(status_code=400, detail="Uploaded DSM hash did not match the prepared upload session.")
        if actual_size != session.fileSizeBytes:
            raise HTTPException(status_code=400, detail="Uploaded DSM size did not match the prepared upload session.")

        existing_descriptor = dataset_store.get_dataset_descriptor(actual_sha256)
        if existing_descriptor is not None:
            return existing_descriptor, True

        source_descriptor = derive_descriptor_from_path(staged_file_path, session.originalName)
        return dataset_store.ingest_dataset_file(
            staged_file_path,
            session.originalName,
            source_descriptor=source_descriptor,
            verified_sha256=actual_sha256,
        )
    finally:
        cleanup_session()
