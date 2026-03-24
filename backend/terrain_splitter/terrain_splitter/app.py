from __future__ import annotations

import io
import base64
import logging
import os
import time
import uuid
from urllib.parse import urlencode
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .debug import write_debug_artifacts
from .dsm_store import create_dsm_dataset_store, derive_descriptor_from_payload
from .dsm_uploads import (
    finalize_dsm_upload,
    prepare_dsm_upload,
    store_local_upload_payload,
    uses_presigned_dsm_upload_flow,
)
from .exact_bridge import create_exact_runtime_bridge
from .features import compute_feature_field
from .grid import build_grid
from .mapbox_tiles import TerrainTileCache, fetch_dem_for_ring, mapbox_token
from .schemas import (
    DebugArtifacts,
    DsmSourceDescriptorModel,
    DsmFinalizeUploadRequest,
    DsmPrepareUploadRequest,
    DsmPrepareUploadResponse,
    DsmStatusResponse,
    ExactOptimizeBearingRequest,
    ExactOptimizeBearingResponse,
    PartitionSolveRequest,
    PartitionSolveResponse,
    PartitionSolutionPreviewModel,
    TerrainSourceModel,
    TerrainBatchRequestModel,
    TerrainBatchResponseModel,
    TerrainBatchTileResponseModel,
)
from .solver_graphcut import solve_partition_hierarchy

BASE_DIR = Path(__file__).resolve().parents[1]


def _runtime_dir(env_name: str, local_dir_name: str) -> Path:
    configured = (Path(path) for path in [os.environ.get(env_name)] if path)
    path = next(configured, BASE_DIR / local_dir_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _should_enable_internal_http() -> bool:
    default_enabled = not bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    return _env_flag("TERRAIN_SPLITTER_ENABLE_INTERNAL_HTTP", default_enabled)


CACHE_DIR = _runtime_dir("TERRAIN_SPLITTER_CACHE_DIR", ".cache")
DEBUG_DIR = _runtime_dir("TERRAIN_SPLITTER_DEBUG_DIR", ".debug")
DSM_DIR = _runtime_dir("TERRAIN_SPLITTER_DSM_DIR", ".dsm")
DSM_UPLOAD_STAGING_DIR = _runtime_dir("TERRAIN_SPLITTER_DSM_UPLOAD_STAGING_DIR", ".dsm-upload-staging")
DSM_DATASET_STORE = create_dsm_dataset_store(DSM_DIR)
EXACT_RUNTIME_BRIDGE = create_exact_runtime_bridge()
EXACT_POSTPROCESS_TOP_K = max(1, int(os.environ.get("TERRAIN_SPLITTER_EXACT_TOP_K", "3")))
ENABLE_INTERNAL_TERRAIN_BATCH_HTTP = _should_enable_internal_http()
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Terrain Splitter Backend", version="0.1.0")
if _env_flag("TERRAIN_SPLITTER_ENABLE_APP_CORS", True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


def _normalize_tile_ref(z: int, x: int, y: int) -> tuple[int, int, int]:
    tiles_per_axis = 1 << z
    wrapped_x = ((x % tiles_per_axis) + tiles_per_axis) % tiles_per_axis
    clamped_y = max(0, min(tiles_per_axis - 1, y))
    return z, wrapped_x, clamped_y


def _encode_rgba_png_bytes(rgba: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(out, format="PNG")
    return out.getvalue()


def _fetch_terrain_rgba_tile(
    cache: TerrainTileCache,
    client: httpx.Client,
    terrain_source: TerrainSourceModel,
    z: int,
    x: int,
    y: int,
) -> np.ndarray:
    z, x, y = _normalize_tile_ref(z, x, y)
    payload = cache.get_or_fetch(client, mapbox_token(), z, x, y)
    image = Image.open(io.BytesIO(payload)).convert("RGBA")
    rgba = np.asarray(image, dtype=np.uint8).copy()
    DSM_DATASET_STORE.apply_terrain_source_to_rgba_tile(terrain_source, z, x, y, rgba)
    return rgba


def _build_padded_dem_rgba(
    cache: TerrainTileCache,
    client: httpx.Client,
    terrain_source: TerrainSourceModel,
    z: int,
    x: int,
    y: int,
    pad_tiles: int,
) -> np.ndarray:
    center_rgba = _fetch_terrain_rgba_tile(cache, client, terrain_source, z, x, y)
    tile_size = center_rgba.shape[0]
    if pad_tiles <= 0:
        return center_rgba
    span = pad_tiles * 2 + 1
    dem_size = tile_size * span
    dem_rgba = np.zeros((dem_size, dem_size, 4), dtype=np.uint8)
    for dy in range(-pad_tiles, pad_tiles + 1):
        for dx in range(-pad_tiles, pad_tiles + 1):
            _, nx, ny = _normalize_tile_ref(z, x + dx, y + dy)
            tile_rgba = _fetch_terrain_rgba_tile(cache, client, terrain_source, z, nx, ny)
            offset_x = (dx + pad_tiles) * tile_size
            offset_y = (dy + pad_tiles) * tile_size
            dem_rgba[offset_y : offset_y + tile_size, offset_x : offset_x + tile_size, :] = tile_rgba
    return dem_rgba


def _build_terrain_batch_response(request: TerrainBatchRequestModel) -> TerrainBatchResponseModel:
    cache = TerrainTileCache(CACHE_DIR)
    entries: list[TerrainBatchTileResponseModel] = []
    with httpx.Client(follow_redirects=True) as client:
        for tile in request.tiles:
            z, x, y = _normalize_tile_ref(tile.z, tile.x, tile.y)
            rgba = _fetch_terrain_rgba_tile(cache, client, request.terrainSource, z, x, y)
            png_bytes = _encode_rgba_png_bytes(rgba)
            dem_png_base64: str | None = None
            dem_size: int | None = None
            dem_pad_tiles: int | None = None
            if tile.padTiles and tile.padTiles > 0:
                dem_rgba = _build_padded_dem_rgba(cache, client, request.terrainSource, z, x, y, tile.padTiles)
                dem_png_base64 = base64.b64encode(_encode_rgba_png_bytes(dem_rgba)).decode("ascii")
                dem_size = int(dem_rgba.shape[0])
                dem_pad_tiles = tile.padTiles
            entries.append(
                TerrainBatchTileResponseModel(
                    z=z,
                    x=x,
                    y=y,
                    size=int(rgba.shape[0]),
                    pngBase64=base64.b64encode(png_bytes).decode("ascii"),
                    demPngBase64=dem_png_base64,
                    demSize=dem_size,
                    demPadTiles=dem_pad_tiles,
                )
            )
    return TerrainBatchResponseModel(tiles=entries)


def _terrain_tile_url_template(request: Request, terrain_source: TerrainSourceModel) -> str:
    query = urlencode(
        {
            "mode": terrain_source.mode,
            **({"datasetId": terrain_source.datasetId} if terrain_source.datasetId else {}),
        }
    )
    return str(request.base_url).rstrip("/") + f"/v1/terrain-rgb/{{z}}/{{x}}/{{y}}.png?{query}"


def _dsm_status_response(
    request: Request,
    descriptor: DsmSourceDescriptorModel,
    *,
    reused_existing: bool,
) -> DsmStatusResponse:
    return DsmStatusResponse(
        datasetId=descriptor.id,
        descriptor=descriptor,
        processingStatus="ready",
        reusedExisting=reused_existing,
        terrainTileUrlTemplate=_terrain_tile_url_template(
            request,
            TerrainSourceModel(mode="blended", datasetId=descriptor.id),
        ),
    )


@app.get("/v1/dsm/datasets/{dataset_id}", response_model=DsmStatusResponse)
def get_dsm_dataset(request: Request, dataset_id: str) -> DsmStatusResponse:
    descriptor = DSM_DATASET_STORE.get_dataset_descriptor(dataset_id)
    if descriptor is None:
        raise HTTPException(status_code=404, detail=f"DSM dataset {dataset_id} was not found.")
    return _dsm_status_response(request, descriptor, reused_existing=True)


@app.post("/v1/dsm/upload", response_model=DsmStatusResponse)
async def upload_dsm(request: Request, file: UploadFile = File(...)) -> DsmStatusResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="DSM upload is empty.")

    try:
        server_descriptor = derive_descriptor_from_payload(
            payload,
            file.filename or "surface.tiff",
        )
        stored_descriptor, reused_existing = DSM_DATASET_STORE.ingest_dataset(
            payload,
            file.filename or server_descriptor.name or "surface.tiff",
            server_descriptor,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to ingest DSM: {exc}") from exc

    return _dsm_status_response(request, stored_descriptor, reused_existing=reused_existing)


@app.post("/v1/dsm/prepare-upload", response_model=DsmPrepareUploadResponse)
def prepare_dsm_upload_route(request: Request, payload: DsmPrepareUploadRequest) -> DsmPrepareUploadResponse:
    prepared = prepare_dsm_upload(
        dataset_store=DSM_DATASET_STORE,
        staging_dir=DSM_UPLOAD_STAGING_DIR,
        base_url=str(request.base_url).rstrip("/"),
        sha256=payload.sha256,
        file_size_bytes=payload.fileSizeBytes,
        original_name=payload.originalName,
        content_type=payload.contentType,
    )
    if prepared.status == "existing" and prepared.descriptor is not None:
        return DsmPrepareUploadResponse(
            status="existing",
            dataset=_dsm_status_response(request, prepared.descriptor, reused_existing=prepared.reused_existing),
        )
    return DsmPrepareUploadResponse(
        status="upload-required",
        uploadId=prepared.upload_id,
        uploadTarget={
            "url": prepared.upload_target_url,
            "method": "PUT",
            "headers": prepared.upload_target_headers or {},
            "expiresAtIso": prepared.expires_at_iso,
        },
    )


@app.post("/v1/dsm/finalize-upload", response_model=DsmStatusResponse)
def finalize_dsm_upload_route(request: Request, payload: DsmFinalizeUploadRequest) -> DsmStatusResponse:
    try:
        descriptor, reused_existing = finalize_dsm_upload(
            dataset_store=DSM_DATASET_STORE,
            staging_dir=DSM_UPLOAD_STAGING_DIR,
            upload_id=payload.uploadId,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to ingest DSM: {exc}") from exc
    return _dsm_status_response(request, descriptor, reused_existing=reused_existing)


if not uses_presigned_dsm_upload_flow(DSM_DATASET_STORE):
    @app.put("/v1/dsm/upload-sessions/{upload_id}", status_code=204)
    async def upload_dsm_session_payload(upload_id: str, request: Request) -> Response:
        await store_local_upload_payload(
            staging_dir=DSM_UPLOAD_STAGING_DIR,
            upload_id=upload_id,
            request=request,
        )
        return Response(status_code=204)


@app.get("/v1/terrain-rgb/{z}/{x}/{y}.png")
def terrain_rgb_tile(z: int, x: int, y: int, mode: str = "mapbox", datasetId: str | None = None) -> Response:
    try:
        terrain_source = TerrainSourceModel(mode=mode, datasetId=datasetId)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cache = TerrainTileCache(CACHE_DIR)
    token = mapbox_token()
    with httpx.Client(follow_redirects=True) as client:
        payload = cache.get_or_fetch(client, token, z, x, y)

    image = Image.open(io.BytesIO(payload)).convert("RGBA")
    rgba = np.asarray(image, dtype=np.uint8).copy()
    try:
        changed = DSM_DATASET_STORE.apply_terrain_source_to_rgba_tile(terrain_source, z, x, y, rgba)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    headers = {"Cache-Control": "no-store"}
    if not changed:
        return Response(content=payload, media_type="image/png", headers=headers)

    out = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(out, format="PNG")
    return Response(content=out.getvalue(), media_type="image/png", headers=headers)


if ENABLE_INTERNAL_TERRAIN_BATCH_HTTP:
    @app.post("/v1/internal/terrain-batch", response_model=TerrainBatchResponseModel)
    def terrain_batch(request: TerrainBatchRequestModel) -> TerrainBatchResponseModel:
        return _build_terrain_batch_response(request)


@app.post("/v1/exact/optimize-bearing", response_model=ExactOptimizeBearingResponse)
def optimize_bearing_exact(request: ExactOptimizeBearingRequest) -> ExactOptimizeBearingResponse:
    if EXACT_RUNTIME_BRIDGE is None:
        raise HTTPException(status_code=503, detail="Backend exact optimization is not available.")
    try:
        payload = EXACT_RUNTIME_BRIDGE.optimize_bearing(request.model_dump(mode="json"))
        best = payload.get("best") or {}
        return ExactOptimizeBearingResponse(
            bearingDeg=best.get("bearingDeg"),
            exactScore=best.get("exactCost"),
            qualityCost=best.get("qualityCost"),
            missionTimeSec=best.get("missionTimeSec"),
            normalizedTimeCost=best.get("normalizedTimeCost"),
            metricKind=best.get("metricKind"),
            seedBearingDeg=float(payload.get("seedBearingDeg", request.seedBearingDeg)),
            lineSpacingM=payload.get("lineSpacingM"),
            diagnostics=best.get("diagnostics") or {},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/partition/solve", response_model=PartitionSolveResponse)
def solve_partition(request: PartitionSolveRequest) -> PartitionSolveResponse:
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    logger.info(
        "[terrain-split-backend][%s] solve request start polygonId=%s payload=%s ringPoints=%d tradeoff=%s debug=%s",
        request_id,
        request.polygonId or "<none>",
        request.payloadKind,
        len(request.ring),
        request.tradeoff,
        request.debug,
    )
    try:
        stage_started_at = time.perf_counter()
        dem, zoom = fetch_dem_for_ring(
            request.ring,
            CACHE_DIR,
            terrain_source=request.terrainSource,
            dsm_store=DSM_DATASET_STORE,
        )
        fetch_dem_ms = (time.perf_counter() - stage_started_at) * 1000.0

        stage_started_at = time.perf_counter()
        grid = build_grid(request.ring, dem)
        build_grid_ms = (time.perf_counter() - stage_started_at) * 1000.0

        stage_started_at = time.perf_counter()
        feature_field = compute_feature_field(grid, dem)
        compute_features_ms = (time.perf_counter() - stage_started_at) * 1000.0

        stage_started_at = time.perf_counter()
        solver_debug_payload: dict[str, Any] | None = {} if request.debug else None
        solutions = solve_partition_hierarchy(
            grid,
            feature_field,
            request.params,
            request.tradeoff,
            request_id=request_id,
            polygon_id=request.polygonId,
            debug_output=solver_debug_payload,
        )
        solve_ms = (time.perf_counter() - stage_started_at) * 1000.0

        stage_started_at = time.perf_counter()
        exact_postprocess_ms = 0.0
        if EXACT_RUNTIME_BRIDGE is not None and len(solutions) > 1:
            try:
                exact_candidates = solutions[: min(EXACT_POSTPROCESS_TOP_K, len(solutions))]
                # Exact-enabled responses intentionally replace the surrogate tail with the
                # exact-reranked shortlist instead of appending discarded surrogate options.
                exact_response = EXACT_RUNTIME_BRIDGE.rerank_solutions(
                    {
                        "polygonId": request.polygonId or request_id,
                        "payloadKind": request.payloadKind,
                        "terrainSource": request.terrainSource.model_dump(mode="json"),
                        "params": request.params.model_dump(mode="json"),
                        "ring": request.ring,
                        "altitudeMode": request.altitudeMode,
                        "minClearanceM": request.minClearanceM,
                        "turnExtendM": request.turnExtendM,
                        "solutions": [solution.model_dump(mode="json") for solution in exact_candidates],
                        "rankingSource": "backend-exact",
                    }
                )
                exact_solutions = [
                    PartitionSolutionPreviewModel.model_validate(solution_payload)
                    for solution_payload in exact_response.get("solutions", [])
                ]
                if exact_solutions:
                    solutions = exact_solutions
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[terrain-split-backend][%s] exact partition rerank failed; using surrogate ordering error=%s",
                    request_id,
                    exc,
                )
        exact_postprocess_ms = (time.perf_counter() - stage_started_at) * 1000.0

        debug_payload = None
        if request.debug:
            stage_started_at = time.perf_counter()
            artifacts = write_debug_artifacts(
                DEBUG_DIR,
                request_id,
                {
                    "request": request.model_dump(mode="json"),
                    "grid": {
                        "zoom": zoom,
                        "gridStepM": grid.grid_step_m,
                        "cellCount": len(grid.cells),
                        "edgeCount": len(grid.edges),
                    },
                    "features": {
                        "dominantPreferredBearingDeg": feature_field.dominant_preferred_bearing_deg,
                        "cellCount": len(feature_field.cells),
                        "cells": [
                            {
                                "index": cell.index,
                                "preferredBearingDeg": cell.preferred_bearing_deg,
                                "slopeMagnitude": cell.slope_magnitude,
                                "breakStrength": cell.break_strength,
                                "confidence": cell.confidence,
                            }
                            for cell in feature_field.cells
                        ],
                    },
                    "solutions": [solution.model_dump(mode="json") for solution in solutions],
                    "solver": solver_debug_payload or {},
                    "timing": {
                        "fetchDemMs": round(fetch_dem_ms, 3),
                        "buildGridMs": round(build_grid_ms, 3),
                        "computeFeaturesMs": round(compute_features_ms, 3),
                        "solveMs": round(solve_ms, 3),
                        "exactPostprocessMs": round(exact_postprocess_ms, 3),
                        "totalMs": round((time.perf_counter() - started_at) * 1000.0, 3),
                    },
                },
            )
            debug_artifacts_ms = (time.perf_counter() - stage_started_at) * 1000.0
            debug_payload = DebugArtifacts(requestId=request_id, artifactPaths=artifacts)
            for solution in solutions:
                solution.debug = debug_payload
        else:
            debug_artifacts_ms = 0.0

        total_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "[terrain-split-backend][%s] solve request finished polygonId=%s payload=%s solutions=%d fetchDemMs=%.1f buildGridMs=%.1f computeFeaturesMs=%.1f solveMs=%.1f exactPostprocessMs=%.1f debugArtifactsMs=%.1f totalMs=%.1f",
            request_id,
            request.polygonId or "<none>",
            request.payloadKind,
            len(solutions),
            fetch_dem_ms,
            build_grid_ms,
            compute_features_ms,
            solve_ms,
            exact_postprocess_ms,
            debug_artifacts_ms,
            total_ms,
        )

        return PartitionSolveResponse(
            requestId=request_id,
            solutions=solutions,
            debug=debug_payload,
        )
    except Exception as exc:  # noqa: BLE001
        total_ms = (time.perf_counter() - started_at) * 1000.0
        logger.exception(
            "[terrain-split-backend][%s] solve request failed polygonId=%s payload=%s totalMs=%.1f error=%s",
            request_id,
            request.polygonId or "<none>",
            request.payloadKind,
            total_ms,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
