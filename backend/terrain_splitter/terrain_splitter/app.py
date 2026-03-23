from __future__ import annotations

import io
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
from .features import compute_feature_field
from .grid import build_grid
from .mapbox_tiles import TerrainTileCache, fetch_dem_for_ring, mapbox_token
from .schemas import (
    DebugArtifacts,
    DsmDatasetListResponse,
    DsmStatusResponse,
    PartitionSolveRequest,
    PartitionSolveResponse,
    TerrainSourceModel,
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


CACHE_DIR = _runtime_dir("TERRAIN_SPLITTER_CACHE_DIR", ".cache")
DEBUG_DIR = _runtime_dir("TERRAIN_SPLITTER_DEBUG_DIR", ".debug")
DSM_DIR = _runtime_dir("TERRAIN_SPLITTER_DSM_DIR", ".dsm")
DSM_DATASET_STORE = create_dsm_dataset_store(DSM_DIR)
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


def _terrain_tile_url_template(request: Request, terrain_source: TerrainSourceModel) -> str:
    query = urlencode(
        {
            "mode": terrain_source.mode,
            **({"datasetId": terrain_source.datasetId} if terrain_source.datasetId else {}),
        }
    )
    return str(request.base_url).rstrip("/") + f"/v1/terrain-rgb/{{z}}/{{x}}/{{y}}.png?{query}"


@app.get("/v1/dsm/datasets", response_model=DsmDatasetListResponse)
def list_dsm_datasets(request: Request) -> DsmDatasetListResponse:
    datasets = [
        DsmStatusResponse(
            datasetId=descriptor.id,
            descriptor=descriptor,
            processingStatus="ready",
            reusedExisting=True,
            terrainTileUrlTemplate=_terrain_tile_url_template(
                request,
                TerrainSourceModel(mode="blended", datasetId=descriptor.id),
            ),
        )
        for descriptor in DSM_DATASET_STORE.list_datasets()
    ]
    return DsmDatasetListResponse(datasets=datasets)


@app.get("/v1/dsm/datasets/{dataset_id}", response_model=DsmStatusResponse)
def get_dsm_dataset(request: Request, dataset_id: str) -> DsmStatusResponse:
    descriptor = DSM_DATASET_STORE.get_dataset_descriptor(dataset_id)
    if descriptor is None:
        raise HTTPException(status_code=404, detail=f"DSM dataset {dataset_id} was not found.")
    return DsmStatusResponse(
        datasetId=descriptor.id,
        descriptor=descriptor,
        processingStatus="ready",
        reusedExisting=True,
        terrainTileUrlTemplate=_terrain_tile_url_template(
            request,
            TerrainSourceModel(mode="blended", datasetId=descriptor.id),
        ),
    )


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

    return DsmStatusResponse(
        datasetId=stored_descriptor.id,
        descriptor=stored_descriptor,
        processingStatus="ready",
        reusedExisting=reused_existing,
        terrainTileUrlTemplate=_terrain_tile_url_template(
            request,
            TerrainSourceModel(mode="blended", datasetId=stored_descriptor.id),
        ),
    )


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
            "[terrain-split-backend][%s] solve request finished polygonId=%s payload=%s solutions=%d fetchDemMs=%.1f buildGridMs=%.1f computeFeaturesMs=%.1f solveMs=%.1f debugArtifactsMs=%.1f totalMs=%.1f",
            request_id,
            request.polygonId or "<none>",
            request.payloadKind,
            len(solutions),
            fetch_dem_ms,
            build_grid_ms,
            compute_features_ms,
            solve_ms,
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
