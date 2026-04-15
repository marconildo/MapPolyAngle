from __future__ import annotations

import base64
import io
import logging
import math
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .costs import line_spacing_for_params
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
from .geometry import ring_to_polygon_mercator
from .grid import build_grid
from .mapbox_tiles import (
    TerrainTileCache,
    choose_grid_step_m,
    fetch_dem_for_ring,
    fetch_dem_for_rings,
    mapbox_token,
)
from .mission_optimizer import optimize_area_sequence
from .schemas import (
    DebugArtifacts,
    DsmFinalizeUploadRequest,
    DsmPrepareUploadRequest,
    DsmPrepareUploadResponse,
    DsmSourceDescriptorModel,
    DsmStatusResponse,
    ExactOptimizeBearingRequest,
    ExactOptimizeBearingResponse,
    MissionOptimizeAreaSequenceRequest,
    MissionOptimizeAreaSequenceResponse,
    PartitionSolutionPreviewModel,
    PartitionSolveRequest,
    PartitionSolveResponse,
    TerrainBatchRequestModel,
    TerrainBatchResponseModel,
    TerrainBatchTileResponseModel,
    TerrainSourceModel,
)
from .solver_graphcut import solve_partition_hierarchy

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parent.parent
LOCAL_RUNTIME_ROOT = Path(
    os.environ.get("TERRAIN_SPLITTER_LOCAL_RUNTIME_ROOT") or (REPO_ROOT / ".terrain-splitter-runtime")
)


def _local_runtime_dir_name(local_dir_name: str) -> str:
    return local_dir_name[1:] if local_dir_name.startswith(".") else local_dir_name


def _migrate_legacy_runtime_dir(local_dir_name: str, destination: Path) -> None:
    legacy_path = BASE_DIR / local_dir_name
    if not legacy_path.exists() or destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_path), str(destination))


def _flatten_nested_legacy_runtime_dir(local_dir_name: str, destination: Path) -> None:
    nested_legacy_path = destination / local_dir_name
    if not nested_legacy_path.exists() or not nested_legacy_path.is_dir():
        return
    for child in nested_legacy_path.iterdir():
        target = destination / child.name
        if child.is_dir() and target.exists() and target.is_dir():
            _merge_directory_contents(child, target)
            continue
        if target.exists():
            continue
        shutil.move(str(child), str(target))
    try:
        nested_legacy_path.rmdir()
    except OSError:
        pass


def _merge_directory_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            if target.exists() and target.is_dir():
                _merge_directory_contents(child, target)
            elif not target.exists():
                shutil.move(str(child), str(target))
        elif not target.exists():
            shutil.move(str(child), str(target))
    try:
        source.rmdir()
    except OSError:
        pass


def _runtime_dir(env_name: str, local_dir_name: str) -> Path:
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured)
    else:
        path = LOCAL_RUNTIME_ROOT / _local_runtime_dir_name(local_dir_name)
        _migrate_legacy_runtime_dir(local_dir_name, path)
    path.mkdir(parents=True, exist_ok=True)
    _flatten_nested_legacy_runtime_dir(local_dir_name, path)
    return path


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _should_enable_internal_http() -> bool:
    default_enabled = not bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    return _env_flag("TERRAIN_SPLITTER_ENABLE_INTERNAL_HTTP", default_enabled)


def _should_use_python_exact_optimize_fanout() -> bool:
    return _env_flag("TERRAIN_SPLITTER_USE_PYTHON_EXACT_OPTIMIZE_FANOUT", False)


CACHE_DIR = _runtime_dir("TERRAIN_SPLITTER_CACHE_DIR", ".cache")
DEBUG_DIR = _runtime_dir("TERRAIN_SPLITTER_DEBUG_DIR", ".debug")
DSM_DIR = _runtime_dir("TERRAIN_SPLITTER_DSM_DIR", ".dsm")
DSM_UPLOAD_STAGING_DIR = _runtime_dir("TERRAIN_SPLITTER_DSM_UPLOAD_STAGING_DIR", ".dsm-upload-staging")
DSM_DATASET_STORE = create_dsm_dataset_store(DSM_DIR)
EXACT_RUNTIME_BRIDGE = create_exact_runtime_bridge()
EXACT_POSTPROCESS_TOP_K = max(1, int(os.environ.get("TERRAIN_SPLITTER_EXACT_TOP_K", "8")))
# Keep these aligned with the exact runtime defaults used by the TS scorer.
EXACT_POSTPROCESS_OPTIMIZE_ZOOM = 14
EXACT_POSTPROCESS_TIME_WEIGHT = 0.1
EXACT_POSTPROCESS_QUALITY_WEIGHT = 1.0 - EXACT_POSTPROCESS_TIME_WEIGHT
EXACT_OPTIMIZE_GLOBAL_COARSE_BEARINGS = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0)
EXACT_OPTIMIZE_LOCAL_COARSE_OFFSETS = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
EXACT_OPTIMIZE_REFINE_STEPS_DEG = (8.0, 4.0, 2.0, 1.0)
EXACT_OPTIMIZE_MIN_IMPROVEMENT = 1e-4
ENABLE_INTERNAL_TERRAIN_BATCH_HTTP = _should_enable_internal_http()
logger = logging.getLogger("uvicorn.error")


def _safe_artifact_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "value"


def _normalize_axial_bearing_deg(value: float) -> float:
    normalized = ((value % 180.0) + 180.0) % 180.0
    return normalized if math.isfinite(normalized) else 0.0


def _rounded_tenths(value: float) -> float:
    return math.floor(value * 10.0 + 0.5) / 10.0


def _candidate_cache_key(bearing_deg: float) -> int:
    return int(round(_normalize_axial_bearing_deg(bearing_deg) * 1000.0))


def _candidate_exact_cost(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict):
        return math.inf
    exact_cost = candidate.get("exactCost")
    return float(exact_cost) if isinstance(exact_cost, (int, float)) and math.isfinite(exact_cost) else math.inf


def _build_exact_optimize_request_payload(request: ExactOptimizeBearingRequest) -> dict[str, Any]:
    return {
        "polygonId": request.polygonId,
        "scopeId": request.polygonId or "optimize-bearing",
        "ring": request.ring,
        "payloadKind": request.payloadKind,
        "params": request.params.model_dump(mode="json"),
        "terrainSource": request.terrainSource.model_dump(mode="json"),
        "altitudeMode": request.altitudeMode,
        "minClearanceM": request.minClearanceM,
        "turnExtendM": request.turnExtendM,
        "exactOptimizeZoom": EXACT_POSTPROCESS_OPTIMIZE_ZOOM,
        "timeWeight": EXACT_POSTPROCESS_TIME_WEIGHT,
    }


def _optimize_bearing_exact_with_bridge_fanout(
    bridge,
    request: ExactOptimizeBearingRequest,
) -> dict[str, Any]:
    request_payload = _build_exact_optimize_request_payload(request)
    normalized_seed_bearing_deg = _normalize_axial_bearing_deg(request.seedBearingDeg)
    half_window_deg = max(1.0, float(request.halfWindowDeg if request.halfWindowDeg is not None else 30.0))
    refine_steps_deg = [step for step in EXACT_OPTIMIZE_REFINE_STEPS_DEG if step <= half_window_deg]
    mode = request.mode
    line_spacing_m = float(line_spacing_for_params(request.params))
    max_inflight = max(1, bridge.candidate_max_inflight())
    candidate_cache: dict[int, tuple[int, Any]] = {}
    next_request_order = 0

    def _register_candidate(executor: ThreadPoolExecutor, bearing_deg: float):
        nonlocal next_request_order
        normalized_bearing_deg = _normalize_axial_bearing_deg(bearing_deg)
        cache_key = _candidate_cache_key(normalized_bearing_deg)
        cached = candidate_cache.get(cache_key)
        if cached is None:
            request_order = next_request_order
            next_request_order += 1
            future = executor.submit(_evaluate_candidate, normalized_bearing_deg)
            cached = (request_order, future)
            candidate_cache[cache_key] = cached
        return normalized_bearing_deg, cached

    def _evaluate_candidate(bearing_deg: float) -> dict[str, Any] | None:
        response_payload = bridge.evaluate_region(
            {**request_payload, "bearingDeg": bearing_deg},
            batch_handle=batch_handle,
        )
        candidate = response_payload.get("candidate")
        return candidate if isinstance(candidate, dict) else None

    def _evaluate_bearings(executor: ThreadPoolExecutor, bearing_degs: list[float]) -> list[dict[str, Any] | None]:
        futures: list[Any] = []
        for bearing_deg in bearing_degs:
            _, cached = _register_candidate(executor, bearing_deg)
            futures.append(cached[1])
        return [future.result() for future in futures]

    batch_handle = bridge.begin_candidate_batch()
    try:
        with ThreadPoolExecutor(max_workers=max_inflight) as executor:
            best: dict[str, Any] | None = None
            best_offset = 0.0

            if mode == "global":
                coarse_bearings = list(dict.fromkeys([*EXACT_OPTIMIZE_GLOBAL_COARSE_BEARINGS, _rounded_tenths(normalized_seed_bearing_deg)]))
                coarse_results = _evaluate_bearings(executor, coarse_bearings)
                for candidate in coarse_results:
                    if candidate is not None and (best is None or _candidate_exact_cost(candidate) < _candidate_exact_cost(best)):
                        best = candidate
            else:
                coarse_offsets = [0.0, *[offset for offset in EXACT_OPTIMIZE_LOCAL_COARSE_OFFSETS if offset != 0.0 and abs(offset) <= half_window_deg + 1e-6]]
                coarse_results = _evaluate_bearings(executor, [normalized_seed_bearing_deg + offset for offset in coarse_offsets])
                for index, candidate in enumerate(coarse_results):
                    if candidate is not None and (best is None or _candidate_exact_cost(candidate) < _candidate_exact_cost(best)):
                        best = candidate
                        best_offset = coarse_offsets[index]

            if best is not None:
                for step_deg in refine_steps_deg:
                    improved = True
                    while improved:
                        improved = False
                        current_best = best
                        if mode == "global":
                            neighbor_specs = [
                                (0.0, float(current_best["bearingDeg"]) - step_deg),
                                (0.0, float(current_best["bearingDeg"]) + step_deg),
                            ]
                        else:
                            neighbor_specs = [
                                (best_offset - step_deg, normalized_seed_bearing_deg + (best_offset - step_deg)),
                                (best_offset + step_deg, normalized_seed_bearing_deg + (best_offset + step_deg)),
                            ]
                        valid_neighbor_specs = [
                            (offset_deg, bearing_deg)
                            for offset_deg, bearing_deg in neighbor_specs
                            if mode == "global" or abs(offset_deg) <= half_window_deg + 1e-6
                        ]
                        neighbor_results = _evaluate_bearings(executor, [bearing_deg for _, bearing_deg in valid_neighbor_specs])
                        next_best: tuple[float, dict[str, Any]] | None = None
                        for index, candidate in enumerate(neighbor_results):
                            if candidate is None:
                                continue
                            if next_best is None or _candidate_exact_cost(candidate) < _candidate_exact_cost(next_best[1]):
                                next_best = (valid_neighbor_specs[index][0], candidate)
                        if next_best is not None and _candidate_exact_cost(next_best[1]) + EXACT_OPTIMIZE_MIN_IMPROVEMENT < _candidate_exact_cost(current_best):
                            best = next_best[1]
                            if mode != "global":
                                best_offset = next_best[0]
                            improved = True

            evaluated_payloads: list[tuple[int, dict[str, Any]]] = []
            for request_order, future in candidate_cache.values():
                candidate = future.result()
                if candidate is not None:
                    evaluated_payloads.append((request_order, candidate))
    finally:
        bridge.end_candidate_batch(batch_handle)

    evaluated_payloads.sort(key=lambda item: (_candidate_exact_cost(item[1]), item[0]))
    return {
        "best": best,
        "evaluated": [candidate for _, candidate in evaluated_payloads],
        "seedBearingDeg": normalized_seed_bearing_deg,
        "lineSpacingM": line_spacing_m,
    }


def _solution_debug_summary(
    solution: PartitionSolutionPreviewModel,
    *,
    surrogate_rank: int,
    exact_requested: bool,
    exact_evaluated: bool,
    not_evaluated_reason: str | None = None,
    exact_rank: int | None = None,
    candidate_error: str | None = None,
) -> dict[str, Any]:
    return {
        "surrogateRank": surrogate_rank,
        "signature": solution.signature,
        "tradeoff": solution.tradeoff,
        "regionCount": solution.regionCount,
        "totalMissionTimeSec": solution.totalMissionTimeSec,
        "normalizedQualityCost": solution.normalizedQualityCost,
        "weightedMeanMismatchDeg": solution.weightedMeanMismatchDeg,
        "hierarchyLevel": solution.hierarchyLevel,
        "largestRegionFraction": solution.largestRegionFraction,
        "meanConvexity": solution.meanConvexity,
        "boundaryBreakAlignment": solution.boundaryBreakAlignment,
        "isFirstPracticalSplit": solution.isFirstPracticalSplit,
        "exactRequested": exact_requested,
        "exactEvaluated": exact_evaluated,
        "notEvaluatedReason": not_evaluated_reason,
        "exactRank": exact_rank,
        "exactScore": solution.exactScore,
        "exactQualityCost": solution.exactQualityCost,
        "exactMissionTimeSec": solution.exactMissionTimeSec,
        "exactMetricKind": solution.exactMetricKind,
        "candidateError": candidate_error,
    }


def _coerce_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _ranked_ms_entries(
    values: dict[str, Any],
    *,
    total_ms: float | None = None,
    top_n: int = 8,
    name_key: str = "name",
    share_key: str = "shareOfTotalMs",
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name, value in values.items():
        numeric = _coerce_finite_float(value)
        if numeric is None or numeric <= 0:
            continue
        entry: dict[str, Any] = {
            name_key: name,
            "ms": round(numeric, 3),
        }
        if total_ms is not None and total_ms > 0:
            entry[share_key] = round(numeric / total_ms, 6)
        entries.append(entry)
    entries.sort(key=lambda item: item["ms"], reverse=True)
    return entries[:top_n]


def _build_request_phase_performance_summary(
    *,
    fetch_dem_ms: float,
    build_grid_ms: float,
    compute_features_ms: float,
    solve_ms: float,
    exact_postprocess_ms: float,
    debug_artifacts_ms: float | None = None,
    total_ms: float | None = None,
) -> dict[str, Any]:
    phases = {
        "solveMs": solve_ms,
        "exactPostprocessMs": exact_postprocess_ms,
        "buildGridMs": build_grid_ms,
        "fetchDemMs": fetch_dem_ms,
        "computeFeaturesMs": compute_features_ms,
    }
    if debug_artifacts_ms is not None:
        phases["debugArtifactsMs"] = debug_artifacts_ms
    known_total_ms = sum(value for value in phases.values() if value > 0)
    effective_total_ms = max(total_ms or 0.0, known_total_ms)
    return {
        "notes": "request phase timings are wall-clock and non-overlapping",
        "knownPhaseTotalMs": round(known_total_ms, 3),
        "totalMs": round(effective_total_ms, 3),
        "unaccountedMs": round(max(0.0, effective_total_ms - known_total_ms), 3),
        "topStages": _ranked_ms_entries(
            phases,
            total_ms=effective_total_ms,
            top_n=6,
            name_key="stage",
            share_key="shareOfRequestMs",
        ),
    }


def _build_solver_performance_summary(
    solver_debug_payload: dict[str, Any] | None,
    *,
    solve_ms: float,
) -> dict[str, Any] | None:
    if not isinstance(solver_debug_payload, dict):
        return None
    solver_summary = solver_debug_payload.get("solverSummary")
    if not isinstance(solver_summary, dict):
        return None
    performance = solver_summary.get("performance")
    if not isinstance(performance, dict):
        return None
    inclusive_timings = {
        key: value
        for key, value in performance.items()
        if key.endswith("_ms")
    }
    objective_ms = _coerce_finite_float(performance.get("objective_ms")) or 0.0
    build_region_ms = _coerce_finite_float(performance.get("build_region_ms")) or 0.0
    flight_time_ms = _coerce_finite_float(performance.get("flight_time_ms")) or 0.0
    exact_geometry_total_ms = _coerce_finite_float(performance.get("exact_geometry_reeval_ms")) or 0.0
    objective_component_timings = {
        "flightTimeMs": _coerce_finite_float(performance.get("flight_time_ms")) or 0.0,
        "lineLiftMs": _coerce_finite_float(performance.get("line_lift_ms")) or 0.0,
        "nodeCostMs": _coerce_finite_float(performance.get("node_cost_ms")) or 0.0,
        "shapeMetricMs": _coerce_finite_float(performance.get("shape_metric_ms")) or 0.0,
    }
    search_stage_timings = {
        "buildRegionMs": build_region_ms,
        "splitCandidateEnumerationMs": _coerce_finite_float(performance.get("split_candidate_enumeration_ms")) or 0.0,
        "recursiveSubsolveMs": _coerce_finite_float(performance.get("recursive_subsolve_ms")) or 0.0,
        "planCombineMs": _coerce_finite_float(performance.get("plan_combine_ms")) or 0.0,
        "frontierPruneMs": _coerce_finite_float(performance.get("frontier_prune_ms")) or 0.0,
        "exactGeometryReevalMs": exact_geometry_total_ms,
        "legacySplitGenerationMs": _coerce_finite_float(performance.get("split_generation_ms")) or 0.0,
    }
    exact_geometry_timings = {
        "regionReevalMs": _coerce_finite_float(performance.get("exact_geometry_region_reeval_ms")) or 0.0,
        "reconstructMs": _coerce_finite_float(performance.get("exact_geometry_reconstruct_ms")) or 0.0,
        "exactRegionObjectiveMs": _coerce_finite_float(performance.get("exact_region_objective_ms")) or 0.0,
    }
    return {
        "notes": "solver performance counters are inclusive and may overlap",
        "topInclusiveMs": _ranked_ms_entries(
            inclusive_timings,
            total_ms=solve_ms,
            top_n=8,
            name_key="metric",
            share_key="shareOfSolveMsInclusive",
        ),
        "flightTimeAnalysis": {
            "flightTimeMs": round(flight_time_ms, 3),
            "objectiveMs": round(objective_ms, 3),
            "buildRegionMs": round(build_region_ms, 3),
            "solveMs": round(solve_ms, 3),
            "shareOfObjectiveMs": round(flight_time_ms / objective_ms, 6) if objective_ms > 0 else None,
            "shareOfBuildRegionMs": round(flight_time_ms / build_region_ms, 6) if build_region_ms > 0 else None,
            "shareOfSolveMs": round(flight_time_ms / solve_ms, 6) if solve_ms > 0 else None,
        },
        "objectiveComponentBreakdown": {
            "notes": "objective component timings are inclusive within objective evaluation",
            "topStages": _ranked_ms_entries(
                objective_component_timings,
                total_ms=objective_ms if objective_ms > 0 else None,
                top_n=6,
                name_key="stage",
                share_key="shareOfObjectiveMs",
            ),
        },
        "searchStageBreakdown": {
            "notes": "search-stage timings are mixed counters gathered during surrogate search; some overlap",
            "topStages": _ranked_ms_entries(
                search_stage_timings,
                total_ms=solve_ms if solve_ms > 0 else None,
                top_n=8,
                name_key="stage",
                share_key="shareOfSolveMs",
            ),
        },
        "exactGeometryBreakdown": {
            "notes": "exact-geometry reevaluation runs after surrogate frontier generation and before app-level exact rerank",
            "planCount": int(_coerce_finite_float(performance.get("exact_geometry_plan_count")) or 0.0),
            "totalMs": round(exact_geometry_total_ms, 3),
            "topStages": _ranked_ms_entries(
                exact_geometry_timings,
                total_ms=exact_geometry_total_ms if exact_geometry_total_ms > 0 else None,
                top_n=6,
                name_key="stage",
                share_key="shareOfExactGeometryMs",
            ),
        },
        "parallel": {
            "rootParallelMs": round(_coerce_finite_float(performance.get("root_parallel_ms")) or 0.0, 3),
            "nestedParallelMs": round(_coerce_finite_float(performance.get("nested_parallel_ms")) or 0.0, 3),
            "rootParallelWorkersUsed": int(_coerce_finite_float(performance.get("root_parallel_workers_used")) or 0.0),
            "nestedParallelWorkersUsedMax": int(_coerce_finite_float(performance.get("nested_parallel_workers_used_max")) or 0.0),
        },
        "cacheHitRates": {
            "regionCacheHitRate": round(
                (_coerce_finite_float(performance.get("region_cache_hits")) or 0.0)
                / max(
                    1.0,
                    (_coerce_finite_float(performance.get("region_cache_hits")) or 0.0)
                    + (_coerce_finite_float(performance.get("region_cache_misses")) or 0.0),
                ),
                6,
            ),
            "regionStaticHitRate": round(
                (_coerce_finite_float(performance.get("region_static_hits")) or 0.0)
                / max(
                    1.0,
                    (_coerce_finite_float(performance.get("region_static_hits")) or 0.0)
                    + (_coerce_finite_float(performance.get("region_static_misses")) or 0.0),
                ),
                6,
            ),
            "regionBearingHitRate": round(
                (_coerce_finite_float(performance.get("region_bearing_hits")) or 0.0)
                / max(
                    1.0,
                    (_coerce_finite_float(performance.get("region_bearing_hits")) or 0.0)
                    + (_coerce_finite_float(performance.get("region_bearing_misses")) or 0.0),
                ),
                6,
            ),
        },
        "counts": {
            "buildRegionCalls": int(_coerce_finite_float(performance.get("build_region_calls")) or 0.0),
            "objectiveCalls": int(_coerce_finite_float(performance.get("objective_calls")) or 0.0),
            "splitAttempts": int(_coerce_finite_float(performance.get("split_attempts")) or 0.0),
            "splitCandidatesReturned": int(_coerce_finite_float(performance.get("split_candidates_returned")) or 0.0),
            "frontierPlanCount": int(_coerce_finite_float(performance.get("frontier_plan_count")) or 0.0),
        },
    }


def _build_exact_candidate_performance_summary(
    *,
    signature: str,
    exact_trace: dict[str, Any] | None,
    bridge_elapsed_ms: float | None = None,
) -> dict[str, Any]:
    timings = exact_trace.get("timings") if isinstance(exact_trace, dict) else {}
    runtime_total_ms = _coerce_finite_float(timings.get("totalElapsedMs")) if isinstance(timings, dict) else None
    preview_elapsed_ms = _coerce_finite_float(timings.get("previewElapsedMs")) if isinstance(timings, dict) else None
    raw_region_elapsed = timings.get("regionSearchElapsedMs") if isinstance(timings, dict) else None
    region_elapsed_ms = [
        numeric
        for value in raw_region_elapsed or []
        if (numeric := _coerce_finite_float(value)) is not None and numeric >= 0
    ]
    region_search_total_ms = sum(region_elapsed_ms)
    runtime_overhead_ms = None
    if runtime_total_ms is not None:
        runtime_overhead_ms = max(0.0, runtime_total_ms - region_search_total_ms - (preview_elapsed_ms or 0.0))
    bridge_overhead_ms = None
    if bridge_elapsed_ms is not None and runtime_total_ms is not None:
        bridge_overhead_ms = max(0.0, bridge_elapsed_ms - runtime_total_ms)
    stage_durations: dict[str, float] = {}
    if region_search_total_ms > 0:
        stage_durations["regionSearchTotalMs"] = region_search_total_ms
    if preview_elapsed_ms is not None and preview_elapsed_ms > 0:
        stage_durations["previewElapsedMs"] = preview_elapsed_ms
    if runtime_overhead_ms is not None and runtime_overhead_ms > 0:
        stage_durations["runtimeOverheadMs"] = runtime_overhead_ms
    if bridge_overhead_ms is not None and bridge_overhead_ms > 0:
        stage_durations["bridgeOverheadMs"] = bridge_overhead_ms
    candidate_total_ms = bridge_elapsed_ms or runtime_total_ms or region_search_total_ms or 0.0
    region_hotspots = [
        {
            "regionIndex": index,
            "ms": round(elapsed_ms, 3),
            "shareOfRegionSearchMs": round(elapsed_ms / region_search_total_ms, 6) if region_search_total_ms > 0 else 0.0,
        }
        for index, elapsed_ms in sorted(
            enumerate(region_elapsed_ms),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    ]
    return {
        "signature": signature,
        "bridgeElapsedMs": round(bridge_elapsed_ms, 3) if bridge_elapsed_ms is not None else None,
        "runtimeTotalElapsedMs": round(runtime_total_ms, 3) if runtime_total_ms is not None else None,
        "regionSearchTotalMs": round(region_search_total_ms, 3),
        "previewElapsedMs": round(preview_elapsed_ms, 3) if preview_elapsed_ms is not None else None,
        "runtimeOverheadMs": round(runtime_overhead_ms, 3) if runtime_overhead_ms is not None else None,
        "bridgeOverheadMs": round(bridge_overhead_ms, 3) if bridge_overhead_ms is not None else None,
        "stageHotspots": _ranked_ms_entries(
            stage_durations,
            total_ms=candidate_total_ms if candidate_total_ms > 0 else None,
            top_n=4,
            name_key="stage",
            share_key="shareOfCandidateMs",
        ),
        "regionHotspots": region_hotspots,
        "longestRegion": region_hotspots[0] if region_hotspots else None,
    }


def _build_exact_rerank_performance_summary(
    *,
    exact_mode: str | None,
    exact_postprocess_ms: float,
    exact_candidates: list[PartitionSolutionPreviewModel],
    exact_rank_by_signature: dict[str, int],
    exact_debug_by_signature: dict[str, dict[str, Any]],
    exact_candidate_elapsed_ms: dict[str, float],
) -> dict[str, Any]:
    candidate_entries: list[dict[str, Any]] = []
    aggregate_stage_totals = {
        "candidateRuntimeTotalMs": 0.0,
        "candidateRegionSearchTotalMs": 0.0,
        "candidatePreviewTotalMs": 0.0,
        "candidateRuntimeOverheadMs": 0.0,
        "candidateBridgeOverheadMs": 0.0,
    }
    total_bridge_ms = 0.0
    for surrogate_rank, candidate in enumerate(exact_candidates, start=1):
        summary = _build_exact_candidate_performance_summary(
            signature=candidate.signature,
            exact_trace=exact_debug_by_signature.get(candidate.signature),
            bridge_elapsed_ms=exact_candidate_elapsed_ms.get(candidate.signature),
        )
        bridge_elapsed_ms = _coerce_finite_float(summary.get("bridgeElapsedMs"))
        runtime_total_ms = _coerce_finite_float(summary.get("runtimeTotalElapsedMs"))
        region_search_total_ms = _coerce_finite_float(summary.get("regionSearchTotalMs")) or 0.0
        preview_elapsed_ms = _coerce_finite_float(summary.get("previewElapsedMs")) or 0.0
        runtime_overhead_ms = _coerce_finite_float(summary.get("runtimeOverheadMs")) or 0.0
        bridge_overhead_ms = _coerce_finite_float(summary.get("bridgeOverheadMs")) or 0.0
        if bridge_elapsed_ms is not None:
            total_bridge_ms += bridge_elapsed_ms
        if runtime_total_ms is not None:
            aggregate_stage_totals["candidateRuntimeTotalMs"] += runtime_total_ms
        aggregate_stage_totals["candidateRegionSearchTotalMs"] += region_search_total_ms
        aggregate_stage_totals["candidatePreviewTotalMs"] += preview_elapsed_ms
        aggregate_stage_totals["candidateRuntimeOverheadMs"] += runtime_overhead_ms
        aggregate_stage_totals["candidateBridgeOverheadMs"] += bridge_overhead_ms
        candidate_entries.append(
            {
                "signature": candidate.signature,
                "surrogateRank": surrogate_rank,
                "exactRank": exact_rank_by_signature.get(candidate.signature),
                "regionCount": candidate.regionCount,
                "bridgeElapsedMs": summary.get("bridgeElapsedMs"),
                "runtimeTotalElapsedMs": summary.get("runtimeTotalElapsedMs"),
                "regionSearchTotalMs": summary.get("regionSearchTotalMs"),
                "previewElapsedMs": summary.get("previewElapsedMs"),
            }
        )
    candidate_entries.sort(
        key=lambda entry: (
            _coerce_finite_float(entry.get("bridgeElapsedMs"))
            or _coerce_finite_float(entry.get("runtimeTotalElapsedMs"))
            or 0.0
        ),
        reverse=True,
    )
    summary: dict[str, Any] = {
        "notes": "exact candidate timings are per-candidate wall-clock and overlap in fanout mode",
        "mode": exact_mode,
        "candidateCount": len(exact_candidates),
        "wallClockExactPostprocessMs": round(exact_postprocess_ms, 3),
        "sumCandidateBridgeElapsedMs": round(total_bridge_ms, 3) if total_bridge_ms > 0 else None,
        "topCandidatesByBridgeElapsedMs": candidate_entries[:5],
        "aggregateStageHotspots": _ranked_ms_entries(
            aggregate_stage_totals,
            total_ms=total_bridge_ms if total_bridge_ms > 0 else None,
            top_n=5,
            name_key="stage",
            share_key="shareOfCandidateBridgeMs",
        ),
    }
    if exact_mode == "fanout" and exact_postprocess_ms > 0 and total_bridge_ms > 0:
        summary["parallelSpeedupEstimate"] = round(total_bridge_ms / exact_postprocess_ms, 3)
    return summary


def _format_hotspot_log_entries(entries: list[dict[str, Any]], *, label_key: str, limit: int = 3) -> str:
    if not entries:
        return "none"
    formatted: list[str] = []
    for entry in entries[:limit]:
        label = entry.get(label_key)
        elapsed_ms = _coerce_finite_float(entry.get("ms") if "ms" in entry else entry.get("bridgeElapsedMs"))
        if label is None or elapsed_ms is None:
            continue
        formatted.append(f"{label}:{elapsed_ms:.1f}ms")
    return ", ".join(formatted) if formatted else "none"

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
        if EXACT_RUNTIME_BRIDGE.supports_candidate_fanout() and _should_use_python_exact_optimize_fanout():
            payload = _optimize_bearing_exact_with_bridge_fanout(EXACT_RUNTIME_BRIDGE, request)
        else:
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


@app.post("/v1/mission/optimize-area-sequence", response_model=MissionOptimizeAreaSequenceResponse)
def optimize_area_sequence_endpoint(
    request: MissionOptimizeAreaSequenceRequest,
) -> MissionOptimizeAreaSequenceResponse:
    request_id = uuid.uuid4().hex[:12]
    try:
        grid_step_m = choose_grid_step_m(
            sum(float(ring_to_polygon_mercator(area.ring).area) for area in request.areas)
        )
        dem, _zoom = fetch_dem_for_rings(
            [area.ring for area in request.areas],
            CACHE_DIR,
            grid_step_m=grid_step_m,
            terrain_source=request.terrainSource,
            dsm_store=DSM_DATASET_STORE,
            lazy_load_missing=True,
        )
        return optimize_area_sequence(request, dem, request_id=request_id)
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
    logger.warning(
        "[terrain-split-autosplit][%s] start polygonId=%s payload=%s terrainMode=%s datasetId=%s exactBridge=%s exactTopK=%d",
        request_id,
        request.polygonId or "<none>",
        request.payloadKind,
        request.terrainSource.mode,
        request.terrainSource.datasetId or "<none>",
        EXACT_RUNTIME_BRIDGE.__class__.__name__ if EXACT_RUNTIME_BRIDGE is not None else "<disabled>",
        EXACT_POSTPROCESS_TOP_K,
    )
    try:
        stage_started_at = time.perf_counter()
        # Keep DEM zoom selection aligned with the area-derived grid resolution.
        grid_step_m = choose_grid_step_m(float(ring_to_polygon_mercator(request.ring).area))
        dem, zoom = fetch_dem_for_ring(
            request.ring,
            CACHE_DIR,
            grid_step_m=grid_step_m,
            terrain_source=request.terrainSource,
            dsm_store=DSM_DATASET_STORE,
        )
        fetch_dem_ms = (time.perf_counter() - stage_started_at) * 1000.0

        stage_started_at = time.perf_counter()
        grid = build_grid(request.ring, dem, grid_step_m=grid_step_m)
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
        polygon_id = request.polygonId or request_id
        terrain_source_payload = request.terrainSource.model_dump(mode="json")
        params_payload = request.params.model_dump(mode="json")
        surrogate_solutions = list(solutions)
        exact_candidates = surrogate_solutions[: min(EXACT_POSTPROCESS_TOP_K, len(surrogate_solutions))]
        exact_solutions: list[PartitionSolutionPreviewModel] = []
        exact_debug_by_signature: dict[str, dict[str, Any]] = {}
        exact_candidate_elapsed_ms: dict[str, float] = {}
        exact_candidate_errors: dict[str, str] = {}
        exact_mode: str | None = None
        exact_status = "skipped"
        exact_skip_reason: str | None = None
        exact_error: str | None = None
        solver_perf_summary = _build_solver_performance_summary(solver_debug_payload, solve_ms=solve_ms)
        if solver_perf_summary is not None and isinstance(solver_debug_payload, dict):
            solver_summary_payload = solver_debug_payload.get("solverSummary")
            if isinstance(solver_summary_payload, dict):
                solver_summary_payload["performanceSummary"] = solver_perf_summary
        if EXACT_RUNTIME_BRIDGE is not None and len(surrogate_solutions) > 1:
            try:
                exact_status = "executed"
                logger.warning(
                    "[terrain-split-autosplit][%s] exact-rerank start polygonId=%s surrogateSolutions=%d candidateSolutions=%d",
                    request_id,
                    request.polygonId or "<none>",
                    len(surrogate_solutions),
                    len(exact_candidates),
                )
                if EXACT_RUNTIME_BRIDGE.supports_candidate_fanout() and len(exact_candidates) > 1:
                    exact_mode = "fanout"
                    max_inflight = min(
                        len(exact_candidates),
                        EXACT_POSTPROCESS_TOP_K,
                        EXACT_RUNTIME_BRIDGE.candidate_max_inflight(),
                    )
                    fastest_mission_time_sec = min(
                        solution.totalMissionTimeSec for solution in exact_candidates
                    )
                    candidate_specs = [
                        (index, solution.model_dump(mode="json"))
                        for index, solution in enumerate(exact_candidates)
                    ]
                    logger.warning(
                        "[terrain-split-autosplit][%s] exact-rerank fanout start polygonId=%s candidateCount=%d maxInflight=%d candidates=%s",
                        request_id,
                        polygon_id,
                        len(candidate_specs),
                        max_inflight,
                        [f"{index}:{payload['signature']}" for index, payload in candidate_specs],
                    )
                    batch_handle = EXACT_RUNTIME_BRIDGE.begin_candidate_batch()

                    def _evaluate_exact_candidate(original_index: int, solution_payload: dict[str, Any]) -> tuple[int, dict[str, Any], float]:
                        candidate_started_at = time.perf_counter()
                        response_payload = EXACT_RUNTIME_BRIDGE.evaluate_solution(
                            {
                                "polygonId": polygon_id,
                                "payloadKind": request.payloadKind,
                                "terrainSource": terrain_source_payload,
                                "params": params_payload,
                                "ring": request.ring,
                                "altitudeMode": request.altitudeMode,
                                "minClearanceM": request.minClearanceM,
                                "turnExtendM": request.turnExtendM,
                                "solution": solution_payload,
                                "fastestMissionTimeSec": fastest_mission_time_sec,
                                "rankingSource": "backend-exact",
                                "exactOptimizeZoom": EXACT_POSTPROCESS_OPTIMIZE_ZOOM,
                                "timeWeight": EXACT_POSTPROCESS_TIME_WEIGHT,
                                "debugTrace": request.debug,
                            },
                            batch_handle=batch_handle,
                        )
                        return original_index, response_payload, (time.perf_counter() - candidate_started_at) * 1000.0

                    candidate_results: list[tuple[int, PartitionSolutionPreviewModel, float]] = []
                    candidate_errors: list[tuple[int, str, Exception]] = []
                    try:
                        with ThreadPoolExecutor(max_workers=max_inflight) as executor:
                            future_to_candidate = {
                                executor.submit(_evaluate_exact_candidate, original_index, solution_payload): (
                                    original_index,
                                    solution_payload["signature"],
                                )
                                for original_index, solution_payload in candidate_specs
                            }
                            for future in as_completed(future_to_candidate):
                                original_index, signature = future_to_candidate[future]
                                try:
                                    resolved_index, response_payload, elapsed_ms = future.result()
                                    solution_payload = response_payload.get("solution")
                                    if not isinstance(solution_payload, dict):
                                        raise RuntimeError("Exact runtime Lambda evaluate-solution returned no solution payload.")
                                    exact_solution = PartitionSolutionPreviewModel.model_validate(solution_payload)
                                    debug_trace = response_payload.get("debugTrace")
                                    if isinstance(debug_trace, dict):
                                        exact_debug_by_signature[signature] = debug_trace
                                    exact_candidate_elapsed_ms[signature] = elapsed_ms
                                    candidate_results.append((resolved_index, exact_solution, elapsed_ms))
                                    logger.warning(
                                        "[terrain-split-autosplit][%s] exact-rerank fanout candidate success polygonId=%s candidateIndex=%d signature=%s elapsedMs=%.1f exactScore=%s",
                                        request_id,
                                        polygon_id,
                                        resolved_index,
                                        signature,
                                        elapsed_ms,
                                        exact_solution.exactScore,
                                    )
                                except Exception as exc:  # noqa: BLE001
                                    candidate_errors.append((original_index, signature, exc))
                                    exact_candidate_errors[signature] = str(exc)
                                    logger.warning(
                                        "[terrain-split-autosplit][%s] exact-rerank fanout candidate failed polygonId=%s candidateIndex=%d signature=%s error=%s",
                                        request_id,
                                        polygon_id,
                                        original_index,
                                        signature,
                                        exc,
                                    )
                    finally:
                        EXACT_RUNTIME_BRIDGE.end_candidate_batch(batch_handle)
                    if candidate_errors or len(candidate_results) != len(candidate_specs):
                        raise RuntimeError(
                            "exact candidate fanout failed for "
                            + ", ".join(
                                f"{index}:{signature}:{error}" for index, signature, error in candidate_errors
                            )
                        )
                    exact_solutions = [
                        solution
                        for _, solution, _ in sorted(
                            candidate_results,
                            key=lambda item: (
                                item[1].exactScore
                                if item[1].exactScore is not None and math.isfinite(item[1].exactScore)
                                else math.inf,
                                item[0],
                            ),
                        )
                    ]
                    logger.warning(
                        "[terrain-split-autosplit][%s] exact-rerank fanout finish polygonId=%s returnedSolutions=%d candidateDurationsMs=%s",
                        request_id,
                        polygon_id,
                        len(exact_solutions),
                        {
                            solution.signature: round(elapsed_ms, 1)
                            for _, solution, elapsed_ms in sorted(candidate_results, key=lambda item: item[0])
                        },
                    )
                else:
                    exact_mode = "single-runtime-rerank"
                    # Exact-enabled responses intentionally replace the surrogate tail with the
                    # exact-reranked shortlist instead of appending discarded surrogate options.
                    exact_response = EXACT_RUNTIME_BRIDGE.rerank_solutions(
                        {
                            "polygonId": polygon_id,
                            "payloadKind": request.payloadKind,
                            "terrainSource": terrain_source_payload,
                            "params": params_payload,
                            "ring": request.ring,
                            "altitudeMode": request.altitudeMode,
                            "minClearanceM": request.minClearanceM,
                            "turnExtendM": request.turnExtendM,
                            "solutions": [solution.model_dump(mode="json") for solution in exact_candidates],
                            "rankingSource": "backend-exact",
                            "exactOptimizeZoom": EXACT_POSTPROCESS_OPTIMIZE_ZOOM,
                            "timeWeight": EXACT_POSTPROCESS_TIME_WEIGHT,
                            "debugTrace": request.debug,
                        }
                    )
                    exact_solutions = [
                        PartitionSolutionPreviewModel.model_validate(solution_payload)
                        for solution_payload in exact_response.get("solutions", [])
                    ]
                    debug_by_signature = exact_response.get("debugBySignature")
                    if isinstance(debug_by_signature, dict):
                        exact_debug_by_signature.update(
                            {
                                str(signature): trace
                                for signature, trace in debug_by_signature.items()
                                if isinstance(trace, dict)
                            }
                        )
                if exact_solutions:
                    solutions = exact_solutions
                logger.warning(
                    "[terrain-split-autosplit][%s] exact-rerank finish polygonId=%s returnedSolutions=%d replaced=%s rankingSources=%s",
                    request_id,
                    request.polygonId or "<none>",
                    len(exact_solutions),
                    bool(exact_solutions),
                    [solution.rankingSource for solution in exact_solutions],
                )
            except Exception as exc:  # noqa: BLE001
                exact_status = "failed"
                exact_error = str(exc)
                logger.exception(
                    "[terrain-split-backend][%s] exact partition rerank failed; using surrogate ordering error=%s",
                    request_id,
                    exc,
                )
                logger.warning(
                    "[terrain-split-autosplit][%s] exact-rerank failed polygonId=%s error=%s",
                    request_id,
                    request.polygonId or "<none>",
                    exc,
                )
        elif EXACT_RUNTIME_BRIDGE is None:
            exact_skip_reason = "no-exact-bridge"
            logger.warning(
                "[terrain-split-autosplit][%s] exact-rerank skipped polygonId=%s reason=no-exact-bridge surrogateSolutions=%d",
                request_id,
                request.polygonId or "<none>",
                len(solutions),
            )
        else:
            exact_skip_reason = "not-enough-solutions"
            logger.warning(
                "[terrain-split-autosplit][%s] exact-rerank skipped polygonId=%s reason=not-enough-solutions surrogateSolutions=%d",
                request_id,
                request.polygonId or "<none>",
                len(solutions),
            )
        exact_postprocess_ms = (time.perf_counter() - stage_started_at) * 1000.0

        debug_payload = None
        request_perf_summary = _build_request_phase_performance_summary(
            fetch_dem_ms=fetch_dem_ms,
            build_grid_ms=build_grid_ms,
            compute_features_ms=compute_features_ms,
            solve_ms=solve_ms,
            exact_postprocess_ms=exact_postprocess_ms,
        )
        exact_rank_by_signature = {
            solution.signature: index + 1 for index, solution in enumerate(exact_solutions)
        }
        exact_rerank_perf_summary = _build_exact_rerank_performance_summary(
            exact_mode=exact_mode,
            exact_postprocess_ms=exact_postprocess_ms,
            exact_candidates=exact_candidates,
            exact_rank_by_signature=exact_rank_by_signature,
            exact_debug_by_signature=exact_debug_by_signature,
            exact_candidate_elapsed_ms=exact_candidate_elapsed_ms,
        )
        if request.debug:
            stage_started_at = time.perf_counter()
            solution_lookup = {solution.signature: solution for solution in solutions}
            exact_artifacts: dict[str, Any] = {}
            summary_candidates = []
            for surrogate_rank, solution in enumerate(surrogate_solutions, start=1):
                signature = solution.signature
                exact_requested = EXACT_RUNTIME_BRIDGE is not None and len(surrogate_solutions) > 1 and surrogate_rank <= len(exact_candidates)
                exact_evaluated = signature in exact_debug_by_signature
                not_evaluated_reason: str | None = None
                if not exact_evaluated:
                    if surrogate_rank > len(exact_candidates):
                        not_evaluated_reason = "outside-top-k"
                    elif exact_skip_reason is not None:
                        not_evaluated_reason = exact_skip_reason
                    elif signature in exact_candidate_errors:
                        not_evaluated_reason = "candidate-failed"
                    elif exact_status == "failed":
                        not_evaluated_reason = "exact-rerank-failed"
                summary_candidates.append(
                    _solution_debug_summary(
                        solution_lookup.get(signature, solution),
                        surrogate_rank=surrogate_rank,
                        exact_requested=exact_requested,
                        exact_evaluated=exact_evaluated,
                        not_evaluated_reason=not_evaluated_reason,
                        exact_rank=exact_rank_by_signature.get(signature),
                        candidate_error=exact_candidate_errors.get(signature),
                    )
                )
                if exact_evaluated:
                    safe_signature = _safe_artifact_component(signature)
                    exact_solution = solution_lookup.get(signature)
                    exact_trace = exact_debug_by_signature[signature]
                    exact_artifacts[f"exact_candidate_{surrogate_rank}_{safe_signature}"] = {
                        "requestId": request_id,
                        "polygonId": polygon_id,
                        "surrogateRank": surrogate_rank,
                        "exactRank": exact_rank_by_signature.get(signature),
                        "signature": signature,
                        "surrogateSolution": solution.model_dump(mode="json"),
                        "refinedSolution": exact_solution.model_dump(mode="json") if exact_solution is not None else None,
                        "runtimeConfig": {
                            "exactOptimizeZoom": exact_trace.get("exactOptimizeZoom", EXACT_POSTPROCESS_OPTIMIZE_ZOOM),
                            "timeWeight": exact_trace.get("timeWeight", EXACT_POSTPROCESS_TIME_WEIGHT),
                            "qualityWeight": exact_trace.get("qualityWeight", EXACT_POSTPROCESS_QUALITY_WEIGHT),
                        },
                        "timings": {
                            "bridgeElapsedMs": round(exact_candidate_elapsed_ms.get(signature), 3)
                            if signature in exact_candidate_elapsed_ms
                            else None,
                            **(exact_trace.get("timings") if isinstance(exact_trace.get("timings"), dict) else {}),
                        },
                        "performanceSummary": _build_exact_candidate_performance_summary(
                            signature=signature,
                            exact_trace=exact_trace,
                            bridge_elapsed_ms=exact_candidate_elapsed_ms.get(signature),
                        ),
                        "partitionScoreBreakdown": exact_trace.get("partitionScoreBreakdown"),
                        "preview": exact_trace.get("preview"),
                        "regions": exact_trace.get("regions"),
                    }
            exact_artifacts["exact_rerank_summary"] = {
                "requestId": request_id,
                "polygonId": request.polygonId,
                "payloadKind": request.payloadKind,
                "topK": len(exact_candidates),
                "exactOptimizeZoom": EXACT_POSTPROCESS_OPTIMIZE_ZOOM,
                "timeWeight": EXACT_POSTPROCESS_TIME_WEIGHT,
                "qualityWeight": EXACT_POSTPROCESS_QUALITY_WEIGHT,
                "rerankMode": exact_mode,
                "exactStatus": exact_status,
                "skipReason": exact_skip_reason,
                "error": exact_error,
                "surrogateSolutionCount": len(surrogate_solutions),
                "candidateSolutions": summary_candidates,
                "exactRankingOrder": [
                    {
                        "exactRank": index + 1,
                        "signature": solution.signature,
                        "exactScore": solution.exactScore,
                    }
                    for index, solution in enumerate(exact_solutions)
                ],
                "returnedOrder": [solution.signature for solution in solutions],
                "winningSignature": solutions[0].signature if solutions else None,
                "performanceSummary": exact_rerank_perf_summary,
            }
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
                    **exact_artifacts,
                    "timing": {
                        "fetchDemMs": round(fetch_dem_ms, 3),
                        "buildGridMs": round(build_grid_ms, 3),
                        "computeFeaturesMs": round(compute_features_ms, 3),
                        "solveMs": round(solve_ms, 3),
                        "exactPostprocessMs": round(exact_postprocess_ms, 3),
                        "totalMs": round((time.perf_counter() - started_at) * 1000.0, 3),
                        "performanceSummary": {
                            "requestPhases": request_perf_summary,
                            "solver": solver_perf_summary,
                            "exactRerank": exact_rerank_perf_summary,
                        },
                    },
                },
            )
            debug_artifacts_ms = (time.perf_counter() - stage_started_at) * 1000.0
            debug_payload = DebugArtifacts(requestId=request_id, artifactPaths=artifacts)
            if artifacts:
                logger.info(
                    "[terrain-split-backend][%s] debug artifacts written dir=%s files=%d",
                    request_id,
                    Path(artifacts[0]).parent,
                    len(artifacts),
                )
            for solution in solutions:
                solution.debug = debug_payload
        else:
            debug_artifacts_ms = 0.0

        total_ms = (time.perf_counter() - started_at) * 1000.0
        final_request_perf_summary = _build_request_phase_performance_summary(
            fetch_dem_ms=fetch_dem_ms,
            build_grid_ms=build_grid_ms,
            compute_features_ms=compute_features_ms,
            solve_ms=solve_ms,
            exact_postprocess_ms=exact_postprocess_ms,
            debug_artifacts_ms=debug_artifacts_ms,
            total_ms=total_ms,
        )
        if request.debug:
            logger.info(
                "[terrain-split-backend][%s] perf hotspots polygonId=%s requestStages=%s solverInclusive=%s exactCandidates=%s",
                request_id,
                request.polygonId or "<none>",
                _format_hotspot_log_entries(
                    final_request_perf_summary.get("topStages", []),
                    label_key="stage",
                ),
                _format_hotspot_log_entries(
                    (solver_perf_summary or {}).get("topInclusiveMs", []),
                    label_key="metric",
                ),
                _format_hotspot_log_entries(
                    exact_rerank_perf_summary.get("topCandidatesByBridgeElapsedMs", []),
                    label_key="signature",
                ),
            )
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
        logger.warning(
            "[terrain-split-autosplit][%s] finish polygonId=%s payload=%s solutions=%d rankingSources=%s fetchDemMs=%.1f solveMs=%.1f exactPostprocessMs=%.1f totalMs=%.1f",
            request_id,
            request.polygonId or "<none>",
            request.payloadKind,
            len(solutions),
            [solution.rankingSource for solution in solutions],
            fetch_dem_ms,
            solve_ms,
            exact_postprocess_ms,
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
