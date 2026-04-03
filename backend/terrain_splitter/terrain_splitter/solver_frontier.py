from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from shapely.geometry import Polygon
from shapely.wkt import loads as load_wkt

from .costs import RegionObjective, RegionStaticInputs, evaluate_region_objective
from .features import CellFeatures, FeatureField
from .geometry import (
    axial_angle_delta_deg,
    clamp,
    hash_signature,
    normalize_axial_bearing,
    polygon_compactness,
    polygon_convexity,
    polygon_to_lnglat_ring,
    ring_to_polygon_mercator,
    simplify_polygon_coverage,
    weighted_axial_mean_deg,
    weighted_mean,
    weighted_quantile,
)
from .grid import GridCell, GridData, GridEdge
from .postprocess import region_polygon_from_cells
from .schemas import FlightParamsModel, PartitionSolutionPreviewModel, RegionPreview

MAX_REGIONS = 8
MAX_SPLIT_OPTIONS = 8
MAX_FRONTIER_STATES = 20
MIN_CHILD_AREA_FRACTION = 0.14
COARSE_NON_LARGEST_FRACTION_MIN = 0.22
INTER_REGION_TRANSITION_SEC = 45.0
REGION_COUNT_PENALTY = 0.02
DOMINANCE_EPS = 1e-6
DEFAULT_BASIC_LINE_LENGTH_SCALE = 0.35
DEFAULT_PRACTICAL_LINE_LENGTH_SCALE = 0.40
LINE_LENGTH_NEAR_MISS_RATIO_BASIC = 0.10
LINE_LENGTH_NEAR_MISS_RATIO_PRACTICAL = 0.08
LINE_LENGTH_NEAR_MISS_MIN_M = 12.0
LINE_LENGTH_NEAR_MISS_MAX_M = 30.0
RELAXED_HARD_MIN_MEAN_LINE_LENGTH_M = 20.0
RELAXED_FALLBACK_TIME_DIVISOR_SEC = 7_500.0
RELAXED_FALLBACK_SOFT_TOTAL_WEIGHT = 0.01
RELAXED_FALLBACK_SOFT_MAX_WEIGHT = 0.002
MAX_RELAXED_FALLBACK_CANDIDATES = 6
PRACTICAL_FRONTIER_BUCKET_KEEP = 5
NON_PRACTICAL_FRONTIER_BUCKET_KEEP = 3
SPLIT_RANK_TIME_DIVISOR_SEC = 6_000.0
SPLIT_NON_IMPROVING_MAX_QUALITY_REGRESSION = 0.03
SPLIT_NON_IMPROVING_MIN_RANK_SCORE = -0.02
DEFAULT_DEPTH_SMALL = 3
DEFAULT_DEPTH_LARGE = 3
DEFAULT_NESTED_LAMBDA_MIN_DEPTH = 2
DEFAULT_NESTED_LAMBDA_MIN_CELLS = 64
DEFAULT_NESTED_LAMBDA_MAX_INFLIGHT = 8
SPLIT_QUANTILES = (0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80)
OUTPUT_RING_SIMPLIFY_FACTOR = 0.4
OUTPUT_RING_SIMPLIFY_MIN_M = 6.0
OUTPUT_RING_SIMPLIFY_MAX_M = 24.0
logger = logging.getLogger("uvicorn.error")
_LAMBDA_CLIENT: Any | None = None
_LAMBDA_CLIENT_MAX_POOL_CONNECTIONS = 0
_LAMBDA_CLIENT_READ_TIMEOUT_SEC = 0
MAX_PERF_HOTSPOTS = 8


@dataclass(slots=True)
class BoundaryStats:
    shared_boundary_m: float
    break_weight_sum: float


@dataclass(slots=True)
class EvaluatedRegion:
    cell_ids: tuple[int, ...]
    polygon: Polygon
    ring: list[tuple[float, float]]
    objective: RegionObjective
    score: float
    hard_invalid: bool


@dataclass(slots=True)
class RegionStatic:
    cell_ids: tuple[int, ...]
    polygon: Polygon
    ring: list[tuple[float, float]]
    cells: tuple[GridCell, ...]
    area_m2: float
    convexity: float
    compactness: float
    static_inputs: RegionStaticInputs


@dataclass(slots=True)
class RegionBearingCore:
    objective: RegionObjective
    score: float


@dataclass(slots=True)
class SplitCandidate:
    left_ids: tuple[int, ...]
    right_ids: tuple[int, ...]
    boundary: BoundaryStats
    direction_deg: float
    threshold: float
    rank_score: float


@dataclass(slots=True)
class RelaxedFallbackCandidate:
    plan: PartitionPlan
    direction_deg: float
    threshold: float
    soft_total_margin: float
    soft_max_margin: float
    fallback_score: float


@dataclass(slots=True)
class PartitionLeafGeometry:
    cell_ids: tuple[int, ...]


@dataclass(slots=True)
class PartitionSplitGeometry:
    direction_deg: float
    threshold: float
    left: PartitionLeafGeometry | PartitionSplitGeometry
    right: PartitionLeafGeometry | PartitionSplitGeometry


@dataclass(slots=True)
class PartitionPlan:
    regions: tuple[EvaluatedRegion, ...]
    quality_cost: float
    mission_time_sec: float
    weighted_mean_mismatch_deg: float
    internal_boundary_m: float
    break_weight_sum: float
    largest_region_fraction: float
    mean_convexity: float
    region_count: int
    geometry_tree: PartitionLeafGeometry | PartitionSplitGeometry | None = None


@dataclass(slots=True)
class SolverContext:
    grid: GridData
    feature_field: FeatureField
    params: FlightParamsModel
    root_area_m2: float
    feature_lookup: dict[int, CellFeatures]
    cell_lookup: dict[int, GridCell]
    neighbors: dict[int, list[int]]
    basic_line_length_scale: float
    practical_line_length_scale: float


@dataclass(slots=True)
class SolverCaches:
    best_bearing_cache: dict[tuple[int, ...], float]
    region_cache: dict[tuple[int, ...], EvaluatedRegion | None]
    region_static_cache: dict[tuple[int, ...], RegionStatic | None]
    region_bearing_core_cache: dict[tuple[tuple[int, ...], int], RegionBearingCore]
    heading_candidates_cache: dict[tuple[int, ...], list[float]]
    polygon_cache: dict[tuple[int, ...], Polygon | None]
    frontier_cache: dict[tuple[tuple[int, ...], int], list[PartitionPlan]]
    basic_rejection_summary: dict[str, Any]
    basic_split_rejection_summary: dict[str, Any]


@dataclass(slots=True)
class RootSplitTask:
    left_ids: tuple[int, ...]
    right_ids: tuple[int, ...]
    boundary: BoundaryStats
    depth: int
    direction_deg: float = 0.0
    threshold: float = 0.0


@dataclass(slots=True)
class SubtreeSolveTask:
    cell_ids: tuple[int, ...]
    depth: int


_PARALLEL_SOLVER_CONTEXT: SolverContext | None = None


def _make_solver_caches() -> SolverCaches:
    return SolverCaches(
        best_bearing_cache={},
        region_cache={},
        region_static_cache={},
        region_bearing_core_cache={},
        heading_candidates_cache={},
        polygon_cache={},
        frontier_cache={},
        basic_rejection_summary={},
        basic_split_rejection_summary={},
    )


def _make_perf() -> defaultdict[str, float]:
    return defaultdict(float)


def _merge_perf(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] += value


def _perf_hotspot_summary(
    entries: list[dict[str, Any]] | None,
    *,
    label_key: str,
    limit: int = 4,
) -> str:
    if not entries:
        return "none"
    parts: list[str] = []
    for entry in entries[:limit]:
        label = entry.get(label_key, "<unknown>")
        elapsed_ms = float(entry.get("elapsedMs", 0.0) or 0.0)
        cell_count = int(entry.get("cellCount", 0) or 0)
        candidate_count = int(entry.get("candidateBearingCount", 0) or 0)
        parts.append(
            f"{label}@{elapsed_ms:.1f}ms cells={cell_count} bearings={candidate_count}"
        )
    return ", ".join(parts)


def _record_perf_hotspot(
    hotspots: dict[str, list[dict[str, Any]]] | None,
    category: str,
    entry: dict[str, Any],
) -> None:
    if hotspots is None:
        return
    bucket = hotspots.setdefault(category, [])
    bucket.append(entry)
    bucket.sort(key=lambda item: float(item.get("elapsedMs", 0.0) or 0.0), reverse=True)
    del bucket[MAX_PERF_HOTSPOTS:]


def _log_perf_hotspots(
    request_id: str | None,
    polygon_id: str | None,
    hotspots: dict[str, list[dict[str, Any]]] | None,
) -> None:
    if not hotspots:
        return
    logger.info(
        "[terrain-split-backend][%s] coarse profiling polygonId=%s buildRegion=%s objective=%s",
        request_id or "<none>",
        polygon_id or "<none>",
        _perf_hotspot_summary(hotspots.get("buildRegion"), label_key="regionSignature"),
        _perf_hotspot_summary(hotspots.get("objective"), label_key="regionSignature"),
    )


def _resolve_root_parallel_workers(requested: int | None) -> int:
    if requested is not None:
        return max(0, int(requested))
    raw = os.environ.get("TERRAIN_SPLITTER_ROOT_PARALLEL_WORKERS")
    if raw is None or raw.strip() == "":
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "[terrain-split-backend] invalid TERRAIN_SPLITTER_ROOT_PARALLEL_WORKERS=%r; falling back to serial",
            raw,
        )
        return 0


def _resolve_root_parallel_mode(requested: Literal["process", "lambda"] | None) -> Literal["process", "lambda"]:
    if requested in {"process", "lambda"}:
        return requested
    raw = (os.environ.get("TERRAIN_SPLITTER_ROOT_PARALLEL_MODE") or "process").strip().lower()
    if raw == "lambda":
        return "lambda"
    return "process"


def _resolve_root_parallel_granularity(requested: Literal["branch", "subtree"] | None) -> Literal["branch", "subtree"]:
    if requested in {"branch", "subtree"}:
        return requested
    raw = (os.environ.get("TERRAIN_SPLITTER_ROOT_PARALLEL_GRANULARITY") or "branch").strip().lower()
    if raw == "subtree":
        return "subtree"
    return "branch"


def _resolve_root_parallel_max_inflight(requested: int | None) -> int | None:
    if requested is not None:
        return max(0, int(requested))
    raw = os.environ.get("TERRAIN_SPLITTER_ROOT_PARALLEL_MAX_INFLIGHT")
    if raw is None or raw.strip() == "":
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "[terrain-split-backend] invalid TERRAIN_SPLITTER_ROOT_PARALLEL_MAX_INFLIGHT=%r; falling back to worker limit",
            raw,
        )
        return None


def _resolve_lambda_invoke_read_timeout_sec(requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
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


def _resolve_nested_lambda_min_cells(requested: int | None) -> int:
    if requested is not None:
        return max(1, int(requested))
    raw = os.environ.get("TERRAIN_SPLITTER_NESTED_LAMBDA_MIN_CELLS")
    if raw is None or raw.strip() == "":
        return DEFAULT_NESTED_LAMBDA_MIN_CELLS
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "[terrain-split-backend] invalid TERRAIN_SPLITTER_NESTED_LAMBDA_MIN_CELLS=%r; falling back to %d",
            raw,
            DEFAULT_NESTED_LAMBDA_MIN_CELLS,
        )
        return DEFAULT_NESTED_LAMBDA_MIN_CELLS


def _resolve_nested_lambda_max_inflight(requested: int | None) -> int:
    if requested is not None:
        return max(0, int(requested))
    raw = os.environ.get("TERRAIN_SPLITTER_NESTED_LAMBDA_MAX_INFLIGHT")
    if raw is None or raw.strip() == "":
        return DEFAULT_NESTED_LAMBDA_MAX_INFLIGHT
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "[terrain-split-backend] invalid TERRAIN_SPLITTER_NESTED_LAMBDA_MAX_INFLIGHT=%r; falling back to %d",
            raw,
            DEFAULT_NESTED_LAMBDA_MAX_INFLIGHT,
        )
        return DEFAULT_NESTED_LAMBDA_MAX_INFLIGHT



def _resolve_lambda_parallel_invocations(
    task_count: int,
    fallback_workers: int,
    max_inflight: int | None,
) -> int:
    if task_count <= 0:
        return 0
    if max_inflight is None:
        return min(task_count, max(1, fallback_workers))
    if max_inflight <= 0:
        return task_count
    return min(task_count, max_inflight)


def _init_parallel_solver_context(context: SolverContext) -> None:
    global _PARALLEL_SOLVER_CONTEXT
    _PARALLEL_SOLVER_CONTEXT = context


def _feature_lookup(feature_field: FeatureField) -> dict[int, CellFeatures]:
    return {cell.index: cell for cell in feature_field.cells}


def _cell_lookup(grid: GridData) -> dict[int, GridCell]:
    return {cell.index: cell for cell in grid.cells}


def _neighbor_lookup(grid: GridData) -> dict[int, list[int]]:
    neighbors: dict[int, list[int]] = defaultdict(list)
    for edge in grid.edges:
        neighbors[edge.a].append(edge.b)
        neighbors[edge.b].append(edge.a)
    return neighbors


def _weighted_circular_mean_deg(values: list[tuple[float, float]]) -> float | None:
    sum_sin = 0.0
    sum_cos = 0.0
    total = 0.0
    for angle_deg, weight in values:
        if not math.isfinite(angle_deg) or weight <= 0:
            continue
        rad = math.radians(angle_deg % 360.0)
        sum_sin += math.sin(rad) * weight
        sum_cos += math.cos(rad) * weight
        total += weight
    if total <= 0:
        return None
    return (math.degrees(math.atan2(sum_sin, sum_cos)) + 360.0) % 360.0


def _region_signature(region: EvaluatedRegion) -> tuple[tuple[int, ...], int]:
    return region.cell_ids, int(round(region.objective.bearing_deg * 1000.0))


def _plan_signature(plan: PartitionPlan) -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple(sorted((_region_signature(region) for region in plan.regions), key=lambda item: (len(item[0]), item[1])))


def _boundary_alignment(boundary: BoundaryStats) -> float:
    if boundary.shared_boundary_m <= 0:
        return 0.0
    return boundary.break_weight_sum / boundary.shared_boundary_m


def _bearing_cache_key(bearing_deg: float) -> int:
    return int(round(normalize_axial_bearing(bearing_deg) * 1000.0))


def _region_score(objective: RegionObjective) -> float:
    compactness_penalty = max(0.0, objective.compactness - 3.4) * 0.06
    convexity_penalty = max(0.0, 0.75 - objective.convexity) * 0.35
    fragmentation_penalty = 0.18 * objective.fragmented_line_fraction + 0.28 * objective.overflight_transit_fraction
    short_line_penalty = 0.12 * objective.short_line_fraction
    time_penalty = objective.total_mission_time_sec / 55_000.0
    return (
        objective.normalized_quality_cost
        + compactness_penalty
        + convexity_penalty
        + fragmentation_penalty
        + short_line_penalty
        + time_penalty
    )


def _capture_efficiency(objective: RegionObjective) -> float:
    return clamp(
        1.0
        - 0.65 * objective.fragmented_line_fraction
        - 0.85 * objective.overflight_transit_fraction
        - 0.45 * objective.short_line_fraction,
        0.0,
        1.0,
    )


def _line_length_tolerance_m(min_line_length_m: float, *, practical: bool) -> float:
    ratio = LINE_LENGTH_NEAR_MISS_RATIO_PRACTICAL if practical else LINE_LENGTH_NEAR_MISS_RATIO_BASIC
    return clamp(min_line_length_m * ratio, LINE_LENGTH_NEAR_MISS_MIN_M, LINE_LENGTH_NEAR_MISS_MAX_M)


def _region_gate_thresholds(*, practical: bool, line_length_scale: float) -> dict[str, float]:
    return {
        "min_child_fraction": MIN_CHILD_AREA_FRACTION,
        "min_flight_line_count": 1.0,
        "line_length_scale": line_length_scale,
        "line_length_floor_m": 10.0 if practical else 8.0,
        "min_convexity": 0.64 if practical else 0.56,
        "max_compactness": 6.25 if practical else 8.5,
        "min_capture_efficiency": 0.22 if practical else 0.12,
        "max_overflight_transit_fraction": 0.55 if practical else 1.0,
    }


def _region_gate_diagnostics(
    region: EvaluatedRegion,
    area_reference_m2: float,
    *,
    practical: bool,
    line_length_scale: float,
) -> dict[str, float | int | bool]:
    thresholds = _region_gate_thresholds(practical=practical, line_length_scale=line_length_scale)
    fraction = region.objective.area_m2 / max(1.0, area_reference_m2)
    line_length_span = max(1.0, region.objective.along_track_length_m, region.objective.cross_track_width_m)
    min_line_length = max(
        thresholds["line_length_floor_m"],
        thresholds["line_length_scale"] * line_length_span,
    )
    line_length_shortfall = max(0.0, min_line_length - region.objective.mean_line_length_m)
    line_length_tolerance = _line_length_tolerance_m(min_line_length, practical=practical)
    return {
        "hard_invalid": region.hard_invalid,
        "fraction": fraction,
        "min_child_fraction": thresholds["min_child_fraction"],
        "flight_line_count": region.objective.flight_line_count,
        "min_flight_line_count": int(thresholds["min_flight_line_count"]),
        "mean_line_length_m": region.objective.mean_line_length_m,
        "min_line_length_m": min_line_length,
        "line_length_shortfall_m": line_length_shortfall,
        "line_length_tolerance_m": line_length_tolerance,
        "line_length_near_miss_allowed": 0.0 < line_length_shortfall <= line_length_tolerance,
        "along_track_length_m": region.objective.along_track_length_m,
        "cross_track_width_m": region.objective.cross_track_width_m,
        "convexity": region.objective.convexity,
        "min_convexity": thresholds["min_convexity"],
        "compactness": region.objective.compactness,
        "max_compactness": thresholds["max_compactness"],
        "capture_efficiency": _capture_efficiency(region.objective),
        "min_capture_efficiency": thresholds["min_capture_efficiency"],
        "overflight_transit_fraction": region.objective.overflight_transit_fraction,
        "max_overflight_transit_fraction": thresholds["max_overflight_transit_fraction"],
        "bearing_deg": region.objective.bearing_deg,
        "area_m2": region.objective.area_m2,
    }


def _region_gate_failure_margins(diagnostics: dict[str, float | int | bool], *, practical: bool) -> dict[str, float]:
    failures: dict[str, float] = {}
    if diagnostics["hard_invalid"]:
        failures["hard_invalid"] = 1.0
    if diagnostics["fraction"] < diagnostics["min_child_fraction"]:
        failures["min_child_fraction"] = float(diagnostics["min_child_fraction"] - diagnostics["fraction"])
    if diagnostics["flight_line_count"] < diagnostics["min_flight_line_count"]:
        failures["flight_line_count"] = float(diagnostics["min_flight_line_count"] - diagnostics["flight_line_count"])
    if diagnostics["convexity"] < diagnostics["min_convexity"]:
        failures["convexity"] = float(diagnostics["min_convexity"] - diagnostics["convexity"])
    if diagnostics["compactness"] > diagnostics["max_compactness"]:
        failures["compactness"] = float(diagnostics["compactness"] - diagnostics["max_compactness"])
    if diagnostics["capture_efficiency"] < diagnostics["min_capture_efficiency"]:
        failures["capture_efficiency"] = float(diagnostics["min_capture_efficiency"] - diagnostics["capture_efficiency"])
    if practical and diagnostics["overflight_transit_fraction"] > diagnostics["max_overflight_transit_fraction"]:
        failures["overflight_transit_fraction"] = float(
            diagnostics["overflight_transit_fraction"] - diagnostics["max_overflight_transit_fraction"]
        )
    line_length_shortfall = float(diagnostics["line_length_shortfall_m"])
    if line_length_shortfall > 0.0:
        line_length_tolerance = float(diagnostics["line_length_tolerance_m"])
        if line_length_shortfall > line_length_tolerance or failures:
            failures["mean_line_length_m"] = line_length_shortfall
    return failures


def _region_gate_passes(diagnostics: dict[str, float | int | bool], *, practical: bool) -> bool:
    return not _region_gate_failure_margins(diagnostics, practical=practical)


def _plan_gate_thresholds() -> dict[str, float]:
    return {
        "min_non_largest_fraction": COARSE_NON_LARGEST_FRACTION_MIN,
        "min_child_fraction": MIN_CHILD_AREA_FRACTION,
        "min_mean_convexity": 0.65,
        "max_two_region_largest_fraction": 0.88,
    }


def _plan_gate_diagnostics(plan: PartitionPlan, root_area_m2: float, *, line_length_scale: float) -> dict[str, Any]:
    fractions = sorted((region.objective.area_m2 / max(1.0, root_area_m2) for region in plan.regions), reverse=True)
    thresholds = _plan_gate_thresholds()
    region_diags = [
        _region_gate_diagnostics(region, root_area_m2, practical=True, line_length_scale=line_length_scale)
        for region in plan.regions
    ]
    return {
        "region_count": plan.region_count,
        "largest_region_fraction": fractions[0] if fractions else 1.0,
        "non_largest_fraction": (1.0 - fractions[0]) if fractions else 0.0,
        "smallest_region_fraction": fractions[-1] if fractions else 0.0,
        "mean_convexity": plan.mean_convexity,
        "min_non_largest_fraction": thresholds["min_non_largest_fraction"],
        "min_child_fraction": thresholds["min_child_fraction"],
        "min_mean_convexity": thresholds["min_mean_convexity"],
        "max_two_region_largest_fraction": thresholds["max_two_region_largest_fraction"],
        "quality_cost": plan.quality_cost,
        "mission_time_sec": plan.mission_time_sec,
        "region_diagnostics": region_diags,
    }


def _plan_gate_failure_margins(diagnostics: dict[str, Any]) -> tuple[dict[str, float], list[tuple[int, dict[str, float | int | bool], dict[str, float]]]]:
    failures: dict[str, float] = {}
    region_failures: list[tuple[int, dict[str, float | int | bool], dict[str, float]]] = []
    if diagnostics["region_count"] <= 1:
        failures["region_count"] = 1.0
        return failures, region_failures
    if diagnostics["non_largest_fraction"] < diagnostics["min_non_largest_fraction"]:
        failures["non_largest_fraction"] = float(
            diagnostics["min_non_largest_fraction"] - diagnostics["non_largest_fraction"]
        )
    if diagnostics["smallest_region_fraction"] < diagnostics["min_child_fraction"]:
        failures["smallest_region_fraction"] = float(
            diagnostics["min_child_fraction"] - diagnostics["smallest_region_fraction"]
        )
    if diagnostics["mean_convexity"] < diagnostics["min_mean_convexity"]:
        failures["mean_convexity"] = float(diagnostics["min_mean_convexity"] - diagnostics["mean_convexity"])
    if diagnostics["region_count"] == 2 and diagnostics["largest_region_fraction"] > diagnostics["max_two_region_largest_fraction"]:
        failures["largest_region_fraction"] = float(
            diagnostics["largest_region_fraction"] - diagnostics["max_two_region_largest_fraction"]
        )

    for index, region_diag in enumerate(diagnostics["region_diagnostics"]):
        region_reasons = _region_gate_failure_margins(region_diag, practical=True)
        if region_reasons:
            region_failures.append((index, region_diag, region_reasons))
    if region_failures:
        failures["region_practicality"] = 1.0
    return failures, region_failures


def _summarize_diagnostic_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _rounded_debug_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _summarize_diagnostic_value(value) for key, value in values.items()}


def _rejection_debug_payload(
    *,
    caches: SolverCaches,
    basic_line_length_scale: float,
    practical_line_length_scale: float,
    practical_plan_rejection_summary: dict[str, Any],
    practical_region_rejection_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "basicRegionValidity": {
            "thresholds": _region_gate_thresholds(practical=False, line_length_scale=basic_line_length_scale),
            **caches.basic_rejection_summary,
        },
        "basicSplitValidity": {
            "thresholds": _region_gate_thresholds(practical=False, line_length_scale=basic_line_length_scale),
            **caches.basic_split_rejection_summary,
        },
        "practicalPlan": {
            "thresholds": _plan_gate_thresholds(),
            **practical_plan_rejection_summary,
        },
        "practicalRegion": {
            "thresholds": _region_gate_thresholds(practical=True, line_length_scale=practical_line_length_scale),
            **practical_region_rejection_summary,
        },
    }


def _plan_debug_signature(plan: PartitionPlan) -> str:
    return hash_signature(
        {
            "regions": [
                {
                    "cellIds": list(region.cell_ids),
                    "bearingDeg": round(region.objective.bearing_deg, 4),
                }
                for region in plan.regions
            ]
        }
    )


def _serialize_plan_debug_snapshot(
    plan: PartitionPlan,
    root_area_m2: float,
    *,
    line_length_scale: float,
) -> dict[str, Any]:
    diagnostics = _plan_gate_diagnostics(plan, root_area_m2, line_length_scale=line_length_scale)
    plan_failures, region_failures = _plan_gate_failure_margins(diagnostics)
    region_failure_lookup = {
        index: {
            "diagnostics": region_diag,
            "failures": failures,
        }
        for index, region_diag, failures in region_failures
    }
    regions: list[dict[str, Any]] = []
    for index, (region, region_diag) in enumerate(zip(plan.regions, diagnostics["region_diagnostics"], strict=True)):
        failure_payload = region_failure_lookup.get(index, {})
        regions.append(
            {
                "regionIndex": index,
                "areaM2": round(region.objective.area_m2, 3),
                "bearingDeg": round(region.objective.bearing_deg, 4),
                "atomCount": len(region.cell_ids),
                "captureEfficiency": _summarize_diagnostic_value(region_diag["capture_efficiency"]),
                "meanLineLengthM": _summarize_diagnostic_value(region_diag["mean_line_length_m"]),
                "minLineLengthM": _summarize_diagnostic_value(region_diag["min_line_length_m"]),
                "lineLengthShortfallM": _summarize_diagnostic_value(region_diag["line_length_shortfall_m"]),
                "lineLengthToleranceM": _summarize_diagnostic_value(region_diag["line_length_tolerance_m"]),
                "lineLengthNearMissAllowed": bool(region_diag["line_length_near_miss_allowed"]),
                "convexity": _summarize_diagnostic_value(region_diag["convexity"]),
                "compactness": _summarize_diagnostic_value(region_diag["compactness"]),
                "overflightTransitFraction": _summarize_diagnostic_value(region_diag["overflight_transit_fraction"]),
                "failures": _rounded_debug_mapping(failure_payload.get("failures", {})),
            }
        )
    return {
        "signature": _plan_debug_signature(plan),
        "regionCount": plan.region_count,
        "qualityCost": round(plan.quality_cost, 6),
        "missionTimeSec": round(plan.mission_time_sec, 3),
        "largestRegionFraction": round(plan.largest_region_fraction, 4),
        "meanConvexity": round(plan.mean_convexity, 4),
        "boundaryBreakAlignment": round(_plan_boundary_alignment(plan), 4),
        "isPractical": not plan_failures,
        "planFailures": _rounded_debug_mapping(plan_failures),
        "regions": regions,
    }


def _populate_solver_debug_output(
    debug_output: dict[str, Any] | None,
    *,
    request_id: str | None,
    polygon_id: str | None,
    grid: GridData,
    requested_tradeoff: float | None,
    max_depth: int,
    basic_line_length_scale: float,
    practical_line_length_scale: float,
    caches: SolverCaches,
    perf: dict[str, float],
    all_plans: list[PartitionPlan],
    practical_plans: list[PartitionPlan],
    returned_plans: list[PartitionPlan],
    returned_previews: list[PartitionSolutionPreviewModel],
    practical_plan_rejection_summary: dict[str, Any],
    practical_region_rejection_summary: dict[str, Any],
    relaxed_fallback: list[RelaxedFallbackCandidate] | None = None,
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    if debug_output is None:
        return
    root_area_m2 = max(1.0, grid.area_m2)
    debug_output.clear()
    debug_output.update(
        {
            "solverSummary": {
                "requestId": request_id,
                "polygonId": polygon_id,
                "requestedTradeoff": requested_tradeoff,
                "gridCellCount": len(grid.cells),
                "gridEdgeCount": len(grid.edges),
                "gridStepM": round(grid.grid_step_m, 4),
                "maxDepth": max_depth,
                "counts": {
                    "allPlans": len(all_plans),
                    "practicalPlans": len(practical_plans),
                    "returnedSolutions": len(returned_previews),
                    "relaxedFallbackCandidates": len(relaxed_fallback or []),
                },
                "constants": {
                    "maxSplitOptions": MAX_SPLIT_OPTIONS,
                    "maxFrontierStates": MAX_FRONTIER_STATES,
                    "interRegionTransitionSec": INTER_REGION_TRANSITION_SEC,
                    "regionCountPenalty": REGION_COUNT_PENALTY,
                    "defaultDepthSmall": DEFAULT_DEPTH_SMALL,
                    "defaultDepthLarge": DEFAULT_DEPTH_LARGE,
                    "basicLineLengthScale": basic_line_length_scale,
                    "practicalLineLengthScale": practical_line_length_scale,
                    "lineLengthNearMissRatioBasic": LINE_LENGTH_NEAR_MISS_RATIO_BASIC,
                    "lineLengthNearMissRatioPractical": LINE_LENGTH_NEAR_MISS_RATIO_PRACTICAL,
                    "defaultNestedLambdaMinDepth": DEFAULT_NESTED_LAMBDA_MIN_DEPTH,
                    "defaultNestedLambdaMinCells": DEFAULT_NESTED_LAMBDA_MIN_CELLS,
                    "defaultNestedLambdaMaxInflight": DEFAULT_NESTED_LAMBDA_MAX_INFLIGHT,
                },
                "performance": _rounded_debug_mapping(dict(perf)),
                "performanceHotspots": perf_hotspots or {},
            },
            "rejectionDiagnostics": _rejection_debug_payload(
                caches=caches,
                basic_line_length_scale=basic_line_length_scale,
                practical_line_length_scale=practical_line_length_scale,
                practical_plan_rejection_summary=practical_plan_rejection_summary,
                practical_region_rejection_summary=practical_region_rejection_summary,
            ),
            "returnedPreviewSignatures": [preview.signature for preview in returned_previews],
            "returnedPlans": [
                _serialize_plan_debug_snapshot(plan, root_area_m2, line_length_scale=practical_line_length_scale)
                for plan in returned_plans
            ],
            "practicalPlanSample": [
                _serialize_plan_debug_snapshot(plan, root_area_m2, line_length_scale=practical_line_length_scale)
                for plan in practical_plans[: min(len(practical_plans), 12)]
            ],
        }
    )
    if relaxed_fallback:
        debug_output["relaxedFallback"] = [
            {
                "signature": _plan_debug_signature(candidate.plan),
                "directionDeg": round(candidate.direction_deg, 4),
                "threshold": round(candidate.threshold, 4),
                "softTotalMargin": round(candidate.soft_total_margin, 4),
                "softMaxMargin": round(candidate.soft_max_margin, 4),
                "fallbackScore": round(candidate.fallback_score, 6),
            }
            for candidate in relaxed_fallback
        ]


def _update_failure_summary(
    summary: dict[str, Any],
    reason: str,
    margin: float,
    snapshot: dict[str, Any],
) -> None:
    counts = summary.setdefault("counts", {})
    counts[reason] = int(counts.get(reason, 0)) + 1
    closest = summary.setdefault("closest", {})
    existing = closest.get(reason)
    if existing is None or margin < existing["margin"]:
        closest[reason] = {
            "margin": round(margin, 4),
            "snapshot": {key: _summarize_diagnostic_value(value) for key, value in snapshot.items()},
        }


def _record_region_failure_summary(
    summary: dict[str, Any],
    region: EvaluatedRegion,
    diagnostics: dict[str, float | int | bool],
    failures: dict[str, float],
) -> None:
    snapshot = {
        "bearing_deg": region.objective.bearing_deg,
        "area_m2": diagnostics["area_m2"],
        "fraction": diagnostics["fraction"],
        "min_child_fraction": diagnostics["min_child_fraction"],
        "flight_line_count": diagnostics["flight_line_count"],
        "mean_line_length_m": diagnostics["mean_line_length_m"],
        "min_line_length_m": diagnostics["min_line_length_m"],
        "along_track_length_m": diagnostics["along_track_length_m"],
        "cross_track_width_m": diagnostics["cross_track_width_m"],
        "convexity": diagnostics["convexity"],
        "min_convexity": diagnostics["min_convexity"],
        "compactness": diagnostics["compactness"],
        "max_compactness": diagnostics["max_compactness"],
        "capture_efficiency": diagnostics["capture_efficiency"],
        "min_capture_efficiency": diagnostics["min_capture_efficiency"],
        "overflight_transit_fraction": diagnostics["overflight_transit_fraction"],
        "max_overflight_transit_fraction": diagnostics["max_overflight_transit_fraction"],
    }
    for reason, margin in failures.items():
        _update_failure_summary(summary, reason, margin, snapshot)


def _region_failure_snapshot(
    diagnostics: dict[str, float | int | bool],
    failures: dict[str, float],
) -> dict[str, Any]:
    return {
        "failureCount": len(failures),
        "failureMargins": {reason: round(margin, 4) for reason, margin in sorted(failures.items())},
        "bearing_deg": _summarize_diagnostic_value(diagnostics["bearing_deg"]),
        "area_m2": _summarize_diagnostic_value(diagnostics["area_m2"]),
        "fraction": _summarize_diagnostic_value(diagnostics["fraction"]),
        "min_child_fraction": _summarize_diagnostic_value(diagnostics["min_child_fraction"]),
        "flight_line_count": diagnostics["flight_line_count"],
        "mean_line_length_m": _summarize_diagnostic_value(diagnostics["mean_line_length_m"]),
        "min_line_length_m": _summarize_diagnostic_value(diagnostics["min_line_length_m"]),
        "along_track_length_m": _summarize_diagnostic_value(diagnostics["along_track_length_m"]),
        "cross_track_width_m": _summarize_diagnostic_value(diagnostics["cross_track_width_m"]),
        "convexity": _summarize_diagnostic_value(diagnostics["convexity"]),
        "min_convexity": _summarize_diagnostic_value(diagnostics["min_convexity"]),
        "compactness": _summarize_diagnostic_value(diagnostics["compactness"]),
        "max_compactness": _summarize_diagnostic_value(diagnostics["max_compactness"]),
        "capture_efficiency": _summarize_diagnostic_value(diagnostics["capture_efficiency"]),
        "min_capture_efficiency": _summarize_diagnostic_value(diagnostics["min_capture_efficiency"]),
        "overflight_transit_fraction": _summarize_diagnostic_value(diagnostics["overflight_transit_fraction"]),
        "max_overflight_transit_fraction": _summarize_diagnostic_value(diagnostics["max_overflight_transit_fraction"]),
    }


def _record_split_failure_summary(
    summary: dict[str, Any],
    *,
    direction_deg: float,
    threshold: float,
    boundary: BoundaryStats,
    left_diagnostics: dict[str, float | int | bool],
    left_failures: dict[str, float],
    right_diagnostics: dict[str, float | int | bool],
    right_failures: dict[str, float],
) -> None:
    summary["splitFailureCount"] = int(summary.get("splitFailureCount", 0)) + 1
    counts = summary.setdefault("counts", {})
    for reason in sorted(set(left_failures) | set(right_failures)):
        counts[reason] = int(counts.get(reason, 0)) + 1

    total_margin = sum(left_failures.values()) + sum(right_failures.values())
    max_margin = max([0.0, *left_failures.values(), *right_failures.values()])
    split_snapshot = {
        "totalMargin": round(total_margin, 4),
        "maxMargin": round(max_margin, 4),
        "direction_deg": round(direction_deg, 4),
        "threshold": round(threshold, 4),
        "shared_boundary_m": round(boundary.shared_boundary_m, 4),
        "break_weight_sum": round(boundary.break_weight_sum, 4),
        "left": _region_failure_snapshot(left_diagnostics, left_failures),
        "right": _region_failure_snapshot(right_diagnostics, right_failures),
    }
    closest_splits = summary.setdefault("closestSplits", [])
    closest_splits.append(split_snapshot)
    closest_splits.sort(key=lambda item: (item["totalMargin"], item["maxMargin"]))
    del closest_splits[5:]


def _record_plan_failure_summary(
    plan_summary: dict[str, Any],
    region_summary: dict[str, Any],
    plan: PartitionPlan,
    diagnostics: dict[str, Any],
    failures: dict[str, float],
    region_failures: list[tuple[int, dict[str, float | int | bool], dict[str, float]]],
) -> None:
    snapshot = {
        "region_count": diagnostics["region_count"],
        "largest_region_fraction": diagnostics["largest_region_fraction"],
        "non_largest_fraction": diagnostics["non_largest_fraction"],
        "min_non_largest_fraction": diagnostics["min_non_largest_fraction"],
        "smallest_region_fraction": diagnostics["smallest_region_fraction"],
        "min_child_fraction": diagnostics["min_child_fraction"],
        "mean_convexity": diagnostics["mean_convexity"],
        "min_mean_convexity": diagnostics["min_mean_convexity"],
        "max_two_region_largest_fraction": diagnostics["max_two_region_largest_fraction"],
        "quality_cost": diagnostics["quality_cost"],
        "mission_time_sec": diagnostics["mission_time_sec"],
    }
    for reason, margin in failures.items():
        _update_failure_summary(plan_summary, reason, margin, snapshot)

    for index, region_diag, region_reasons in region_failures:
        region_snapshot = {
            "region_index": index,
            **snapshot,
            "bearing_deg": region_diag["bearing_deg"],
            "area_m2": region_diag["area_m2"],
            "fraction": region_diag["fraction"],
            "flight_line_count": region_diag["flight_line_count"],
            "mean_line_length_m": region_diag["mean_line_length_m"],
            "min_line_length_m": region_diag["min_line_length_m"],
            "along_track_length_m": region_diag["along_track_length_m"],
            "cross_track_width_m": region_diag["cross_track_width_m"],
            "convexity": region_diag["convexity"],
            "min_convexity": region_diag["min_convexity"],
            "compactness": region_diag["compactness"],
            "max_compactness": region_diag["max_compactness"],
            "capture_efficiency": region_diag["capture_efficiency"],
            "min_capture_efficiency": region_diag["min_capture_efficiency"],
            "overflight_transit_fraction": region_diag["overflight_transit_fraction"],
            "max_overflight_transit_fraction": region_diag["max_overflight_transit_fraction"],
        }
        for reason, margin in region_reasons.items():
            _update_failure_summary(region_summary, reason, margin, region_snapshot)


def _relaxed_region_failure_sets(
    diagnostics: dict[str, float | int | bool],
    *,
    practical: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    failures = _region_gate_failure_margins(diagnostics, practical=practical)
    hard_failures: dict[str, float] = {}
    soft_failures: dict[str, float] = {}
    mean_line_length_m = float(diagnostics["mean_line_length_m"])
    for reason, margin in failures.items():
        if reason in {"hard_invalid", "min_child_fraction", "flight_line_count"}:
            hard_failures[reason] = margin
        elif reason == "convexity":
            hard_failures[reason] = margin
        elif reason == "mean_line_length_m" and mean_line_length_m < RELAXED_HARD_MIN_MEAN_LINE_LENGTH_M:
            hard_failures["mean_line_length_hard_floor_m"] = RELAXED_HARD_MIN_MEAN_LINE_LENGTH_M - mean_line_length_m
        else:
            soft_failures[reason] = margin
    return hard_failures, soft_failures


def _score_relaxed_fallback_candidate(
    plan: PartitionPlan,
    baseline_plan: PartitionPlan,
    *,
    soft_total_margin: float,
    soft_max_margin: float,
) -> float:
    quality_regression = max(0.0, plan.quality_cost - baseline_plan.quality_cost)
    mission_time_penalty = max(0.0, plan.mission_time_sec - baseline_plan.mission_time_sec) / RELAXED_FALLBACK_TIME_DIVISOR_SEC
    return (
        plan.quality_cost
        + quality_regression
        + mission_time_penalty
        + RELAXED_FALLBACK_SOFT_TOTAL_WEIGHT * soft_total_margin
        + RELAXED_FALLBACK_SOFT_MAX_WEIGHT * soft_max_margin
    )


def _relaxed_plan_soft_failure_values(
    plan: PartitionPlan,
    root_area_m2: float,
    *,
    line_length_scale: float,
) -> list[float]:
    diagnostics = _plan_gate_diagnostics(plan, root_area_m2, line_length_scale=line_length_scale)
    failures, region_failures = _plan_gate_failure_margins(diagnostics)
    soft_values: list[float] = []
    for reason, margin in failures.items():
        if reason in {"region_count", "region_practicality"}:
            continue
        soft_values.append(float(margin))
    for _, region_diag, region_reasons in region_failures:
        hard_failures, soft_failures = _relaxed_region_failure_sets(region_diag, practical=True)
        if hard_failures:
            return [float("inf")]
        soft_values.extend(float(margin) for margin in soft_failures.values())
    return soft_values


def _region_basic_validity(
    region: EvaluatedRegion,
    parent_area_m2: float,
    line_length_scale: float,
) -> bool:
    diagnostics = _region_gate_diagnostics(region, parent_area_m2, practical=False, line_length_scale=line_length_scale)
    return _region_gate_passes(diagnostics, practical=False)


def _region_practical(
    region: EvaluatedRegion,
    root_area_m2: float,
    line_length_scale: float,
) -> bool:
    diagnostics = _region_gate_diagnostics(region, root_area_m2, practical=True, line_length_scale=line_length_scale)
    return _region_gate_passes(diagnostics, practical=True)


def _plan_boundary_alignment(plan: PartitionPlan) -> float:
    if plan.internal_boundary_m <= 0:
        return 0.0
    return plan.break_weight_sum / plan.internal_boundary_m


def _plan_is_practical(plan: PartitionPlan, root_area_m2: float, line_length_scale: float) -> bool:
    diagnostics = _plan_gate_diagnostics(plan, root_area_m2, line_length_scale=line_length_scale)
    failures, _ = _plan_gate_failure_margins(diagnostics)
    return not failures


def _dominates(
    a: PartitionPlan,
    b: PartitionPlan,
    root_area_m2: float,
    line_length_scale: float,
    status_cache: dict[tuple[tuple[tuple[int, ...], int], ...], int],
) -> bool:
    status_a = status_cache.get(_plan_signature(a))
    if status_a is None:
        status_a = 2 if a.region_count <= 1 else (1 if _plan_is_practical(a, root_area_m2, line_length_scale) else 0)
        status_cache[_plan_signature(a)] = status_a
    status_b = status_cache.get(_plan_signature(b))
    if status_b is None:
        status_b = 2 if b.region_count <= 1 else (1 if _plan_is_practical(b, root_area_m2, line_length_scale) else 0)
        status_cache[_plan_signature(b)] = status_b

    # Keep "don't split" baselines as first-class options for parent composition.
    if status_a == 2 or status_b == 2:
        return False

    # Non-practical plans are useful for search, but they should not suppress
    # practical plans from the frontier because they can never be returned.
    if status_a == 0 and status_b == 1:
        return False

    better_or_equal = (
        a.quality_cost <= b.quality_cost + DOMINANCE_EPS
        and a.mission_time_sec <= b.mission_time_sec + DOMINANCE_EPS
    )
    strictly_better = (
        a.quality_cost < b.quality_cost - DOMINANCE_EPS
        or a.mission_time_sec < b.mission_time_sec - DOMINANCE_EPS
    )
    return better_or_equal and strictly_better


def _thin_frontier(plans: list[PartitionPlan], root_area_m2: float, line_length_scale: float) -> list[PartitionPlan]:
    baseline_bucket: list[PartitionPlan] = []
    practical_buckets: dict[int, list[PartitionPlan]] = defaultdict(list)
    non_practical_buckets: dict[int, list[PartitionPlan]] = defaultdict(list)
    for plan in plans:
        if plan.region_count <= 1:
            baseline_bucket.append(plan)
        elif _plan_is_practical(plan, root_area_m2, line_length_scale):
            practical_buckets[plan.region_count].append(plan)
        else:
            non_practical_buckets[plan.region_count].append(plan)

    retained: list[PartitionPlan] = []
    if baseline_bucket:
        baseline_bucket.sort(key=lambda plan: (plan.mission_time_sec, plan.quality_cost))
        retained.extend(baseline_bucket[:1])

    def append_unique(bucket_retained: list[PartitionPlan], source: list[PartitionPlan], limit: int) -> None:
        seen = {_plan_signature(plan) for plan in bucket_retained}
        for plan in source:
            signature = _plan_signature(plan)
            if signature in seen:
                continue
            bucket_retained.append(plan)
            seen.add(signature)
            if len(bucket_retained) >= limit:
                break

    for region_count, bucket in practical_buckets.items():
        quality_sorted = sorted(bucket, key=lambda plan: (plan.quality_cost, plan.mission_time_sec))
        time_sorted = sorted(bucket, key=lambda plan: (plan.mission_time_sec, plan.quality_cost))
        bucket_retained: list[PartitionPlan] = []
        append_unique(bucket_retained, quality_sorted[:1], PRACTICAL_FRONTIER_BUCKET_KEEP)
        append_unique(bucket_retained, time_sorted[:1], PRACTICAL_FRONTIER_BUCKET_KEEP)
        append_unique(bucket_retained, quality_sorted, PRACTICAL_FRONTIER_BUCKET_KEEP)
        retained.extend(bucket_retained[:PRACTICAL_FRONTIER_BUCKET_KEEP])

    remaining = max(0, MAX_FRONTIER_STATES - len(retained))
    if remaining > 0:
        for region_count, bucket in non_practical_buckets.items():
            quality_sorted = sorted(bucket, key=lambda plan: (plan.quality_cost, plan.mission_time_sec))
            time_sorted = sorted(bucket, key=lambda plan: (plan.mission_time_sec, plan.quality_cost))
            bucket_retained: list[PartitionPlan] = []
            append_unique(bucket_retained, quality_sorted[:1], NON_PRACTICAL_FRONTIER_BUCKET_KEEP)
            append_unique(bucket_retained, time_sorted[:1], NON_PRACTICAL_FRONTIER_BUCKET_KEEP)
            append_unique(bucket_retained, quality_sorted, NON_PRACTICAL_FRONTIER_BUCKET_KEEP)
            retained.extend(bucket_retained[: min(NON_PRACTICAL_FRONTIER_BUCKET_KEEP, remaining)])
            remaining = max(0, MAX_FRONTIER_STATES - len(retained))
            if remaining <= 0:
                break

    retained.sort(key=lambda plan: (plan.region_count, plan.mission_time_sec, plan.quality_cost))
    return retained[:MAX_FRONTIER_STATES]


def _pareto_frontier(plans: list[PartitionPlan], root_area_m2: float, line_length_scale: float) -> list[PartitionPlan]:
    unique: dict[tuple[tuple[tuple[int, ...], int], ...], PartitionPlan] = {}
    for plan in plans:
        signature = _plan_signature(plan)
        existing = unique.get(signature)
        if existing is None or plan.quality_cost < existing.quality_cost - DOMINANCE_EPS:
            unique[signature] = plan
    status_cache: dict[tuple[tuple[tuple[int, ...], int], ...], int] = {}
    nondominated: list[PartitionPlan] = []
    for plan in sorted(unique.values(), key=lambda item: (item.mission_time_sec, item.quality_cost, item.region_count)):
        if any(_dominates(existing, plan, root_area_m2, line_length_scale, status_cache) for existing in nondominated):
            continue
        nondominated = [
            existing
            for existing in nondominated
            if not _dominates(plan, existing, root_area_m2, line_length_scale, status_cache)
        ]
        nondominated.append(plan)
    return _thin_frontier(nondominated, root_area_m2, line_length_scale)


def _pareto_frontier_with_perf(
    plans: list[PartitionPlan],
    root_area_m2: float,
    line_length_scale: float,
    perf: dict[str, float],
) -> list[PartitionPlan]:
    prune_started_at = time.perf_counter()
    try:
        return _pareto_frontier(plans, root_area_m2, line_length_scale)
    finally:
        perf["frontier_prune_ms"] += (time.perf_counter() - prune_started_at) * 1000.0


def _solve_region_recursive_child(
    cell_ids: tuple[int, ...],
    depth: int,
    context: SolverContext,
    caches: SolverCaches,
    perf: dict[str, float],
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = None,
    *,
    allow_nested_lambda_fanout: bool,
) -> list[PartitionPlan]:
    subsolve_started_at = time.perf_counter()
    try:
        return _solve_region_recursive(
            cell_ids,
            depth,
            context,
            caches,
            perf,
            perf_hotspots,
            allow_nested_lambda_fanout=allow_nested_lambda_fanout,
        ) or []
    finally:
        perf["recursive_subsolve_ms"] += (time.perf_counter() - subsolve_started_at) * 1000.0


def _extend_combined_candidates(
    destination: list[PartitionPlan],
    left_frontier: list[PartitionPlan],
    right_frontier: list[PartitionPlan],
    boundary: BoundaryStats,
    split: SplitCandidate,
    perf: dict[str, float],
) -> None:
    combine_started_at = time.perf_counter()
    try:
        for left_plan in left_frontier:
            for right_plan in right_frontier:
                combined = _combine_plans(left_plan, right_plan, boundary, split)
                if combined.region_count > MAX_REGIONS:
                    perf["combine_region_limit_rejections"] += 1
                    continue
                destination.append(combined)
                perf["combined_plan_candidates"] += 1
    finally:
        perf["plan_combine_ms"] += (time.perf_counter() - combine_started_at) * 1000.0


def _principal_axis_bearing(cell_ids: tuple[int, ...], cell_lookup: dict[int, GridCell]) -> float | None:
    if len(cell_ids) < 2:
        return None
    weights = np.asarray([max(1e-6, cell_lookup[cell_id].area_m2) for cell_id in cell_ids], dtype=np.float64)
    points = np.asarray([[cell_lookup[cell_id].x, cell_lookup[cell_id].y] for cell_id in cell_ids], dtype=np.float64)
    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    covariance = np.cov(centered.T, aweights=weights)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    bearing_deg = math.degrees(math.atan2(axis[0], axis[1]))
    return normalize_axial_bearing(bearing_deg)


def _region_heading_candidates(
    cell_ids: tuple[int, ...],
    feature_lookup: dict[int, CellFeatures],
    cell_lookup: dict[int, GridCell],
    feature_field: FeatureField,
) -> list[float]:
    weighted_bearings: list[tuple[float, float]] = []
    weighted_aspects: list[tuple[float, float]] = []
    histogram_bins = [0.0] * 12
    for cell_id in cell_ids:
        feature = feature_lookup.get(cell_id)
        cell = cell_lookup.get(cell_id)
        if feature is None or cell is None:
            continue
        weight = max(1e-6, cell.area_m2 * (0.2 + 0.8 * feature.confidence))
        weighted_bearings.append((feature.preferred_bearing_deg, weight))
        weighted_aspects.append((feature.aspect_deg, weight))
        idx = int(normalize_axial_bearing(feature.preferred_bearing_deg) // 15.0) % len(histogram_bins)
        histogram_bins[idx] += weight

    candidates: list[float] = []

    def add(angle_deg: float | None) -> None:
        if angle_deg is None or not math.isfinite(angle_deg):
            return
        normalized = normalize_axial_bearing(angle_deg)
        if any(axial_angle_delta_deg(normalized, existing) < 9.0 for existing in candidates):
            return
        candidates.append(normalized)

    add(weighted_axial_mean_deg(weighted_bearings))
    mean_aspect = _weighted_circular_mean_deg(weighted_aspects)
    add(mean_aspect)
    add(None if mean_aspect is None else mean_aspect + 90.0)
    add(feature_field.dominant_preferred_bearing_deg)
    add(feature_field.dominant_aspect_deg)
    add(None if feature_field.dominant_aspect_deg is None else feature_field.dominant_aspect_deg + 90.0)
    principal_axis = _principal_axis_bearing(cell_ids, cell_lookup)
    add(principal_axis)
    add(None if principal_axis is None else principal_axis + 90.0)

    for idx, weight in sorted(enumerate(histogram_bins), key=lambda item: item[1], reverse=True):
        if weight <= 0:
            continue
        center_angle = idx * 15.0 + 7.5
        add(center_angle)
        if len(candidates) >= 8:
            break

    if not candidates:
        candidates = [feature_field.dominant_preferred_bearing_deg or 0.0]
    return candidates[:8]


def _cut_direction_candidates(
    baseline_region: EvaluatedRegion,
    cell_ids: tuple[int, ...],
    feature_lookup: dict[int, CellFeatures],
    cell_lookup: dict[int, GridCell],
    feature_field: FeatureField,
) -> list[float]:
    candidates: list[float] = []

    def add(angle_deg: float | None) -> None:
        if angle_deg is None or not math.isfinite(angle_deg):
            return
        normalized = normalize_axial_bearing(angle_deg)
        if any(axial_angle_delta_deg(normalized, existing) < 8.0 for existing in candidates):
            return
        candidates.append(normalized)

    add(baseline_region.objective.bearing_deg + 90.0)
    add(baseline_region.objective.bearing_deg)
    for bearing in _region_heading_candidates(cell_ids, feature_lookup, cell_lookup, feature_field):
        add(bearing)
        add(bearing + 90.0)

    principal_axis = _principal_axis_bearing(cell_ids, cell_lookup)
    add(principal_axis)
    add(None if principal_axis is None else principal_axis + 90.0)

    for fallback in (0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0):
        add(fallback)
    return candidates[:16]


def _boundary_stats_for_split(
    left_ids: set[int],
    right_ids: set[int],
    grid: GridData,
    feature_lookup: dict[int, CellFeatures],
) -> BoundaryStats:
    shared_boundary_m = 0.0
    break_weight_sum = 0.0
    for edge in grid.edges:
        if (edge.a in left_ids and edge.b in right_ids) or (edge.a in right_ids and edge.b in left_ids):
            shared_boundary_m += edge.shared_boundary_m
            feature_a = feature_lookup.get(edge.a)
            feature_b = feature_lookup.get(edge.b)
            if feature_a is None or feature_b is None:
                continue
            aspect_delta = abs(((feature_a.aspect_deg - feature_b.aspect_deg + 180.0) % 360.0) - 180.0)
            bearing_delta = axial_angle_delta_deg(feature_a.preferred_bearing_deg, feature_b.preferred_bearing_deg)
            break_strength = 0.5 * (feature_a.break_strength + feature_b.break_strength)
            break_weight_sum += edge.shared_boundary_m * (
                0.55 * break_strength
                + 0.30 * aspect_delta
                + 0.15 * bearing_delta
            )
    return BoundaryStats(shared_boundary_m=shared_boundary_m, break_weight_sum=break_weight_sum)


def _connected_components_for_subset(
    cell_ids: tuple[int, ...],
    neighbors: dict[int, list[int]],
) -> list[list[int]]:
    remaining = set(cell_ids)
    components: list[list[int]] = []
    while remaining:
        seed = next(iter(remaining))
        queue = deque([seed])
        remaining.remove(seed)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in neighbors.get(current, []):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
        components.append(sorted(component))
    return components


def _build_region_polygon(
    cell_ids: tuple[int, ...],
    grid: GridData,
    neighbors: dict[int, list[int]],
    polygon_cache: dict[tuple[int, ...], Polygon | None],
) -> Polygon | None:
    cached = polygon_cache.get(cell_ids)
    if cached is not None or cell_ids in polygon_cache:
        return cached
    if not cell_ids:
        polygon_cache[cell_ids] = None
        return None
    components = _connected_components_for_subset(cell_ids, neighbors)
    if len(components) != 1:
        polygon_cache[cell_ids] = None
        return None
    try:
        polygon = region_polygon_from_cells(grid, list(cell_ids))
    except Exception:  # noqa: BLE001
        polygon_cache[cell_ids] = None
        return None
    if polygon.is_empty or not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        polygon_cache[cell_ids] = None
        return None
    polygon_cache[cell_ids] = polygon
    return polygon


def _build_region(
    cell_ids: tuple[int, ...],
    boundary_alignment: float,
    grid: GridData,
    neighbors: dict[int, list[int]],
    feature_lookup: dict[int, CellFeatures],
    cell_lookup: dict[int, GridCell],
    feature_field: FeatureField,
    params: FlightParamsModel,
    best_bearing_cache: dict[tuple[int, ...], float],
    region_cache: dict[tuple[int, ...], EvaluatedRegion | None],
    region_static_cache: dict[tuple[int, ...], RegionStatic | None],
    region_bearing_core_cache: dict[tuple[tuple[int, ...], int], RegionBearingCore],
    heading_candidates_cache: dict[tuple[int, ...], list[float]],
    polygon_cache: dict[tuple[int, ...], Polygon | None],
    perf: dict[str, float],
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = None,
) -> EvaluatedRegion | None:
    cached_region = region_cache.get(cell_ids)
    if cached_region is not None or cell_ids in region_cache:
        perf["region_cache_hits"] += 1
        if cached_region is None:
            return None
        if abs(cached_region.objective.boundary_break_alignment - boundary_alignment) <= 1e-9:
            return cached_region
        return replace(
            cached_region,
            objective=replace(cached_region.objective, boundary_break_alignment=boundary_alignment),
        )

    perf["region_cache_misses"] += 1
    perf["build_region_calls"] += 1
    build_started_at = time.perf_counter()
    static_region = region_static_cache.get(cell_ids)
    reused_static_region = static_region is not None or cell_ids in region_static_cache
    if static_region is not None or cell_ids in region_static_cache:
        perf["region_static_hits"] += 1
        if static_region is None:
            perf["region_static_null_hits"] += 1
            perf["build_region_ms"] += (time.perf_counter() - build_started_at) * 1000.0
            region_cache[cell_ids] = None
            return None
    else:
        perf["region_static_misses"] += 1
        static_started_at = time.perf_counter()
        polygon_started_at = time.perf_counter()
        polygon = _build_region_polygon(cell_ids, grid, neighbors, polygon_cache)
        perf["build_region_polygon_ms"] += (time.perf_counter() - polygon_started_at) * 1000.0
        if polygon is None:
            perf["build_region_polygon_failures"] += 1
            perf["region_static_build_ms"] += (time.perf_counter() - static_started_at) * 1000.0
            perf["build_region_ms"] += (time.perf_counter() - build_started_at) * 1000.0
            region_static_cache[cell_ids] = None
            region_cache[cell_ids] = None
            return None
        static_region = RegionStatic(
            cell_ids=cell_ids,
            polygon=polygon,
            ring=polygon_to_lnglat_ring(polygon),
            cells=tuple(cell_lookup[cell_id] for cell_id in cell_ids),
            area_m2=float(polygon.area),
            convexity=polygon_convexity(polygon),
            compactness=polygon_compactness(polygon),
            static_inputs=RegionStaticInputs(
                x=np.asarray([cell_lookup[cell_id].x for cell_id in cell_ids], dtype=np.float64),
                y=np.asarray([cell_lookup[cell_id].y for cell_id in cell_ids], dtype=np.float64),
                area_m2=np.asarray([max(1e-6, cell_lookup[cell_id].area_m2) for cell_id in cell_ids], dtype=np.float64),
                terrain_z=np.asarray([cell_lookup[cell_id].terrain_z for cell_id in cell_ids], dtype=np.float64),
                preferred_bearing_deg=np.asarray(
                    [feature_lookup[cell_id].preferred_bearing_deg for cell_id in cell_ids],
                    dtype=np.float64,
                ),
                confidence=np.asarray([feature_lookup[cell_id].confidence for cell_id in cell_ids], dtype=np.float64),
                slope_magnitude=np.asarray([feature_lookup[cell_id].slope_magnitude for cell_id in cell_ids], dtype=np.float64),
                grid_step_m=grid.grid_step_m,
            ),
        )
        region_static_cache[cell_ids] = static_region
        perf["region_static_build_ms"] += (time.perf_counter() - static_started_at) * 1000.0
    if static_region is None:
        perf["build_region_ms"] += (time.perf_counter() - build_started_at) * 1000.0
        region_cache[cell_ids] = None
        return None
    polygon = static_region.polygon
    ring = static_region.ring
    cells = static_region.cells
    candidate_bearings = heading_candidates_cache.get(cell_ids)
    if candidate_bearings is None:
        candidate_bearings = _region_heading_candidates(cell_ids, feature_lookup, cell_lookup, feature_field)
        heading_candidates_cache[cell_ids] = candidate_bearings
    if cell_ids in best_bearing_cache:
        cached = best_bearing_cache[cell_ids]
        candidate_bearings = [cached] + [bearing for bearing in candidate_bearings if axial_angle_delta_deg(bearing, cached) > 1e-6]

    best_objective: RegionObjective | None = None
    best_score = float("inf")
    objective_elapsed_total_ms = 0.0
    region_signature = hash_signature({"cellIds": list(cell_ids)})[:16]
    for bearing_deg in candidate_bearings:
        bearing_key = (cell_ids, _bearing_cache_key(bearing_deg))
        cached_core = region_bearing_core_cache.get(bearing_key)
        if cached_core is not None:
            perf["region_bearing_hits"] += 1
            if abs(boundary_alignment) > 1e-9:
                perf["region_bearing_rewraps"] += 1
                objective = replace(cached_core.objective, boundary_break_alignment=boundary_alignment)
            else:
                objective = cached_core.objective
            score = cached_core.score
        else:
            perf["region_bearing_misses"] += 1
            perf["objective_calls"] += 1
            objective_started_at = time.perf_counter()
            objective = evaluate_region_objective(
                cells,
                feature_lookup,
                bearing_deg,
                params,
                polygon,
                0.0,
                perf=perf,
                precomputed_area_m2=static_region.area_m2,
                precomputed_convexity=static_region.convexity,
                precomputed_compactness=static_region.compactness,
                precomputed_static_inputs=static_region.static_inputs,
            )
            objective_elapsed_ms = (time.perf_counter() - objective_started_at) * 1000.0
            objective_elapsed_total_ms += objective_elapsed_ms
            perf["objective_ms"] += objective_elapsed_ms
            perf["region_bearing_core_ms"] += objective_elapsed_ms
            score = _region_score(objective)
            region_bearing_core_cache[bearing_key] = RegionBearingCore(objective=objective, score=score)
            _record_perf_hotspot(
                perf_hotspots,
                "objective",
                {
                    "regionSignature": region_signature,
                    "elapsedMs": round(objective_elapsed_ms, 3),
                    "cellCount": len(cell_ids),
                    "candidateBearingCount": len(candidate_bearings),
                    "bearingDeg": round(float(bearing_deg), 4),
                    "areaM2": round(float(static_region.area_m2), 3),
                    "flightLineCount": int(objective.flight_line_count),
                    "missionTimeSec": round(float(objective.total_mission_time_sec), 3),
                    "normalizedQualityCost": round(float(objective.normalized_quality_cost), 6),
                },
            )
            if abs(boundary_alignment) > 1e-9:
                perf["region_bearing_rewraps"] += 1
                objective = replace(objective, boundary_break_alignment=boundary_alignment)
        if score < best_score:
            best_score = score
            best_objective = objective

    if best_objective is None:
        perf["build_region_ms"] += (time.perf_counter() - build_started_at) * 1000.0
        region_cache[cell_ids] = None
        return None
    best_bearing_cache[cell_ids] = best_objective.bearing_deg
    region = EvaluatedRegion(
        cell_ids=cell_ids,
        polygon=polygon,
        ring=ring,
        objective=best_objective,
        score=best_score,
        hard_invalid=False,
    )
    build_region_elapsed_ms = (time.perf_counter() - build_started_at) * 1000.0
    perf["build_region_ms"] += build_region_elapsed_ms
    _record_perf_hotspot(
        perf_hotspots,
        "buildRegion",
        {
            "regionSignature": region_signature,
            "elapsedMs": round(build_region_elapsed_ms, 3),
            "cellCount": len(cell_ids),
            "candidateBearingCount": len(candidate_bearings),
            "areaM2": round(float(static_region.area_m2), 3),
            "bestBearingDeg": round(float(best_objective.bearing_deg), 4),
            "bestScore": round(float(best_score), 6),
            "objectiveMs": round(objective_elapsed_total_ms, 3),
            "reusedStaticRegion": bool(reused_static_region and static_region is not None),
        },
    )
    region_cache[cell_ids] = region
    return region


def _plan_from_regions(
    regions: tuple[EvaluatedRegion, ...],
    internal_boundary_m: float,
    break_weight_sum: float,
    geometry_tree: PartitionLeafGeometry | PartitionSplitGeometry | None = None,
) -> PartitionPlan:
    total_area = sum(region.objective.area_m2 for region in regions)
    quality_terms = []
    mismatch_terms = []
    convexity_terms = []
    for region in regions:
        weight = max(1e-6, region.objective.area_m2)
        shape_penalty = (
            0.22 * max(0.0, 0.74 - region.objective.convexity)
            + 0.05 * max(0.0, region.objective.compactness - 3.0)
            + 0.22 * region.objective.fragmented_line_fraction
            + 0.28 * region.objective.overflight_transit_fraction
            + 0.02 * region.objective.short_line_fraction
        )
        quality_terms.append((region.objective.normalized_quality_cost + shape_penalty, weight))
        mismatch_terms.append((region.objective.weighted_mean_mismatch_deg, weight))
        convexity_terms.append((region.objective.convexity, weight))

    mission_time_sec = sum(region.objective.total_mission_time_sec for region in regions)
    if len(regions) > 1:
        mission_time_sec += INTER_REGION_TRANSITION_SEC * (len(regions) - 1)
    if geometry_tree is None and len(regions) == 1:
        geometry_tree = PartitionLeafGeometry(cell_ids=regions[0].cell_ids)

    return PartitionPlan(
        regions=tuple(sorted(regions, key=lambda region: region.objective.area_m2, reverse=True)),
        quality_cost=weighted_mean(quality_terms) + REGION_COUNT_PENALTY * max(0, len(regions) - 1),
        mission_time_sec=mission_time_sec,
        weighted_mean_mismatch_deg=weighted_mean(mismatch_terms),
        internal_boundary_m=internal_boundary_m,
        break_weight_sum=break_weight_sum,
        largest_region_fraction=max((region.objective.area_m2 / max(1.0, total_area) for region in regions), default=1.0),
        mean_convexity=weighted_mean(convexity_terms),
        region_count=len(regions),
        geometry_tree=geometry_tree,
    )


def _combine_plans(left: PartitionPlan, right: PartitionPlan, boundary: BoundaryStats, split: SplitCandidate) -> PartitionPlan:
    return _plan_from_regions(
        left.regions + right.regions,
        internal_boundary_m=left.internal_boundary_m + right.internal_boundary_m + boundary.shared_boundary_m,
        break_weight_sum=left.break_weight_sum + right.break_weight_sum + boundary.break_weight_sum,
        geometry_tree=PartitionSplitGeometry(
            direction_deg=split.direction_deg,
            threshold=split.threshold,
            left=left.geometry_tree,
            right=right.geometry_tree,
        ),
    )


def _projection_values(
    cell_ids: tuple[int, ...],
    cell_lookup: dict[int, GridCell],
    direction_deg: float,
) -> list[tuple[int, float, float]]:
    rad = math.radians(direction_deg)
    ux = math.sin(rad)
    uy = math.cos(rad)
    return [
        (
            cell_id,
            cell_lookup[cell_id].x * ux + cell_lookup[cell_id].y * uy,
            max(1e-6, cell_lookup[cell_id].area_m2),
        )
        for cell_id in cell_ids
    ]


def _split_cell_ids_by_projection_values(
    projected: list[tuple[int, float, float]],
    quantile: float,
) -> tuple[tuple[int, ...], tuple[int, ...], float] | None:
    if len(projected) < 4:
        return None
    min_projection = min(value for _, value, _ in projected)
    max_projection = max(value for _, value, _ in projected)
    if max_projection - min_projection < 1e-6:
        return None
    threshold = weighted_quantile(((value, weight) for _, value, weight in projected), quantile)
    left = tuple(sorted(cell_id for cell_id, value, _ in projected if value <= threshold))
    right = tuple(sorted(cell_id for cell_id, value, _ in projected if value > threshold))
    if not left or not right:
        return None
    return left, right, threshold


def _clip_polygon_by_half_plane(
    polygon: Polygon,
    direction_deg: float,
    threshold: float,
    *,
    keep_left: bool,
) -> Polygon | None:
    if polygon.is_empty:
        return None
    min_x, min_y, max_x, max_y = polygon.bounds
    span = max(max_x - min_x, max_y - min_y, 1.0) * 4.0 + 1000.0
    rad = math.radians(direction_deg)
    nx = math.sin(rad)
    ny = math.cos(rad)
    tx = -ny
    ty = nx
    centroid = polygon.representative_point()
    centroid_projection = centroid.x * nx + centroid.y * ny
    projection_delta = threshold - centroid_projection
    line_x = centroid.x + nx * projection_delta
    line_y = centroid.y + ny * projection_delta
    side_sign = -1.0 if keep_left else 1.0
    far_x = line_x + nx * span * side_sign
    far_y = line_y + ny * span * side_sign
    half_plane = Polygon(
        [
            (far_x - tx * span, far_y - ty * span),
            (far_x + tx * span, far_y + ty * span),
            (line_x + tx * span, line_y + ty * span),
            (line_x - tx * span, line_y - ty * span),
        ]
    )
    clipped = polygon.intersection(half_plane)
    if clipped.is_empty:
        return None
    if not clipped.is_valid:
        clipped = clipped.buffer(0)
    if clipped.is_empty:
        return None
    if clipped.geom_type == "Polygon":
        return clipped
    if clipped.geom_type == "MultiPolygon":
        return max(clipped.geoms, key=lambda geom: geom.area)
    return None


def _reconstruct_plan_polygons_from_tree(
    polygon: Polygon,
    node: PartitionLeafGeometry | PartitionSplitGeometry,
) -> dict[tuple[int, ...], Polygon] | None:
    if polygon.is_empty:
        return None
    if isinstance(node, PartitionLeafGeometry):
        return {node.cell_ids: polygon}

    left_polygon = _clip_polygon_by_half_plane(
        polygon,
        node.direction_deg,
        node.threshold,
        keep_left=True,
    )
    right_polygon = _clip_polygon_by_half_plane(
        polygon,
        node.direction_deg,
        node.threshold,
        keep_left=False,
    )
    if left_polygon is None or right_polygon is None:
        return None

    left_mapping = _reconstruct_plan_polygons_from_tree(left_polygon, node.left)
    right_mapping = _reconstruct_plan_polygons_from_tree(right_polygon, node.right)
    if left_mapping is None or right_mapping is None:
        return None
    return {**left_mapping, **right_mapping}


def _reevaluate_region_with_polygon(
    region: EvaluatedRegion,
    polygon: Polygon,
    context: SolverContext,
    caches: SolverCaches,
    perf: dict[str, float],
) -> EvaluatedRegion | None:
    if polygon.is_empty:
        return None
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.geom_type != "Polygon":
        return None

    static_region = caches.region_static_cache.get(region.cell_ids)
    if static_region is None:
        baseline_region = _build_region_for_context(
            region.cell_ids,
            region.objective.boundary_break_alignment,
            context,
            caches,
            perf,
        )
        if baseline_region is None:
            return None
        static_region = caches.region_static_cache.get(region.cell_ids)
        if static_region is None:
            return None

    candidate_bearings = caches.heading_candidates_cache.get(region.cell_ids)
    if candidate_bearings is None:
        candidate_bearings = _region_heading_candidates(
            region.cell_ids,
            context.feature_lookup,
            context.cell_lookup,
            context.feature_field,
        )
        caches.heading_candidates_cache[region.cell_ids] = candidate_bearings

    ordered_bearings = [region.objective.bearing_deg]
    ordered_bearings.extend(
        bearing
        for bearing in candidate_bearings
        if axial_angle_delta_deg(bearing, region.objective.bearing_deg) > 1e-6
    )

    area_m2 = float(polygon.area)
    convexity = polygon_convexity(polygon)
    compactness = polygon_compactness(polygon)
    ring = polygon_to_lnglat_ring(polygon)
    best_objective: RegionObjective | None = None
    best_score = float("inf")
    boundary_alignment = region.objective.boundary_break_alignment
    for bearing_deg in ordered_bearings:
        perf["exact_region_objective_calls"] += 1
        objective_started_at = time.perf_counter()
        objective = evaluate_region_objective(
            static_region.cells,
            context.feature_lookup,
            bearing_deg,
            context.params,
            polygon,
            boundary_alignment,
            perf=perf,
            precomputed_area_m2=area_m2,
            precomputed_convexity=convexity,
            precomputed_compactness=compactness,
            precomputed_static_inputs=static_region.static_inputs,
        )
        perf["exact_region_objective_ms"] += (time.perf_counter() - objective_started_at) * 1000.0
        score = _region_score(objective)
        if score < best_score:
            best_score = score
            best_objective = objective

    if best_objective is None:
        return None

    return EvaluatedRegion(
        cell_ids=region.cell_ids,
        polygon=polygon,
        ring=ring,
        objective=best_objective,
        score=best_score,
        hard_invalid=False,
    )


def _reevaluate_plan_with_exact_geometry(
    plan: PartitionPlan,
    context: SolverContext,
    caches: SolverCaches,
    perf: dict[str, float],
) -> PartitionPlan | None:
    perf["exact_geometry_plan_count"] += 1
    reevaluate_started_at = time.perf_counter()
    try:
        if plan.geometry_tree is None:
            return plan
        reconstruct_started_at = time.perf_counter()
        reconstructed = _reconstruct_plan_polygons_from_tree(context.grid.polygon_mercator, plan.geometry_tree)
        perf["exact_geometry_reconstruct_ms"] += (time.perf_counter() - reconstruct_started_at) * 1000.0
        if reconstructed is None:
            return None

        exact_regions: list[EvaluatedRegion] = []
        for region in plan.regions:
            polygon = reconstructed.get(region.cell_ids)
            if polygon is None or polygon.is_empty:
                return None
            region_started_at = time.perf_counter()
            exact_region = _reevaluate_region_with_polygon(region, polygon, context, caches, perf)
            perf["exact_geometry_region_reeval_ms"] += (time.perf_counter() - region_started_at) * 1000.0
            if exact_region is None:
                return None
            exact_regions.append(exact_region)

        return _plan_from_regions(
            tuple(exact_regions),
            internal_boundary_m=plan.internal_boundary_m,
            break_weight_sum=plan.break_weight_sum,
            geometry_tree=plan.geometry_tree,
        )
    finally:
        perf["exact_geometry_reeval_ms"] += (time.perf_counter() - reevaluate_started_at) * 1000.0


def _generate_split_candidates(
    baseline_region: EvaluatedRegion,
    baseline_plan: PartitionPlan,
    cell_ids: tuple[int, ...],
    root_area_m2: float,
    grid: GridData,
    feature_lookup: dict[int, CellFeatures],
    cell_lookup: dict[int, GridCell],
    feature_field: FeatureField,
    params: FlightParamsModel,
    neighbors: dict[int, list[int]],
    best_bearing_cache: dict[tuple[int, ...], float],
    region_cache: dict[tuple[int, ...], EvaluatedRegion | None],
    region_static_cache: dict[tuple[int, ...], RegionStatic | None],
    region_bearing_core_cache: dict[tuple[tuple[int, ...], int], RegionBearingCore],
    heading_candidates_cache: dict[tuple[int, ...], list[float]],
    polygon_cache: dict[tuple[int, ...], Polygon | None],
    basic_rejection_summary: dict[str, Any],
    basic_split_rejection_summary: dict[str, Any],
    line_length_scale: float,
    perf: dict[str, float],
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = None,
) -> list[SplitCandidate]:
    enumerate_started_at = time.perf_counter()
    try:
        parent_area = max(1.0, baseline_region.objective.area_m2)
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        candidates: list[SplitCandidate] = []
        directions = _cut_direction_candidates(baseline_region, cell_ids, feature_lookup, cell_lookup, feature_field)
        perf["split_direction_count"] += len(directions)
        for direction_deg in directions:
            projected = _projection_values(cell_ids, cell_lookup, direction_deg)
            for quantile in SPLIT_QUANTILES:
                perf["split_attempts"] += 1
                split = _split_cell_ids_by_projection_values(projected, quantile)
                if split is None:
                    perf["split_projection_failures"] += 1
                    continue
                left_ids, right_ids, threshold = split
                left_area = sum(cell_lookup[cell_id].area_m2 for cell_id in left_ids)
                right_area = sum(cell_lookup[cell_id].area_m2 for cell_id in right_ids)
                left_fraction = left_area / parent_area
                right_fraction = right_area / parent_area
                if min(left_fraction, right_fraction) < MIN_CHILD_AREA_FRACTION:
                    perf["split_small_child_rejections"] += 1
                    continue
                signature = (left_ids, right_ids) if left_ids < right_ids else (right_ids, left_ids)
                if signature in seen:
                    perf["split_duplicate_rejections"] += 1
                    continue
                seen.add(signature)

                if len(_connected_components_for_subset(left_ids, neighbors)) != 1:
                    perf["split_disconnected_rejections"] += 1
                    continue
                if len(_connected_components_for_subset(right_ids, neighbors)) != 1:
                    perf["split_disconnected_rejections"] += 1
                    continue

                boundary = _boundary_stats_for_split(set(left_ids), set(right_ids), grid, feature_lookup)
                if boundary.shared_boundary_m <= grid.grid_step_m * 0.8:
                    perf["split_boundary_rejections"] += 1
                    continue
                boundary_alignment = _boundary_alignment(boundary)
                left_region = _build_region(
                    left_ids,
                    boundary_alignment,
                    grid,
                    neighbors,
                    feature_lookup,
                    cell_lookup,
                    feature_field,
                    params,
                    best_bearing_cache,
                    region_cache,
                    region_static_cache,
                    region_bearing_core_cache,
                    heading_candidates_cache,
                    polygon_cache,
                    perf,
                    perf_hotspots,
                )
                right_region = _build_region(
                    right_ids,
                    boundary_alignment,
                    grid,
                    neighbors,
                    feature_lookup,
                    cell_lookup,
                    feature_field,
                    params,
                    best_bearing_cache,
                    region_cache,
                    region_static_cache,
                    region_bearing_core_cache,
                    heading_candidates_cache,
                    polygon_cache,
                    perf,
                    perf_hotspots,
                )
                if left_region is None or right_region is None:
                    perf["split_region_build_rejections"] += 1
                    continue
                left_basic_diag = _region_gate_diagnostics(left_region, parent_area, practical=False, line_length_scale=line_length_scale)
                right_basic_diag = _region_gate_diagnostics(right_region, parent_area, practical=False, line_length_scale=line_length_scale)
                left_basic_failures = _region_gate_failure_margins(left_basic_diag, practical=False)
                right_basic_failures = _region_gate_failure_margins(right_basic_diag, practical=False)
                if left_basic_failures or right_basic_failures:
                    perf["split_basic_validity_rejections"] += 1
                    if left_basic_failures:
                        _record_region_failure_summary(basic_rejection_summary, left_region, left_basic_diag, left_basic_failures)
                    if right_basic_failures:
                        _record_region_failure_summary(basic_rejection_summary, right_region, right_basic_diag, right_basic_failures)
                    _record_split_failure_summary(
                        basic_split_rejection_summary,
                        direction_deg=direction_deg,
                        threshold=threshold,
                        boundary=boundary,
                        left_diagnostics=left_basic_diag,
                        left_failures=left_basic_failures,
                        right_diagnostics=right_basic_diag,
                        right_failures=right_basic_failures,
                    )
                    continue
                immediate_plan = _plan_from_regions((left_region, right_region), boundary.shared_boundary_m, boundary.break_weight_sum)
                quality_gain = baseline_plan.quality_cost - immediate_plan.quality_cost
                time_delta = immediate_plan.mission_time_sec - baseline_plan.mission_time_sec
                rank_score = (
                    quality_gain
                    - max(0.0, time_delta) / SPLIT_RANK_TIME_DIVISOR_SEC
                    + boundary_alignment / 28.0
                    - abs(left_fraction - right_fraction) * 0.2
                )
                if quality_gain <= -SPLIT_NON_IMPROVING_MAX_QUALITY_REGRESSION and rank_score <= SPLIT_NON_IMPROVING_MIN_RANK_SCORE:
                    perf["split_non_improving_rejections"] += 1
                    continue
                candidates.append(
                    SplitCandidate(
                        left_ids=left_ids,
                        right_ids=right_ids,
                        boundary=boundary,
                        direction_deg=direction_deg,
                        threshold=threshold,
                        rank_score=rank_score,
                    )
                )
                perf["split_candidates_kept"] += 1
        candidates.sort(key=lambda candidate: candidate.rank_score, reverse=True)
        perf["split_candidates_returned"] += min(len(candidates), MAX_SPLIT_OPTIONS)
        return candidates[:MAX_SPLIT_OPTIONS]
    finally:
        perf["split_candidate_enumeration_ms"] += (time.perf_counter() - enumerate_started_at) * 1000.0


def _build_region_for_context(
    cell_ids: tuple[int, ...],
    boundary_alignment: float,
    context: SolverContext,
    caches: SolverCaches,
    perf: dict[str, float],
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = None,
) -> EvaluatedRegion | None:
    return _build_region(
        cell_ids,
        boundary_alignment,
        context.grid,
        context.neighbors,
        context.feature_lookup,
        context.cell_lookup,
        context.feature_field,
        context.params,
        caches.best_bearing_cache,
        caches.region_cache,
        caches.region_static_cache,
        caches.region_bearing_core_cache,
        caches.heading_candidates_cache,
        caches.polygon_cache,
        perf,
        perf_hotspots,
    )


def _generate_split_candidates_for_context(
    baseline_region: EvaluatedRegion,
    baseline_plan: PartitionPlan,
    cell_ids: tuple[int, ...],
    context: SolverContext,
    caches: SolverCaches,
    perf: dict[str, float],
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = None,
) -> list[SplitCandidate]:
    return _generate_split_candidates(
        baseline_region,
        baseline_plan,
        cell_ids,
        context.root_area_m2,
        context.grid,
        context.feature_lookup,
        context.cell_lookup,
        context.feature_field,
        context.params,
        context.neighbors,
        caches.best_bearing_cache,
        caches.region_cache,
        caches.region_static_cache,
        caches.region_bearing_core_cache,
        caches.heading_candidates_cache,
        caches.polygon_cache,
        caches.basic_rejection_summary,
        caches.basic_split_rejection_summary,
        context.basic_line_length_scale,
        perf,
        perf_hotspots,
    )


def _generate_relaxed_root_fallback_candidates(
    baseline_region: EvaluatedRegion,
    baseline_plan: PartitionPlan,
    root_cell_ids: tuple[int, ...],
    context: SolverContext,
    caches: SolverCaches,
    perf: dict[str, float],
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = None,
) -> list[RelaxedFallbackCandidate]:
    parent_area = max(1.0, baseline_region.objective.area_m2)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    retained: dict[tuple[tuple[tuple[int, ...], int], ...], RelaxedFallbackCandidate] = {}
    directions = _cut_direction_candidates(
        baseline_region,
        root_cell_ids,
        context.feature_lookup,
        context.cell_lookup,
        context.feature_field,
    )
    for direction_deg in directions:
        projected = _projection_values(root_cell_ids, context.cell_lookup, direction_deg)
        for quantile in SPLIT_QUANTILES:
            split = _split_cell_ids_by_projection_values(projected, quantile)
            if split is None:
                continue
            left_ids, right_ids, threshold = split
            left_area = sum(context.cell_lookup[cell_id].area_m2 for cell_id in left_ids)
            right_area = sum(context.cell_lookup[cell_id].area_m2 for cell_id in right_ids)
            if min(left_area, right_area) / parent_area < MIN_CHILD_AREA_FRACTION:
                continue
            signature = (left_ids, right_ids) if left_ids < right_ids else (right_ids, left_ids)
            if signature in seen:
                continue
            seen.add(signature)
            if len(_connected_components_for_subset(left_ids, context.neighbors)) != 1:
                continue
            if len(_connected_components_for_subset(right_ids, context.neighbors)) != 1:
                continue
            boundary = _boundary_stats_for_split(set(left_ids), set(right_ids), context.grid, context.feature_lookup)
            if boundary.shared_boundary_m <= context.grid.grid_step_m * 0.8:
                continue
            boundary_alignment = _boundary_alignment(boundary)
            left_region = _build_region_for_context(left_ids, boundary_alignment, context, caches, perf, perf_hotspots)
            right_region = _build_region_for_context(right_ids, boundary_alignment, context, caches, perf, perf_hotspots)
            if left_region is None or right_region is None:
                continue

            split_candidate = SplitCandidate(
                left_ids=left_ids,
                right_ids=right_ids,
                boundary=boundary,
                direction_deg=direction_deg,
                threshold=threshold,
                rank_score=0.0,
            )
            left_plan = _plan_from_regions((left_region,), 0.0, 0.0)
            right_plan = _plan_from_regions((right_region,), 0.0, 0.0)
            combine_started_at = time.perf_counter()
            candidate_plan = _combine_plans(left_plan, right_plan, boundary, split_candidate)
            perf["plan_combine_ms"] += (time.perf_counter() - combine_started_at) * 1000.0
            exact_plan = _reevaluate_plan_with_exact_geometry(candidate_plan, context, caches, perf)
            if exact_plan is None or exact_plan.region_count <= 1:
                continue
            quality_gain = baseline_plan.quality_cost - exact_plan.quality_cost
            if quality_gain <= DOMINANCE_EPS:
                continue

            soft_values: list[float] = []
            hard_rejected = False
            for region in exact_plan.regions:
                diagnostics = _region_gate_diagnostics(
                    region,
                    context.root_area_m2,
                    practical=False,
                    line_length_scale=context.basic_line_length_scale,
                )
                hard_failures, soft_failures = _relaxed_region_failure_sets(diagnostics, practical=False)
                if hard_failures:
                    hard_rejected = True
                    break
                soft_values.extend(float(margin) for margin in soft_failures.values())
            if hard_rejected:
                continue

            plan_soft_values = _relaxed_plan_soft_failure_values(
                exact_plan,
                context.root_area_m2,
                line_length_scale=context.practical_line_length_scale,
            )
            if any(math.isinf(value) for value in plan_soft_values):
                continue
            soft_values.extend(plan_soft_values)
            soft_total_margin = float(sum(soft_values))
            soft_max_margin = float(max(soft_values, default=0.0))
            fallback_score = _score_relaxed_fallback_candidate(
                exact_plan,
                baseline_plan,
                soft_total_margin=soft_total_margin,
                soft_max_margin=soft_max_margin,
            )
            plan_signature = _plan_signature(exact_plan)
            existing = retained.get(plan_signature)
            candidate = RelaxedFallbackCandidate(
                plan=exact_plan,
                direction_deg=direction_deg,
                threshold=threshold,
                soft_total_margin=soft_total_margin,
                soft_max_margin=soft_max_margin,
                fallback_score=fallback_score,
            )
            if existing is None or candidate.fallback_score < existing.fallback_score - DOMINANCE_EPS:
                retained[plan_signature] = candidate

    candidates = sorted(
        retained.values(),
        key=lambda candidate: (
            candidate.fallback_score,
            candidate.soft_total_margin,
            candidate.soft_max_margin,
            candidate.plan.quality_cost,
            candidate.plan.mission_time_sec,
        ),
    )
    return candidates[:MAX_RELAXED_FALLBACK_CANDIDATES]


def _solve_region_recursive(
    cell_ids: tuple[int, ...],
    depth: int,
    context: SolverContext,
    caches: SolverCaches,
    perf: dict[str, float],
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = None,
    *,
    allow_nested_lambda_fanout: bool = False,
) -> list[PartitionPlan]:
    perf["solve_region_calls"] += 1
    key = (cell_ids, depth)
    cached = caches.frontier_cache.get(key)
    if cached is not None:
        perf["solve_region_cache_hits"] += 1
        return cached

    baseline_region = _build_region_for_context(cell_ids, 0.0, context, caches, perf, perf_hotspots)
    if baseline_region is None:
        caches.frontier_cache[key] = []
        return []

    baseline_plan = _plan_from_regions((baseline_region,), 0.0, 0.0)
    if depth <= 0 or len(cell_ids) < 4 or baseline_region.objective.flight_line_count < 1:
        caches.frontier_cache[key] = [baseline_plan]
        perf["baseline_leaf_plans"] += 1
        return [baseline_plan]

    candidates = [baseline_plan]
    split_gen_started_at = time.perf_counter()
    split_candidates = _generate_split_candidates_for_context(
        baseline_region,
        baseline_plan,
        cell_ids,
        context,
        caches,
        perf,
        perf_hotspots,
    )
    if allow_nested_lambda_fanout and _should_use_nested_lambda_fanout(cell_ids, depth, split_candidates):
        try:
            candidates.extend(
                _solve_split_candidates_via_nested_lambda(
                    split_candidates,
                    depth,
                    context,
                    perf,
                )
            )
        except Exception as exc:  # noqa: BLE001
            perf["nested_parallel_failures"] += 1
            logger.warning(
                "[terrain-split-backend] nested lambda fan-out failed cells=%d depth=%d; falling back to serial",
                len(cell_ids),
                depth,
                exc_info=exc,
            )
            for split in split_candidates:
                left_frontier = _solve_region_recursive_child(
                    split.left_ids,
                    depth - 1,
                    context,
                    caches,
                    perf,
                    perf_hotspots,
                    allow_nested_lambda_fanout=allow_nested_lambda_fanout,
                )
                right_frontier = _solve_region_recursive_child(
                    split.right_ids,
                    depth - 1,
                    context,
                    caches,
                    perf,
                    perf_hotspots,
                    allow_nested_lambda_fanout=allow_nested_lambda_fanout,
                )
                if not left_frontier or not right_frontier:
                    continue
                _extend_combined_candidates(
                    candidates,
                    left_frontier,
                    right_frontier,
                    split.boundary,
                    split,
                    perf,
                )
    else:
        for split in split_candidates:
            left_frontier = _solve_region_recursive_child(
                split.left_ids,
                depth - 1,
                context,
                caches,
                perf,
                perf_hotspots,
                allow_nested_lambda_fanout=allow_nested_lambda_fanout,
            )
            right_frontier = _solve_region_recursive_child(
                split.right_ids,
                depth - 1,
                context,
                caches,
                perf,
                perf_hotspots,
                allow_nested_lambda_fanout=allow_nested_lambda_fanout,
            )
            if not left_frontier or not right_frontier:
                continue
            _extend_combined_candidates(
                candidates,
                left_frontier,
                right_frontier,
                split.boundary,
                split,
                perf,
            )
    perf["split_generation_ms"] += (time.perf_counter() - split_gen_started_at) * 1000.0

    frontier = _pareto_frontier_with_perf(candidates, context.root_area_m2, context.practical_line_length_scale, perf)
    perf["frontier_plan_count"] += len(frontier)
    caches.frontier_cache[key] = frontier
    return frontier


def _solve_root_split_branch(task: RootSplitTask) -> tuple[list[PartitionPlan], dict[str, float]]:
    if _PARALLEL_SOLVER_CONTEXT is None:
        raise RuntimeError("Parallel solver context is not initialized.")
    context = _PARALLEL_SOLVER_CONTEXT
    return _solve_root_split_branch_with_context(task, context)


def _solve_root_split_branch_with_context(
    task: RootSplitTask,
    context: SolverContext,
) -> tuple[list[PartitionPlan], dict[str, float]]:
    caches = _make_solver_caches()
    perf = _make_perf()
    left_frontier = _solve_region_recursive_child(
        task.left_ids,
        task.depth,
        context,
        caches,
        perf,
        allow_nested_lambda_fanout=True,
    )
    right_frontier = _solve_region_recursive_child(
        task.right_ids,
        task.depth,
        context,
        caches,
        perf,
        allow_nested_lambda_fanout=True,
    )
    if not left_frontier or not right_frontier:
        return [], dict(perf)
    combined_candidates: list[PartitionPlan] = []
    _extend_combined_candidates(
        combined_candidates,
        left_frontier,
        right_frontier,
        task.boundary,
        SplitCandidate(
            left_ids=task.left_ids,
            right_ids=task.right_ids,
            boundary=task.boundary,
            direction_deg=task.direction_deg,
            threshold=task.threshold,
            rank_score=0.0,
        ),
        perf,
    )
    branch_frontier = _pareto_frontier_with_perf(
        combined_candidates,
        context.root_area_m2,
        context.practical_line_length_scale,
        perf,
    )
    perf["frontier_plan_count"] += len(branch_frontier)
    return branch_frontier, dict(perf)


def _solve_subtree_task(task: SubtreeSolveTask) -> tuple[list[PartitionPlan], dict[str, float]]:
    if _PARALLEL_SOLVER_CONTEXT is None:
        raise RuntimeError("Parallel solver context is not initialized.")
    return _solve_subtree_task_with_context(task, _PARALLEL_SOLVER_CONTEXT)


def _solve_subtree_task_with_context(
    task: SubtreeSolveTask,
    context: SolverContext,
) -> tuple[list[PartitionPlan], dict[str, float]]:
    caches = _make_solver_caches()
    perf = _make_perf()
    frontier = _solve_region_recursive(
        task.cell_ids,
        task.depth,
        context,
        caches,
        perf,
        allow_nested_lambda_fanout=True,
    ) or []
    return frontier, dict(perf)


def _serialize_polygon(polygon: Polygon) -> str:
    return polygon.wkt


def _serialize_grid(grid: GridData) -> dict[str, Any]:
    return {
        "ring": grid.ring,
        "polygonWkt": _serialize_polygon(grid.polygon_mercator),
        "cells": [
            {
                "index": cell.index,
                "row": cell.row,
                "col": cell.col,
                "x": cell.x,
                "y": cell.y,
                "lng": cell.lng,
                "lat": cell.lat,
                "areaM2": cell.area_m2,
                "terrainZ": cell.terrain_z,
                "polygonWkt": _serialize_polygon(cell.polygon),
            }
            for cell in grid.cells
        ],
        "edges": [
            {
                "a": edge.a,
                "b": edge.b,
                "sharedBoundaryM": edge.shared_boundary_m,
            }
            for edge in grid.edges
        ],
        "gridStepM": grid.grid_step_m,
        "areaM2": grid.area_m2,
    }


def _deserialize_grid(payload: dict[str, Any]) -> GridData:
    return GridData(
        ring=[tuple(coord) for coord in payload["ring"]],
        polygon_mercator=load_wkt(payload["polygonWkt"]),
        cells=[
            GridCell(
                index=cell["index"],
                row=cell["row"],
                col=cell["col"],
                x=cell["x"],
                y=cell["y"],
                lng=cell["lng"],
                lat=cell["lat"],
                area_m2=cell["areaM2"],
                terrain_z=cell["terrainZ"],
                polygon=load_wkt(cell["polygonWkt"]),
            )
            for cell in payload["cells"]
        ],
        edges=[
            GridEdge(
                a=edge["a"],
                b=edge["b"],
                shared_boundary_m=edge["sharedBoundaryM"],
            )
            for edge in payload["edges"]
        ],
        grid_step_m=payload["gridStepM"],
        area_m2=payload["areaM2"],
    )


def _serialize_feature_field(feature_field: FeatureField) -> dict[str, Any]:
    return {
        "cells": [
            {
                "index": cell.index,
                "preferredBearingDeg": cell.preferred_bearing_deg,
                "slopeMagnitude": cell.slope_magnitude,
                "breakStrength": cell.break_strength,
                "confidence": cell.confidence,
                "aspectDeg": cell.aspect_deg,
            }
            for cell in feature_field.cells
        ],
        "dominantPreferredBearingDeg": feature_field.dominant_preferred_bearing_deg,
        "dominantAspectDeg": feature_field.dominant_aspect_deg,
    }


def _deserialize_feature_field(payload: dict[str, Any]) -> FeatureField:
    return FeatureField(
        cells=[
            CellFeatures(
                index=cell["index"],
                preferred_bearing_deg=cell["preferredBearingDeg"],
                slope_magnitude=cell["slopeMagnitude"],
                break_strength=cell["breakStrength"],
                confidence=cell["confidence"],
                aspect_deg=cell["aspectDeg"],
            )
            for cell in payload["cells"]
        ],
        dominant_preferred_bearing_deg=payload.get("dominantPreferredBearingDeg"),
        dominant_aspect_deg=payload.get("dominantAspectDeg"),
    )


def _serialize_solver_context(context: SolverContext) -> dict[str, Any]:
    return {
        "grid": _serialize_grid(context.grid),
        "featureField": _serialize_feature_field(context.feature_field),
        "params": context.params.model_dump(mode="json"),
        "basicLineLengthScale": context.basic_line_length_scale,
        "practicalLineLengthScale": context.practical_line_length_scale,
    }


def _deserialize_solver_context(payload: dict[str, Any]) -> SolverContext:
    grid = _deserialize_grid(payload["grid"])
    feature_field = _deserialize_feature_field(payload["featureField"])
    return SolverContext(
        grid=grid,
        feature_field=feature_field,
        params=FlightParamsModel.model_validate(payload["params"]),
        root_area_m2=max(1.0, grid.area_m2),
        feature_lookup=_feature_lookup(feature_field),
        cell_lookup=_cell_lookup(grid),
        neighbors=_neighbor_lookup(grid),
        basic_line_length_scale=float(payload.get("basicLineLengthScale", DEFAULT_BASIC_LINE_LENGTH_SCALE)),
        practical_line_length_scale=float(payload.get("practicalLineLengthScale", DEFAULT_PRACTICAL_LINE_LENGTH_SCALE)),
    )


def _serialize_boundary(boundary: BoundaryStats) -> dict[str, float]:
    return {
        "sharedBoundaryM": boundary.shared_boundary_m,
        "breakWeightSum": boundary.break_weight_sum,
    }


def _deserialize_boundary(payload: dict[str, Any]) -> BoundaryStats:
    return BoundaryStats(
        shared_boundary_m=payload["sharedBoundaryM"],
        break_weight_sum=payload["breakWeightSum"],
    )


def _serialize_root_split_task(task: RootSplitTask) -> dict[str, Any]:
    return {
        "leftIds": list(task.left_ids),
        "rightIds": list(task.right_ids),
        "boundary": _serialize_boundary(task.boundary),
        "directionDeg": task.direction_deg,
        "threshold": task.threshold,
        "depth": task.depth,
    }


def _deserialize_root_split_task(payload: dict[str, Any]) -> RootSplitTask:
    return RootSplitTask(
        left_ids=tuple(payload["leftIds"]),
        right_ids=tuple(payload["rightIds"]),
        boundary=_deserialize_boundary(payload["boundary"]),
        direction_deg=payload.get("directionDeg", 0.0),
        threshold=payload.get("threshold", 0.0),
        depth=payload["depth"],
    )


def _serialize_subtree_task(task: SubtreeSolveTask) -> dict[str, Any]:
    return {
        "cellIds": list(task.cell_ids),
        "depth": task.depth,
    }


def _deserialize_subtree_task(payload: dict[str, Any]) -> SubtreeSolveTask:
    return SubtreeSolveTask(
        cell_ids=tuple(payload["cellIds"]),
        depth=payload["depth"],
    )


def _serialize_region_objective(objective: RegionObjective) -> dict[str, Any]:
    return {
        "bearingDeg": objective.bearing_deg,
        "normalizedQualityCost": objective.normalized_quality_cost,
        "totalMissionTimeSec": objective.total_mission_time_sec,
        "weightedMeanMismatchDeg": objective.weighted_mean_mismatch_deg,
        "areaM2": objective.area_m2,
        "convexity": objective.convexity,
        "compactness": objective.compactness,
        "boundaryBreakAlignment": objective.boundary_break_alignment,
        "flightLineCount": objective.flight_line_count,
        "lineSpacingM": objective.line_spacing_m,
        "alongTrackLengthM": objective.along_track_length_m,
        "crossTrackWidthM": objective.cross_track_width_m,
        "fragmentedLineFraction": objective.fragmented_line_fraction,
        "overflightTransitFraction": objective.overflight_transit_fraction,
        "shortLineFraction": objective.short_line_fraction,
        "meanLineLengthM": objective.mean_line_length_m,
        "medianLineLengthM": objective.median_line_length_m,
        "meanLineLiftM": objective.mean_line_lift_m,
        "p90LineLiftM": objective.p90_line_lift_m,
        "maxLineLiftM": objective.max_line_lift_m,
        "elevatedAreaFraction": objective.elevated_area_fraction,
        "severeLiftAreaFraction": objective.severe_lift_area_fraction,
    }


def _deserialize_region_objective(payload: dict[str, Any]) -> RegionObjective:
    return RegionObjective(
        bearing_deg=payload["bearingDeg"],
        normalized_quality_cost=payload["normalizedQualityCost"],
        total_mission_time_sec=payload["totalMissionTimeSec"],
        weighted_mean_mismatch_deg=payload["weightedMeanMismatchDeg"],
        area_m2=payload["areaM2"],
        convexity=payload["convexity"],
        compactness=payload["compactness"],
        boundary_break_alignment=payload["boundaryBreakAlignment"],
        flight_line_count=payload["flightLineCount"],
        line_spacing_m=payload["lineSpacingM"],
        along_track_length_m=payload["alongTrackLengthM"],
        cross_track_width_m=payload["crossTrackWidthM"],
        fragmented_line_fraction=payload["fragmentedLineFraction"],
        overflight_transit_fraction=payload["overflightTransitFraction"],
        short_line_fraction=payload["shortLineFraction"],
        mean_line_length_m=payload["meanLineLengthM"],
        median_line_length_m=payload["medianLineLengthM"],
        mean_line_lift_m=payload["meanLineLiftM"],
        p90_line_lift_m=payload["p90LineLiftM"],
        max_line_lift_m=payload["maxLineLiftM"],
        elevated_area_fraction=payload["elevatedAreaFraction"],
        severe_lift_area_fraction=payload["severeLiftAreaFraction"],
    )


def _serialize_evaluated_region(region: EvaluatedRegion) -> dict[str, Any]:
    return {
        "cellIds": list(region.cell_ids),
        "ring": region.ring,
        "objective": _serialize_region_objective(region.objective),
        "score": region.score,
        "hardInvalid": region.hard_invalid,
    }


def _serialize_geometry_tree(node: PartitionLeafGeometry | PartitionSplitGeometry) -> dict[str, Any]:
    if isinstance(node, PartitionLeafGeometry):
        return {
            "type": "leaf",
            "cellIds": list(node.cell_ids),
        }
    return {
        "type": "split",
        "directionDeg": node.direction_deg,
        "threshold": node.threshold,
        "left": _serialize_geometry_tree(node.left),
        "right": _serialize_geometry_tree(node.right),
    }


def _deserialize_geometry_tree(payload: dict[str, Any]) -> PartitionLeafGeometry | PartitionSplitGeometry:
    if payload["type"] == "leaf":
        return PartitionLeafGeometry(cell_ids=tuple(payload["cellIds"]))
    return PartitionSplitGeometry(
        direction_deg=payload["directionDeg"],
        threshold=payload["threshold"],
        left=_deserialize_geometry_tree(payload["left"]),
        right=_deserialize_geometry_tree(payload["right"]),
    )


def _deserialize_evaluated_region(payload: dict[str, Any]) -> EvaluatedRegion:
    ring = [tuple(coord) for coord in payload["ring"]]
    return EvaluatedRegion(
        cell_ids=tuple(payload["cellIds"]),
        polygon=ring_to_polygon_mercator(ring) if ring else Polygon(),
        ring=ring,
        objective=_deserialize_region_objective(payload["objective"]),
        score=payload["score"],
        hard_invalid=payload["hardInvalid"],
    )


def _serialize_partition_plan(plan: PartitionPlan) -> dict[str, Any]:
    return {
        "regions": [_serialize_evaluated_region(region) for region in plan.regions],
        "qualityCost": plan.quality_cost,
        "missionTimeSec": plan.mission_time_sec,
        "weightedMeanMismatchDeg": plan.weighted_mean_mismatch_deg,
        "internalBoundaryM": plan.internal_boundary_m,
        "breakWeightSum": plan.break_weight_sum,
        "largestRegionFraction": plan.largest_region_fraction,
        "meanConvexity": plan.mean_convexity,
        "regionCount": plan.region_count,
        "geometryTree": _serialize_geometry_tree(plan.geometry_tree) if plan.geometry_tree is not None else None,
    }


def _deserialize_partition_plan(payload: dict[str, Any]) -> PartitionPlan:
    return PartitionPlan(
        regions=tuple(_deserialize_evaluated_region(region) for region in payload["regions"]),
        quality_cost=payload["qualityCost"],
        mission_time_sec=payload["missionTimeSec"],
        weighted_mean_mismatch_deg=payload["weightedMeanMismatchDeg"],
        internal_boundary_m=payload["internalBoundaryM"],
        break_weight_sum=payload["breakWeightSum"],
        largest_region_fraction=payload["largestRegionFraction"],
        mean_convexity=payload["meanConvexity"],
        region_count=payload["regionCount"],
        geometry_tree=_deserialize_geometry_tree(payload["geometryTree"]) if payload.get("geometryTree") is not None else None,
    )


def solve_root_split_branch_event(payload: dict[str, Any]) -> dict[str, Any]:
    context = _deserialize_solver_context(payload["context"])
    task = _deserialize_root_split_task(payload["task"])
    frontier, perf = _solve_root_split_branch_with_context(task, context)
    return {
        "plans": [_serialize_partition_plan(plan) for plan in frontier],
        "perf": perf,
    }


def solve_subtree_task_event(payload: dict[str, Any]) -> dict[str, Any]:
    context = _deserialize_solver_context(payload["context"])
    task = _deserialize_subtree_task(payload["task"])
    frontier, perf = _solve_subtree_task_with_context(task, context)
    return {
        "plans": [_serialize_partition_plan(plan) for plan in frontier],
        "perf": perf,
    }


def _lambda_client(max_pool_connections: int, read_timeout_sec: int) -> Any:
    global _LAMBDA_CLIENT, _LAMBDA_CLIENT_MAX_POOL_CONNECTIONS, _LAMBDA_CLIENT_READ_TIMEOUT_SEC
    required_pool_connections = max(8, int(max_pool_connections))
    required_read_timeout_sec = max(1, int(read_timeout_sec))
    if (
        _LAMBDA_CLIENT is not None
        and _LAMBDA_CLIENT_MAX_POOL_CONNECTIONS >= required_pool_connections
        and _LAMBDA_CLIENT_READ_TIMEOUT_SEC == required_read_timeout_sec
    ):
        return _LAMBDA_CLIENT
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("boto3 is required for Lambda root fan-out mode.") from exc

    _LAMBDA_CLIENT = boto3.client(
        "lambda",
        config=Config(
            max_pool_connections=required_pool_connections,
            read_timeout=required_read_timeout_sec,
        ),
    )
    _LAMBDA_CLIENT_MAX_POOL_CONNECTIONS = required_pool_connections
    _LAMBDA_CLIENT_READ_TIMEOUT_SEC = required_read_timeout_sec
    return _LAMBDA_CLIENT


def _invoke_root_split_branch_lambda(
    function_name: str,
    serialized_context: dict[str, Any],
    task: RootSplitTask,
    client: Any,
) -> tuple[list[PartitionPlan], dict[str, float]]:
    payload = {
        "terrainSplitterInternal": "root-split",
        "payload": {
            "context": serialized_context,
            "task": _serialize_root_split_task(task),
        },
    }
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    if "FunctionError" in response:
        error_payload = response["Payload"].read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Lambda root split invocation failed: {error_payload}")
    raw_payload = response["Payload"].read()
    decoded = json.loads(raw_payload.decode("utf-8"))
    return (
        [_deserialize_partition_plan(plan) for plan in decoded["plans"]],
        {key: float(value) for key, value in decoded.get("perf", {}).items()},
    )


def _invoke_subtree_lambda(
    function_name: str,
    serialized_context: dict[str, Any],
    task: SubtreeSolveTask,
    client: Any,
) -> tuple[list[PartitionPlan], dict[str, float]]:
    payload = {
        "terrainSplitterInternal": "subtree",
        "payload": {
            "context": serialized_context,
            "task": _serialize_subtree_task(task),
        },
    }
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    if "FunctionError" in response:
        error_payload = response["Payload"].read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Lambda subtree invocation failed: {error_payload}")
    raw_payload = response["Payload"].read()
    decoded = json.loads(raw_payload.decode("utf-8"))
    return (
        [_deserialize_partition_plan(plan) for plan in decoded["plans"]],
        {key: float(value) for key, value in decoded.get("perf", {}).items()},
    )


def _solve_root_splits_via_lambda(
    tasks: list[RootSplitTask],
    context: SolverContext,
    max_workers: int,
) -> tuple[list[PartitionPlan], dict[str, float]]:
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if not function_name:
        raise RuntimeError("AWS_LAMBDA_FUNCTION_NAME is required for Lambda root fan-out mode.")
    perf = _make_perf()
    plans: list[PartitionPlan] = []
    serialized_context = _serialize_solver_context(context)
    read_timeout_sec = _resolve_lambda_invoke_read_timeout_sec(None)
    client = _lambda_client(max_workers, read_timeout_sec)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_invoke_root_split_branch_lambda, function_name, serialized_context, task, client)
            for task in tasks
        ]
        for future in futures:
            branch_frontier, branch_perf = future.result()
            plans.extend(branch_frontier)
            _merge_perf(perf, branch_perf)
    return plans, perf


def _solve_subtrees_via_lambda(
    tasks: list[tuple[int, str, SubtreeSolveTask]],
    context: SolverContext,
    max_workers: int,
) -> tuple[dict[int, dict[str, list[PartitionPlan]]], dict[str, float]]:
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if not function_name:
        raise RuntimeError("AWS_LAMBDA_FUNCTION_NAME is required for Lambda subtree fan-out mode.")
    perf = _make_perf()
    subtree_results: dict[int, dict[str, list[PartitionPlan]]] = defaultdict(dict)
    serialized_context = _serialize_solver_context(context)
    read_timeout_sec = _resolve_lambda_invoke_read_timeout_sec(None)
    client = _lambda_client(max_workers, read_timeout_sec)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            (index, side, executor.submit(_invoke_subtree_lambda, function_name, serialized_context, task, client))
            for index, side, task in tasks
        ]
        for index, side, future in futures:
            frontier, subtree_perf = future.result()
            subtree_results[index][side] = frontier
            _merge_perf(perf, subtree_perf)
    return subtree_results, perf


def _should_use_nested_lambda_fanout(
    cell_ids: tuple[int, ...],
    depth: int,
    split_candidates: list[SplitCandidate],
) -> bool:
    if depth < DEFAULT_NESTED_LAMBDA_MIN_DEPTH:
        return False
    if len(cell_ids) < _resolve_nested_lambda_min_cells(None):
        return False
    if len(split_candidates) < 2:
        return False
    if _resolve_root_parallel_mode(None) != "lambda":
        return False
    if _resolve_root_parallel_granularity(None) != "subtree":
        return False
    if _resolve_root_parallel_workers(None) <= 1:
        return False
    if _resolve_nested_lambda_max_inflight(None) <= 1:
        return False
    if not os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return False
    return True


def _solve_split_candidates_via_nested_lambda(
    split_candidates: list[SplitCandidate],
    depth: int,
    context: SolverContext,
    perf: dict[str, float],
) -> list[PartitionPlan]:
    subtree_tasks: list[tuple[int, str, SubtreeSolveTask]] = []
    for index, split in enumerate(split_candidates):
        subtree_tasks.append((index, "left", SubtreeSolveTask(cell_ids=split.left_ids, depth=depth - 1)))
        subtree_tasks.append((index, "right", SubtreeSolveTask(cell_ids=split.right_ids, depth=depth - 1)))
    requested_workers = _resolve_root_parallel_workers(None)
    nested_max_inflight = _resolve_nested_lambda_max_inflight(None)
    subtree_workers = _resolve_lambda_parallel_invocations(
        len(subtree_tasks),
        min(max(requested_workers, len(split_candidates)), len(subtree_tasks)),
        nested_max_inflight,
    )
    perf["nested_parallel_invocations"] += 1
    perf["nested_parallel_split_count"] += len(split_candidates)
    perf["nested_parallel_subtree_tasks"] += len(subtree_tasks)
    perf["nested_parallel_workers_used_max"] = max(perf["nested_parallel_workers_used_max"], subtree_workers)
    nested_started_at = time.perf_counter()
    subtree_results, lambda_perf = _solve_subtrees_via_lambda(
        subtree_tasks,
        context,
        subtree_workers,
    )
    perf["nested_parallel_ms"] += (time.perf_counter() - nested_started_at) * 1000.0
    _merge_perf(perf, lambda_perf)

    candidates: list[PartitionPlan] = []
    for index, split in enumerate(split_candidates):
        left_frontier = subtree_results.get(index, {}).get("left", [])
        right_frontier = subtree_results.get(index, {}).get("right", [])
        if not left_frontier or not right_frontier:
            continue
        _extend_combined_candidates(
            candidates,
            left_frontier,
            right_frontier,
            split.boundary,
            split,
            perf,
        )
    return candidates


def solve_partition_hierarchy(
    grid: GridData,
    feature_field: FeatureField,
    params: FlightParamsModel,
    requested_tradeoff: float | None = None,
    *,
    request_id: str | None = None,
    polygon_id: str | None = None,
    root_parallel_workers: int | None = None,
    root_parallel_mode: Literal["process", "lambda"] | None = None,
    root_parallel_granularity: Literal["branch", "subtree"] | None = None,
    root_parallel_max_inflight: int | None = None,
    line_length_scale: float | None = None,
    debug_output: dict[str, Any] | None = None,
) -> list[PartitionSolutionPreviewModel]:
    practical_line_length_scale = (
        DEFAULT_PRACTICAL_LINE_LENGTH_SCALE if line_length_scale is None else float(line_length_scale)
    )
    basic_line_length_scale = DEFAULT_BASIC_LINE_LENGTH_SCALE

    solve_started_at = time.perf_counter()
    if not grid.cells:
        return []

    feature_lookup = _feature_lookup(feature_field)
    cell_lookup = _cell_lookup(grid)
    neighbors = _neighbor_lookup(grid)
    perf = _make_perf()
    profile_hotspots_enabled = debug_output is not None or (
        os.environ.get("TERRAIN_SPLITTER_PROFILE_HOTSPOTS", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    perf_hotspots: dict[str, list[dict[str, Any]]] | None = (
        {"buildRegion": [], "objective": []} if profile_hotspots_enabled else None
    )
    caches = _make_solver_caches()
    root_area_m2 = max(1.0, grid.area_m2)
    context = SolverContext(
        grid=grid,
        feature_field=feature_field,
        params=params,
        root_area_m2=root_area_m2,
        feature_lookup=feature_lookup,
        cell_lookup=cell_lookup,
        neighbors=neighbors,
        basic_line_length_scale=basic_line_length_scale,
        practical_line_length_scale=practical_line_length_scale,
    )
    max_depth = DEFAULT_DEPTH_SMALL if len(grid.cells) <= 140 else DEFAULT_DEPTH_LARGE

    root_cell_ids = tuple(sorted(cell_lookup))
    requested_workers = _resolve_root_parallel_workers(root_parallel_workers)
    parallel_mode = _resolve_root_parallel_mode(root_parallel_mode)
    parallel_granularity = _resolve_root_parallel_granularity(root_parallel_granularity)
    lambda_max_inflight = _resolve_root_parallel_max_inflight(root_parallel_max_inflight)
    all_plans: list[PartitionPlan]
    if requested_workers <= 1:
        all_plans = _solve_region_recursive(root_cell_ids, max_depth, context, caches, perf, perf_hotspots)
    else:
        baseline_region = _build_region_for_context(root_cell_ids, 0.0, context, caches, perf, perf_hotspots)
        if baseline_region is None:
            all_plans = []
        else:
            baseline_plan = _plan_from_regions((baseline_region,), 0.0, 0.0)
            if max_depth <= 0 or len(root_cell_ids) < 4 or baseline_region.objective.flight_line_count < 1:
                perf["baseline_leaf_plans"] += 1
                all_plans = [baseline_plan]
            else:
                root_splits = _generate_split_candidates_for_context(
                    baseline_region,
                    baseline_plan,
                    root_cell_ids,
                    context,
                    caches,
                    perf,
                )
                usable_workers = min(requested_workers, len(root_splits))
                if usable_workers <= 1 or not root_splits:
                    all_plans = _solve_region_recursive(root_cell_ids, max_depth, context, caches, perf, perf_hotspots)
                else:
                    perf["root_parallel_requested_workers"] = requested_workers
                    perf["root_parallel_workers_used"] = usable_workers
                    perf["root_parallel_split_count"] = len(root_splits)
                    if lambda_max_inflight is not None:
                        perf["root_parallel_max_inflight"] = lambda_max_inflight
                    branch_started_at = time.perf_counter()
                    branch_candidates: list[PartitionPlan] = []
                    tasks = [
                        RootSplitTask(
                            left_ids=split.left_ids,
                            right_ids=split.right_ids,
                            boundary=split.boundary,
                            direction_deg=split.direction_deg,
                            threshold=split.threshold,
                            depth=max_depth - 1,
                        )
                        for split in root_splits
                    ]
                    try:
                        if parallel_mode == "lambda":
                            if parallel_granularity == "subtree":
                                subtree_tasks: list[tuple[int, str, SubtreeSolveTask]] = []
                                for index, split in enumerate(root_splits):
                                    subtree_tasks.append((index, "left", SubtreeSolveTask(cell_ids=split.left_ids, depth=max_depth - 1)))
                                    subtree_tasks.append((index, "right", SubtreeSolveTask(cell_ids=split.right_ids, depth=max_depth - 1)))
                                subtree_workers = _resolve_lambda_parallel_invocations(
                                    len(subtree_tasks),
                                    min(max(requested_workers, len(tasks)), len(subtree_tasks)),
                                    lambda_max_inflight,
                                )
                                perf["root_parallel_subtree_tasks"] = len(subtree_tasks)
                                perf["root_parallel_workers_used"] = subtree_workers
                                subtree_results, lambda_perf = _solve_subtrees_via_lambda(
                                    subtree_tasks,
                                    context,
                                    subtree_workers,
                                )
                                _merge_perf(perf, lambda_perf)
                                for index, split in enumerate(root_splits):
                                    left_frontier = subtree_results.get(index, {}).get("left", [])
                                    right_frontier = subtree_results.get(index, {}).get("right", [])
                                    if not left_frontier or not right_frontier:
                                        continue
                                    _extend_combined_candidates(
                                        branch_candidates,
                                        left_frontier,
                                        right_frontier,
                                        split.boundary,
                                        split,
                                        perf,
                                    )
                            else:
                                branch_workers = _resolve_lambda_parallel_invocations(
                                    len(tasks),
                                    usable_workers,
                                    lambda_max_inflight,
                                )
                                perf["root_parallel_workers_used"] = branch_workers
                                lambda_plans, lambda_perf = _solve_root_splits_via_lambda(tasks, context, branch_workers)
                                branch_candidates.extend(lambda_plans)
                                _merge_perf(perf, lambda_perf)
                        else:
                            if parallel_granularity == "subtree":
                                subtree_tasks: list[tuple[int, str, SubtreeSolveTask]] = []
                                for index, split in enumerate(root_splits):
                                    subtree_tasks.append((index, "left", SubtreeSolveTask(cell_ids=split.left_ids, depth=max_depth - 1)))
                                    subtree_tasks.append((index, "right", SubtreeSolveTask(cell_ids=split.right_ids, depth=max_depth - 1)))
                                subtree_workers = min(max(requested_workers, len(tasks)), len(subtree_tasks))
                                perf["root_parallel_subtree_tasks"] = len(subtree_tasks)
                                perf["root_parallel_workers_used"] = subtree_workers
                                subtree_results: dict[int, dict[str, list[PartitionPlan]]] = defaultdict(dict)
                                with ProcessPoolExecutor(
                                    max_workers=subtree_workers,
                                    initializer=_init_parallel_solver_context,
                                    initargs=(context,),
                                ) as executor:
                                    for (index, side, _), (subtree_frontier, subtree_perf) in zip(
                                        subtree_tasks,
                                        executor.map(_solve_subtree_task, [task for _, _, task in subtree_tasks]),
                                        strict=True,
                                    ):
                                        subtree_results[index][side] = subtree_frontier
                                        _merge_perf(perf, subtree_perf)
                                for index, split in enumerate(root_splits):
                                    left_frontier = subtree_results.get(index, {}).get("left", [])
                                    right_frontier = subtree_results.get(index, {}).get("right", [])
                                    if not left_frontier or not right_frontier:
                                        continue
                                    _extend_combined_candidates(
                                        branch_candidates,
                                        left_frontier,
                                        right_frontier,
                                        split.boundary,
                                        split,
                                        perf,
                                    )
                            else:
                                with ProcessPoolExecutor(
                                    max_workers=usable_workers,
                                    initializer=_init_parallel_solver_context,
                                    initargs=(context,),
                                ) as executor:
                                    for branch_frontier, branch_perf in executor.map(_solve_root_split_branch, tasks):
                                        branch_candidates.extend(branch_frontier)
                                        _merge_perf(perf, branch_perf)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[terrain-split-backend][%s] root-parallel solve failed polygonId=%s mode=%s granularity=%s; falling back to serial",
                            request_id or "<none>",
                            polygon_id or "<none>",
                            parallel_mode,
                            parallel_granularity,
                        )
                        perf["root_parallel_failures"] += 1
                        all_plans = _solve_region_recursive(root_cell_ids, max_depth, context, caches, perf, perf_hotspots)
                    else:
                        perf["root_parallel_ms"] += (time.perf_counter() - branch_started_at) * 1000.0
                        all_plans = _pareto_frontier_with_perf(
                            [baseline_plan] + branch_candidates,
                            root_area_m2,
                            practical_line_length_scale,
                            perf,
                        )
                        perf["frontier_plan_count"] += len(all_plans)
    if not all_plans:
        total_ms = (time.perf_counter() - solve_started_at) * 1000.0
        logger.info(
            "[terrain-split-backend][%s] solver finished polygonId=%s cells=%d maxDepth=%d totalMs=%.1f allPlans=0 rootParallelMode=%s rootParallelGranularity=%s rootParallelRequested=%d rootParallelUsed=%d rootParallelSplits=%d solveRegionCalls=%d cacheHits=%d regionCacheHits=%d regionCacheMisses=%d regionStaticHits=%d regionStaticMisses=%d regionBearingHits=%d regionBearingMisses=%d buildRegionCalls=%d objectiveCalls=%d splitAttempts=%d kept=%d",
            request_id or "<none>",
            polygon_id or "<none>",
            len(grid.cells),
            max_depth,
            total_ms,
            parallel_mode,
            parallel_granularity,
            requested_workers,
            int(perf["root_parallel_workers_used"]),
            int(perf["root_parallel_split_count"]),
            int(perf["solve_region_calls"]),
            int(perf["solve_region_cache_hits"]),
            int(perf["region_cache_hits"]),
            int(perf["region_cache_misses"]),
            int(perf["region_static_hits"]),
            int(perf["region_static_misses"]),
            int(perf["region_bearing_hits"]),
            int(perf["region_bearing_misses"]),
            int(perf["build_region_calls"]),
            int(perf["objective_calls"]),
            int(perf["split_attempts"]),
            int(perf["split_candidates_kept"]),
        )
        _populate_solver_debug_output(
            debug_output,
            request_id=request_id,
            polygon_id=polygon_id,
            grid=grid,
            requested_tradeoff=requested_tradeoff,
            max_depth=max_depth,
            basic_line_length_scale=basic_line_length_scale,
            practical_line_length_scale=practical_line_length_scale,
            caches=caches,
            perf=perf,
            all_plans=[],
            practical_plans=[],
            returned_plans=[],
            returned_previews=[],
            practical_plan_rejection_summary={},
            practical_region_rejection_summary={},
            perf_hotspots=perf_hotspots,
        )
        _log_perf_hotspots(request_id, polygon_id, perf_hotspots)
        return []

    baseline = min(
        (plan for plan in all_plans if plan.region_count == 1),
        key=lambda plan: (plan.quality_cost, plan.mission_time_sec),
        default=None,
    )
    exact_all_plans: list[PartitionPlan] = []
    for plan in all_plans:
        exact_plan = _reevaluate_plan_with_exact_geometry(plan, context, caches, perf)
        if exact_plan is not None:
            exact_all_plans.append(exact_plan)
    if exact_all_plans:
        all_plans = _pareto_frontier_with_perf(exact_all_plans, root_area_m2, practical_line_length_scale, perf)
        baseline = min(
            (plan for plan in all_plans if plan.region_count == 1),
            key=lambda plan: (plan.quality_cost, plan.mission_time_sec),
            default=None,
        )

    practical_plan_rejection_summary: dict[str, Any] = {}
    practical_region_rejection_summary: dict[str, Any] = {}
    for plan in all_plans:
        if plan.region_count <= 1:
            continue
        plan_diag = _plan_gate_diagnostics(plan, root_area_m2, line_length_scale=practical_line_length_scale)
        plan_failures, region_failures = _plan_gate_failure_margins(plan_diag)
        if plan_failures:
            _record_plan_failure_summary(
                practical_plan_rejection_summary,
                practical_region_rejection_summary,
                plan,
                plan_diag,
                plan_failures,
                region_failures,
            )
    practical = [plan for plan in all_plans if _plan_is_practical(plan, root_area_m2, practical_line_length_scale)]
    if not practical:
        relaxed_fallback: list[RelaxedFallbackCandidate] = []
        if baseline is not None:
            relaxed_fallback = _generate_relaxed_root_fallback_candidates(
                baseline.regions[0],
                baseline,
                root_cell_ids,
                context,
                caches,
                perf,
            )
        if relaxed_fallback:
            logger.info(
                "[terrain-split-backend][%s] relaxed fallback accepted polygonId=%s lineLengthScale=%.2f candidateCount=%d bestScore=%.4f bestSoftTotal=%.4f bestSoftMax=%.4f bestDirectionDeg=%.4f bestThreshold=%.4f",
                request_id or "<none>",
                polygon_id or "<none>",
                practical_line_length_scale,
                len(relaxed_fallback),
                relaxed_fallback[0].fallback_score,
                relaxed_fallback[0].soft_total_margin,
                relaxed_fallback[0].soft_max_margin,
                relaxed_fallback[0].direction_deg,
                relaxed_fallback[0].threshold,
            )
            practical_filtered = [candidate.plan for candidate in relaxed_fallback[:1]]
            time_min = min(plan.mission_time_sec for plan in practical_filtered)
            time_max = max(plan.mission_time_sec for plan in practical_filtered)
            previews: list[PartitionSolutionPreviewModel] = []
            for index, plan in enumerate(practical_filtered):
                preview_polygons = [region.polygon for region in plan.regions]
                simplify_tolerance_m = clamp(
                    grid.grid_step_m * OUTPUT_RING_SIMPLIFY_FACTOR,
                    OUTPUT_RING_SIMPLIFY_MIN_M,
                    OUTPUT_RING_SIMPLIFY_MAX_M,
                )
                simplified_polygons = simplify_polygon_coverage(
                    [polygon for polygon in preview_polygons if polygon is not None],
                    simplify_tolerance_m,
                    simplify_boundary=True,
                )
                simplified_rings = [polygon_to_lnglat_ring(polygon) for polygon in simplified_polygons]
                boundary_break_alignment = _plan_boundary_alignment(plan)
                if time_max - time_min <= 1e-6:
                    tradeoff = requested_tradeoff if requested_tradeoff is not None else 0.5
                else:
                    tradeoff = clamp((plan.mission_time_sec - time_min) / (time_max - time_min), 0.0, 1.0)
                previews.append(
                    PartitionSolutionPreviewModel(
                        signature=hash_signature(
                            {
                                "regions": [
                                    {
                                        "ring": ring,
                                        "bearingDeg": round(region.objective.bearing_deg, 4),
                                        "areaM2": round(region.objective.area_m2, 3),
                                    }
                                    for region, ring in zip(plan.regions, simplified_rings)
                                ]
                            }
                        ),
                        tradeoff=float(tradeoff),
                        regionCount=plan.region_count,
                        totalMissionTimeSec=plan.mission_time_sec,
                        normalizedQualityCost=plan.quality_cost,
                        weightedMeanMismatchDeg=plan.weighted_mean_mismatch_deg,
                        hierarchyLevel=index + 1,
                        largestRegionFraction=plan.largest_region_fraction,
                        meanConvexity=plan.mean_convexity,
                        boundaryBreakAlignment=boundary_break_alignment,
                        isFirstPracticalSplit=False,
                        regions=[
                            RegionPreview(
                                areaM2=region.objective.area_m2,
                                bearingDeg=region.objective.bearing_deg,
                                atomCount=len(region.cell_ids),
                                ring=ring,
                                convexity=region.objective.convexity,
                                compactness=region.objective.compactness,
                                baseAltitudeAGL=params.altitudeAGL,
                            )
                            for region, ring in zip(plan.regions, simplified_rings)
                        ],
                    )
                )
            _populate_solver_debug_output(
                debug_output,
                request_id=request_id,
                polygon_id=polygon_id,
                grid=grid,
                requested_tradeoff=requested_tradeoff,
                max_depth=max_depth,
                basic_line_length_scale=basic_line_length_scale,
                practical_line_length_scale=practical_line_length_scale,
                caches=caches,
                perf=perf,
                all_plans=all_plans,
                practical_plans=practical_filtered,
                returned_plans=practical_filtered,
                returned_previews=previews,
                practical_plan_rejection_summary=practical_plan_rejection_summary,
                practical_region_rejection_summary=practical_region_rejection_summary,
                relaxed_fallback=relaxed_fallback,
                perf_hotspots=perf_hotspots,
            )
            _log_perf_hotspots(request_id, polygon_id, perf_hotspots)
            return previews
        rejection_payload = _rejection_debug_payload(
            caches=caches,
            basic_line_length_scale=basic_line_length_scale,
            practical_line_length_scale=practical_line_length_scale,
            practical_plan_rejection_summary=practical_plan_rejection_summary,
            practical_region_rejection_summary=practical_region_rejection_summary,
        )
        logger.info(
            "[terrain-split-backend][%s] rejection diagnostics polygonId=%s details=%s",
            request_id or "<none>",
            polygon_id or "<none>",
            json.dumps(
                rejection_payload,
                sort_keys=True,
            ),
        )
        total_ms = (time.perf_counter() - solve_started_at) * 1000.0
        logger.info(
            "[terrain-split-backend][%s] solver finished polygonId=%s cells=%d maxDepth=%d totalMs=%.1f allPlans=%d practicalPlans=0 rootParallelMode=%s rootParallelGranularity=%s rootParallelRequested=%d rootParallelUsed=%d rootParallelSplits=%d rootParallelSubtreeTasks=%d rootParallelMs=%.1f nestedParallelInvocations=%d nestedParallelWorkersMax=%d nestedParallelTasks=%d nestedParallelMs=%.1f nestedParallelFailures=%d solveRegionCalls=%d cacheHits=%d regionCacheHits=%d regionCacheMisses=%d regionStaticHits=%d regionStaticMisses=%d regionBearingHits=%d regionBearingMisses=%d buildRegionCalls=%d buildRegionMs=%.1f regionStaticBuildMs=%.1f polygonMs=%.1f objectiveCalls=%d objectiveMs=%.1f regionBearingCoreMs=%.1f exactRegionObjectiveCalls=%d exactRegionObjectiveMs=%.1f nodeCostMs=%.1f lineLiftMs=%.1f flightTimeMs=%.1f shapeMetricMs=%.1f splitGenMs=%.1f splitAttempts=%d kept=%d returned=%d smallChildRejects=%d duplicateRejects=%d disconnectedRejects=%d boundaryRejects=%d regionBuildRejects=%d validityRejects=%d nonImprovingRejects=%d combinedCandidates=%d frontierStates=%d",
            request_id or "<none>",
            polygon_id or "<none>",
            len(grid.cells),
            max_depth,
            total_ms,
            len(all_plans),
            parallel_mode,
            parallel_granularity,
            requested_workers,
            int(perf["root_parallel_workers_used"]),
            int(perf["root_parallel_split_count"]),
            int(perf["root_parallel_subtree_tasks"]),
            perf["root_parallel_ms"],
            int(perf["nested_parallel_invocations"]),
            int(perf["nested_parallel_workers_used_max"]),
            int(perf["nested_parallel_subtree_tasks"]),
            perf["nested_parallel_ms"],
            int(perf["nested_parallel_failures"]),
            int(perf["solve_region_calls"]),
            int(perf["solve_region_cache_hits"]),
            int(perf["region_cache_hits"]),
            int(perf["region_cache_misses"]),
            int(perf["region_static_hits"]),
            int(perf["region_static_misses"]),
            int(perf["region_bearing_hits"]),
            int(perf["region_bearing_misses"]),
            int(perf["build_region_calls"]),
            perf["build_region_ms"],
            perf["region_static_build_ms"],
            perf["build_region_polygon_ms"],
            int(perf["objective_calls"]),
            perf["objective_ms"],
            perf["region_bearing_core_ms"],
            int(perf["exact_region_objective_calls"]),
            perf["exact_region_objective_ms"],
            perf["node_cost_ms"],
            perf["line_lift_ms"],
            perf["flight_time_ms"],
            perf["shape_metric_ms"],
            perf["split_generation_ms"],
            int(perf["split_attempts"]),
            int(perf["split_candidates_kept"]),
            int(perf["split_candidates_returned"]),
            int(perf["split_small_child_rejections"]),
            int(perf["split_duplicate_rejections"]),
            int(perf["split_disconnected_rejections"]),
            int(perf["split_boundary_rejections"]),
            int(perf["split_region_build_rejections"]),
            int(perf["split_basic_validity_rejections"]),
            int(perf["split_non_improving_rejections"]),
            int(perf["combined_plan_candidates"]),
            int(perf["frontier_plan_count"]),
        )
        _populate_solver_debug_output(
            debug_output,
            request_id=request_id,
            polygon_id=polygon_id,
            grid=grid,
            requested_tradeoff=requested_tradeoff,
            max_depth=max_depth,
            basic_line_length_scale=basic_line_length_scale,
            practical_line_length_scale=practical_line_length_scale,
            caches=caches,
            perf=perf,
            all_plans=all_plans,
            practical_plans=[],
            returned_plans=[],
            returned_previews=[],
            practical_plan_rejection_summary=practical_plan_rejection_summary,
            practical_region_rejection_summary=practical_region_rejection_summary,
            perf_hotspots=perf_hotspots,
        )
        _log_perf_hotspots(request_id, polygon_id, perf_hotspots)
        return []

    comparison_pool = practical[:]
    if baseline is not None:
        comparison_pool.append(baseline)
    filtered = _pareto_frontier_with_perf(comparison_pool, root_area_m2, practical_line_length_scale, perf)
    practical_filtered = [
        plan
        for plan in filtered
        if plan.region_count > 1 and _plan_is_practical(plan, root_area_m2, practical_line_length_scale)
    ]
    if not practical_filtered:
        _populate_solver_debug_output(
            debug_output,
            request_id=request_id,
            polygon_id=polygon_id,
            grid=grid,
            requested_tradeoff=requested_tradeoff,
            max_depth=max_depth,
            basic_line_length_scale=basic_line_length_scale,
            practical_line_length_scale=practical_line_length_scale,
            caches=caches,
            perf=perf,
            all_plans=all_plans,
            practical_plans=practical,
            returned_plans=[],
            returned_previews=[],
            practical_plan_rejection_summary=practical_plan_rejection_summary,
            practical_region_rejection_summary=practical_region_rejection_summary,
            perf_hotspots=perf_hotspots,
        )
        _log_perf_hotspots(request_id, polygon_id, perf_hotspots)
        return []

    best_two_region = min(
        (plan for plan in practical if plan.region_count == 2),
        key=lambda plan: (plan.quality_cost, plan.mission_time_sec),
        default=None,
    )
    if best_two_region is not None and not any(_plan_signature(plan) == _plan_signature(best_two_region) for plan in practical_filtered):
        practical_filtered.append(best_two_region)

    practical_filtered.sort(key=lambda plan: (plan.region_count, plan.mission_time_sec, plan.quality_cost))
    time_min = min(plan.mission_time_sec for plan in practical_filtered)
    time_max = max(plan.mission_time_sec for plan in practical_filtered)
    first_practical_signature = _plan_signature(practical_filtered[0]) if practical_filtered else None

    previews: list[PartitionSolutionPreviewModel] = []
    for index, plan in enumerate(practical_filtered):
        preview_polygons = [region.polygon for region in plan.regions]
        simplify_tolerance_m = clamp(
            grid.grid_step_m * OUTPUT_RING_SIMPLIFY_FACTOR,
            OUTPUT_RING_SIMPLIFY_MIN_M,
            OUTPUT_RING_SIMPLIFY_MAX_M,
        )
        simplified_polygons = simplify_polygon_coverage(
            [polygon for polygon in preview_polygons if polygon is not None],
            simplify_tolerance_m,
            simplify_boundary=True,
        )
        simplified_rings = [polygon_to_lnglat_ring(polygon) for polygon in simplified_polygons]
        boundary_break_alignment = _plan_boundary_alignment(plan)
        if time_max - time_min <= 1e-6:
            tradeoff = requested_tradeoff if requested_tradeoff is not None else 0.5
        else:
            tradeoff = clamp((plan.mission_time_sec - time_min) / (time_max - time_min), 0.0, 1.0)
        preview = PartitionSolutionPreviewModel(
            signature=hash_signature(
                {
                    "regions": [
                        {
                            "ring": ring,
                            "bearingDeg": round(region.objective.bearing_deg, 4),
                            "areaM2": round(region.objective.area_m2, 3),
                        }
                        for region, ring in zip(plan.regions, simplified_rings)
                    ]
                }
            ),
            tradeoff=float(tradeoff),
            regionCount=plan.region_count,
            totalMissionTimeSec=plan.mission_time_sec,
            normalizedQualityCost=plan.quality_cost,
            weightedMeanMismatchDeg=plan.weighted_mean_mismatch_deg,
            hierarchyLevel=index + 1,
            largestRegionFraction=plan.largest_region_fraction,
            meanConvexity=plan.mean_convexity,
            boundaryBreakAlignment=boundary_break_alignment,
            isFirstPracticalSplit=first_practical_signature is not None and _plan_signature(plan) == first_practical_signature,
            regions=[
                RegionPreview(
                    areaM2=region.objective.area_m2,
                    bearingDeg=region.objective.bearing_deg,
                    atomCount=len(region.cell_ids),
                    ring=ring,
                    convexity=region.objective.convexity,
                    compactness=region.objective.compactness,
                    baseAltitudeAGL=params.altitudeAGL,
                )
                for region, ring in zip(plan.regions, simplified_rings)
            ],
        )
        previews.append(preview)
    total_ms = (time.perf_counter() - solve_started_at) * 1000.0
    region_cache_attempts = perf["region_cache_hits"] + perf["region_cache_misses"]
    region_static_attempts = perf["region_static_hits"] + perf["region_static_misses"]
    region_bearing_attempts = perf["region_bearing_hits"] + perf["region_bearing_misses"]
    logger.info(
        "[terrain-split-backend][%s] rejection diagnostics polygonId=%s details=%s",
        request_id or "<none>",
        polygon_id or "<none>",
        json.dumps(
            _rejection_debug_payload(
                caches=caches,
                basic_line_length_scale=basic_line_length_scale,
                practical_line_length_scale=practical_line_length_scale,
                practical_plan_rejection_summary=practical_plan_rejection_summary,
                practical_region_rejection_summary=practical_region_rejection_summary,
            ),
            sort_keys=True,
        ),
    )
    logger.info(
        "[terrain-split-backend][%s] solver finished polygonId=%s cells=%d maxDepth=%d totalMs=%.1f allPlans=%d practicalPlans=%d returnedSolutions=%d rootParallelMode=%s rootParallelGranularity=%s rootParallelRequested=%d rootParallelUsed=%d rootParallelSplits=%d rootParallelSubtreeTasks=%d rootParallelMs=%.1f rootParallelFailures=%d nestedParallelInvocations=%d nestedParallelWorkersMax=%d nestedParallelTasks=%d nestedParallelMs=%.1f nestedParallelFailures=%d solveRegionCalls=%d cacheHits=%d regionCacheHits=%d regionCacheMisses=%d regionCacheHitRate=%.3f regionStaticHits=%d regionStaticMisses=%d regionStaticHitRate=%.3f regionStaticNullHits=%d regionBearingHits=%d regionBearingMisses=%d regionBearingHitRate=%.3f regionBearingRewraps=%d buildRegionCalls=%d buildRegionMs=%.1f regionStaticBuildMs=%.1f polygonMs=%.1f polygonFailures=%d objectiveCalls=%d objectiveMs=%.1f regionBearingCoreMs=%.1f exactRegionObjectiveCalls=%d exactRegionObjectiveMs=%.1f nodeCostMs=%.1f lineLiftMs=%.1f flightTimeMs=%.1f shapeMetricMs=%.1f splitGenMs=%.1f splitDirections=%d splitAttempts=%d kept=%d returned=%d smallChildRejects=%d duplicateRejects=%d disconnectedRejects=%d boundaryRejects=%d regionBuildRejects=%d validityRejects=%d nonImprovingRejects=%d combinedCandidates=%d frontierStates=%d",
        request_id or "<none>",
        polygon_id or "<none>",
        len(grid.cells),
        max_depth,
        total_ms,
        len(all_plans),
        len(practical),
        len(previews),
        parallel_mode,
        parallel_granularity,
        requested_workers,
        int(perf["root_parallel_workers_used"]),
        int(perf["root_parallel_split_count"]),
        int(perf["root_parallel_subtree_tasks"]),
        perf["root_parallel_ms"],
        int(perf["root_parallel_failures"]),
        int(perf["nested_parallel_invocations"]),
        int(perf["nested_parallel_workers_used_max"]),
        int(perf["nested_parallel_subtree_tasks"]),
        perf["nested_parallel_ms"],
        int(perf["nested_parallel_failures"]),
        int(perf["solve_region_calls"]),
        int(perf["solve_region_cache_hits"]),
        int(perf["region_cache_hits"]),
        int(perf["region_cache_misses"]),
        (perf["region_cache_hits"] / region_cache_attempts) if region_cache_attempts > 0 else 0.0,
        int(perf["region_static_hits"]),
        int(perf["region_static_misses"]),
        (perf["region_static_hits"] / region_static_attempts) if region_static_attempts > 0 else 0.0,
        int(perf["region_static_null_hits"]),
        int(perf["region_bearing_hits"]),
        int(perf["region_bearing_misses"]),
        (perf["region_bearing_hits"] / region_bearing_attempts) if region_bearing_attempts > 0 else 0.0,
        int(perf["region_bearing_rewraps"]),
        int(perf["build_region_calls"]),
        perf["build_region_ms"],
        perf["region_static_build_ms"],
        perf["build_region_polygon_ms"],
        int(perf["build_region_polygon_failures"]),
        int(perf["objective_calls"]),
        perf["objective_ms"],
        perf["region_bearing_core_ms"],
        int(perf["exact_region_objective_calls"]),
        perf["exact_region_objective_ms"],
        perf["node_cost_ms"],
        perf["line_lift_ms"],
        perf["flight_time_ms"],
        perf["shape_metric_ms"],
        perf["split_generation_ms"],
        int(perf["split_direction_count"]),
        int(perf["split_attempts"]),
        int(perf["split_candidates_kept"]),
        int(perf["split_candidates_returned"]),
        int(perf["split_small_child_rejections"]),
        int(perf["split_duplicate_rejections"]),
        int(perf["split_disconnected_rejections"]),
        int(perf["split_boundary_rejections"]),
        int(perf["split_region_build_rejections"]),
        int(perf["split_basic_validity_rejections"]),
        int(perf["split_non_improving_rejections"]),
        int(perf["combined_plan_candidates"]),
        int(perf["frontier_plan_count"]),
    )
    _populate_solver_debug_output(
        debug_output,
        request_id=request_id,
        polygon_id=polygon_id,
        grid=grid,
        requested_tradeoff=requested_tradeoff,
        max_depth=max_depth,
        basic_line_length_scale=basic_line_length_scale,
        practical_line_length_scale=practical_line_length_scale,
        caches=caches,
        perf=perf,
        all_plans=all_plans,
        practical_plans=practical,
        returned_plans=practical_filtered,
        returned_previews=previews,
        practical_plan_rejection_summary=practical_plan_rejection_summary,
        practical_region_rejection_summary=practical_region_rejection_summary,
        perf_hotspots=perf_hotspots,
    )
    _log_perf_hotspots(request_id, polygon_id, perf_hotspots)
    return previews
