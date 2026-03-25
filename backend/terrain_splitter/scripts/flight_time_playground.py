from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiLineString, Polygon

from terrain_splitter.costs import estimate_region_flight_time, line_spacing_for_params
from terrain_splitter.features import compute_feature_field
from terrain_splitter.geometry import deg_to_rad, project_extents, ring_to_polygon_mercator
from terrain_splitter.grid import build_grid
from terrain_splitter.mapbox_tiles import fetch_dem_for_ring
from terrain_splitter.schemas import FlightParamsModel, TerrainSourceModel
from terrain_splitter.solver_frontier import (
    SolverContext,
    _build_region_for_context,
    _cell_lookup,
    _feature_lookup,
    _make_perf,
    _make_solver_caches,
    _neighbor_lookup,
    _region_heading_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEBUG_ROOT = REPO_ROOT / ".terrain-splitter-runtime" / "debug"
DEFAULT_PLAYGROUND_ROOT = REPO_ROOT / ".terrain-splitter-runtime" / "playground"
DEFAULT_CACHE_ROOT = REPO_ROOT / ".terrain-splitter-runtime" / "cache"


@dataclass(slots=True)
class SegmentTrace:
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    length_m: float


@dataclass(slots=True)
class GapTrace:
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    length_m: float


@dataclass(slots=True)
class SweepTrace:
    sweep_index: int
    offset_m: float
    center_xy: tuple[float, float]
    line_start_xy: tuple[float, float]
    line_end_xy: tuple[float, float]
    segments: list[SegmentTrace]
    gaps: list[GapTrace]
    total_inside_length_m: float
    total_gap_length_m: float


@dataclass(slots=True)
class ScanlineEdge:
    x1: float
    y1: float
    dx_dy: float
    min_y: float
    max_y: float


@dataclass(slots=True)
class BenchmarkStats:
    iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float


def _latest_debug_dir(debug_root: Path) -> Path:
    candidates = [path for path in debug_root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No debug artifact directories found under {debug_root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _solution_region_bearings(debug_dir: Path) -> list[float]:
    solutions_path = debug_dir / "solutions.json"
    if not solutions_path.exists():
        return [0.0, 45.0, 90.0, 135.0]
    solutions = _load_json(solutions_path)
    if not isinstance(solutions, list) or not solutions:
        return [0.0, 45.0, 90.0, 135.0]
    first = solutions[0]
    if not isinstance(first, dict):
        return [0.0, 45.0, 90.0, 135.0]
    regions = first.get("regions")
    if not isinstance(regions, list):
        return [0.0, 45.0, 90.0, 135.0]
    bearings: list[float] = []
    seen: set[int] = set()
    for region in regions:
        if not isinstance(region, dict):
            continue
        raw = region.get("bearingDeg")
        if not isinstance(raw, (int, float)) or not math.isfinite(raw):
            continue
        normalized = ((float(raw) % 180.0) + 180.0) % 180.0
        key = round(normalized * 1000)
        if key in seen:
            continue
        seen.add(key)
        bearings.append(normalized)
    return bearings or [0.0, 45.0, 90.0, 135.0]


def _build_root_heading_fixture(
    request_payload: dict[str, Any],
    *,
    grid_step_m: float | None,
    debug_dir: Path,
) -> dict[str, Any]:
    ring = [tuple(coord) for coord in request_payload["ring"]]
    params = FlightParamsModel.model_validate(request_payload["params"])
    terrain_source = TerrainSourceModel.model_validate(request_payload.get("terrainSource") or {"mode": "mapbox"})
    dem, zoom = fetch_dem_for_ring(
        ring,
        DEFAULT_CACHE_ROOT,
        grid_step_m=grid_step_m,
        terrain_source=terrain_source,
    )
    grid = build_grid(ring, dem, grid_step_m=grid_step_m)
    feature_field = compute_feature_field(grid, dem)
    feature_lookup = _feature_lookup(feature_field)
    cell_lookup = _cell_lookup(grid)
    root_cell_ids = tuple(sorted(cell_lookup))
    heading_candidates = _region_heading_candidates(root_cell_ids, feature_lookup, cell_lookup, feature_field)
    context = SolverContext(
        grid=grid,
        feature_field=feature_field,
        params=params,
        root_area_m2=max(1.0, grid.area_m2),
        feature_lookup=feature_lookup,
        cell_lookup=cell_lookup,
        neighbors=_neighbor_lookup(grid),
        basic_line_length_scale=0.35,
        practical_line_length_scale=0.40,
    )
    baseline_region = _build_region_for_context(
        root_cell_ids,
        0.0,
        context,
        _make_solver_caches(),
        _make_perf(),
    )
    return {
        "source": "solver-root-heading-candidates",
        "requestPolygonId": request_payload.get("polygonId"),
        "gridStepM": grid.grid_step_m,
        "demZoom": zoom,
        "cellCount": len(grid.cells),
        "headingCandidatesDeg": heading_candidates,
        "bestBearingDeg": baseline_region.objective.bearing_deg if baseline_region is not None else None,
        "solutionRegionBearingsDeg": _solution_region_bearings(debug_dir),
    }


def _build_synthetic_concave_u_case() -> dict[str, Any]:
    ring = [
        (0.0, 0.0),
        (1800.0, 0.0),
        (1800.0, 1800.0),
        (1250.0, 1800.0),
        (1250.0, 550.0),
        (550.0, 550.0),
        (550.0, 1800.0),
        (0.0, 1800.0),
        (0.0, 0.0),
    ]
    polygon = Polygon(ring)
    params = FlightParamsModel(
        payloadKind="camera",
        altitudeAGL=120.0,
        frontOverlap=70.0,
        sideOverlap=70.0,
        cameraKey="SONY_RX1R2",
        speedMps=12.0,
    )
    bearings = [0.0, 45.0, 90.0]
    return {
        "caseId": "synthetic_concave_u",
        "polygonId": "synthetic-concave-u",
        "polygon": polygon,
        "params": params,
        "fixturePack": {
            "source": "synthetic-concave-u",
            "requestPolygonId": "synthetic-concave-u",
            "selectedSource": "synthetic-default-bearings",
            "selectedBearingsDeg": bearings,
            "shape": "concave-u",
            "vertexCount": len(ring) - 1,
            "bounds": {
                "minX": polygon.bounds[0],
                "minY": polygon.bounds[1],
                "maxX": polygon.bounds[2],
                "maxY": polygon.bounds[3],
            },
            "notes": "Concave U-shaped polygon designed to create fragmented sweep intersections for east-west flight lines.",
        },
    }


def _build_synthetic_v_notch_case() -> dict[str, Any]:
    ring = [
        (0.0, 0.0),
        (2200.0, 0.0),
        (2200.0, 2200.0),
        (1425.0, 2200.0),
        (1100.0, 900.0),
        (775.0, 2200.0),
        (0.0, 2200.0),
        (0.0, 0.0),
    ]
    polygon = Polygon(ring)
    params = FlightParamsModel(
        payloadKind="camera",
        altitudeAGL=120.0,
        frontOverlap=70.0,
        sideOverlap=70.0,
        cameraKey="SONY_RX1R2",
        speedMps=12.0,
    )
    bearings = [0.0, 45.0, 90.0]
    return {
        "caseId": "synthetic_v_notch",
        "polygonId": "synthetic-v-notch",
        "polygon": polygon,
        "params": params,
        "fixturePack": {
            "source": "synthetic-v-notch",
            "requestPolygonId": "synthetic-v-notch",
            "selectedSource": "synthetic-default-bearings",
            "selectedBearingsDeg": bearings,
            "shape": "v-notch",
            "vertexCount": len(ring) - 1,
            "bounds": {
                "minX": polygon.bounds[0],
                "minY": polygon.bounds[1],
                "maxX": polygon.bounds[2],
                "maxY": polygon.bounds[3],
            },
            "notes": "Convex outer box with a sharp inward V-shaped notch from the top edge to stress near-merge and fragmentation behavior.",
        },
    }


def _segment_endpoints(segment: LineString) -> tuple[tuple[float, float], tuple[float, float]]:
    coords = list(segment.coords)
    start = (float(coords[0][0]), float(coords[0][1]))
    end = (float(coords[-1][0]), float(coords[-1][1]))
    return start, end


def _local_to_world(
    local_x: float,
    local_y: float,
    *,
    center_xy: tuple[float, float],
    ux: float,
    uy: float,
    px: float,
    py: float,
) -> tuple[float, float]:
    return (
        center_xy[0] + ux * local_x + px * local_y,
        center_xy[1] + uy * local_x + py * local_y,
    )


def _scanline_edges(
    polygon: Polygon,
    *,
    center_xy: tuple[float, float],
    ux: float,
    uy: float,
    px: float,
    py: float,
) -> list[ScanlineEdge]:
    edges: list[ScanlineEdge] = []

    def add_ring(coords) -> None:
        coord_list = list(coords)
        for start, end in zip(coord_list, coord_list[1:]):
            sx = float(start[0]) - center_xy[0]
            sy = float(start[1]) - center_xy[1]
            ex = float(end[0]) - center_xy[0]
            ey = float(end[1]) - center_xy[1]
            x1 = sx * ux + sy * uy
            y1 = sx * px + sy * py
            x2 = ex * ux + ey * uy
            y2 = ex * px + ey * py
            delta_y = y2 - y1
            if abs(delta_y) <= 1e-12:
                continue
            edges.append(
                ScanlineEdge(
                    x1=x1,
                    y1=y1,
                    dx_dy=(x2 - x1) / delta_y,
                    min_y=min(y1, y2),
                    max_y=max(y1, y2),
                )
            )

    add_ring(polygon.exterior.coords)
    for interior in polygon.interiors:
        add_ring(interior.coords)
    return edges


def _scanline_intervals(edges: list[ScanlineEdge], offset_m: float) -> list[tuple[float, float]]:
    intersections: list[float] = []
    for edge in edges:
        if edge.min_y <= offset_m < edge.max_y:
            intersections.append(edge.x1 + (offset_m - edge.y1) * edge.dx_dy)
    if len(intersections) < 2:
        return []
    intersections.sort()
    intervals: list[tuple[float, float]] = []
    limit = len(intersections) - (len(intersections) % 2)
    for index in range(0, limit, 2):
        start_x = intersections[index]
        end_x = intersections[index + 1]
        if end_x - start_x <= 1e-9:
            continue
        intervals.append((start_x, end_x))
    return intervals


def _run_scanline_method(
    polygon: Polygon,
    bearing_deg: float,
    params: FlightParamsModel,
    *,
    collect_trace: bool,
) -> dict[str, Any]:
    line_spacing = line_spacing_for_params(params)
    along_len, cross_width = project_extents(polygon, bearing_deg)
    lengths: list[float] = []
    gap_lengths: list[float] = []
    fragmented = 0
    sweeps: list[SweepTrace] = []
    center = polygon.centroid
    center_xy = (float(center.x), float(center.y))
    perp_rad = deg_to_rad((bearing_deg + 90.0) % 360.0)
    along_rad = deg_to_rad(bearing_deg)
    ux, uy = math.sin(along_rad), math.cos(along_rad)
    px, py = math.sin(perp_rad), math.cos(perp_rad)
    line_count_est = max(1, int(math.ceil(cross_width / max(1.0, line_spacing))))
    half_span = max(along_len, cross_width) * 0.75
    edges = _scanline_edges(polygon, center_xy=center_xy, ux=ux, uy=uy, px=px, py=py)

    for sweep_index in range(-line_count_est - 1, line_count_est + 2):
        offset = sweep_index * line_spacing
        intervals = _scanline_intervals(edges, offset)
        if not intervals:
            continue
        if len(intervals) > 1:
            fragmented += len(intervals) - 1
        segment_traces: list[SegmentTrace] = []
        gap_traces: list[GapTrace] = []
        if len(intervals) > 1:
            for index in range(1, len(intervals)):
                previous = intervals[index - 1]
                current = intervals[index]
                gap_length = current[0] - previous[1]
                if gap_length <= 1e-9:
                    continue
                gap_lengths.append(gap_length)
                if collect_trace:
                    gap_traces.append(
                        GapTrace(
                            start_xy=_local_to_world(
                                previous[1],
                                offset,
                                center_xy=center_xy,
                                ux=ux,
                                uy=uy,
                                px=px,
                                py=py,
                            ),
                            end_xy=_local_to_world(
                                current[0],
                                offset,
                                center_xy=center_xy,
                                ux=ux,
                                uy=uy,
                                px=px,
                                py=py,
                            ),
                            length_m=float(gap_length),
                        )
                    )
        for start_x, end_x in intervals:
            length_m = end_x - start_x
            lengths.append(length_m)
            if collect_trace:
                segment_traces.append(
                    SegmentTrace(
                        start_xy=_local_to_world(
                            start_x,
                            offset,
                            center_xy=center_xy,
                            ux=ux,
                            uy=uy,
                            px=px,
                            py=py,
                        ),
                        end_xy=_local_to_world(
                            end_x,
                            offset,
                            center_xy=center_xy,
                            ux=ux,
                            uy=uy,
                            px=px,
                            py=py,
                        ),
                        length_m=float(length_m),
                    )
                )
        if collect_trace:
            cx = center_xy[0] + px * offset
            cy = center_xy[1] + py * offset
            line_start_xy = _local_to_world(
                -half_span,
                offset,
                center_xy=center_xy,
                ux=ux,
                uy=uy,
                px=px,
                py=py,
            )
            line_end_xy = _local_to_world(
                half_span,
                offset,
                center_xy=center_xy,
                ux=ux,
                uy=uy,
                px=px,
                py=py,
            )
            sweeps.append(
                SweepTrace(
                    sweep_index=sweep_index,
                    offset_m=float(offset),
                    center_xy=(float(cx), float(cy)),
                    line_start_xy=line_start_xy,
                    line_end_xy=line_end_xy,
                    segments=segment_traces,
                    gaps=gap_traces,
                    total_inside_length_m=float(sum(segment.length_m for segment in segment_traces)),
                    total_gap_length_m=float(sum(gap.length_m for gap in gap_traces)),
                )
            )

    total_length = sum(lengths)
    total_gap = sum(gap_lengths)
    speed = params.speedMps or 12.0
    turn_count = max(0, len(lengths) - 1) + fragmented
    aggregate = {
        "line_spacing_m": float(line_spacing),
        "line_count": float(len(lengths)),
        "fragmented_line_count": float(fragmented),
        "fragmented_line_fraction": (fragmented / len(lengths)) if lengths else 0.0,
        "inter_segment_gap_length_m": float(total_gap),
        "overflight_transit_fraction": total_gap / max(1.0, total_length + total_gap),
        "turn_count": float(turn_count),
        "total_flight_line_length_m": float(total_length),
        "mean_line_length_m": float(statistics.fmean(lengths)) if lengths else 0.0,
        "median_line_length_m": float(statistics.median(lengths)) if lengths else 0.0,
        "short_line_fraction": (
            sum(1 for length in lengths if length < max(80.0, line_spacing * 5.0)) / len(lengths)
        ) if lengths else 1.0,
        "cruise_speed_mps": float(speed),
        "total_mission_time_sec": (total_length + total_gap) / max(1.0, speed) + turn_count * 8.0 + 25.0,
        "along_track_length_m": float(along_len),
        "cross_track_width_m": float(cross_width),
        "sweep_count": len(sweeps) if collect_trace else 0,
    }
    result: dict[str, Any] = {"aggregate": aggregate}
    if collect_trace:
        result["sweeps"] = [asdict(sweep) for sweep in sweeps]
    return result


def trace_current_method(
    polygon: Polygon,
    bearing_deg: float,
    params: FlightParamsModel,
) -> dict[str, Any]:
    line_spacing = line_spacing_for_params(params)
    along_len, cross_width = project_extents(polygon, bearing_deg)
    lengths: list[float] = []
    gap_lengths: list[float] = []
    fragmented = 0
    sweeps: list[SweepTrace] = []
    center = polygon.centroid
    perp_rad = deg_to_rad((bearing_deg + 90.0) % 360.0)
    along_rad = deg_to_rad(bearing_deg)
    ux, uy = math.sin(along_rad), math.cos(along_rad)
    px, py = math.sin(perp_rad), math.cos(perp_rad)
    line_count_est = max(1, int(math.ceil(cross_width / max(1.0, line_spacing))))
    half_span = max(along_len, cross_width) * 0.75

    for sweep_index in range(-line_count_est - 1, line_count_est + 2):
        offset = sweep_index * line_spacing
        cx = center.x + px * offset
        cy = center.y + py * offset
        line_start = (cx - ux * half_span, cy - uy * half_span)
        line_end = (cx + ux * half_span, cy + uy * half_span)
        line = LineString([line_start, line_end])
        intersection = polygon.intersection(line)
        if intersection.is_empty:
            continue
        segments: list[LineString] = []
        if isinstance(intersection, LineString):
            segments = [intersection]
        elif isinstance(intersection, MultiLineString):
            segments = [segment for segment in intersection.geoms if segment.length > 0]
        else:
            continue
        segments = [segment for segment in segments if segment.length > 0]
        if not segments:
            continue
        segments.sort(key=lambda seg: seg.bounds[0] + seg.bounds[2])

        segment_traces: list[SegmentTrace] = []
        gap_traces: list[GapTrace] = []
        if len(segments) > 1:
            fragmented += len(segments) - 1
            for index in range(1, len(segments)):
                previous = segments[index - 1]
                current = segments[index]
                previous_end = _segment_endpoints(previous)[1]
                current_start = _segment_endpoints(current)[0]
                gap_length = current.distance(previous)
                gap_lengths.append(gap_length)
                gap_traces.append(
                    GapTrace(
                        start_xy=previous_end,
                        end_xy=current_start,
                        length_m=float(gap_length),
                    )
                )

        for segment in segments:
            start_xy, end_xy = _segment_endpoints(segment)
            length_m = float(segment.length)
            lengths.append(length_m)
            segment_traces.append(
                SegmentTrace(
                    start_xy=start_xy,
                    end_xy=end_xy,
                    length_m=length_m,
                )
            )

        sweeps.append(
            SweepTrace(
                sweep_index=sweep_index,
                offset_m=float(offset),
                center_xy=(float(cx), float(cy)),
                line_start_xy=(float(line_start[0]), float(line_start[1])),
                line_end_xy=(float(line_end[0]), float(line_end[1])),
                segments=segment_traces,
                gaps=gap_traces,
                total_inside_length_m=float(sum(segment.length for segment in segments)),
                total_gap_length_m=float(sum(gap.length_m for gap in gap_traces)),
            )
        )

    total_length = sum(lengths)
    total_gap = sum(gap_lengths)
    speed = params.speedMps or 12.0
    turn_count = max(0, len(lengths) - 1) + fragmented
    aggregate = {
        "line_spacing_m": float(line_spacing),
        "line_count": float(len(lengths)),
        "fragmented_line_count": float(fragmented),
        "fragmented_line_fraction": (fragmented / len(lengths)) if lengths else 0.0,
        "inter_segment_gap_length_m": float(total_gap),
        "overflight_transit_fraction": total_gap / max(1.0, total_length + total_gap),
        "turn_count": float(turn_count),
        "total_flight_line_length_m": float(total_length),
        "mean_line_length_m": float(statistics.fmean(lengths)) if lengths else 0.0,
        "median_line_length_m": float(statistics.median(lengths)) if lengths else 0.0,
        "short_line_fraction": (
            sum(1 for length in lengths if length < max(80.0, line_spacing * 5.0)) / len(lengths)
        ) if lengths else 1.0,
        "cruise_speed_mps": float(speed),
        "total_mission_time_sec": (total_length + total_gap) / max(1.0, speed) + turn_count * 8.0 + 25.0,
        "along_track_length_m": float(along_len),
        "cross_track_width_m": float(cross_width),
        "sweep_count": len(sweeps),
    }
    return {
        "aggregate": aggregate,
        "sweeps": [asdict(sweep) for sweep in sweeps],
    }


def trace_scanline_method(
    polygon: Polygon,
    bearing_deg: float,
    params: FlightParamsModel,
) -> dict[str, Any]:
    return _run_scanline_method(polygon, bearing_deg, params, collect_trace=True)


def estimate_region_flight_time_scanline(
    polygon: Polygon,
    bearing_deg: float,
    params: FlightParamsModel,
) -> dict[str, float]:
    return _run_scanline_method(polygon, bearing_deg, params, collect_trace=False)["aggregate"]


def _benchmark(name: str, func, iterations: int) -> dict[str, Any]:
    durations_ms: list[float] = []
    last_result = None
    for _ in range(iterations):
        started_at = time.perf_counter()
        last_result = func()
        durations_ms.append((time.perf_counter() - started_at) * 1000.0)
    ordered = sorted(durations_ms)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.9) - 1))
    stats = BenchmarkStats(
        iterations=iterations,
        mean_ms=float(statistics.fmean(durations_ms)),
        median_ms=float(statistics.median(durations_ms)),
        min_ms=float(min(durations_ms)),
        max_ms=float(max(durations_ms)),
        p90_ms=float(ordered[p90_index]),
    )
    return {
        "name": name,
        "stats": asdict(stats),
        "lastResult": last_result,
    }


def _benchmark_suite(
    name: str,
    funcs: list[tuple[float, Any]],
    iterations: int,
) -> dict[str, Any]:
    durations_ms: list[float] = []
    per_bearing_totals: dict[float, list[float]] = {bearing: [] for bearing, _ in funcs}
    for _ in range(iterations):
        suite_started_at = time.perf_counter()
        for bearing, func in funcs:
            started_at = time.perf_counter()
            func()
            per_bearing_totals[bearing].append((time.perf_counter() - started_at) * 1000.0)
        durations_ms.append((time.perf_counter() - suite_started_at) * 1000.0)
    ordered = sorted(durations_ms)
    p90_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.9) - 1))
    return {
        "name": name,
        "stats": asdict(
            BenchmarkStats(
                iterations=iterations,
                mean_ms=float(statistics.fmean(durations_ms)),
                median_ms=float(statistics.median(durations_ms)),
                min_ms=float(min(durations_ms)),
                max_ms=float(max(durations_ms)),
                p90_ms=float(ordered[p90_index]),
            )
        ),
        "perBearingMeanMs": {
            f"{bearing:.6f}": round(float(statistics.fmean(values)), 6)
            for bearing, values in per_bearing_totals.items()
            if values
        },
    }


def _validate(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    diffs: dict[str, dict[str, float]] = {}
    shared_keys = sorted(set(reference) & set(candidate))
    missing_from_reference = sorted(set(candidate) - set(reference))
    missing_from_candidate = sorted(set(reference) - set(candidate))
    for key in shared_keys:
        left = reference.get(key)
        right = candidate.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            diffs[key] = {
                "reference": float(left),
                "candidate": float(right),
                "absDiff": abs(float(left) - float(right)),
            }
    max_numeric_diff = max(
        (payload["absDiff"] for payload in diffs.values() if math.isfinite(payload["absDiff"])),
        default=0.0,
    )
    return {
        "matchesReferenceOutputsExactly": max_numeric_diff == 0.0,
        "hasOnlyPlaygroundExtras": not missing_from_reference,
        "maxNumericAbsDiff": max_numeric_diff,
        "missingFromReference": missing_from_reference,
        "missingFromCandidate": missing_from_candidate,
        "diffs": diffs,
    }


def _make_transform(bounds: tuple[float, float, float, float], size: tuple[int, int], padding_px: int = 40):
    min_x, min_y, max_x, max_y = bounds
    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    canvas_w, canvas_h = size
    usable_w = max(1, canvas_w - padding_px * 2)
    usable_h = max(1, canvas_h - padding_px * 2)
    scale = min(usable_w / width, usable_h / height)

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x = padding_px + (point[0] - min_x) * scale
        y = canvas_h - padding_px - (point[1] - min_y) * scale
        return (x, y)

    return transform


def _draw_overview(
    polygon: Polygon,
    trace: dict[str, Any],
    bearing_deg: float,
    method_label: str,
    output_path: Path,
) -> None:
    image = Image.new("RGB", (1800, 1400), "white")
    draw = ImageDraw.Draw(image)
    transform = _make_transform(polygon.bounds, image.size)

    polygon_points = [transform((float(x), float(y))) for x, y in polygon.exterior.coords]
    draw.polygon(polygon_points, outline=(20, 20, 20), width=4)

    for sweep in trace["sweeps"]:
        line_start = transform(tuple(sweep["line_start_xy"]))
        line_end = transform(tuple(sweep["line_end_xy"]))
        draw.line([line_start, line_end], fill=(205, 205, 205), width=1)
        for gap in sweep["gaps"]:
            draw.line(
                [transform(tuple(gap["start_xy"])), transform(tuple(gap["end_xy"]))],
                fill=(210, 70, 70),
                width=2,
            )
        for segment in sweep["segments"]:
            draw.line(
                [transform(tuple(segment["start_xy"])), transform(tuple(segment["end_xy"]))],
                fill=(50, 90, 210),
                width=3,
            )

    aggregate = trace["aggregate"]
    summary_lines = [
        f"{method_label} bearing {bearing_deg:.4f} deg",
        f"Line spacing {aggregate['line_spacing_m']:.2f} m",
        f"Lines {int(aggregate['line_count'])}, fragments {int(aggregate['fragmented_line_count'])}",
        f"Flight length {aggregate['total_flight_line_length_m']:.1f} m",
        f"Gap length {aggregate['inter_segment_gap_length_m']:.1f} m",
        f"Mission time {aggregate['total_mission_time_sec']:.1f} s",
    ]
    x0 = 48
    y0 = 40
    for index, text in enumerate(summary_lines):
        draw.text((x0, y0 + index * 22), text, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _draw_bar_chart(
    trace: dict[str, Any],
    bearing_deg: float,
    method_label: str,
    output_path: Path,
) -> None:
    sweeps = trace["sweeps"]
    image = Image.new("RGB", (1800, 900), "white")
    draw = ImageDraw.Draw(image)
    if not sweeps:
        draw.text((40, 40), f"No sweeps for {method_label} bearing {bearing_deg:.4f}", fill=(0, 0, 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return

    totals = [float(sweep["total_inside_length_m"]) for sweep in sweeps]
    gaps = [float(sweep["total_gap_length_m"]) for sweep in sweeps]
    max_value = max(max(totals), max(gaps), 1.0)
    left_pad = 70
    bottom_pad = 70
    top_pad = 50
    chart_w = image.size[0] - left_pad - 40
    chart_h = image.size[1] - top_pad - bottom_pad
    step = chart_w / max(1, len(sweeps))
    bar_w = max(2, int(step * 0.35))
    draw.line([(left_pad, top_pad), (left_pad, top_pad + chart_h)], fill=(80, 80, 80), width=2)
    draw.line(
        [(left_pad, top_pad + chart_h), (left_pad + chart_w, top_pad + chart_h)],
        fill=(80, 80, 80),
        width=2,
    )
    for index, sweep in enumerate(sweeps):
        base_x = left_pad + index * step + step * 0.15
        inside_h = (totals[index] / max_value) * chart_h
        gap_h = (gaps[index] / max_value) * chart_h
        draw.rectangle(
            [
                (base_x, top_pad + chart_h - inside_h),
                (base_x + bar_w, top_pad + chart_h),
            ],
            fill=(50, 90, 210),
        )
        draw.rectangle(
            [
                (base_x + bar_w + 2, top_pad + chart_h - gap_h),
                (base_x + bar_w * 2 + 2, top_pad + chart_h),
            ],
            fill=(210, 70, 70),
        )
        if len(sweeps) <= 32:
            draw.text((base_x, top_pad + chart_h + 8), str(sweep["sweep_index"]), fill=(0, 0, 0))
    draw.text(
        (left_pad, 20),
        f"{method_label} bearing {bearing_deg:.4f} deg: inside length vs gap length per sweep",
        fill=(0, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _compose_side_by_side(left_path: Path, right_path: Path, output_path: Path) -> None:
    left = Image.open(left_path)
    right = Image.open(right_path)
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Playground for flight-time estimation on a real backend debug polygon.")
    parser.add_argument(
        "--synthetic-case",
        choices=["concave-u", "v-notch"],
        help="Run a built-in synthetic polygon instead of loading a backend debug request.",
    )
    parser.add_argument(
        "--request-debug-dir",
        type=Path,
        help="Path to a backend debug artifact directory. Defaults to the latest request under .terrain-splitter-runtime/debug.",
    )
    parser.add_argument(
        "--bearing-deg",
        action="append",
        type=float,
        help="Bearing(s) to analyze. Repeat to analyze multiple bearings. Defaults to the returned regions of the first solution.",
    )
    parser.add_argument(
        "--fixture-mode",
        choices=["solver-root-heading-candidates", "solution-0-region-bearings"],
        default="solver-root-heading-candidates",
        help="How to choose default bearings when --bearing-deg is not supplied.",
    )
    parser.add_argument(
        "--bench-iterations",
        type=int,
        default=20,
        help="Number of benchmark iterations per method.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated playground outputs. Defaults to .terrain-splitter-runtime/playground/<request-id>/.",
    )
    args = parser.parse_args()

    if args.synthetic_case in {"concave-u", "v-notch"}:
        synthetic_case = (
            _build_synthetic_concave_u_case()
            if args.synthetic_case == "concave-u"
            else _build_synthetic_v_notch_case()
        )
        debug_dir = None
        request_payload = None
        params = synthetic_case["params"]
        polygon = synthetic_case["polygon"]
        grid_payload: dict[str, Any] = {}
        if args.bearing_deg:
            bearings = [float(value) for value in args.bearing_deg]
            selected_fixture_source = "cli"
        else:
            bearings = [float(value) for value in synthetic_case["fixturePack"]["selectedBearingsDeg"]]
            selected_fixture_source = "synthetic-default-bearings"
        request_id = synthetic_case["caseId"]
        fixture_pack = {
            **synthetic_case["fixturePack"],
            "selectedSource": selected_fixture_source,
            "selectedBearingsDeg": bearings,
        }
        polygon_id = synthetic_case["polygonId"]
    else:
        debug_dir = args.request_debug_dir or _latest_debug_dir(DEFAULT_DEBUG_ROOT)
        request_payload = _load_json(debug_dir / "request.json")
        if not isinstance(request_payload, dict):
            raise ValueError("request.json must contain an object payload.")
        ring = request_payload["ring"]
        params = FlightParamsModel.model_validate(request_payload["params"])
        polygon = ring_to_polygon_mercator((tuple(coord) for coord in ring))
        grid_payload = _load_json(debug_dir / "grid.json") if (debug_dir / "grid.json").exists() else {}
        if not isinstance(grid_payload, dict):
            grid_payload = {}
        fixture_pack = _build_root_heading_fixture(
            request_payload,
            grid_step_m=float(grid_payload["gridStepM"]) if isinstance(grid_payload.get("gridStepM"), (int, float)) else None,
            debug_dir=debug_dir,
        )
        if args.bearing_deg:
            bearings = [float(value) for value in args.bearing_deg]
            selected_fixture_source = "cli"
        elif args.fixture_mode == "solution-0-region-bearings":
            bearings = _solution_region_bearings(debug_dir)
            selected_fixture_source = "solution-0-region-bearings"
        else:
            bearings = [float(value) for value in fixture_pack["headingCandidatesDeg"]]
            selected_fixture_source = "solver-root-heading-candidates"
        request_id = debug_dir.name
        fixture_pack = {
            **fixture_pack,
            "selectedSource": selected_fixture_source,
            "selectedBearingsDeg": bearings,
        }
        polygon_id = request_payload.get("polygonId")

    output_dir = args.output_dir or (DEFAULT_PLAYGROUND_ROOT / request_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "requestId": request_id,
        "debugDir": str(debug_dir) if debug_dir is not None else None,
        "outputDir": str(output_dir),
        "polygonId": polygon_id,
        "grid": grid_payload or None,
        "fixturePack": fixture_pack,
        "bearings": [],
    }
    traced_suite_funcs: list[tuple[float, Any]] = []
    backend_suite_funcs: list[tuple[float, Any]] = []
    scanline_trace_suite_funcs: list[tuple[float, Any]] = []
    scanline_estimate_suite_funcs: list[tuple[float, Any]] = []

    for bearing_deg in bearings:
        normalized_bearing = ((bearing_deg % 180.0) + 180.0) % 180.0
        traced = trace_current_method(polygon, normalized_bearing, params)
        scanline_trace = trace_scanline_method(polygon, normalized_bearing, params)
        backend_result = estimate_region_flight_time(polygon, normalized_bearing, params)
        scanline_estimate = estimate_region_flight_time_scanline(polygon, normalized_bearing, params)
        validation = _validate(traced["aggregate"], backend_result)
        scanline_validation_vs_backend = _validate(scanline_trace["aggregate"], backend_result)
        scanline_validation_vs_traced = _validate(scanline_trace["aggregate"], traced["aggregate"])
        traced_benchmark = _benchmark(
            "traced_current_method",
            lambda: trace_current_method(polygon, normalized_bearing, params),
            args.bench_iterations,
        )
        backend_benchmark = _benchmark(
            "estimate_region_flight_time",
            lambda: estimate_region_flight_time(polygon, normalized_bearing, params),
            args.bench_iterations,
        )
        scanline_trace_benchmark = _benchmark(
            "trace_scanline_method",
            lambda: trace_scanline_method(polygon, normalized_bearing, params),
            args.bench_iterations,
        )
        scanline_estimate_benchmark = _benchmark(
            "estimate_region_flight_time_scanline",
            lambda: estimate_region_flight_time_scanline(polygon, normalized_bearing, params),
            args.bench_iterations,
        )

        bearing_slug = f"{normalized_bearing:07.3f}".replace(".", "_")
        bearing_dir = output_dir / f"bearing_{bearing_slug}"
        current_overview_path = bearing_dir / "overview_current.png"
        scanline_overview_path = bearing_dir / "overview_scanline.png"
        current_lengths_path = bearing_dir / "line_lengths_current.png"
        scanline_lengths_path = bearing_dir / "line_lengths_scanline.png"
        _draw_overview(polygon, traced, normalized_bearing, "Current GEOS", current_overview_path)
        _draw_overview(polygon, scanline_trace, normalized_bearing, "Scanline", scanline_overview_path)
        _draw_bar_chart(traced, normalized_bearing, "Current GEOS", current_lengths_path)
        _draw_bar_chart(scanline_trace, normalized_bearing, "Scanline", scanline_lengths_path)
        _compose_side_by_side(current_overview_path, scanline_overview_path, bearing_dir / "overview_compare.png")
        _compose_side_by_side(current_lengths_path, scanline_lengths_path, bearing_dir / "line_lengths_compare.png")
        _compose_side_by_side(current_overview_path, scanline_overview_path, bearing_dir / "overview.png")
        _compose_side_by_side(current_lengths_path, scanline_lengths_path, bearing_dir / "line_lengths.png")
        _write_json(
            bearing_dir / "summary.json",
            {
                "bearingDeg": normalized_bearing,
                "aggregates": {
                    "currentTrace": traced["aggregate"],
                    "scanlineTrace": scanline_trace["aggregate"],
                    "backendReference": backend_result,
                    "scanlineEstimate": scanline_estimate,
                },
                "backendReference": backend_result,
                "validation": {
                    "currentTraceVsBackendReference": validation,
                    "scanlineTraceVsBackendReference": scanline_validation_vs_backend,
                    "scanlineTraceVsCurrentTrace": scanline_validation_vs_traced,
                },
                "benchmarks": {
                    "traced": traced_benchmark["stats"],
                    "backendReference": backend_benchmark["stats"],
                    "scanlineTrace": scanline_trace_benchmark["stats"],
                    "scanlineEstimate": scanline_estimate_benchmark["stats"],
                },
                "speedups": {
                    "scanlineTraceVsCurrentTrace": (
                        traced_benchmark["stats"]["mean_ms"] / scanline_trace_benchmark["stats"]["mean_ms"]
                    ) if scanline_trace_benchmark["stats"]["mean_ms"] > 0 else None,
                    "scanlineEstimateVsBackendReference": (
                        backend_benchmark["stats"]["mean_ms"] / scanline_estimate_benchmark["stats"]["mean_ms"]
                    ) if scanline_estimate_benchmark["stats"]["mean_ms"] > 0 else None,
                },
            },
        )
        _write_json(
            bearing_dir / "sweeps_current.json",
            {
                "bearingDeg": normalized_bearing,
                "sweeps": traced["sweeps"],
            },
        )
        _write_json(
            bearing_dir / "sweeps_scanline.json",
            {
                "bearingDeg": normalized_bearing,
                "sweeps": scanline_trace["sweeps"],
            },
        )
        report["bearings"].append(
            {
                "bearingDeg": normalized_bearing,
                "outputDir": str(bearing_dir),
                "validation": {
                    "currentTraceVsBackendReference": validation,
                    "scanlineTraceVsBackendReference": scanline_validation_vs_backend,
                    "scanlineTraceVsCurrentTrace": scanline_validation_vs_traced,
                },
                "benchmarks": {
                    "traced": traced_benchmark["stats"],
                    "backendReference": backend_benchmark["stats"],
                    "scanlineTrace": scanline_trace_benchmark["stats"],
                    "scanlineEstimate": scanline_estimate_benchmark["stats"],
                },
                "speedups": {
                    "scanlineTraceVsCurrentTrace": (
                        traced_benchmark["stats"]["mean_ms"] / scanline_trace_benchmark["stats"]["mean_ms"]
                    ) if scanline_trace_benchmark["stats"]["mean_ms"] > 0 else None,
                    "scanlineEstimateVsBackendReference": (
                        backend_benchmark["stats"]["mean_ms"] / scanline_estimate_benchmark["stats"]["mean_ms"]
                    ) if scanline_estimate_benchmark["stats"]["mean_ms"] > 0 else None,
                },
                "aggregate": traced["aggregate"],
            }
        )
        traced_suite_funcs.append((normalized_bearing, lambda b=normalized_bearing: trace_current_method(polygon, b, params)))
        backend_suite_funcs.append((normalized_bearing, lambda b=normalized_bearing: estimate_region_flight_time(polygon, b, params)))
        scanline_trace_suite_funcs.append((normalized_bearing, lambda b=normalized_bearing: trace_scanline_method(polygon, b, params)))
        scanline_estimate_suite_funcs.append(
            (normalized_bearing, lambda b=normalized_bearing: estimate_region_flight_time_scanline(polygon, b, params))
        )

    report["suiteBenchmarks"] = {
        "traced": _benchmark_suite("traced_current_method_suite", traced_suite_funcs, args.bench_iterations),
        "backendReference": _benchmark_suite("estimate_region_flight_time_suite", backend_suite_funcs, args.bench_iterations),
        "scanlineTrace": _benchmark_suite("trace_scanline_method_suite", scanline_trace_suite_funcs, args.bench_iterations),
        "scanlineEstimate": _benchmark_suite(
            "estimate_region_flight_time_scanline_suite",
            scanline_estimate_suite_funcs,
            args.bench_iterations,
        ),
    }
    report["suiteSpeedups"] = {
        "scanlineTraceVsCurrentTrace": (
            report["suiteBenchmarks"]["traced"]["stats"]["mean_ms"]
            / report["suiteBenchmarks"]["scanlineTrace"]["stats"]["mean_ms"]
        ) if report["suiteBenchmarks"]["scanlineTrace"]["stats"]["mean_ms"] > 0 else None,
        "scanlineEstimateVsBackendReference": (
            report["suiteBenchmarks"]["backendReference"]["stats"]["mean_ms"]
            / report["suiteBenchmarks"]["scanlineEstimate"]["stats"]["mean_ms"]
        ) if report["suiteBenchmarks"]["scanlineEstimate"]["stats"]["mean_ms"] > 0 else None,
    }
    _write_json(output_dir / "fixture_pack.json", report["fixturePack"])
    _write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
