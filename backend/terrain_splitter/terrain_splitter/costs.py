from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, MultiLineString, Polygon

from .features import CellFeatures
from .geometry import (
    axial_angle_delta_deg,
    calculate_gsd,
    clamp,
    deg_to_rad,
    forward_spacing_camera,
    lidar_line_spacing,
    lidar_swath_width,
    line_spacing_camera,
    polygon_compactness,
    polygon_convexity,
    project_extents,
)
from .grid import GridCell
from .schemas import FlightParamsModel

CAMERA_REGISTRY = {
    "SONY_RX1R2": {"f_m": 0.035, "sx_m": 4.88e-6, "sy_m": 4.88e-6, "w_px": 7952, "h_px": 5304},
    "DJI_ZENMUSE_P1_24MM": {"f_m": 5626.690009970837 * 4.27246e-6, "sx_m": 4.27246e-6, "sy_m": 4.27246e-6, "w_px": 8192, "h_px": 5460},
    "ILX_LR1_INSPECT_85MM": {"f_m": 0.085, "sx_m": 35.7e-3 / 9504, "sy_m": 23.8e-3 / 6336, "w_px": 9504, "h_px": 6336},
    "MAP61_17MM": {"f_m": 0.017, "sx_m": 35.7e-3 / 9504, "sy_m": 23.8e-3 / 6336, "w_px": 9504, "h_px": 6336},
    "RGB61_24MM": {"f_m": 0.024, "sx_m": 36e-3 / 9504, "sy_m": 24e-3 / 6336, "w_px": 9504, "h_px": 6336},
}

LIDAR_DEFAULTS = {
    "default_speed_mps": 16.0,
    "mapping_fov_deg": 90.0,
    "default_max_range_m": 200.0,
    "effective_point_rates": {"single": 160_000.0, "dual": 320_000.0, "triple": 480_000.0},
}


@dataclass(slots=True)
class NodeCost:
    quality_cost: float
    hole_risk: float
    low_coverage_risk: float
    mean_coverage_score: float
    bearing_prior_loss: float


@dataclass(slots=True)
class LineLiftSummary:
    line_count: int
    mean_line_lift_m: float
    p90_line_lift_m: float
    max_line_lift_m: float
    elevated_area_fraction: float
    severe_lift_area_fraction: float


@dataclass(slots=True)
class RegionObjective:
    bearing_deg: float
    normalized_quality_cost: float
    total_mission_time_sec: float
    weighted_mean_mismatch_deg: float
    area_m2: float
    convexity: float
    compactness: float
    boundary_break_alignment: float
    flight_line_count: int
    line_spacing_m: float
    along_track_length_m: float
    cross_track_width_m: float
    fragmented_line_fraction: float
    overflight_transit_fraction: float
    short_line_fraction: float
    mean_line_length_m: float
    median_line_length_m: float
    mean_line_lift_m: float
    p90_line_lift_m: float
    max_line_lift_m: float
    elevated_area_fraction: float
    severe_lift_area_fraction: float


@dataclass(slots=True)
class RegionStaticInputs:
    x: np.ndarray
    y: np.ndarray
    area_m2: np.ndarray
    terrain_z: np.ndarray
    preferred_bearing_deg: np.ndarray
    confidence: np.ndarray
    slope_magnitude: np.ndarray
    grid_step_m: float


def _camera_model(params: FlightParamsModel) -> dict[str, float]:
    return CAMERA_REGISTRY.get(params.cameraKey or "SONY_RX1R2", CAMERA_REGISTRY["SONY_RX1R2"])


def line_spacing_for_params(params: FlightParamsModel) -> float:
    if params.payloadKind == "lidar":
        return max(
            1.0,
            lidar_line_spacing(
                params.altitudeAGL,
                params.sideOverlap,
                params.mappingFovDeg or LIDAR_DEFAULTS["mapping_fov_deg"],
            ),
        )
    camera = _camera_model(params)
    rotate = round((((params.cameraYawOffsetDeg or 0.0) % 180.0) + 180.0) % 180.0) == 90
    return max(
        1.0,
        line_spacing_camera(
            params.altitudeAGL,
            params.sideOverlap,
            camera["f_m"],
            camera["sx_m"],
            camera["sy_m"],
            int(camera["w_px"]),
            int(camera["h_px"]),
            rotate_90=rotate,
        ),
    )


def forward_spacing_for_params(params: FlightParamsModel) -> float | None:
    if params.payloadKind == "lidar":
        return None
    camera = _camera_model(params)
    rotate = round((((params.cameraYawOffsetDeg or 0.0) % 180.0) + 180.0) % 180.0) == 90
    return max(
        1.0,
        forward_spacing_camera(
            params.altitudeAGL,
            params.frontOverlap,
            camera["f_m"],
            camera["sx_m"],
            camera["sy_m"],
            int(camera["w_px"]),
            int(camera["h_px"]),
            rotate_90=rotate,
        ),
    )


def lidar_target_density(params: FlightParamsModel) -> float:
    mapping_fov = params.mappingFovDeg or LIDAR_DEFAULTS["mapping_fov_deg"]
    speed = params.speedMps or LIDAR_DEFAULTS["default_speed_mps"]
    point_rate = LIDAR_DEFAULTS["effective_point_rates"][params.lidarReturnMode or "single"]
    single_pass = point_rate / max(1e-6, speed * lidar_swath_width(params.altitudeAGL, mapping_fov))
    if params.pointDensityPtsM2 and params.pointDensityPtsM2 > 0:
        return params.pointDensityPtsM2
    factor = max(1e-6, 1.0 - params.sideOverlap / 100.0)
    return single_pass / factor


def _weighted_mean_array(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return 0.0
    masked_weights = weights[mask]
    return float(np.sum(values[mask] * masked_weights) / np.sum(masked_weights))


def _weighted_quantile_array(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return 0.0
    filtered_values = values[mask]
    filtered_weights = weights[mask]
    order = np.argsort(filtered_values)
    sorted_values = filtered_values[order]
    sorted_weights = filtered_weights[order]
    total_weight = float(np.sum(sorted_weights))
    if total_weight <= 0:
        return 0.0
    target = clamp(q, 0.0, 1.0) * total_weight
    cumulative = np.cumsum(sorted_weights)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, sorted_values.shape[0] - 1)
    return float(sorted_values[index])


def _axial_angle_delta_array_deg(values_deg: np.ndarray, bearing_deg: float) -> np.ndarray:
    aa = np.mod(values_deg, 180.0)
    bb = ((bearing_deg % 180.0) + 180.0) % 180.0
    delta = np.abs(aa - bb)
    return np.minimum(delta, 180.0 - delta)


def evaluate_sensor_node_cost(
    preferred_bearing_deg: float,
    slope_magnitude: float,
    confidence: float,
    bearing_deg: float,
    params: FlightParamsModel,
) -> NodeCost:
    delta_deg = axial_angle_delta_deg(preferred_bearing_deg, bearing_deg)
    mismatch_loss = math.sin(deg_to_rad(delta_deg)) ** 2 * (0.35 + 0.65 * confidence)
    cross_slope_factor = clamp(abs(math.sin(deg_to_rad(delta_deg))) * (slope_magnitude / 0.12), 0.0, 1.8)

    if params.payloadKind == "lidar":
        mapping_fov = params.mappingFovDeg or LIDAR_DEFAULTS["mapping_fov_deg"]
        max_range_m = params.maxLidarRangeM or LIDAR_DEFAULTS["default_max_range_m"]
        target_density = lidar_target_density(params)
        half_swath_width_m = lidar_swath_width(params.altitudeAGL, mapping_fov) * 0.5
        terrain_lift_factor = cross_slope_factor * (0.55 + 0.45 * confidence)
        induced_altitude_m = params.altitudeAGL * (1.0 + 0.65 * terrain_lift_factor)
        slant_range_m = math.sqrt(induced_altitude_m * induced_altitude_m + half_swath_width_m * half_swath_width_m)
        range_overshoot = max(0.0, slant_range_m / max(1.0, max_range_m) - 1.0)
        hole_risk = clamp(range_overshoot * 1.25 + max(0.0, mismatch_loss + 0.35 * terrain_lift_factor - 1.10), 0.0, 1.0)
        density_factor = clamp(
            1.0 - 1.05 * mismatch_loss - 0.42 * terrain_lift_factor - 1.6 * range_overshoot,
            0.0,
            1.15,
        )
        mean_deficit = max(0.0, 1.0 - density_factor)
        low_coverage_risk = clamp((1.0 - density_factor) * 0.75 + hole_risk * 0.9, 0.0, 1.0)
        return NodeCost(
            quality_cost=0.85 * mean_deficit + 2.2 * hole_risk + 1.75 * low_coverage_risk + 0.35 * mismatch_loss,
            hole_risk=hole_risk,
            low_coverage_risk=low_coverage_risk,
            mean_coverage_score=density_factor * target_density,
            bearing_prior_loss=mismatch_loss,
        )

    underlap_risk = clamp(max(0.0, mismatch_loss - 0.3) * 1.8 + max(0.0, cross_slope_factor - 1.0) * 0.5, 0.0, 1.0)
    return NodeCost(
        quality_cost=max(0.0, 1.35 * mismatch_loss + 0.35 * cross_slope_factor) + 1.05 * underlap_risk,
        hole_risk=0.0,
        low_coverage_risk=underlap_risk,
        mean_coverage_score=max(0.0, 1.0 - mismatch_loss - 0.2 * cross_slope_factor),
        bearing_prior_loss=mismatch_loss,
    )


def summarize_line_lift(
    cells: Sequence[GridCell],
    bearing_deg: float,
    params: FlightParamsModel,
    confidences: dict[int, float],
    *,
    static_inputs: RegionStaticInputs | None = None,
) -> LineLiftSummary:
    if not cells:
        return LineLiftSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    line_spacing = line_spacing_for_params(params)
    if static_inputs is not None:
        x = static_inputs.x
        y = static_inputs.y
        area = static_inputs.area_m2
        terrain_z = static_inputs.terrain_z
        confidence = static_inputs.confidence
    else:
        x = np.asarray([cell.x for cell in cells], dtype=np.float64)
        y = np.asarray([cell.y for cell in cells], dtype=np.float64)
        area = np.asarray([cell.area_m2 for cell in cells], dtype=np.float64)
        terrain_z = np.asarray([cell.terrain_z for cell in cells], dtype=np.float64)
        confidence = np.asarray([confidences.get(cell.index, 0.0) for cell in cells], dtype=np.float64)

    center_x = _weighted_mean_array(x, area)
    center_y = _weighted_mean_array(y, area)
    perp_rad = deg_to_rad((bearing_deg + 90.0) % 360.0)
    px = math.sin(perp_rad)
    py = math.cos(perp_rad)
    cross_track = (x - center_x) * px + (y - center_y) * py
    bin_index = np.rint(cross_track / max(1.0, line_spacing)).astype(np.int64)
    order = np.argsort(bin_index, kind="mergesort")
    sorted_bins = bin_index[order]
    sorted_terrain = terrain_z[order]
    unique_bins, start_idx = np.unique(sorted_bins, return_index=True)
    support_by_bin = np.maximum.reduceat(sorted_terrain, start_idx)
    inverse = np.searchsorted(unique_bins, bin_index)
    line_lift = np.maximum(0.0, support_by_bin[inverse] - terrain_z)
    weights = np.maximum(1e-6, area * (0.35 + 0.65 * confidence))

    elevated_threshold = max(10.0, params.altitudeAGL * 0.12)
    severe_threshold = max(25.0, params.altitudeAGL * 0.28)
    total_area = float(np.sum(area))
    elevated_area = float(np.sum(area[line_lift >= elevated_threshold]))
    severe_area = float(np.sum(area[line_lift >= severe_threshold]))
    return LineLiftSummary(
        line_count=int(unique_bins.shape[0]),
        mean_line_lift_m=_weighted_mean_array(line_lift, weights),
        p90_line_lift_m=_weighted_quantile_array(line_lift, weights, 0.9),
        max_line_lift_m=float(np.max(line_lift)) if line_lift.size > 0 else 0.0,
        elevated_area_fraction=elevated_area / total_area if total_area > 0 else 0.0,
        severe_lift_area_fraction=severe_area / total_area if total_area > 0 else 0.0,
    )


def estimate_region_flight_time(
    polygon: Polygon,
    bearing_deg: float,
    params: FlightParamsModel,
    *,
    static_inputs: RegionStaticInputs | None = None,
) -> dict[str, float]:
    line_spacing = line_spacing_for_params(params)
    along_len, cross_width = project_extents(polygon, bearing_deg)
    lengths: list[float] = []
    gaps: list[float] = []
    fragmented = 0
    center = polygon.centroid
    perp_rad = deg_to_rad((bearing_deg + 90.0) % 360.0)
    along_rad = deg_to_rad(bearing_deg)
    ux, uy = math.sin(along_rad), math.cos(along_rad)
    px, py = math.sin(perp_rad), math.cos(perp_rad)
    line_count_est = max(1, int(math.ceil(cross_width / max(1.0, line_spacing))))
    half_span = max(along_len, cross_width) * 0.75

    for i in range(-line_count_est - 1, line_count_est + 2):
        offset = i * line_spacing
        cx = center.x + px * offset
        cy = center.y + py * offset
        line = LineString([(cx - ux * half_span, cy - uy * half_span), (cx + ux * half_span, cy + uy * half_span)])
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
        if len(segments) > 1:
            fragmented += len(segments) - 1
            for j in range(1, len(segments)):
                gaps.append(segments[j].distance(segments[j - 1]))
        lengths.extend(segment.length for segment in segments)

    total_length = sum(lengths)
    total_gap = sum(gaps)
    speed = params.speedMps or (LIDAR_DEFAULTS["default_speed_mps"] if params.payloadKind == "lidar" else 12.0)
    turn_count = max(0, len(lengths) - 1) + fragmented
    return {
        "line_spacing_m": line_spacing,
        "line_count": float(len(lengths)),
        "fragmented_line_count": float(fragmented),
        "fragmented_line_fraction": (fragmented / len(lengths)) if lengths else 0.0,
        "inter_segment_gap_length_m": total_gap,
        "overflight_transit_fraction": total_gap / max(1.0, total_length + total_gap),
        "turn_count": float(turn_count),
        "total_flight_line_length_m": total_length,
        "mean_line_length_m": float(np.mean(lengths)) if lengths else 0.0,
        "median_line_length_m": float(np.median(lengths)) if lengths else 0.0,
        "short_line_fraction": (sum(1 for length in lengths if length < max(80.0, line_spacing * 5.0)) / len(lengths)) if lengths else 1.0,
        "cruise_speed_mps": speed,
        "total_mission_time_sec": (total_length + total_gap) / max(1.0, speed) + turn_count * 8.0 + 25.0,
    }


def evaluate_region_objective(
    cells: Sequence[GridCell],
    features_by_index: dict[int, CellFeatures],
    bearing_deg: float,
    params: FlightParamsModel,
    polygon: Polygon,
    boundary_break_alignment: float,
    *,
    perf: dict[str, float] | None = None,
    precomputed_area_m2: float | None = None,
    precomputed_convexity: float | None = None,
    precomputed_compactness: float | None = None,
    precomputed_static_inputs: RegionStaticInputs | None = None,
) -> RegionObjective:
    if not cells:
        return RegionObjective(
            bearing_deg=bearing_deg,
            normalized_quality_cost=10.0,
            total_mission_time_sec=0.0,
            weighted_mean_mismatch_deg=90.0,
            area_m2=0.0,
            convexity=0.0,
            compactness=10.0,
            boundary_break_alignment=0.0,
            flight_line_count=0,
            line_spacing_m=0.0,
            along_track_length_m=0.0,
            cross_track_width_m=0.0,
            fragmented_line_fraction=1.0,
            overflight_transit_fraction=1.0,
            short_line_fraction=1.0,
            mean_line_length_m=0.0,
            median_line_length_m=0.0,
            mean_line_lift_m=0.0,
            p90_line_lift_m=0.0,
            max_line_lift_m=0.0,
            elevated_area_fraction=0.0,
            severe_lift_area_fraction=0.0,
        )

    node_cost_started_at = time.perf_counter() if perf is not None else 0.0
    if precomputed_static_inputs is None:
        preferred = []
        confidence = []
        slope = []
        base_area = []
        x_vals = []
        y_vals = []
        terrain_z = []
        for cell in cells:
            features = features_by_index.get(cell.index)
            if features is None:
                continue
            preferred.append(features.preferred_bearing_deg)
            confidence.append(features.confidence)
            slope.append(features.slope_magnitude)
            base_area.append(max(1e-6, cell.area_m2))
            x_vals.append(cell.x)
            y_vals.append(cell.y)
            terrain_z.append(cell.terrain_z)
        static_inputs = RegionStaticInputs(
            x=np.asarray(x_vals, dtype=np.float64),
            y=np.asarray(y_vals, dtype=np.float64),
            area_m2=np.asarray(base_area, dtype=np.float64),
            terrain_z=np.asarray(terrain_z, dtype=np.float64),
            preferred_bearing_deg=np.asarray(preferred, dtype=np.float64),
            confidence=np.asarray(confidence, dtype=np.float64),
            slope_magnitude=np.asarray(slope, dtype=np.float64),
            grid_step_m=1.0,
        )
    else:
        static_inputs = precomputed_static_inputs

    base_weights = np.maximum(1e-6, static_inputs.area_m2)
    confidence = static_inputs.confidence
    weighted_mismatch_weights = base_weights * (0.35 + 0.65 * confidence)
    delta_deg = _axial_angle_delta_array_deg(static_inputs.preferred_bearing_deg, bearing_deg)
    sin_delta = np.sin(np.deg2rad(delta_deg))
    mismatch_loss = (sin_delta ** 2) * (0.35 + 0.65 * confidence)
    cross_slope_factor = np.clip(np.abs(sin_delta) * (static_inputs.slope_magnitude / 0.12), 0.0, 1.8)
    if perf is not None:
        perf["node_cost_ms"] += (time.perf_counter() - node_cost_started_at) * 1000.0

    line_lift_started_at = time.perf_counter() if perf is not None else 0.0
    line_lift = summarize_line_lift(cells, bearing_deg, params, {}, static_inputs=static_inputs)
    if perf is not None:
        perf["line_lift_ms"] += (time.perf_counter() - line_lift_started_at) * 1000.0

    flight_time_started_at = time.perf_counter() if perf is not None else 0.0
    flight = estimate_region_flight_time(polygon, bearing_deg, params, static_inputs=static_inputs)
    if perf is not None:
        perf["flight_time_ms"] += (time.perf_counter() - flight_time_started_at) * 1000.0

    shape_started_at = time.perf_counter() if perf is not None else 0.0
    area_m2 = float(polygon.area) if precomputed_area_m2 is None else precomputed_area_m2
    convexity = polygon_convexity(polygon) if precomputed_convexity is None else precomputed_convexity
    compactness = polygon_compactness(polygon) if precomputed_compactness is None else precomputed_compactness
    if perf is not None:
        perf["shape_metric_ms"] += (time.perf_counter() - shape_started_at) * 1000.0
    weighted_mean_mismatch_deg = _weighted_mean_array(delta_deg, weighted_mismatch_weights)

    if params.payloadKind == "lidar":
        target_density = lidar_target_density(params)
        mapping_fov = params.mappingFovDeg or LIDAR_DEFAULTS["mapping_fov_deg"]
        max_range_m = params.maxLidarRangeM or LIDAR_DEFAULTS["default_max_range_m"]
        half_swath_width_m = lidar_swath_width(params.altitudeAGL, mapping_fov) * 0.5
        terrain_lift_factor = cross_slope_factor * (0.55 + 0.45 * confidence)
        induced_altitude_m = params.altitudeAGL * (1.0 + 0.65 * terrain_lift_factor)
        slant_range_m = np.sqrt(induced_altitude_m * induced_altitude_m + half_swath_width_m * half_swath_width_m)
        range_overshoot = np.maximum(0.0, slant_range_m / max(1.0, max_range_m) - 1.0)
        hole_risk = np.clip(range_overshoot * 1.25 + np.maximum(0.0, mismatch_loss + 0.35 * terrain_lift_factor - 1.10), 0.0, 1.0)
        density_factor = np.clip(1.0 - 1.05 * mismatch_loss - 0.42 * terrain_lift_factor - 1.6 * range_overshoot, 0.0, 1.15)
        low_coverage_risk = np.clip((1.0 - density_factor) * 0.75 + hole_risk * 0.9, 0.0, 1.0)
        mean_coverage_score = density_factor * target_density
        mean_density = _weighted_mean_array(mean_coverage_score, base_weights)
        p10_density = _weighted_quantile_array(mean_coverage_score, base_weights, 0.1)
        hole_fraction = _weighted_mean_array(hole_risk, base_weights)
        low_fraction = _weighted_mean_array(low_coverage_risk, base_weights)
        mean_deficit = max(0.0, 1.0 - mean_density / max(1.0, target_density))
        p10_deficit = max(0.0, 1.0 - p10_density / max(1.0, target_density))
        line_lift_cost = (
            0.55 * line_lift.mean_line_lift_m / max(1.0, params.altitudeAGL)
            + 0.95 * line_lift.elevated_area_fraction
            + 1.35 * line_lift.severe_lift_area_fraction
        )
        normalized_quality_cost = (
            0.9 * mean_deficit
            + 1.7 * p10_deficit
            + 2.7 * hole_fraction
            + 1.8 * low_fraction
            + line_lift_cost
            + 0.22 * flight["overflight_transit_fraction"]
        )
    else:
        target_gsd = calculate_gsd(
            params.altitudeAGL,
            _camera_model(params)["f_m"],
            _camera_model(params)["sx_m"],
            _camera_model(params)["sy_m"],
        )
        underlap_risk = np.clip(np.maximum(0.0, mismatch_loss - 0.3) * 1.8 + np.maximum(0.0, cross_slope_factor - 1.0) * 0.5, 0.0, 1.0)
        node_quality = np.maximum(0.0, 1.35 * mismatch_loss + 0.35 * cross_slope_factor) + 1.05 * underlap_risk
        mean_node_cost = _weighted_mean_array(node_quality, base_weights)
        line_lift_cost = (
            0.7 * line_lift.mean_line_lift_m / max(1.0, params.altitudeAGL)
            + 0.8 * line_lift.elevated_area_fraction
            + 1.1 * line_lift.severe_lift_area_fraction
        )
        forward_spacing = forward_spacing_for_params(params) or 1.0
        overlap_pressure = clamp((line_lift.max_line_lift_m / max(5.0, params.altitudeAGL)) * (forward_spacing / 120.0), 0.0, 2.0)
        normalized_quality_cost = mean_node_cost + line_lift_cost + 0.55 * overlap_pressure + target_gsd * 0.0

    along_track_length_m, cross_track_width_m = project_extents(polygon, bearing_deg)

    return RegionObjective(
        bearing_deg=bearing_deg,
        normalized_quality_cost=normalized_quality_cost,
        total_mission_time_sec=flight["total_mission_time_sec"],
        weighted_mean_mismatch_deg=weighted_mean_mismatch_deg,
        area_m2=area_m2,
        convexity=convexity,
        compactness=compactness,
        boundary_break_alignment=boundary_break_alignment,
        flight_line_count=int(round(flight["line_count"])),
        line_spacing_m=flight["line_spacing_m"],
        along_track_length_m=along_track_length_m,
        cross_track_width_m=cross_track_width_m,
        fragmented_line_fraction=flight["fragmented_line_fraction"],
        overflight_transit_fraction=flight["overflight_transit_fraction"],
        short_line_fraction=flight["short_line_fraction"],
        mean_line_length_m=flight["mean_line_length_m"],
        median_line_length_m=flight["median_line_length_m"],
        mean_line_lift_m=line_lift.mean_line_lift_m,
        p90_line_lift_m=line_lift.p90_line_lift_m,
        max_line_lift_m=line_lift.max_line_lift_m,
        elevated_area_fraction=line_lift.elevated_area_fraction,
        severe_lift_area_fraction=line_lift.severe_lift_area_fraction,
    )
