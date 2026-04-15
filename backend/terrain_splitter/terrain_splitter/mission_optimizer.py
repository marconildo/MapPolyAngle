from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from .costs import LIDAR_DEFAULTS, _scanline_edges, _scanline_intervals, line_spacing_for_params
from .geometry import (
    deg_to_rad,
    haversine_m,
    lnglat_to_mercator,
    mercator_to_lnglat,
    normalize_axial_bearing,
    project_extents,
    ring_to_polygon_mercator,
)
from .mapbox_tiles import TerrainDEM
from .schemas import (
    FlightParamsModel,
    MissionAreaRequest,
    MissionAreaTraversalModel,
    MissionAreaTraversalRequestModel,
    MissionConnectionLoiterStepModel,
    MissionConnectionModel,
    MissionOptimizeAreaSequenceRequest,
    MissionOptimizeAreaSequenceResponse,
    MissionTransferCostModel,
    MissionTraversalLoiterModel,
)

DEFAULT_SEGMENT_SAMPLE_SPACING_M = 30.0
DEFAULT_CONNECTION_TURN_RADIUS_M = 30.0
CONNECTION_ARC_POINT_SPACING_M = 8.0
CONNECTION_COIL_POINT_SPACING_M = 8.0
CONNECTION_CLIMB_PER_METER = 0.25
ALTITUDE_EPSILON_M = 0.25
ALTITUDE_STEP_EPSILON_M = 0.5

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _LocalPoint:
    x: float
    y: float


@dataclass(slots=True)
class _LocalPoint3D:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class _TraversalLoiter:
    center_point: tuple[float, float]
    center_mercator: tuple[float, float]
    radius_m: float
    direction: str


@dataclass(slots=True)
class _TraversalOption:
    polygon_id: str
    area_index: int
    flipped: bool
    bearing_deg: float
    params: FlightParamsModel
    altitude_agl: float
    start_point: tuple[float, float]
    end_point: tuple[float, float]
    start_mercator: tuple[float, float]
    end_mercator: tuple[float, float]
    start_terrain_m: float
    end_terrain_m: float
    start_altitude_wgs84_m: float
    end_altitude_wgs84_m: float
    lead_in: _TraversalLoiter
    lead_out: _TraversalLoiter


@dataclass(slots=True)
class _ConnectionSample:
    point: tuple[float, float]
    mercator: tuple[float, float]
    terrain_m: float


@dataclass(slots=True)
class _FallbackConnectionGeometry:
    line: list[tuple[float, float]]
    trajectory: list[tuple[float, float]]
    loiter_steps: list[MissionConnectionLoiterStepModel]


@dataclass(slots=True)
class _LocalLoiterDescriptor:
    center: _LocalPoint
    radius_m: float
    direction: str


@dataclass(slots=True)
class _LocalConnectorShape:
    line: list[_LocalPoint]
    trajectory3d: list[_LocalPoint3D]
    loiter_steps: list[MissionConnectionLoiterStepModel]


@dataclass(slots=True)
class _LocalConnectorPlanarShape:
    line: list[_LocalPoint]
    lead_out_base_2d: list[_LocalPoint]
    connector_path: list[_LocalPoint]
    lead_in_base_2d: list[_LocalPoint]
    from_loiter: _LocalLoiterDescriptor
    to_loiter: _LocalLoiterDescriptor
    start_tangent: _LocalPoint
    end_tangent: _LocalPoint
    connector_entry: _LocalPoint
    connector_exit: _LocalPoint


@dataclass(slots=True)
class _TransferBandDiagnostics:
    fits: bool
    lower_bound_max_wgs84_m: float
    upper_bound_min_wgs84_m: float
    max_below_band_m: float
    max_above_band_m: float


@dataclass(slots=True)
class _ConnectionCandidate:
    from_area_index: int
    to_area_index: int
    from_flipped: bool
    to_flipped: bool
    model: MissionConnectionModel

    @property
    def transfer_time_sec(self) -> float:
        return float(self.model.transferTimeSec)

    @property
    def objective_cost(self) -> float:
        return float(self.model.transferCost)


@dataclass(slots=True)
class _TransferCostConfig:
    horizontal_speed_mps: float
    climb_rate_mps: float
    descent_rate_mps: float
    horizontal_energy_rate: float
    climb_energy_rate: float
    descent_energy_rate: float


@dataclass(slots=True)
class _TransferCostBreakdown:
    horizontal_distance_m: float
    climb_m: float
    descent_m: float
    horizontal_time_sec: float
    climb_time_sec: float
    descent_time_sec: float
    total_time_sec: float
    total_cost: float


def build_combined_bounds_ring(areas: list[MissionAreaRequest]) -> list[tuple[float, float]]:
    min_lng = math.inf
    min_lat = math.inf
    max_lng = -math.inf
    max_lat = -math.inf
    for area in areas:
        for lng, lat in area.ring:
            min_lng = min(min_lng, float(lng))
            min_lat = min(min_lat, float(lat))
            max_lng = max(max_lng, float(lng))
            max_lat = max(max_lat, float(lat))
    if not math.isfinite(min_lng) or not math.isfinite(min_lat) or not math.isfinite(max_lng) or not math.isfinite(max_lat):
        raise ValueError("Unable to derive bounds for mission areas.")
    return [
        (min_lng, min_lat),
        (max_lng, min_lat),
        (max_lng, max_lat),
        (min_lng, max_lat),
        (min_lng, min_lat),
    ]


def _cruise_speed_mps(params: FlightParamsModel) -> float:
    if params.payloadKind == "lidar":
        return max(1.0, float(params.speedMps or LIDAR_DEFAULTS["default_speed_mps"]))
    return max(1.0, float(params.speedMps or 12.0))


def _resolve_transfer_cost_config(
    from_option: _TraversalOption,
    to_option: _TraversalOption,
    transfer_cost: MissionTransferCostModel | None,
) -> _TransferCostConfig:
    horizontal_speed_mps = float(transfer_cost.horizontalSpeedMps) if transfer_cost and transfer_cost.horizontalSpeedMps else max(
        1.0,
        0.5 * (_cruise_speed_mps(from_option.params) + _cruise_speed_mps(to_option.params)),
    )
    climb_rate_mps = float(transfer_cost.climbRateMps) if transfer_cost is not None else 4.0
    descent_rate_mps = float(transfer_cost.descentRateMps) if transfer_cost is not None else 6.0
    horizontal_energy_rate = float(transfer_cost.horizontalEnergyRate) if transfer_cost is not None else 1.0
    climb_energy_rate = float(transfer_cost.climbEnergyRate) if transfer_cost is not None else 2.5
    descent_energy_rate = float(transfer_cost.descentEnergyRate) if transfer_cost is not None else 0.6
    return _TransferCostConfig(
        horizontal_speed_mps=max(1.0, horizontal_speed_mps),
        climb_rate_mps=max(0.1, climb_rate_mps),
        descent_rate_mps=max(0.1, descent_rate_mps),
        horizontal_energy_rate=max(0.0, horizontal_energy_rate),
        climb_energy_rate=max(0.0, climb_energy_rate),
        descent_energy_rate=max(0.0, descent_energy_rate),
    )


def _segment_time_sec(
    horizontal_distance_m: float,
    delta_altitude_m: float,
    timing: _TransferCostConfig,
) -> float:
    horizontal_time_sec = horizontal_distance_m / timing.horizontal_speed_mps if horizontal_distance_m > ALTITUDE_EPSILON_M else 0.0
    climb_time_sec = max(0.0, delta_altitude_m) / timing.climb_rate_mps if delta_altitude_m > ALTITUDE_EPSILON_M else 0.0
    descent_time_sec = max(0.0, -delta_altitude_m) / timing.descent_rate_mps if delta_altitude_m < -ALTITUDE_EPSILON_M else 0.0
    return max(horizontal_time_sec, climb_time_sec, descent_time_sec)


def _local3d_transfer_cost(points: list[_LocalPoint3D], timing: _TransferCostConfig) -> _TransferCostBreakdown:
    horizontal_distance_m = 0.0
    climb_m = 0.0
    descent_m = 0.0
    horizontal_time_sec = 0.0
    climb_time_sec = 0.0
    descent_time_sec = 0.0
    total_time_sec = 0.0
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        segment_horizontal_distance_m = math.hypot(end.x - start.x, end.y - start.y)
        delta_altitude_m = end.z - start.z
        horizontal_distance_m += segment_horizontal_distance_m
        horizontal_time_sec += segment_horizontal_distance_m / timing.horizontal_speed_mps if segment_horizontal_distance_m > ALTITUDE_EPSILON_M else 0.0
        if delta_altitude_m > 0:
            climb_m += delta_altitude_m
            climb_time_sec += delta_altitude_m / timing.climb_rate_mps
        else:
            descent_m += -delta_altitude_m
            descent_time_sec += (-delta_altitude_m) / timing.descent_rate_mps
        total_time_sec += _segment_time_sec(segment_horizontal_distance_m, delta_altitude_m, timing)
    return _TransferCostBreakdown(
        horizontal_distance_m=horizontal_distance_m,
        climb_m=climb_m,
        descent_m=descent_m,
        horizontal_time_sec=horizontal_time_sec,
        climb_time_sec=climb_time_sec,
        descent_time_sec=descent_time_sec,
        total_time_sec=total_time_sec,
        total_cost=(
            horizontal_time_sec * timing.horizontal_energy_rate
            + climb_time_sec * timing.climb_energy_rate
            + descent_time_sec * timing.descent_energy_rate
        ),
    )


def _lnglat3d_transfer_cost(
    points: list[tuple[float, float, float]],
    timing: _TransferCostConfig,
) -> _TransferCostBreakdown:
    horizontal_distance_m = 0.0
    climb_m = 0.0
    descent_m = 0.0
    horizontal_time_sec = 0.0
    climb_time_sec = 0.0
    descent_time_sec = 0.0
    total_time_sec = 0.0
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        start_x, start_y = lnglat_to_mercator(start[0], start[1])
        end_x, end_y = lnglat_to_mercator(end[0], end[1])
        segment_horizontal_distance_m = math.hypot(end_x - start_x, end_y - start_y)
        delta_altitude_m = end[2] - start[2]
        horizontal_distance_m += segment_horizontal_distance_m
        horizontal_time_sec += segment_horizontal_distance_m / timing.horizontal_speed_mps if segment_horizontal_distance_m > ALTITUDE_EPSILON_M else 0.0
        if delta_altitude_m > 0:
            climb_m += delta_altitude_m
            climb_time_sec += delta_altitude_m / timing.climb_rate_mps
        else:
            descent_m += -delta_altitude_m
            descent_time_sec += (-delta_altitude_m) / timing.descent_rate_mps
        total_time_sec += _segment_time_sec(segment_horizontal_distance_m, delta_altitude_m, timing)
    return _TransferCostBreakdown(
        horizontal_distance_m=horizontal_distance_m,
        climb_m=climb_m,
        descent_m=descent_m,
        horizontal_time_sec=horizontal_time_sec,
        climb_time_sec=climb_time_sec,
        descent_time_sec=descent_time_sec,
        total_time_sec=total_time_sec,
        total_cost=(
            horizontal_time_sec * timing.horizontal_energy_rate
            + climb_time_sec * timing.climb_energy_rate
            + descent_time_sec * timing.descent_energy_rate
        ),
    )


def _normalize360(angle_deg: float) -> float:
    return ((angle_deg % 360.0) + 360.0) % 360.0


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max_value, max(min_value, value))


def _invert_maneuver_direction(direction: str) -> str:
    return "counterclockwise" if direction == "clockwise" else "clockwise"


def _sample_terrain_at_lnglat(lng: float, lat: float, dem: TerrainDEM) -> float:
    x, y = lnglat_to_mercator(lng, lat)
    return float(dem.sample_mercator(x, y))


def _calculate_bearing_deg(start_point: tuple[float, float], end_point: tuple[float, float]) -> float:
    if start_point == end_point:
        return 0.0
    lng1, lat1 = math.radians(start_point[0]), math.radians(start_point[1])
    lng2, lat2 = math.radians(end_point[0]), math.radians(end_point[1])
    delta_lng = lng2 - lng1
    y = math.sin(delta_lng) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lng)
    return _normalize360(math.degrees(math.atan2(y, x)))


def _sample_segment_terrain_stats(
    start_mercator: tuple[float, float],
    end_mercator: tuple[float, float],
    dem: TerrainDEM,
    sample_spacing_m: float = DEFAULT_SEGMENT_SAMPLE_SPACING_M,
) -> tuple[float, float]:
    distance_m = math.hypot(end_mercator[0] - start_mercator[0], end_mercator[1] - start_mercator[1])
    sample_count = max(2, int(math.ceil(distance_m / max(5.0, sample_spacing_m))) + 1)
    terrain_values: list[float] = []
    for sample_index in range(sample_count):
        t = sample_index / (sample_count - 1)
        x = start_mercator[0] + (end_mercator[0] - start_mercator[0]) * t
        y = start_mercator[1] + (end_mercator[1] - start_mercator[1]) * t
        terrain = float(dem.sample_mercator(x, y))
        if math.isfinite(terrain):
            terrain_values.append(terrain)
    if not terrain_values:
        return 0.0, 0.0
    return min(terrain_values), max(terrain_values)


def _local_to_mercator(
    center_x: float,
    center_y: float,
    along_x: float,
    along_y: float,
    cross_x: float,
    cross_y: float,
    along_m: float,
    cross_m: float,
) -> tuple[float, float]:
    return (
        center_x + along_m * along_x + cross_m * cross_x,
        center_y + along_m * along_y + cross_m * cross_y,
    )


def _loiter_from_request_model(model: MissionTraversalLoiterModel) -> _TraversalLoiter:
    center_point = (float(model.centerPoint[0]), float(model.centerPoint[1]))
    return _TraversalLoiter(
        center_point=center_point,
        center_mercator=lnglat_to_mercator(*center_point),
        radius_m=max(1.0, float(model.radiusM)),
        direction=str(model.direction),
    )


def _build_fallback_loiter_descriptor(
    point: tuple[float, float],
    heading_deg: float,
    radius_m: float,
    *,
    direction: str = "clockwise",
) -> _TraversalLoiter:
    center_bearing_deg = _normalize360(heading_deg + 90.0)
    center_x, center_y = lnglat_to_mercator(*point)
    center_bearing_rad = math.radians(center_bearing_deg)
    center_mercator = (
        center_x + radius_m * math.cos(center_bearing_rad),
        center_y + radius_m * math.sin(center_bearing_rad),
    )
    return _TraversalLoiter(
        center_point=mercator_to_lnglat(*center_mercator),
        center_mercator=center_mercator,
        radius_m=max(1.0, radius_m),
        direction=direction,
    )


def _traversal_option_from_request(
    area_index: int,
    area: MissionAreaRequest,
    traversal: MissionAreaTraversalRequestModel,
    *,
    flipped: bool,
    bearing_deg: float,
) -> _TraversalOption:
    return _TraversalOption(
        polygon_id=area.polygonId,
        area_index=area_index,
        flipped=flipped,
        bearing_deg=bearing_deg,
        params=area.params,
        altitude_agl=float(traversal.altitudeAGL),
        start_point=(float(traversal.startPoint[0]), float(traversal.startPoint[1])),
        end_point=(float(traversal.endPoint[0]), float(traversal.endPoint[1])),
        start_mercator=lnglat_to_mercator(*traversal.startPoint),
        end_mercator=lnglat_to_mercator(*traversal.endPoint),
        start_terrain_m=float(traversal.startTerrainElevationWgs84M),
        end_terrain_m=float(traversal.endTerrainElevationWgs84M),
        start_altitude_wgs84_m=float(traversal.startAltitudeWgs84M),
        end_altitude_wgs84_m=float(traversal.endAltitudeWgs84M),
        lead_in=_loiter_from_request_model(traversal.leadIn),
        lead_out=_loiter_from_request_model(traversal.leadOut),
    )


def build_area_traversal_options(
    area_index: int,
    area: MissionAreaRequest,
    dem: TerrainDEM,
    *,
    altitude_mode: str,
    min_clearance_m: float,
) -> tuple[_TraversalOption, _TraversalOption]:
    bearing_deg = normalize_axial_bearing(area.bearingDeg)
    if area.forwardTraversal is not None and area.flippedTraversal is not None:
        return (
            _traversal_option_from_request(
                area_index,
                area,
                area.forwardTraversal,
                flipped=False,
                bearing_deg=bearing_deg,
            ),
            _traversal_option_from_request(
                area_index,
                area,
                area.flippedTraversal,
                flipped=True,
                bearing_deg=bearing_deg,
            ),
        )

    polygon = ring_to_polygon_mercator(area.ring)
    center = polygon.centroid
    center_x = float(center.x)
    center_y = float(center.y)
    line_spacing_m = line_spacing_for_params(area.params)
    _along_len_m, cross_width_m = project_extents(polygon, bearing_deg)
    along_rad = deg_to_rad(bearing_deg)
    cross_rad = deg_to_rad((bearing_deg + 90.0) % 360.0)
    along_x, along_y = math.sin(along_rad), math.cos(along_rad)
    cross_x, cross_y = math.sin(cross_rad), math.cos(cross_rad)
    line_count_est = max(1, int(math.ceil(cross_width_m / max(1.0, line_spacing_m))))
    edges = _scanline_edges(
        polygon,
        center_x=center_x,
        center_y=center_y,
        ux=along_x,
        uy=along_y,
        px=cross_x,
        py=cross_y,
    )

    sweeps: list[dict[str, object]] = []
    for sweep_index in range(-line_count_est - 1, line_count_est + 2):
        offset_m = sweep_index * line_spacing_m
        intervals = _scanline_intervals(edges, offset_m)
        if not intervals:
            continue
        direction_forward = len(sweeps) % 2 == 0
        start_along_m = intervals[0][0] if direction_forward else intervals[-1][1]
        end_along_m = intervals[-1][1] if direction_forward else intervals[0][0]
        start_mercator = _local_to_mercator(
            center_x,
            center_y,
            along_x,
            along_y,
            cross_x,
            cross_y,
            start_along_m,
            offset_m,
        )
        end_mercator = _local_to_mercator(
            center_x,
            center_y,
            along_x,
            along_y,
            cross_x,
            cross_y,
            end_along_m,
            offset_m,
        )
        start_point = mercator_to_lnglat(*start_mercator)
        end_point = mercator_to_lnglat(*end_mercator)
        start_terrain_m = _sample_terrain_at_lnglat(start_point[0], start_point[1], dem)
        end_terrain_m = _sample_terrain_at_lnglat(end_point[0], end_point[1], dem)
        min_terrain_m, max_terrain_m = _sample_segment_terrain_stats(start_mercator, end_mercator, dem)
        if altitude_mode == "min-clearance":
            sweep_altitude_wgs84_m = max(
                min_terrain_m + float(area.params.altitudeAGL),
                max_terrain_m + float(min_clearance_m),
            )
        else:
            sweep_altitude_wgs84_m = max_terrain_m + float(area.params.altitudeAGL)
        sweeps.append(
            {
                "start_point": start_point,
                "end_point": end_point,
                "start_mercator": start_mercator,
                "end_mercator": end_mercator,
                "start_terrain_m": start_terrain_m,
                "end_terrain_m": end_terrain_m,
                "altitude_wgs84_m": sweep_altitude_wgs84_m,
            }
        )

    turn_radius_m = max(DEFAULT_CONNECTION_TURN_RADIUS_M, line_spacing_m * 0.5)

    if not sweeps:
        centroid_point = (float(area.ring[0][0]), float(area.ring[0][1]))
        centroid_mercator = lnglat_to_mercator(*centroid_point)
        terrain_m = _sample_terrain_at_lnglat(centroid_point[0], centroid_point[1], dem)
        altitude_wgs84_m = terrain_m + float(area.params.altitudeAGL)
        lead_in = _build_fallback_loiter_descriptor(centroid_point, bearing_deg, turn_radius_m)
        lead_out = _build_fallback_loiter_descriptor(centroid_point, bearing_deg, turn_radius_m)
        forward = _TraversalOption(
            polygon_id=area.polygonId,
            area_index=area_index,
            flipped=False,
            bearing_deg=bearing_deg,
            params=area.params,
            altitude_agl=float(area.params.altitudeAGL),
            start_point=centroid_point,
            end_point=centroid_point,
            start_mercator=centroid_mercator,
            end_mercator=centroid_mercator,
            start_terrain_m=terrain_m,
            end_terrain_m=terrain_m,
            start_altitude_wgs84_m=altitude_wgs84_m,
            end_altitude_wgs84_m=altitude_wgs84_m,
            lead_in=lead_in,
            lead_out=lead_out,
        )
        flipped = _TraversalOption(
            polygon_id=forward.polygon_id,
            area_index=forward.area_index,
            flipped=True,
            bearing_deg=forward.bearing_deg,
            params=forward.params,
            altitude_agl=forward.altitude_agl,
            start_point=forward.start_point,
            end_point=forward.end_point,
            start_mercator=forward.start_mercator,
            end_mercator=forward.end_mercator,
            start_terrain_m=forward.start_terrain_m,
            end_terrain_m=forward.end_terrain_m,
            start_altitude_wgs84_m=forward.start_altitude_wgs84_m,
            end_altitude_wgs84_m=forward.end_altitude_wgs84_m,
            lead_in=_TraversalLoiter(
                center_point=forward.lead_out.center_point,
                center_mercator=forward.lead_out.center_mercator,
                radius_m=forward.lead_out.radius_m,
                direction=_invert_maneuver_direction(forward.lead_out.direction),
            ),
            lead_out=_TraversalLoiter(
                center_point=forward.lead_in.center_point,
                center_mercator=forward.lead_in.center_mercator,
                radius_m=forward.lead_in.radius_m,
                direction=_invert_maneuver_direction(forward.lead_in.direction),
            ),
        )
        return forward, flipped

    first_sweep = sweeps[0]
    last_sweep = sweeps[-1]
    first_start_point = tuple(first_sweep["start_point"])
    first_end_point = tuple(first_sweep["end_point"])
    last_start_point = tuple(last_sweep["start_point"])
    last_end_point = tuple(last_sweep["end_point"])
    forward_start_heading_deg = _calculate_bearing_deg(first_start_point, first_end_point)
    forward_end_heading_deg = _calculate_bearing_deg(last_start_point, last_end_point)

    forward_lead_in = _build_fallback_loiter_descriptor(first_start_point, forward_start_heading_deg, turn_radius_m)
    forward_lead_out = _build_fallback_loiter_descriptor(last_end_point, forward_end_heading_deg, turn_radius_m)

    forward = _TraversalOption(
        polygon_id=area.polygonId,
        area_index=area_index,
        flipped=False,
        bearing_deg=bearing_deg,
        params=area.params,
        altitude_agl=float(area.params.altitudeAGL),
        start_point=first_start_point,
        end_point=last_end_point,
        start_mercator=tuple(first_sweep["start_mercator"]),
        end_mercator=tuple(last_sweep["end_mercator"]),
        start_terrain_m=float(first_sweep["start_terrain_m"]),
        end_terrain_m=float(last_sweep["end_terrain_m"]),
        start_altitude_wgs84_m=float(first_sweep["altitude_wgs84_m"]),
        end_altitude_wgs84_m=float(last_sweep["altitude_wgs84_m"]),
        lead_in=forward_lead_in,
        lead_out=forward_lead_out,
    )
    flipped = _TraversalOption(
        polygon_id=area.polygonId,
        area_index=area_index,
        flipped=True,
        bearing_deg=bearing_deg,
        params=area.params,
        altitude_agl=float(area.params.altitudeAGL),
        start_point=last_end_point,
        end_point=first_start_point,
        start_mercator=tuple(last_sweep["end_mercator"]),
        end_mercator=tuple(first_sweep["start_mercator"]),
        start_terrain_m=float(last_sweep["end_terrain_m"]),
        end_terrain_m=float(first_sweep["start_terrain_m"]),
        start_altitude_wgs84_m=float(last_sweep["altitude_wgs84_m"]),
        end_altitude_wgs84_m=float(first_sweep["altitude_wgs84_m"]),
        lead_in=_TraversalLoiter(
            center_point=forward_lead_out.center_point,
            center_mercator=forward_lead_out.center_mercator,
            radius_m=forward_lead_out.radius_m,
            direction=_invert_maneuver_direction(forward_lead_out.direction),
        ),
        lead_out=_TraversalLoiter(
            center_point=forward_lead_in.center_point,
            center_mercator=forward_lead_in.center_mercator,
            radius_m=forward_lead_in.radius_m,
            direction=_invert_maneuver_direction(forward_lead_in.direction),
        ),
    )
    return forward, flipped


def _build_connection_samples(
    start_mercator: tuple[float, float],
    end_mercator: tuple[float, float],
    dem: TerrainDEM,
    sample_spacing_m: float = DEFAULT_SEGMENT_SAMPLE_SPACING_M,
) -> list[_ConnectionSample]:
    distance_m = math.hypot(end_mercator[0] - start_mercator[0], end_mercator[1] - start_mercator[1])
    sample_count = max(2, int(math.ceil(distance_m / max(5.0, sample_spacing_m))) + 1)
    samples: list[_ConnectionSample] = []
    for sample_index in range(sample_count):
        t = sample_index / (sample_count - 1)
        x = start_mercator[0] + (end_mercator[0] - start_mercator[0]) * t
        y = start_mercator[1] + (end_mercator[1] - start_mercator[1]) * t
        terrain_m = float(dem.sample_mercator(x, y))
        if not math.isfinite(terrain_m):
            continue
        samples.append(
            _ConnectionSample(
                point=mercator_to_lnglat(x, y),
                mercator=(x, y),
                terrain_m=terrain_m,
            )
        )
    return samples


def _almost_equal_local(a: _LocalPoint, b: _LocalPoint, epsilon: float = 1e-6) -> bool:
    return abs(a.x - b.x) <= epsilon and abs(a.y - b.y) <= epsilon


def _append_unique_local(points: list[_LocalPoint], next_point: _LocalPoint) -> None:
    if not points or not _almost_equal_local(points[-1], next_point):
        points.append(next_point)


def _append_unique_3d(points: list[_LocalPoint3D], next_point: _LocalPoint3D) -> None:
    if not points:
        points.append(next_point)
        return
    previous = points[-1]
    if (
        abs(previous.x - next_point.x) > 1e-6
        or abs(previous.y - next_point.y) > 1e-6
        or abs(previous.z - next_point.z) > 1e-6
    ):
        points.append(next_point)


def _distance_local(a: _LocalPoint, b: _LocalPoint) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def _angle_of_point(center: _LocalPoint, point: _LocalPoint) -> float:
    return math.atan2(point.y - center.y, point.x - center.x)


def _point_on_circle(center: _LocalPoint, radius_m: float, angle_rad: float) -> _LocalPoint:
    return _LocalPoint(
        x=center.x + radius_m * math.cos(angle_rad),
        y=center.y + radius_m * math.sin(angle_rad),
    )


def _point_along_bearing(center: _LocalPoint, distance_m: float, angle_rad: float) -> _LocalPoint:
    return _LocalPoint(
        x=center.x + distance_m * math.cos(angle_rad),
        y=center.y + distance_m * math.sin(angle_rad),
    )


def _normalize_arc_delta(start_angle_rad: float, end_angle_rad: float, direction: str) -> float:
    delta_rad = end_angle_rad - start_angle_rad
    if direction == "clockwise":
        while delta_rad >= 0:
            delta_rad -= math.pi * 2
    else:
        while delta_rad <= 0:
            delta_rad += math.pi * 2
    return delta_rad


def _calculate_circle_tangent_point(
    external_point: _LocalPoint,
    loiter: _LocalLoiterDescriptor,
) -> _LocalPoint:
    dx = external_point.x - loiter.center.x
    dy = external_point.y - loiter.center.y
    distance_to_center = math.hypot(dx, dy)
    if not (distance_to_center > loiter.radius_m + 1e-6):
        return _point_on_circle(loiter.center, loiter.radius_m, math.atan2(dy, dx) + math.pi)

    center_to_external_angle = math.atan2(dy, dx)
    center_to_tangent_angle = math.acos(_clamp(loiter.radius_m / distance_to_center, -1.0, 1.0))
    sign = 1.0 if loiter.direction == "clockwise" else -1.0
    return _point_on_circle(
        loiter.center,
        loiter.radius_m,
        center_to_external_angle + sign * center_to_tangent_angle,
    )


def _build_direct_common_tangent_path(
    from_loiter: _LocalLoiterDescriptor,
    to_loiter: _LocalLoiterDescriptor,
) -> list[_LocalPoint] | None:
    dx = to_loiter.center.x - from_loiter.center.x
    dy = to_loiter.center.y - from_loiter.center.y
    distance_between_centers = math.hypot(dx, dy)
    if not (distance_between_centers > 1e-6):
        return None

    delta_radius = from_loiter.radius_m - to_loiter.radius_m
    bearing_rad = math.atan2(dy, dx)
    if distance_between_centers <= abs(delta_radius) + 1e-6:
        if from_loiter.radius_m < to_loiter.radius_m:
            auxiliary_radius_m = (to_loiter.radius_m - (from_loiter.radius_m - distance_between_centers)) / 2.0
            auxiliary_center = _point_along_bearing(
                to_loiter.center,
                from_loiter.radius_m - distance_between_centers + auxiliary_radius_m,
                bearing_rad,
            )
            regular_tangent = _build_direct_common_tangent_path(
                from_loiter,
                _LocalLoiterDescriptor(
                    center=auxiliary_center,
                    radius_m=auxiliary_radius_m,
                    direction=from_loiter.direction,
                ),
            )
            if not regular_tangent:
                return None
            circular_tangent = _sample_arc_local(
                auxiliary_center,
                auxiliary_radius_m,
                regular_tangent[-1],
                _point_on_circle(auxiliary_center, auxiliary_radius_m, bearing_rad),
                from_loiter.direction,
            )
            return _merge_local_segments(regular_tangent, circular_tangent[1:])

        auxiliary_radius_m = (from_loiter.radius_m - (to_loiter.radius_m - distance_between_centers)) / 2.0
        auxiliary_center = _point_along_bearing(
            from_loiter.center,
            to_loiter.radius_m - distance_between_centers + auxiliary_radius_m,
            bearing_rad + math.pi,
        )
        regular_tangent = _build_direct_common_tangent_path(
            _LocalLoiterDescriptor(
                center=auxiliary_center,
                radius_m=auxiliary_radius_m,
                direction=from_loiter.direction,
            ),
            to_loiter,
        )
        if not regular_tangent:
            return None
        circular_tangent = _sample_arc_local(
            auxiliary_center,
            auxiliary_radius_m,
            _point_on_circle(auxiliary_center, auxiliary_radius_m, bearing_rad + math.pi),
            regular_tangent[0],
            from_loiter.direction,
        )
        return _merge_local_segments(circular_tangent, regular_tangent[1:])

    angle_offset_rad = math.acos(_clamp(delta_radius / distance_between_centers, -1.0, 1.0))
    if from_loiter.direction == "clockwise":
        tangent_angle_rad = bearing_rad - angle_offset_rad
    else:
        tangent_angle_rad = bearing_rad + angle_offset_rad

    return [
        _point_on_circle(from_loiter.center, from_loiter.radius_m, tangent_angle_rad),
        _point_on_circle(to_loiter.center, to_loiter.radius_m, tangent_angle_rad),
    ]


def _build_transverse_common_tangent_path(
    from_loiter: _LocalLoiterDescriptor,
    to_loiter: _LocalLoiterDescriptor,
) -> list[_LocalPoint] | None:
    dx = to_loiter.center.x - from_loiter.center.x
    dy = to_loiter.center.y - from_loiter.center.y
    distance_between_centers = math.hypot(dx, dy)
    bearing_rad = math.atan2(dy, dx)
    if not (distance_between_centers > from_loiter.radius_m + to_loiter.radius_m + 1e-6):
        auxiliary_radius_m = (distance_between_centers - from_loiter.radius_m + to_loiter.radius_m) / 2.0
        auxiliary_center = _point_along_bearing(
            from_loiter.center,
            from_loiter.radius_m + auxiliary_radius_m,
            bearing_rad,
        )
        return _sample_arc_local(
            auxiliary_center,
            auxiliary_radius_m,
            _point_on_circle(auxiliary_center, auxiliary_radius_m, bearing_rad + math.pi),
            _point_on_circle(auxiliary_center, auxiliary_radius_m, bearing_rad),
            _invert_maneuver_direction(from_loiter.direction),
        )

    angle_offset_rad = math.acos(
        _clamp((from_loiter.radius_m + to_loiter.radius_m) / distance_between_centers, -1.0, 1.0),
    )
    if from_loiter.direction == "clockwise":
        start_angle_rad = bearing_rad - angle_offset_rad
        end_angle_rad = bearing_rad + math.pi - angle_offset_rad
    else:
        start_angle_rad = bearing_rad + angle_offset_rad
        end_angle_rad = bearing_rad - math.pi + angle_offset_rad

    return [
        _point_on_circle(from_loiter.center, from_loiter.radius_m, start_angle_rad),
        _point_on_circle(to_loiter.center, to_loiter.radius_m, end_angle_rad),
    ]


def _sample_arc_local(
    center: _LocalPoint,
    radius_m: float,
    start_point: _LocalPoint,
    end_point: _LocalPoint,
    direction: str,
) -> list[_LocalPoint]:
    if _almost_equal_local(start_point, end_point):
        return [start_point]
    start_angle_rad = _angle_of_point(center, start_point)
    end_angle_rad = _angle_of_point(center, end_point)
    delta_angle_rad = _normalize_arc_delta(start_angle_rad, end_angle_rad, direction)
    arc_length_m = abs(delta_angle_rad) * radius_m
    segment_count = max(12, int(math.ceil(arc_length_m / CONNECTION_ARC_POINT_SPACING_M)))
    return [
        _point_on_circle(center, radius_m, start_angle_rad + delta_angle_rad * (index / segment_count))
        for index in range(segment_count + 1)
    ]


def _sample_coil_local(
    center: _LocalPoint,
    radius_m: float,
    start_point: _LocalPoint,
    direction: str,
    angular_length_rad: float,
) -> list[_LocalPoint]:
    if not (angular_length_rad > 1e-9):
        return [start_point]
    start_angle_rad = _angle_of_point(center, start_point)
    signed_angular_length_rad = -abs(angular_length_rad) if direction == "clockwise" else abs(angular_length_rad)
    arc_length_m = abs(signed_angular_length_rad) * radius_m
    segment_count = max(24, int(math.ceil(arc_length_m / CONNECTION_COIL_POINT_SPACING_M)))
    return [
        _point_on_circle(center, radius_m, start_angle_rad + signed_angular_length_rad * (index / segment_count))
        for index in range(segment_count + 1)
    ]


def _polyline_distance_local(points: list[_LocalPoint]) -> float:
    return sum(_distance_local(points[index - 1], points[index]) for index in range(1, len(points)))


def _polyline_distance_3d(points: list[_LocalPoint3D]) -> float:
    total = 0.0
    for index in range(1, len(points)):
        dx = points[index].x - points[index - 1].x
        dy = points[index].y - points[index - 1].y
        dz = points[index].z - points[index - 1].z
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def _elevate_polyline(points: list[_LocalPoint], start_altitude_wgs84_m: float, end_altitude_wgs84_m: float) -> list[_LocalPoint3D]:
    if not points:
        return []
    if len(points) == 1:
        return [_LocalPoint3D(x=points[0].x, y=points[0].y, z=end_altitude_wgs84_m)]
    total_distance_m = _polyline_distance_local(points)
    if not (total_distance_m > 1e-6):
        return [_LocalPoint3D(x=point.x, y=point.y, z=end_altitude_wgs84_m) for point in points]

    elevated = [_LocalPoint3D(x=points[0].x, y=points[0].y, z=start_altitude_wgs84_m)]
    traversed_distance_m = 0.0
    for index in range(1, len(points)):
        traversed_distance_m += _distance_local(points[index - 1], points[index])
        t = traversed_distance_m / total_distance_m
        elevated.append(
            _LocalPoint3D(
                x=points[index].x,
                y=points[index].y,
                z=start_altitude_wgs84_m + t * (end_altitude_wgs84_m - start_altitude_wgs84_m),
            )
        )
    return elevated


def _merge_local_segments(*segments: list[_LocalPoint]) -> list[_LocalPoint]:
    merged: list[_LocalPoint] = []
    for segment in segments:
        for point in segment:
            _append_unique_local(merged, point)
    return merged


def _merge_3d_segments(*segments: list[_LocalPoint3D]) -> list[_LocalPoint3D]:
    merged: list[_LocalPoint3D] = []
    for segment in segments:
        for point in segment:
            _append_unique_3d(merged, point)
    return merged


def _local2d_to_lnglat(points: list[_LocalPoint]) -> list[tuple[float, float]]:
    return [mercator_to_lnglat(point.x, point.y) for point in points]


def _local3d_to_lnglat(points: list[_LocalPoint3D]) -> list[tuple[float, float, float]]:
    return [(*mercator_to_lnglat(point.x, point.y), point.z) for point in points]


def _append_line_point(points: list[tuple[float, float]], next_point: tuple[float, float]) -> None:
    if not points or points[-1] != next_point:
        points.append(next_point)


def _append_loiter_step(
    steps: list[MissionConnectionLoiterStepModel],
    point: tuple[float, float],
    target_altitude_wgs84_m: float,
    terrain_elevation_wgs84_m: float,
    previous_altitude_wgs84_m: float,
    loiter_radius_m: float,
) -> None:
    if abs(target_altitude_wgs84_m - previous_altitude_wgs84_m) <= ALTITUDE_EPSILON_M:
        return
    loop_circumference_m = 2 * math.pi * max(1.0, loiter_radius_m)
    altitude_per_loop_m = loop_circumference_m * CONNECTION_CLIMB_PER_METER
    steps.append(
        MissionConnectionLoiterStepModel(
            point=point,
            targetAltitudeWgs84M=target_altitude_wgs84_m,
            terrainElevationWgs84M=terrain_elevation_wgs84_m,
            heightAboveGroundM=target_altitude_wgs84_m - terrain_elevation_wgs84_m,
            direction="climb" if target_altitude_wgs84_m > previous_altitude_wgs84_m else "descent",
            loopCount=max(1, int(math.ceil(abs(target_altitude_wgs84_m - previous_altitude_wgs84_m) / altitude_per_loop_m))),
        )
    )


def _interpolate_stepped_fallback_boundary_anchor(
    *,
    from_sample: _ConnectionSample,
    to_sample: _ConnectionSample,
    current_altitude_wgs84_m: float,
    transfer_min_clearance_m: float,
    resolved_max_height_above_ground_m: float,
    mode: str,
) -> _ConnectionSample:
    from_boundary_wgs84_m = from_sample.terrain_m + (
        transfer_min_clearance_m if mode == "ascend" else resolved_max_height_above_ground_m
    )
    to_boundary_wgs84_m = to_sample.terrain_m + (
        transfer_min_clearance_m if mode == "ascend" else resolved_max_height_above_ground_m
    )
    boundary_delta_wgs84_m = to_boundary_wgs84_m - from_boundary_wgs84_m
    if not math.isfinite(boundary_delta_wgs84_m) or abs(boundary_delta_wgs84_m) <= 1e-6:
        return from_sample

    t = _clamp((current_altitude_wgs84_m - from_boundary_wgs84_m) / boundary_delta_wgs84_m, 0.0, 1.0)
    if not (1e-6 < t < 1.0 - 1e-6):
        return to_sample if t >= 1.0 - 1e-6 else from_sample

    from_x, from_y = from_sample.mercator
    to_x, to_y = to_sample.mercator
    x = from_x + (to_x - from_x) * t
    y = from_y + (to_y - from_y) * t
    return _ConnectionSample(
        point=mercator_to_lnglat(x, y),
        mercator=(x, y),
        terrain_m=from_sample.terrain_m + (to_sample.terrain_m - from_sample.terrain_m) * t,
    )


def _resolve_loiter_segment_bearing_deg(
    corridor_points: list[tuple[float, float]],
    point_index: int,
    fallback_heading_deg: float,
) -> float:
    current_point = corridor_points[point_index]
    next_point = corridor_points[point_index + 1] if point_index + 1 < len(corridor_points) else None
    if next_point is not None and current_point != next_point:
        return _calculate_bearing_deg(current_point, next_point)

    previous_point = corridor_points[point_index - 1] if point_index > 0 else None
    if previous_point is not None and previous_point != current_point:
        return _calculate_bearing_deg(previous_point, current_point)

    return _normalize360(fallback_heading_deg)


def _resolve_stepped_fallback_target_altitude(
    *,
    span_lower_bound_wgs84_m: float,
    span_upper_bound_wgs84_m: float,
    target_altitude_wgs84_m: float,
) -> float:
    # Any altitude inside the overlap span reaches the same future segment set, so
    # stay as close as possible to the destination target to avoid corrective loops.
    return _clamp(target_altitude_wgs84_m, span_lower_bound_wgs84_m, span_upper_bound_wgs84_m)


def _build_bezier_segment(
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    start_heading_deg: float,
    end_heading_deg: float,
    loiter_radius_m: float,
) -> list[tuple[float, float]]:
    if start_point == end_point:
        return [start_point]
    direct_distance_m = haversine_m(start_point, end_point)
    if direct_distance_m <= 1.0:
        return [start_point, end_point]

    control_distance_m = min(max(loiter_radius_m, direct_distance_m * 0.28), direct_distance_m * 0.45)
    start_x, start_y = lnglat_to_mercator(*start_point)
    end_x, end_y = lnglat_to_mercator(*end_point)
    start_heading_rad = math.radians(_normalize360(start_heading_deg))
    end_heading_rad = math.radians(_normalize360(end_heading_deg + 180.0))
    start_control = (
        start_x + control_distance_m * math.sin(start_heading_rad),
        start_y + control_distance_m * math.cos(start_heading_rad),
    )
    end_control = (
        end_x + control_distance_m * math.sin(end_heading_rad),
        end_y + control_distance_m * math.cos(end_heading_rad),
    )
    segment_point_count = max(12, int(math.ceil(direct_distance_m / CONNECTION_ARC_POINT_SPACING_M)))
    segment: list[tuple[float, float]] = []
    for point_index in range(segment_point_count + 1):
        t = point_index / segment_point_count
        mt = 1.0 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2 = t * t
        t3 = t2 * t
        x = mt3 * start_x + 3 * mt2 * t * start_control[0] + 3 * mt * t2 * end_control[0] + t3 * end_x
        y = mt3 * start_y + 3 * mt2 * t * start_control[1] + 3 * mt * t2 * end_control[1] + t3 * end_y
        segment.append(mercator_to_lnglat(x, y))
    return segment


def _build_full_loiter_loop(
    tangent_point: tuple[float, float],
    line_bearing_deg: float,
    direction: str,
    loop_count: int,
    loiter_radius_m: float,
) -> list[tuple[float, float]]:
    clockwise = direction == "climb"
    center_bearing_deg = _normalize360(line_bearing_deg + (90.0 if clockwise else -90.0))
    tangent_x, tangent_y = lnglat_to_mercator(*tangent_point)
    center_bearing_rad = math.radians(center_bearing_deg)
    center_x = tangent_x + loiter_radius_m * math.sin(center_bearing_rad)
    center_y = tangent_y + loiter_radius_m * math.cos(center_bearing_rad)
    start_radial_bearing_deg = _calculate_bearing_deg(mercator_to_lnglat(center_x, center_y), tangent_point)
    total_delta_deg = (360.0 if clockwise else -360.0) * max(1, loop_count)
    arc_length_m = abs(total_delta_deg) * math.pi * loiter_radius_m / 180.0
    point_count = max(24, int(math.ceil(arc_length_m / CONNECTION_COIL_POINT_SPACING_M)))
    arc: list[tuple[float, float]] = []
    for point_index in range(point_count + 1):
        radial_bearing_deg = _normalize360(start_radial_bearing_deg + (total_delta_deg * point_index) / point_count)
        angle_rad = math.radians(radial_bearing_deg)
        arc.append(mercator_to_lnglat(center_x + loiter_radius_m * math.sin(angle_rad), center_y + loiter_radius_m * math.cos(angle_rad)))
    return arc


def _append_segment(trajectory: list[tuple[float, float]], segment: list[tuple[float, float]]) -> None:
    for point in segment:
        _append_line_point(trajectory, point)


def _build_stepped_fallback_trajectory(
    from_option: _TraversalOption,
    to_option: _TraversalOption,
    loiter_steps: list[MissionConnectionLoiterStepModel],
) -> list[tuple[float, float]]:
    loiter_radius_m = max(
        DEFAULT_CONNECTION_TURN_RADIUS_M,
        from_option.lead_out.radius_m,
        to_option.lead_in.radius_m,
    )
    corridor_points = [from_option.end_point, *(step.point for step in loiter_steps), to_option.start_point]
    trajectory = [from_option.end_point]

    if len(corridor_points) == 2:
        direct_bearing_deg = _calculate_bearing_deg(from_option.end_point, to_option.start_point)
        direct_distance_m = haversine_m(from_option.end_point, to_option.start_point)
        shoulder_distance_m = min(max(loiter_radius_m * 1.25, 40.0), direct_distance_m * 0.35)
        from_x, from_y = lnglat_to_mercator(*from_option.end_point)
        to_x, to_y = lnglat_to_mercator(*to_option.start_point)
        direct_bearing_rad = math.radians(direct_bearing_deg)
        straight_start = mercator_to_lnglat(
            from_x + shoulder_distance_m * math.sin(direct_bearing_rad),
            from_y + shoulder_distance_m * math.cos(direct_bearing_rad),
        )
        reverse_bearing_rad = math.radians(_normalize360(direct_bearing_deg + 180.0))
        straight_end = mercator_to_lnglat(
            to_x + shoulder_distance_m * math.sin(reverse_bearing_rad),
            to_y + shoulder_distance_m * math.cos(reverse_bearing_rad),
        )
        _append_segment(
            trajectory,
            _build_bezier_segment(
                from_option.end_point,
                straight_start,
                _calculate_bearing_deg(from_option.start_point, from_option.end_point),
                direct_bearing_deg,
                loiter_radius_m,
            )[1:],
        )
        _append_segment(trajectory, [straight_start, straight_end])
        _append_segment(
            trajectory,
            _build_bezier_segment(
                straight_end,
                to_option.start_point,
                direct_bearing_deg,
                _calculate_bearing_deg(to_option.start_point, to_option.end_point),
                loiter_radius_m,
            )[1:],
        )
        return trajectory

    first_straight_bearing_deg = _calculate_bearing_deg(corridor_points[1], corridor_points[min(2, len(corridor_points) - 1)])
    _append_segment(
        trajectory,
        _build_bezier_segment(
            from_option.end_point,
            corridor_points[1],
            _calculate_bearing_deg(from_option.start_point, from_option.end_point),
            first_straight_bearing_deg,
            loiter_radius_m,
        )[1:],
    )

    for point_index in range(1, len(corridor_points) - 1):
        current_point = corridor_points[point_index]
        next_point = corridor_points[point_index + 1]
        outgoing_bearing_deg = _resolve_loiter_segment_bearing_deg(
            corridor_points,
            point_index,
            _calculate_bearing_deg(to_option.start_point, to_option.end_point),
        )
        loiter_step = loiter_steps[point_index - 1] if point_index - 1 < len(loiter_steps) else None
        if loiter_step is not None:
            _append_segment(
                trajectory,
                _build_full_loiter_loop(
                    current_point,
                    outgoing_bearing_deg,
                    loiter_step.direction,
                    max(1, loiter_step.loopCount or 1),
                    loiter_radius_m,
                )[1:],
            )

        if point_index < len(corridor_points) - 2:
            _append_segment(trajectory, [current_point, next_point])
        else:
            _append_segment(
                trajectory,
                _build_bezier_segment(
                    current_point,
                    to_option.start_point,
                    outgoing_bearing_deg,
                    _calculate_bearing_deg(to_option.start_point, to_option.end_point),
                    loiter_radius_m,
                )[1:],
            )

    return trajectory


def _build_stepped_fallback_trajectory_3d(
    from_option: _TraversalOption,
    to_option: _TraversalOption,
    loiter_steps: list[MissionConnectionLoiterStepModel],
) -> list[tuple[float, float, float]]:
    loiter_radius_m = max(
        DEFAULT_CONNECTION_TURN_RADIUS_M,
        from_option.lead_out.radius_m,
        to_option.lead_in.radius_m,
    )

    def append_segment_3d(
        points: list[tuple[float, float, float]],
        anchor_point: tuple[float, float],
        anchor_altitude_wgs84_m: float,
        segment: list[tuple[float, float]],
        target_altitude_wgs84_m: float | None = None,
    ) -> None:
        if not segment:
            return
        resolved_target_altitude_wgs84_m = (
            anchor_altitude_wgs84_m if target_altitude_wgs84_m is None else target_altitude_wgs84_m
        )
        total_distance_m = 0.0
        previous_point = anchor_point
        for point in segment:
            total_distance_m += haversine_m(previous_point, point)
            previous_point = point

        traversed_distance_m = 0.0
        previous_point = anchor_point
        for point in segment:
            traversed_distance_m += haversine_m(previous_point, point)
            previous_point = point
            t = traversed_distance_m / total_distance_m if total_distance_m > 1e-6 else 1.0
            _append_unique_3d(
                points,
                _LocalPoint3D(
                    x=lnglat_to_mercator(*point)[0],
                    y=lnglat_to_mercator(*point)[1],
                    z=anchor_altitude_wgs84_m + t * (resolved_target_altitude_wgs84_m - anchor_altitude_wgs84_m),
                ),
            )

    corridor_points = [from_option.end_point, *(step.point for step in loiter_steps), to_option.start_point]
    points = [_LocalPoint3D(from_option.end_mercator[0], from_option.end_mercator[1], from_option.end_altitude_wgs84_m)]

    if len(corridor_points) == 2:
        fallback = _build_stepped_fallback_trajectory(from_option, to_option, loiter_steps)
        total_distance_m = sum(haversine_m(fallback[index - 1], fallback[index]) for index in range(1, len(fallback)))
        if not (total_distance_m > 1e-6):
            return _local3d_to_lnglat(points)
        traversed_distance_m = 0.0
        for index in range(1, len(fallback)):
            traversed_distance_m += haversine_m(fallback[index - 1], fallback[index])
            t = traversed_distance_m / total_distance_m
            lng, lat = fallback[index]
            x, y = lnglat_to_mercator(lng, lat)
            _append_unique_3d(
                points,
                _LocalPoint3D(
                    x=x,
                    y=y,
                    z=from_option.end_altitude_wgs84_m
                    + t * (to_option.start_altitude_wgs84_m - from_option.end_altitude_wgs84_m),
                ),
            )
        return _local3d_to_lnglat(points)

    first_straight_bearing_deg = _calculate_bearing_deg(
        corridor_points[1],
        corridor_points[min(2, len(corridor_points) - 1)],
    )
    append_segment_3d(
        points,
        from_option.end_point,
        from_option.end_altitude_wgs84_m,
        _build_bezier_segment(
            from_option.end_point,
            corridor_points[1],
            _calculate_bearing_deg(from_option.start_point, from_option.end_point),
            first_straight_bearing_deg,
            loiter_radius_m,
        )[1:],
    )

    current_altitude_wgs84_m = from_option.end_altitude_wgs84_m
    for point_index in range(1, len(corridor_points) - 1):
        current_point = corridor_points[point_index]
        next_point = corridor_points[point_index + 1]
        outgoing_bearing_deg = _resolve_loiter_segment_bearing_deg(
            corridor_points,
            point_index,
            _calculate_bearing_deg(to_option.start_point, to_option.end_point),
        )
        loiter_step = loiter_steps[point_index - 1]

        if loiter_step:
            append_segment_3d(
                points,
                current_point,
                current_altitude_wgs84_m,
                _build_full_loiter_loop(
                    current_point,
                    outgoing_bearing_deg,
                    loiter_step.direction,
                    max(1, loiter_step.loopCount or 1),
                    loiter_radius_m,
                )[1:],
                loiter_step.targetAltitudeWgs84M,
            )
            current_altitude_wgs84_m = loiter_step.targetAltitudeWgs84M

        if point_index < len(corridor_points) - 2:
            append_segment_3d(points, current_point, current_altitude_wgs84_m, [next_point], current_altitude_wgs84_m)
        else:
            append_segment_3d(
                points,
                current_point,
                current_altitude_wgs84_m,
                _build_bezier_segment(
                    current_point,
                    to_option.start_point,
                    outgoing_bearing_deg,
                    _calculate_bearing_deg(to_option.start_point, to_option.end_point),
                    loiter_radius_m,
                )[1:],
                current_altitude_wgs84_m,
            )

    return _local3d_to_lnglat(points)


def _build_stepped_fallback_connection_geometry(
    from_option: _TraversalOption,
    to_option: _TraversalOption,
    dem: TerrainDEM,
    resolved_max_height_above_ground_m: float,
    transfer_min_clearance_m: float,
) -> _FallbackConnectionGeometry | None:
    samples = _build_connection_samples(from_option.end_mercator, to_option.start_mercator, dem)
    if len(samples) < 2:
        return None

    loiter_radius_m = max(
        DEFAULT_CONNECTION_TURN_RADIUS_M,
        from_option.lead_out.radius_m,
        to_option.lead_in.radius_m,
    )
    line = [from_option.end_point]
    loiter_steps: list[MissionConnectionLoiterStepModel] = []
    current_altitude_wgs84_m = from_option.end_altitude_wgs84_m
    current_index = 0
    last_index = len(samples) - 1
    guard = 0
    sample_bounds = [
        (
            sample.terrain_m + transfer_min_clearance_m,
            sample.terrain_m + resolved_max_height_above_ground_m,
        )
        for sample in samples
    ]

    def altitude_fits_sample_band(altitude_wgs84_m: float, sample: _ConnectionSample) -> bool:
        lower = sample.terrain_m + transfer_min_clearance_m - ALTITUDE_EPSILON_M
        upper = sample.terrain_m + resolved_max_height_above_ground_m + ALTITUDE_EPSILON_M
        return lower <= altitude_wgs84_m <= upper

    def find_latest_final_altitude_anchor() -> _ConnectionSample | None:
        suffix_fits_target_altitude = True
        for sample_index in range(last_index - 1, current_index, -1):
            suffix_fits_target_altitude = (
                suffix_fits_target_altitude
                and altitude_fits_sample_band(to_option.start_altitude_wgs84_m, samples[sample_index])
            )
            if not suffix_fits_target_altitude:
                continue
            if samples[sample_index].point == to_option.start_point:
                continue
            return samples[sample_index]
        return None

    while current_index < last_index and guard < len(samples) + 8:
        guard += 1
        furthest_reachable_index = current_index
        while (
            furthest_reachable_index + 1 <= last_index
            and altitude_fits_sample_band(current_altitude_wgs84_m, samples[furthest_reachable_index + 1])
        ):
            furthest_reachable_index += 1

        if furthest_reachable_index >= last_index:
            break

        next_sample_index = furthest_reachable_index + 1
        must_ascend = current_altitude_wgs84_m < sample_bounds[next_sample_index][0] - ALTITUDE_EPSILON_M
        must_descend = current_altitude_wgs84_m > sample_bounds[next_sample_index][1] + ALTITUDE_EPSILON_M
        span_lower_bound, span_upper_bound = sample_bounds[furthest_reachable_index]
        span_end_index = furthest_reachable_index
        while span_end_index + 1 <= last_index:
            candidate_lower_bound, candidate_upper_bound = sample_bounds[span_end_index + 1]
            next_span_lower_bound = max(span_lower_bound, candidate_lower_bound)
            next_span_upper_bound = min(span_upper_bound, candidate_upper_bound)
            if next_span_lower_bound > next_span_upper_bound + ALTITUDE_EPSILON_M:
                break
            span_lower_bound = next_span_lower_bound
            span_upper_bound = next_span_upper_bound
            span_end_index += 1

        next_altitude_wgs84_m = _resolve_stepped_fallback_target_altitude(
            span_lower_bound_wgs84_m=span_lower_bound,
            span_upper_bound_wgs84_m=span_upper_bound,
            target_altitude_wgs84_m=to_option.start_altitude_wgs84_m,
        )

        if abs(next_altitude_wgs84_m - current_altitude_wgs84_m) <= ALTITUDE_STEP_EPSILON_M:
            _append_line_point(line, to_option.start_point)
            break

        step_sample = (
            _interpolate_stepped_fallback_boundary_anchor(
                from_sample=samples[furthest_reachable_index],
                to_sample=samples[next_sample_index],
                current_altitude_wgs84_m=current_altitude_wgs84_m,
                transfer_min_clearance_m=transfer_min_clearance_m,
                resolved_max_height_above_ground_m=resolved_max_height_above_ground_m,
                mode="ascend" if must_ascend else "descent",
            )
            if (must_ascend or must_descend)
            else samples[furthest_reachable_index]
        )
        _append_line_point(line, step_sample.point)
        _append_loiter_step(
            loiter_steps,
            step_sample.point,
            next_altitude_wgs84_m,
            step_sample.terrain_m,
            current_altitude_wgs84_m,
            loiter_radius_m,
        )
        current_altitude_wgs84_m = next_altitude_wgs84_m
        current_index = furthest_reachable_index

    if abs(to_option.start_altitude_wgs84_m - current_altitude_wgs84_m) > ALTITUDE_STEP_EPSILON_M:
        final_anchor = find_latest_final_altitude_anchor()
        if final_anchor is not None:
            _append_line_point(line, final_anchor.point)
            _append_loiter_step(
                loiter_steps,
                final_anchor.point,
                to_option.start_altitude_wgs84_m,
                final_anchor.terrain_m,
                current_altitude_wgs84_m,
                loiter_radius_m,
            )
            current_altitude_wgs84_m = to_option.start_altitude_wgs84_m

    _append_line_point(line, to_option.start_point)

    final_terrain_elevation_wgs84_m = samples[last_index].terrain_m
    if abs(to_option.start_altitude_wgs84_m - current_altitude_wgs84_m) > ALTITUDE_STEP_EPSILON_M:
        _append_loiter_step(
            loiter_steps,
            to_option.start_point,
            to_option.start_altitude_wgs84_m,
            final_terrain_elevation_wgs84_m,
            current_altitude_wgs84_m,
            loiter_radius_m,
        )

    trajectory = _build_stepped_fallback_trajectory(from_option, to_option, loiter_steps)
    return _FallbackConnectionGeometry(
        line=line,
        trajectory=trajectory,
        loiter_steps=loiter_steps,
    )


def _make_loiter_step_at_circle(
    loiter: _TraversalLoiter,
    target_altitude_wgs84_m: float,
    previous_altitude_wgs84_m: float,
    dem: TerrainDEM,
) -> MissionConnectionLoiterStepModel:
    terrain_elevation_wgs84_m = float(dem.sample_mercator(*loiter.center_mercator))
    loop_circumference_m = 2 * math.pi * max(1.0, loiter.radius_m)
    altitude_per_loop_m = loop_circumference_m * CONNECTION_CLIMB_PER_METER
    return MissionConnectionLoiterStepModel(
        point=loiter.center_point,
        targetAltitudeWgs84M=target_altitude_wgs84_m,
        terrainElevationWgs84M=terrain_elevation_wgs84_m,
        heightAboveGroundM=target_altitude_wgs84_m - terrain_elevation_wgs84_m,
        direction="climb" if target_altitude_wgs84_m > previous_altitude_wgs84_m else "descent",
        loopCount=max(1, int(math.ceil(abs(target_altitude_wgs84_m - previous_altitude_wgs84_m) / altitude_per_loop_m))),
    )


def _build_wic_connector_planar_shape(
    from_option: _TraversalOption,
    to_option: _TraversalOption,
) -> _LocalConnectorPlanarShape | None:
    from_end = _LocalPoint(*from_option.end_mercator)
    to_start = _LocalPoint(*to_option.start_mercator)
    from_loiter = _LocalLoiterDescriptor(
        center=_LocalPoint(*from_option.lead_out.center_mercator),
        radius_m=max(1.0, from_option.lead_out.radius_m),
        direction=from_option.lead_out.direction,
    )
    to_loiter = _LocalLoiterDescriptor(
        center=_LocalPoint(*to_option.lead_in.center_mercator),
        radius_m=max(1.0, to_option.lead_in.radius_m),
        direction=to_option.lead_in.direction,
    )

    start_tangent = _calculate_circle_tangent_point(
        from_end,
        _LocalLoiterDescriptor(
            center=from_loiter.center,
            radius_m=from_loiter.radius_m,
            direction=_invert_maneuver_direction(from_loiter.direction),
        ),
    )
    end_tangent = _calculate_circle_tangent_point(to_start, to_loiter)
    if from_loiter.direction == to_loiter.direction:
        connector_path = _build_direct_common_tangent_path(from_loiter, to_loiter)
    else:
        connector_path = _build_transverse_common_tangent_path(from_loiter, to_loiter)
    if connector_path is None or len(connector_path) == 0:
        return None
    connector_entry = connector_path[0]
    connector_exit = connector_path[-1]
    lead_out_base_2d = _merge_local_segments(
        [from_end, start_tangent],
        _sample_arc_local(
            from_loiter.center,
            from_loiter.radius_m,
            start_tangent,
            connector_entry,
            from_loiter.direction,
        )[1:],
    )
    lead_in_base_2d = _merge_local_segments(
        [connector_exit],
        _sample_arc_local(
            to_loiter.center,
            to_loiter.radius_m,
            connector_exit,
            end_tangent,
            to_loiter.direction,
        )[1:],
        [to_start],
    )
    return _LocalConnectorPlanarShape(
        line=_merge_local_segments([from_end], connector_path, [to_start]),
        lead_out_base_2d=lead_out_base_2d,
        connector_path=connector_path,
        lead_in_base_2d=lead_in_base_2d,
        from_loiter=from_loiter,
        to_loiter=to_loiter,
        start_tangent=start_tangent,
        end_tangent=end_tangent,
        connector_entry=connector_entry,
        connector_exit=connector_exit,
    )


def _build_wic_connector_shape(
    planar: _LocalConnectorPlanarShape,
    from_option: _TraversalOption,
    to_option: _TraversalOption,
    dem: TerrainDEM,
    transfer_altitude_wgs84_m: float | None = None,
) -> _LocalConnectorShape:
    from_altitude_wgs84_m = from_option.end_altitude_wgs84_m
    to_altitude_wgs84_m = to_option.start_altitude_wgs84_m
    if transfer_altitude_wgs84_m is None:
        transfer_altitude_wgs84_m = max(from_altitude_wgs84_m, to_altitude_wgs84_m)
    from_coil_angular_length_rad = (
        abs(transfer_altitude_wgs84_m - from_altitude_wgs84_m)
        / (CONNECTION_CLIMB_PER_METER * max(1.0, planar.from_loiter.radius_m))
        if abs(transfer_altitude_wgs84_m - from_altitude_wgs84_m) > ALTITUDE_EPSILON_M
        else 0.0
    )
    to_coil_angular_length_rad = (
        abs(transfer_altitude_wgs84_m - to_altitude_wgs84_m)
        / (CONNECTION_CLIMB_PER_METER * max(1.0, planar.to_loiter.radius_m))
        if abs(transfer_altitude_wgs84_m - to_altitude_wgs84_m) > ALTITUDE_EPSILON_M
        else 0.0
    )

    lead_out_2d: list[_LocalPoint] = [planar.lead_out_base_2d[0], planar.start_tangent]
    if from_coil_angular_length_rad > 0:
        lead_out_2d = _merge_local_segments(
            lead_out_2d,
            _sample_coil_local(
                planar.from_loiter.center,
                planar.from_loiter.radius_m,
                planar.start_tangent,
                planar.from_loiter.direction,
                from_coil_angular_length_rad,
            )[1:],
        )
    lead_out_2d = _merge_local_segments(
        lead_out_2d,
        _sample_arc_local(
            planar.from_loiter.center,
            planar.from_loiter.radius_m,
            lead_out_2d[-1],
            planar.connector_entry,
            planar.from_loiter.direction,
        )[1:],
    )

    lead_in_2d: list[_LocalPoint] = [planar.connector_exit]
    if to_coil_angular_length_rad > 0:
        lead_in_2d = _merge_local_segments(
            lead_in_2d,
            _sample_coil_local(
                planar.to_loiter.center,
                planar.to_loiter.radius_m,
                planar.connector_exit,
                planar.to_loiter.direction,
                to_coil_angular_length_rad,
            )[1:],
        )
    lead_in_2d = _merge_local_segments(
        lead_in_2d,
        _sample_arc_local(
            planar.to_loiter.center,
            planar.to_loiter.radius_m,
            lead_in_2d[-1],
            planar.end_tangent,
            planar.to_loiter.direction,
        )[1:],
    )
    _append_unique_local(lead_in_2d, planar.lead_in_base_2d[-1])

    lead_out_3d = _elevate_polyline(
        lead_out_2d,
        from_altitude_wgs84_m,
        transfer_altitude_wgs84_m,
    )
    middle_3d = _elevate_polyline(
        planar.connector_path,
        transfer_altitude_wgs84_m,
        transfer_altitude_wgs84_m,
    )
    lead_in_3d = _elevate_polyline(
        lead_in_2d,
        transfer_altitude_wgs84_m,
        to_altitude_wgs84_m,
    )

    loiter_steps: list[MissionConnectionLoiterStepModel] = []
    if from_coil_angular_length_rad > 0:
        loiter_steps.append(
            _make_loiter_step_at_circle(
                from_option.lead_out,
                transfer_altitude_wgs84_m,
                from_altitude_wgs84_m,
                dem,
            )
        )
    if to_coil_angular_length_rad > 0:
        loiter_steps.append(
            _make_loiter_step_at_circle(
                to_option.lead_in,
                to_altitude_wgs84_m,
                transfer_altitude_wgs84_m,
                dem,
            )
        )

    return _LocalConnectorShape(
        line=planar.line,
        trajectory3d=_merge_3d_segments(lead_out_3d, middle_3d, lead_in_3d),
        loiter_steps=loiter_steps,
    )


def _evaluate_transfer_band(
    trajectory3d: list[_LocalPoint3D],
    dem: TerrainDEM,
    transfer_min_clearance_m: float,
    resolved_max_height_above_ground_m: float,
) -> _TransferBandDiagnostics:
    if len(trajectory3d) < 2:
        return _TransferBandDiagnostics(
            fits=False,
            lower_bound_max_wgs84_m=math.inf,
            upper_bound_min_wgs84_m=-math.inf,
            max_below_band_m=0.0,
            max_above_band_m=0.0,
        )
    lower_bound_max_wgs84_m = -math.inf
    upper_bound_min_wgs84_m = math.inf
    max_below_band_m = 0.0
    max_above_band_m = 0.0
    for index in range(1, len(trajectory3d)):
        start = trajectory3d[index - 1]
        end = trajectory3d[index]
        segment_distance_m = math.hypot(end.x - start.x, end.y - start.y)
        sample_count = max(2, int(math.ceil(segment_distance_m / DEFAULT_SEGMENT_SAMPLE_SPACING_M)) + 1)
        for sample_index in range(sample_count):
            t = sample_index / (sample_count - 1) if sample_count > 1 else 1.0
            x = start.x + (end.x - start.x) * t
            y = start.y + (end.y - start.y) * t
            altitude_wgs84_m = start.z + (end.z - start.z) * t
            terrain_elevation_wgs84_m = float(dem.sample_mercator(x, y))
            if not math.isfinite(terrain_elevation_wgs84_m):
                return _TransferBandDiagnostics(
                    fits=False,
                    lower_bound_max_wgs84_m=lower_bound_max_wgs84_m,
                    upper_bound_min_wgs84_m=upper_bound_min_wgs84_m,
                    max_below_band_m=math.inf,
                    max_above_band_m=math.inf,
                )
            lower_bound = terrain_elevation_wgs84_m + transfer_min_clearance_m - ALTITUDE_EPSILON_M
            upper_bound = terrain_elevation_wgs84_m + resolved_max_height_above_ground_m + ALTITUDE_EPSILON_M
            lower_bound_max_wgs84_m = max(lower_bound_max_wgs84_m, lower_bound)
            upper_bound_min_wgs84_m = min(upper_bound_min_wgs84_m, upper_bound)
            max_below_band_m = max(max_below_band_m, lower_bound - altitude_wgs84_m)
            max_above_band_m = max(max_above_band_m, altitude_wgs84_m - upper_bound)
    return _TransferBandDiagnostics(
        fits=max_below_band_m <= ALTITUDE_EPSILON_M and max_above_band_m <= ALTITUDE_EPSILON_M,
        lower_bound_max_wgs84_m=lower_bound_max_wgs84_m,
        upper_bound_min_wgs84_m=upper_bound_min_wgs84_m,
        max_below_band_m=max_below_band_m,
        max_above_band_m=max_above_band_m,
    )


def _path_fits_transfer_band(
    trajectory3d: list[_LocalPoint3D],
    dem: TerrainDEM,
    transfer_min_clearance_m: float,
    resolved_max_height_above_ground_m: float,
) -> bool:
    return _evaluate_transfer_band(
        trajectory3d,
        dem,
        transfer_min_clearance_m,
        resolved_max_height_above_ground_m,
    ).fits


def _build_adjusted_wic_connector_shape(
    planar: _LocalConnectorPlanarShape,
    from_option: _TraversalOption,
    to_option: _TraversalOption,
    dem: TerrainDEM,
    transfer_min_clearance_m: float,
    resolved_max_height_above_ground_m: float,
) -> _LocalConnectorShape | None:
    default_transfer_altitude_wgs84_m = max(from_option.end_altitude_wgs84_m, to_option.start_altitude_wgs84_m)
    base_trajectory3d = _elevate_polyline(
        _merge_local_segments(planar.lead_out_base_2d, planar.connector_path, planar.lead_in_base_2d),
        default_transfer_altitude_wgs84_m,
        default_transfer_altitude_wgs84_m,
    )
    diagnostics = _evaluate_transfer_band(
        base_trajectory3d,
        dem,
        transfer_min_clearance_m,
        resolved_max_height_above_ground_m,
    )
    if not (diagnostics.lower_bound_max_wgs84_m <= diagnostics.upper_bound_min_wgs84_m):
        return None

    def _clamped(value: float) -> float:
        return _clamp(value, diagnostics.lower_bound_max_wgs84_m, diagnostics.upper_bound_min_wgs84_m)

    candidates: list[float] = []
    for candidate in (
        _clamped(default_transfer_altitude_wgs84_m),
        _clamped(from_option.end_altitude_wgs84_m),
        _clamped(to_option.start_altitude_wgs84_m),
        diagnostics.upper_bound_min_wgs84_m,
        diagnostics.lower_bound_max_wgs84_m,
        0.5 * (diagnostics.lower_bound_max_wgs84_m + diagnostics.upper_bound_min_wgs84_m),
    ):
        if not math.isfinite(candidate):
            continue
        if all(abs(existing - candidate) > ALTITUDE_EPSILON_M for existing in candidates):
            candidates.append(candidate)

    for cruise_altitude_wgs84_m in candidates:
        shape = _build_wic_connector_shape(
            planar,
            from_option,
            to_option,
            dem,
            transfer_altitude_wgs84_m=cruise_altitude_wgs84_m,
        )
        if _path_fits_transfer_band(
            shape.trajectory3d,
            dem,
            transfer_min_clearance_m,
            resolved_max_height_above_ground_m,
        ):
            return shape

    return None


def build_connection_candidate(
    from_option: _TraversalOption,
    to_option: _TraversalOption,
    dem: TerrainDEM,
    *,
    max_height_above_ground_m: float,
    transfer_cost: MissionTransferCostModel | None = None,
) -> _ConnectionCandidate:
    requested_max_hag_m = float(max_height_above_ground_m)
    endpoint_hag_from = from_option.end_altitude_wgs84_m - from_option.end_terrain_m
    endpoint_hag_to = to_option.start_altitude_wgs84_m - to_option.start_terrain_m
    resolved_max_hag_m = max(requested_max_hag_m, endpoint_hag_from, endpoint_hag_to)
    transfer_min_clearance_m = min(from_option.altitude_agl, to_option.altitude_agl)
    timing = _resolve_transfer_cost_config(from_option, to_option, transfer_cost)

    planar = _build_wic_connector_planar_shape(from_option, to_option)
    wic_shape = _build_wic_connector_shape(planar, from_option, to_option, dem) if planar is not None else None
    if wic_shape is not None and _path_fits_transfer_band(
        wic_shape.trajectory3d,
        dem,
        transfer_min_clearance_m,
        resolved_max_hag_m,
    ):
        transfer_distance_m = _polyline_distance_3d(wic_shape.trajectory3d)
        transfer_cost_breakdown = _local3d_transfer_cost(wic_shape.trajectory3d, timing)
        trajectory3d = _local3d_to_lnglat(wic_shape.trajectory3d)
        line = _local2d_to_lnglat(wic_shape.line)
        if line:
            line[0] = from_option.end_point
            line[-1] = to_option.start_point
        trajectory = [(lng, lat) for lng, lat, _alt in trajectory3d]
        if trajectory:
            trajectory[0] = from_option.end_point
            trajectory[-1] = to_option.start_point
        model = MissionConnectionModel(
            fromPolygonId=from_option.polygon_id,
            toPolygonId=to_option.polygon_id,
            connectionMode="wic",
            fromFlipped=from_option.flipped,
            toFlipped=to_option.flipped,
            line=line,
            trajectory=trajectory,
            trajectory3D=trajectory3d,
            loiterSteps=wic_shape.loiter_steps,
            requestedMaxHeightAboveGroundM=requested_max_hag_m,
            transferDistanceM=transfer_distance_m,
            transferTimeSec=transfer_cost_breakdown.total_time_sec,
            transferCost=transfer_cost_breakdown.total_cost,
            transferMinClearanceM=transfer_min_clearance_m,
            startAltitudeWgs84M=from_option.end_altitude_wgs84_m,
            endAltitudeWgs84M=to_option.start_altitude_wgs84_m,
            resolvedMaxHeightAboveGroundM=resolved_max_hag_m,
            transferHorizontalDistanceM=transfer_cost_breakdown.horizontal_distance_m,
            transferClimbM=transfer_cost_breakdown.climb_m,
            transferDescentM=transfer_cost_breakdown.descent_m,
            transferHorizontalTimeSec=transfer_cost_breakdown.horizontal_time_sec,
            transferClimbTimeSec=transfer_cost_breakdown.climb_time_sec,
            transferDescentTimeSec=transfer_cost_breakdown.descent_time_sec,
            transferHorizontalSpeedMps=timing.horizontal_speed_mps,
            transferClimbRateMps=timing.climb_rate_mps,
            transferDescentRateMps=timing.descent_rate_mps,
            transferHorizontalEnergyRate=timing.horizontal_energy_rate,
            transferClimbEnergyRate=timing.climb_energy_rate,
            transferDescentEnergyRate=timing.descent_energy_rate,
        )
        return _ConnectionCandidate(
            from_area_index=from_option.area_index,
            to_area_index=to_option.area_index,
            from_flipped=from_option.flipped,
            to_flipped=to_option.flipped,
            model=model,
        )

    adjusted_wic_shape = (
        _build_adjusted_wic_connector_shape(
            planar,
            from_option,
            to_option,
            dem,
            transfer_min_clearance_m,
            resolved_max_hag_m,
        )
        if planar is not None
        else None
    )
    if adjusted_wic_shape is not None:
        transfer_distance_m = _polyline_distance_3d(adjusted_wic_shape.trajectory3d)
        transfer_cost_breakdown = _local3d_transfer_cost(adjusted_wic_shape.trajectory3d, timing)
        trajectory3d = _local3d_to_lnglat(adjusted_wic_shape.trajectory3d)
        line = _local2d_to_lnglat(adjusted_wic_shape.line)
        if line:
            line[0] = from_option.end_point
            line[-1] = to_option.start_point
        trajectory = [(lng, lat) for lng, lat, _alt in trajectory3d]
        if trajectory:
            trajectory[0] = from_option.end_point
            trajectory[-1] = to_option.start_point
        model = MissionConnectionModel(
            fromPolygonId=from_option.polygon_id,
            toPolygonId=to_option.polygon_id,
            connectionMode="wic-adjusted",
            fromFlipped=from_option.flipped,
            toFlipped=to_option.flipped,
            line=line,
            trajectory=trajectory,
            trajectory3D=trajectory3d,
            loiterSteps=adjusted_wic_shape.loiter_steps,
            requestedMaxHeightAboveGroundM=requested_max_hag_m,
            transferDistanceM=transfer_distance_m,
            transferTimeSec=transfer_cost_breakdown.total_time_sec,
            transferCost=transfer_cost_breakdown.total_cost,
            transferMinClearanceM=transfer_min_clearance_m,
            startAltitudeWgs84M=from_option.end_altitude_wgs84_m,
            endAltitudeWgs84M=to_option.start_altitude_wgs84_m,
            resolvedMaxHeightAboveGroundM=resolved_max_hag_m,
            transferHorizontalDistanceM=transfer_cost_breakdown.horizontal_distance_m,
            transferClimbM=transfer_cost_breakdown.climb_m,
            transferDescentM=transfer_cost_breakdown.descent_m,
            transferHorizontalTimeSec=transfer_cost_breakdown.horizontal_time_sec,
            transferClimbTimeSec=transfer_cost_breakdown.climb_time_sec,
            transferDescentTimeSec=transfer_cost_breakdown.descent_time_sec,
            transferHorizontalSpeedMps=timing.horizontal_speed_mps,
            transferClimbRateMps=timing.climb_rate_mps,
            transferDescentRateMps=timing.descent_rate_mps,
            transferHorizontalEnergyRate=timing.horizontal_energy_rate,
            transferClimbEnergyRate=timing.climb_energy_rate,
            transferDescentEnergyRate=timing.descent_energy_rate,
        )
        return _ConnectionCandidate(
            from_area_index=from_option.area_index,
            to_area_index=to_option.area_index,
            from_flipped=from_option.flipped,
            to_flipped=to_option.flipped,
            model=model,
        )

    fallback = _build_stepped_fallback_connection_geometry(
        from_option,
        to_option,
        dem,
        resolved_max_hag_m,
        transfer_min_clearance_m,
    )
    if fallback is None:
        direct_distance_m = max(0.0, haversine_m(from_option.end_point, to_option.start_point))
        direct_cost_breakdown = _lnglat3d_transfer_cost(
            [
                (from_option.end_point[0], from_option.end_point[1], from_option.end_altitude_wgs84_m),
                (to_option.start_point[0], to_option.start_point[1], to_option.start_altitude_wgs84_m),
            ],
            timing,
        )
        model = MissionConnectionModel(
            fromPolygonId=from_option.polygon_id,
            toPolygonId=to_option.polygon_id,
            connectionMode="direct-fallback",
            fromFlipped=from_option.flipped,
            toFlipped=to_option.flipped,
            line=[from_option.end_point, to_option.start_point],
            trajectory=[from_option.end_point, to_option.start_point],
            trajectory3D=[
                (from_option.end_point[0], from_option.end_point[1], from_option.end_altitude_wgs84_m),
                (to_option.start_point[0], to_option.start_point[1], to_option.start_altitude_wgs84_m),
            ],
            loiterSteps=[],
            requestedMaxHeightAboveGroundM=requested_max_hag_m,
            transferDistanceM=direct_distance_m,
            transferTimeSec=direct_cost_breakdown.total_time_sec,
            transferCost=direct_cost_breakdown.total_cost,
            transferMinClearanceM=transfer_min_clearance_m,
            startAltitudeWgs84M=from_option.end_altitude_wgs84_m,
            endAltitudeWgs84M=to_option.start_altitude_wgs84_m,
            resolvedMaxHeightAboveGroundM=resolved_max_hag_m,
            transferHorizontalDistanceM=direct_cost_breakdown.horizontal_distance_m,
            transferClimbM=direct_cost_breakdown.climb_m,
            transferDescentM=direct_cost_breakdown.descent_m,
            transferHorizontalTimeSec=direct_cost_breakdown.horizontal_time_sec,
            transferClimbTimeSec=direct_cost_breakdown.climb_time_sec,
            transferDescentTimeSec=direct_cost_breakdown.descent_time_sec,
            transferHorizontalSpeedMps=timing.horizontal_speed_mps,
            transferClimbRateMps=timing.climb_rate_mps,
            transferDescentRateMps=timing.descent_rate_mps,
            transferHorizontalEnergyRate=timing.horizontal_energy_rate,
            transferClimbEnergyRate=timing.climb_energy_rate,
            transferDescentEnergyRate=timing.descent_energy_rate,
        )
    else:
        fallback3d = _build_stepped_fallback_trajectory_3d(
            from_option,
            to_option,
            fallback.loiter_steps,
        )
        transfer_distance_m = 0.0
        for index in range(1, len(fallback3d)):
            start = fallback3d[index - 1]
            end = fallback3d[index]
            start_x, start_y = lnglat_to_mercator(start[0], start[1])
            end_x, end_y = lnglat_to_mercator(end[0], end[1])
            dz = end[2] - start[2]
            transfer_distance_m += math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2 + dz ** 2)
        fallback_cost_breakdown = _lnglat3d_transfer_cost(fallback3d, timing)

        model = MissionConnectionModel(
            fromPolygonId=from_option.polygon_id,
            toPolygonId=to_option.polygon_id,
            connectionMode="stepped-fallback",
            fromFlipped=from_option.flipped,
            toFlipped=to_option.flipped,
            line=fallback.line,
            trajectory=fallback.trajectory,
            trajectory3D=fallback3d,
            loiterSteps=fallback.loiter_steps,
            requestedMaxHeightAboveGroundM=requested_max_hag_m,
            transferDistanceM=transfer_distance_m,
            transferTimeSec=fallback_cost_breakdown.total_time_sec,
            transferCost=fallback_cost_breakdown.total_cost,
            transferMinClearanceM=transfer_min_clearance_m,
            startAltitudeWgs84M=from_option.end_altitude_wgs84_m,
            endAltitudeWgs84M=to_option.start_altitude_wgs84_m,
            resolvedMaxHeightAboveGroundM=resolved_max_hag_m,
            transferHorizontalDistanceM=fallback_cost_breakdown.horizontal_distance_m,
            transferClimbM=fallback_cost_breakdown.climb_m,
            transferDescentM=fallback_cost_breakdown.descent_m,
            transferHorizontalTimeSec=fallback_cost_breakdown.horizontal_time_sec,
            transferClimbTimeSec=fallback_cost_breakdown.climb_time_sec,
            transferDescentTimeSec=fallback_cost_breakdown.descent_time_sec,
            transferHorizontalSpeedMps=timing.horizontal_speed_mps,
            transferClimbRateMps=timing.climb_rate_mps,
            transferDescentRateMps=timing.descent_rate_mps,
            transferHorizontalEnergyRate=timing.horizontal_energy_rate,
            transferClimbEnergyRate=timing.climb_energy_rate,
            transferDescentEnergyRate=timing.descent_energy_rate,
        )

    return _ConnectionCandidate(
        from_area_index=from_option.area_index,
        to_area_index=to_option.area_index,
        from_flipped=from_option.flipped,
        to_flipped=to_option.flipped,
        model=model,
    )


def _total_path_cost(
    sequence: list[tuple[int, bool]],
    edge_lookup: dict[tuple[int, bool, int, bool], _ConnectionCandidate],
) -> float:
    total = 0.0
    for (from_area, from_flip), (to_area, to_flip) in zip(sequence, sequence[1:]):
        total += edge_lookup[(from_area, from_flip, to_area, to_flip)].objective_cost
    return total


def _solve_exact_path(
    area_count: int,
    edge_lookup: dict[tuple[int, bool, int, bool], _ConnectionCandidate],
) -> list[tuple[int, bool]]:
    if area_count <= 1:
        return [(0, False)] if area_count == 1 else []

    full_mask = (1 << area_count) - 1
    dp: dict[tuple[int, int, bool], float] = {}
    previous: dict[tuple[int, int, bool], tuple[int, int, bool] | None] = {}

    for area_index in range(area_count):
        for flipped in (False, True):
            key = (1 << area_index, area_index, flipped)
            dp[key] = 0.0
            previous[key] = None

    for mask in range(1, full_mask + 1):
        for area_index in range(area_count):
            if not (mask & (1 << area_index)):
                continue
            for flipped in (False, True):
                key = (mask, area_index, flipped)
                current_cost = dp.get(key)
                if current_cost is None:
                    continue
                for next_area_index in range(area_count):
                    if mask & (1 << next_area_index):
                        continue
                    next_mask = mask | (1 << next_area_index)
                    for next_flipped in (False, True):
                        edge = edge_lookup[(area_index, flipped, next_area_index, next_flipped)]
                        next_key = (next_mask, next_area_index, next_flipped)
                        candidate_cost = current_cost + edge.objective_cost
                        if candidate_cost + 1e-9 < dp.get(next_key, math.inf):
                            dp[next_key] = candidate_cost
                            previous[next_key] = key

    best_key: tuple[int, int, bool] | None = None
    best_cost = math.inf
    for area_index in range(area_count):
        for flipped in (False, True):
            key = (full_mask, area_index, flipped)
            cost = dp.get(key)
            if cost is not None and cost < best_cost:
                best_cost = cost
                best_key = key

    if best_key is None:
        return [(index, False) for index in range(area_count)]

    sequence: list[tuple[int, bool]] = []
    cursor = best_key
    while cursor is not None:
        sequence.append((cursor[1], cursor[2]))
        cursor = previous.get(cursor)
    sequence.reverse()
    return sequence


def _solve_greedy_path(
    area_count: int,
    edge_lookup: dict[tuple[int, bool, int, bool], _ConnectionCandidate],
) -> list[tuple[int, bool]]:
    if area_count <= 1:
        return [(0, False)] if area_count == 1 else []

    best_sequence: list[tuple[int, bool]] | None = None
    best_cost = math.inf

    for start_area_index in range(area_count):
        for start_flipped in (False, True):
            visited = {start_area_index}
            sequence = [(start_area_index, start_flipped)]
            while len(sequence) < area_count:
                last_area_index, last_flipped = sequence[-1]
                best_next: tuple[int, bool] | None = None
                best_next_cost = math.inf
                for next_area_index in range(area_count):
                    if next_area_index in visited:
                        continue
                    for next_flipped in (False, True):
                        edge = edge_lookup[(last_area_index, last_flipped, next_area_index, next_flipped)]
                        if edge.objective_cost < best_next_cost:
                            best_next_cost = edge.objective_cost
                            best_next = (next_area_index, next_flipped)
                if best_next is None:
                    break
                visited.add(best_next[0])
                sequence.append(best_next)

            improved = True
            while improved:
                improved = False
                for position in range(len(sequence)):
                    candidate = list(sequence)
                    area_index, flipped = candidate[position]
                    candidate[position] = (area_index, not flipped)
                    if _total_path_cost(candidate, edge_lookup) + 1e-9 < _total_path_cost(sequence, edge_lookup):
                        sequence = candidate
                        improved = True

            for left_index in range(len(sequence)):
                for right_index in range(left_index + 1, len(sequence)):
                    candidate = list(sequence)
                    candidate[left_index], candidate[right_index] = candidate[right_index], candidate[left_index]
                    if _total_path_cost(candidate, edge_lookup) + 1e-9 < _total_path_cost(sequence, edge_lookup):
                        sequence = candidate

            sequence_cost = _total_path_cost(sequence, edge_lookup)
            if sequence_cost < best_cost:
                best_cost = sequence_cost
                best_sequence = sequence

    return best_sequence or [(index, False) for index in range(area_count)]


def optimize_area_sequence(
    request: MissionOptimizeAreaSequenceRequest,
    dem: TerrainDEM,
    *,
    request_id: str,
) -> MissionOptimizeAreaSequenceResponse:
    if len(request.areas) == 1:
        only_area = request.areas[0]
        forward, _ = build_area_traversal_options(
            0,
            only_area,
            dem,
            altitude_mode=request.altitudeMode,
            min_clearance_m=float(request.minClearanceM),
        )
        return MissionOptimizeAreaSequenceResponse(
            requestId=request_id,
            solveMode="exact-dp",
            solvedExactly=True,
            areas=[
                MissionAreaTraversalModel(
                    polygonId=only_area.polygonId,
                    orderIndex=0,
                    flipped=False,
                    bearingDeg=forward.bearing_deg,
                    startPoint=forward.start_point,
                    endPoint=forward.end_point,
                    startAltitudeWgs84M=forward.start_altitude_wgs84_m,
                    endAltitudeWgs84M=forward.end_altitude_wgs84_m,
                )
            ],
            connections=[],
            totalTransferDistanceM=0.0,
            totalTransferTimeSec=0.0,
            totalTransferCost=0.0,
        )
        logger.info(
            "[terrain-split-sequence][%s] single-area passthrough polygonId=%s bearingDeg=%.1f",
            request_id,
            only_area.polygonId,
            forward.bearing_deg,
        )
        return response

    traversal_options_by_area: list[dict[bool, _TraversalOption]] = []
    for area_index, area in enumerate(request.areas):
        forward, flipped = build_area_traversal_options(
            area_index,
            area,
            dem,
            altitude_mode=request.altitudeMode,
            min_clearance_m=float(request.minClearanceM),
        )
        traversal_options_by_area.append({False: forward, True: flipped})

    edge_lookup: dict[tuple[int, bool, int, bool], _ConnectionCandidate] = {}
    for from_area_index in range(len(request.areas)):
        for to_area_index in range(len(request.areas)):
            if from_area_index == to_area_index:
                continue
            for from_flipped in (False, True):
                for to_flipped in (False, True):
                    edge_lookup[(from_area_index, from_flipped, to_area_index, to_flipped)] = build_connection_candidate(
                        traversal_options_by_area[from_area_index][from_flipped],
                        traversal_options_by_area[to_area_index][to_flipped],
                        dem,
                        max_height_above_ground_m=float(request.maxHeightAboveGroundM),
                        transfer_cost=request.transferCost,
                    )

    if len(request.areas) <= request.exactSearchMaxAreas:
        sequence = _solve_exact_path(len(request.areas), edge_lookup)
        solve_mode = "exact-dp"
        solved_exactly = True
    else:
        sequence = _solve_greedy_path(len(request.areas), edge_lookup)
        solve_mode = "greedy-fallback"
        solved_exactly = False

    area_models: list[MissionAreaTraversalModel] = []
    connection_models: list[MissionConnectionModel] = []
    total_transfer_distance_m = 0.0
    total_transfer_time_sec = 0.0
    total_transfer_cost = 0.0

    for order_index, (area_index, flipped) in enumerate(sequence):
        option = traversal_options_by_area[area_index][flipped]
        area_models.append(
            MissionAreaTraversalModel(
                polygonId=option.polygon_id,
                orderIndex=order_index,
                flipped=flipped,
                bearingDeg=option.bearing_deg,
                startPoint=option.start_point,
                endPoint=option.end_point,
                startAltitudeWgs84M=option.start_altitude_wgs84_m,
                endAltitudeWgs84M=option.end_altitude_wgs84_m,
            )
        )
        if order_index == 0:
            continue
        previous_area_index, previous_flipped = sequence[order_index - 1]
        edge = edge_lookup[(previous_area_index, previous_flipped, area_index, flipped)]
        connection_models.append(edge.model)
        total_transfer_distance_m += float(edge.model.transferDistanceM)
        total_transfer_time_sec += float(edge.model.transferTimeSec)
        total_transfer_cost += float(edge.model.transferCost)

    response = MissionOptimizeAreaSequenceResponse(
        requestId=request_id,
        solveMode=solve_mode,
        solvedExactly=solved_exactly,
        areas=area_models,
        connections=connection_models,
        totalTransferDistanceM=total_transfer_distance_m,
        totalTransferTimeSec=total_transfer_time_sec,
        totalTransferCost=total_transfer_cost,
    )
    logger.info(
        "[terrain-split-sequence][%s] solved mode=%s exact=%s areaCount=%d order=%s totalTransferCost=%.3f totalTransferTimeSec=%.3f totalTransferDistanceM=%.1f",
        request_id,
        solve_mode,
        "true" if solved_exactly else "false",
        len(area_models),
        " -> ".join(
            f"{area.polygonId}({'flip' if area.flipped else 'fwd'})"
            for area in area_models
        ),
        total_transfer_cost,
        total_transfer_time_sec,
        total_transfer_distance_m,
    )
    return response
