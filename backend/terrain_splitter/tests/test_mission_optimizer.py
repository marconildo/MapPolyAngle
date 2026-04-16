from __future__ import annotations

import itertools
import math

import pytest

from terrain_splitter import mission_optimizer
from terrain_splitter.mission_optimizer import (
    build_area_traversal_options,
    build_connection_candidate,
    optimize_area_sequence,
)
from terrain_splitter.schemas import (
    FlightParamsModel,
    MissionAreaRequest,
    MissionAreaTraversalRequestModel,
    MissionOptimizeAreaSequenceRequest,
    MissionSequenceEndpointModel,
    MissionTraversalLoiterModel,
)


class FakeDem:
    def sample_mercator(self, x: float, y: float) -> float:
        return 100.0 + x * 0.00003 + y * 0.00001


class FlatDem:
    def sample_mercator(self, x: float, y: float) -> float:
        return 100.0


CAMERA_PARAMS = FlightParamsModel(
    payloadKind="camera",
    altitudeAGL=80.0,
    frontOverlap=70.0,
    sideOverlap=70.0,
    cameraKey="SONY_RX1R2",
    speedMps=12.0,
)


def _rectangle(min_lng: float, min_lat: float, width_deg: float, height_deg: float) -> list[tuple[float, float]]:
    return [
        (min_lng, min_lat),
        (min_lng + width_deg, min_lat),
        (min_lng + width_deg, min_lat + height_deg),
        (min_lng, min_lat + height_deg),
        (min_lng, min_lat),
    ]


def _make_area(polygon_id: str, min_lng: float, min_lat: float, *, bearing_deg: float = 0.0) -> MissionAreaRequest:
    return MissionAreaRequest(
        polygonId=polygon_id,
        ring=_rectangle(min_lng, min_lat, 0.0018, 0.0012),
        bearingDeg=bearing_deg,
        payloadKind="camera",
        params=CAMERA_PARAMS.model_copy(deep=True),
    )


def _loiter(center_point: tuple[float, float], *, radius_m: float = 1.0, direction: str = "clockwise") -> MissionTraversalLoiterModel:
    return MissionTraversalLoiterModel(
        centerPoint=center_point,
        radiusM=radius_m,
        direction=direction,
    )


def _make_provided_area(
    polygon_id: str,
    *,
    ring_origin_lng: float,
    ring_origin_lat: float,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    lead_in_center: tuple[float, float],
    lead_out_center: tuple[float, float],
    lead_in_radius_m: float,
    lead_out_radius_m: float,
    lead_in_direction: str = "clockwise",
    lead_out_direction: str = "clockwise",
    altitude_agl: float = 300.0,
    altitude_wgs84_m: float = 500.0,
) -> MissionAreaRequest:
    return MissionAreaRequest(
        polygonId=polygon_id,
        ring=_rectangle(ring_origin_lng, ring_origin_lat, 0.0018, 0.0012),
        bearingDeg=0.0,
        payloadKind="camera",
        params=CAMERA_PARAMS.model_copy(update={"altitudeAGL": altitude_agl}, deep=True),
        forwardTraversal=MissionAreaTraversalRequestModel(
            altitudeAGL=altitude_agl,
            startPoint=start_point,
            endPoint=end_point,
            startTerrainElevationWgs84M=altitude_wgs84_m - altitude_agl,
            endTerrainElevationWgs84M=altitude_wgs84_m - altitude_agl,
            startAltitudeWgs84M=altitude_wgs84_m,
            endAltitudeWgs84M=altitude_wgs84_m,
            leadIn=_loiter(lead_in_center, radius_m=lead_in_radius_m, direction=lead_in_direction),
            leadOut=_loiter(lead_out_center, radius_m=lead_out_radius_m, direction=lead_out_direction),
        ),
        flippedTraversal=MissionAreaTraversalRequestModel(
            altitudeAGL=altitude_agl,
            startPoint=end_point,
            endPoint=start_point,
            startTerrainElevationWgs84M=altitude_wgs84_m - altitude_agl,
            endTerrainElevationWgs84M=altitude_wgs84_m - altitude_agl,
            startAltitudeWgs84M=altitude_wgs84_m,
            endAltitudeWgs84M=altitude_wgs84_m,
            leadIn=_loiter(lead_out_center, radius_m=lead_out_radius_m, direction=_inverse_direction(lead_out_direction)),
            leadOut=_loiter(lead_in_center, radius_m=lead_in_radius_m, direction=_inverse_direction(lead_in_direction)),
        ),
    )


def _inverse_direction(direction: str) -> str:
    return "counterclockwise" if direction == "clockwise" else "clockwise"


def _endpoint(
    point: tuple[float, float],
    *,
    altitude_wgs84_m: float,
    heading_deg: float = 0.0,
    loiter_radius_m: float = 30.0,
) -> MissionSequenceEndpointModel:
    return MissionSequenceEndpointModel(
        point=point,
        altitudeWgs84M=altitude_wgs84_m,
        headingDeg=heading_deg,
        loiterRadiusM=loiter_radius_m,
    )


def test_two_area_exact_solver_picks_lower_cost_flip() -> None:
    dem = FakeDem()
    area_a = _make_area("A", 0.0000, 0.0000)
    area_b = _make_area("B", 0.0030, 0.0000)
    request = MissionOptimizeAreaSequenceRequest(
        areas=[area_a, area_b],
        altitudeMode="legacy",
        minClearanceM=60.0,
        maxHeightAboveGroundM=120.0,
        exactSearchMaxAreas=12,
    )

    response = optimize_area_sequence(request, dem, request_id="req-test")

    options = [
        build_area_traversal_options(index, area, dem, altitude_mode="legacy", min_clearance_m=60.0)
        for index, area in enumerate([area_a, area_b])
    ]
    edge_lookup = {}
    for from_index in range(2):
        for to_index in range(2):
            if from_index == to_index:
                continue
            for from_flipped in (False, True):
                for to_flipped in (False, True):
                    edge_lookup[(from_index, from_flipped, to_index, to_flipped)] = build_connection_candidate(
                        options[from_index][1 if from_flipped else 0],
                        options[to_index][1 if to_flipped else 0],
                        dem,
                        max_height_above_ground_m=120.0,
                    )

    best_cost = math.inf
    optimal_sequences: set[tuple[tuple[int, bool], ...]] = set()
    for sequence in (
        ((0, False), (1, False)),
        ((0, False), (1, True)),
        ((0, True), (1, False)),
        ((0, True), (1, True)),
        ((1, False), (0, False)),
        ((1, False), (0, True)),
        ((1, True), (0, False)),
        ((1, True), (0, True)),
    ):
        cost = edge_lookup[(sequence[0][0], sequence[0][1], sequence[1][0], sequence[1][1])].objective_cost
        if cost + 1e-9 < best_cost:
            best_cost = cost
            optimal_sequences = {sequence}
        elif math.isclose(cost, best_cost, rel_tol=1e-9, abs_tol=1e-9):
            optimal_sequences.add(sequence)

    response_sequence = tuple(
        (0 if area.polygonId == "A" else 1, area.flipped)
        for area in response.areas
    )

    assert response.solveMode == "exact-dp"
    assert response.solvedExactly is True
    assert math.isclose(response.totalTransferCost, best_cost, rel_tol=1e-9, abs_tol=1e-9)
    assert response_sequence in optimal_sequences


def test_exact_solver_matches_bruteforce_best_path_cost() -> None:
    dem = FakeDem()
    areas = [
        _make_area("A", 0.0000, 0.0000),
        _make_area("B", 0.0030, 0.0000),
        _make_area("C", 0.0060, 0.0004),
        _make_area("D", 0.0094, -0.0002),
    ]
    request = MissionOptimizeAreaSequenceRequest(
        areas=areas,
        altitudeMode="legacy",
        minClearanceM=60.0,
        maxHeightAboveGroundM=120.0,
        exactSearchMaxAreas=12,
    )

    response = optimize_area_sequence(request, dem, request_id="req-best")

    options = [
        build_area_traversal_options(index, area, dem, altitude_mode="legacy", min_clearance_m=60.0)
        for index, area in enumerate(areas)
    ]
    edge_lookup = {}
    for from_index in range(len(areas)):
        for to_index in range(len(areas)):
            if from_index == to_index:
                continue
            for from_flipped in (False, True):
                for to_flipped in (False, True):
                    edge_lookup[(from_index, from_flipped, to_index, to_flipped)] = build_connection_candidate(
                        options[from_index][1 if from_flipped else 0],
                        options[to_index][1 if to_flipped else 0],
                        dem,
                        max_height_above_ground_m=120.0,
                    )

    best_cost = math.inf
    for permutation in itertools.permutations(range(len(areas))):
        for flip_choices in itertools.product((False, True), repeat=len(areas)):
            sequence = list(zip(permutation, flip_choices))
            cost = 0.0
            for (from_index, from_flipped), (to_index, to_flipped) in zip(sequence, sequence[1:]):
                cost += edge_lookup[(from_index, from_flipped, to_index, to_flipped)].objective_cost
            best_cost = min(best_cost, cost)

    assert response.solveMode == "exact-dp"
    assert math.isclose(response.totalTransferCost, best_cost, rel_tol=1e-9, abs_tol=1e-9)


def test_connection_cost_penalizes_climb_more_than_descent_for_fixed_wing() -> None:
    dem = FlatDem()
    area_low = _make_provided_area(
        "low",
        ring_origin_lng=0.0,
        ring_origin_lat=0.0,
        start_point=(0.0000, 0.0000),
        end_point=(0.0018, 0.0000),
        lead_in_center=(0.0000, -0.0002),
        lead_out_center=(0.0018, -0.0002),
        lead_in_radius_m=30.0,
        lead_out_radius_m=30.0,
        altitude_agl=120.0,
        altitude_wgs84_m=520.0,
    )
    area_high = _make_provided_area(
        "high",
        ring_origin_lng=0.0040,
        ring_origin_lat=0.0,
        start_point=(0.0040, 0.0000),
        end_point=(0.0058, 0.0000),
        lead_in_center=(0.0040, -0.0002),
        lead_out_center=(0.0058, -0.0002),
        lead_in_radius_m=30.0,
        lead_out_radius_m=30.0,
        altitude_agl=120.0,
        altitude_wgs84_m=700.0,
    )
    low_forward, _ = build_area_traversal_options(0, area_low, dem, altitude_mode="legacy", min_clearance_m=60.0)
    high_forward, _ = build_area_traversal_options(1, area_high, dem, altitude_mode="legacy", min_clearance_m=60.0)

    climb_connection = build_connection_candidate(
        low_forward,
        high_forward,
        dem,
        max_height_above_ground_m=220.0,
    )
    descent_connection = build_connection_candidate(
        high_forward,
        low_forward,
        dem,
        max_height_above_ground_m=220.0,
    )

    assert climb_connection.model.transferClimbM > 0
    assert descent_connection.model.transferDescentM > 0
    assert climb_connection.model.transferCost > descent_connection.model.transferCost
    assert climb_connection.model.transferClimbRateMps == 4.0
    assert descent_connection.model.transferDescentRateMps == 6.0


def test_connection_builder_inserts_loiter_steps_on_rising_terrain() -> None:
    class RisingDem:
        def sample_mercator(self, x: float, y: float) -> float:
            return 100.0 + x * 0.1

    dem = RisingDem()
    area_a = _make_area("A", 0.0000, 0.0000)
    area_b = _make_area("B", 0.0120, 0.0000)
    from_option, _ = build_area_traversal_options(0, area_a, dem, altitude_mode="legacy", min_clearance_m=60.0)
    to_option, _ = build_area_traversal_options(1, area_b, dem, altitude_mode="legacy", min_clearance_m=60.0)

    connection = build_connection_candidate(from_option, to_option, dem, max_height_above_ground_m=120.0)

    assert len(connection.model.loiterSteps) > 0
    assert len(connection.model.line) > 2
    assert all(
        step.heightAboveGroundM <= connection.model.resolvedMaxHeightAboveGroundM + 0.5
        for step in connection.model.loiterSteps
    )


def test_interpolate_stepped_fallback_boundary_anchor() -> None:
    ascend_anchor = mission_optimizer._interpolate_stepped_fallback_boundary_anchor(
        from_sample=mission_optimizer._ConnectionSample(point=(0.0, 0.0), mercator=(0.0, 0.0), terrain_m=100.0),
        to_sample=mission_optimizer._ConnectionSample(point=(0.001, 0.0), mercator=(100.0, 0.0), terrain_m=140.0),
        current_altitude_wgs84_m=200.0,
        transfer_min_clearance_m=80.0,
        resolved_max_height_above_ground_m=120.0,
        mode="ascend",
    )
    assert math.isclose(ascend_anchor.mercator[0], 50.0, abs_tol=1e-9)
    assert math.isclose(ascend_anchor.mercator[1], 0.0, abs_tol=1e-9)
    assert math.isclose(ascend_anchor.terrain_m, 120.0, abs_tol=1e-9)

    descend_anchor = mission_optimizer._interpolate_stepped_fallback_boundary_anchor(
        from_sample=mission_optimizer._ConnectionSample(point=(0.0, 0.0), mercator=(0.0, 0.0), terrain_m=100.0),
        to_sample=mission_optimizer._ConnectionSample(point=(0.001, 0.0), mercator=(100.0, 0.0), terrain_m=140.0),
        current_altitude_wgs84_m=240.0,
        transfer_min_clearance_m=80.0,
        resolved_max_height_above_ground_m=120.0,
        mode="descent",
    )
    assert math.isclose(descend_anchor.mercator[0], 50.0, abs_tol=1e-9)
    assert math.isclose(descend_anchor.mercator[1], 0.0, abs_tol=1e-9)
    assert math.isclose(descend_anchor.terrain_m, 120.0, abs_tol=1e-9)


def test_resolve_stepped_fallback_target_altitude() -> None:
    assert mission_optimizer._resolve_stepped_fallback_target_altitude(
        span_lower_bound_wgs84_m=376.0,
        span_upper_bound_wgs84_m=383.0,
        target_altitude_wgs84_m=191.0,
    ) == 376.0
    assert mission_optimizer._resolve_stepped_fallback_target_altitude(
        span_lower_bound_wgs84_m=376.0,
        span_upper_bound_wgs84_m=383.0,
        target_altitude_wgs84_m=380.0,
    ) == 380.0


def test_resolve_loiter_segment_bearing_uses_incoming_segment_for_duplicate_endpoint() -> None:
    bearing_deg = mission_optimizer._resolve_loiter_segment_bearing_deg(
        corridor_points=[
            (0.0, 0.0),
            (0.001, 0.0),
            (0.001, 0.0),
        ],
        point_index=1,
        fallback_heading_deg=0.0,
    )

    assert math.isclose(bearing_deg, 90.0, abs_tol=1e-6)


def test_stepped_fallback_moves_final_loiter_before_destination_when_suffix_allows_it() -> None:
    class DescendingDem:
        def sample_mercator(self, x: float, y: float) -> float:
            if x < 500.0:
                return 320.0
            if x < 1500.0:
                return 240.0
            return 100.0

    dem = DescendingDem()
    area_a = _make_provided_area(
        "A",
        ring_origin_lng=0.0,
        ring_origin_lat=0.0,
        start_point=(0.0010, 0.0030),
        end_point=(0.0030, 0.0030),
        lead_in_center=(0.0010, 0.00282),
        lead_out_center=(0.0030, 0.00282),
        lead_in_radius_m=20.0,
        lead_out_radius_m=20.0,
        altitude_agl=80.0,
        altitude_wgs84_m=400.0,
    )
    area_b = _make_provided_area(
        "B",
        ring_origin_lng=0.012,
        ring_origin_lat=0.0,
        start_point=(0.0170, 0.0030),
        end_point=(0.0190, 0.0030),
        lead_in_center=(0.0170, 0.00282),
        lead_out_center=(0.0190, 0.00282),
        lead_in_radius_m=20.0,
        lead_out_radius_m=20.0,
        altitude_agl=80.0,
        altitude_wgs84_m=180.0,
    )

    from_option, _ = build_area_traversal_options(0, area_a, dem, altitude_mode="legacy", min_clearance_m=60.0)
    to_option, _ = build_area_traversal_options(1, area_b, dem, altitude_mode="legacy", min_clearance_m=60.0)
    connection = build_connection_candidate(from_option, to_option, dem, max_height_above_ground_m=120.0)

    assert connection.model.connectionMode == "stepped-fallback"
    assert len(connection.model.loiterSteps) == 1
    assert connection.model.loiterSteps[0].point != to_option.start_point
    assert len(connection.model.line) > 2
    assert math.isclose(connection.model.trajectory[-1][0], to_option.start_point[0], abs_tol=1e-9)
    assert math.isclose(connection.model.trajectory[-1][1], to_option.start_point[1], abs_tol=1e-9)


def test_stepped_fallback_avoids_redundant_band_edge_overshoot() -> None:
    class PiecewiseDem:
        def sample_mercator(self, x: float, y: float) -> float:
            if x < 600.0:
                return 244.0
            if x < 1200.0:
                return 277.0
            if x < 1800.0:
                return 157.0
            return 60.0

    dem = PiecewiseDem()
    area_a = _make_provided_area(
        "A",
        ring_origin_lng=0.0,
        ring_origin_lat=0.0,
        start_point=(0.0005, 0.0030),
        end_point=(0.0010, 0.0030),
        lead_in_center=(0.0005, 0.00282),
        lead_out_center=(0.0010, 0.00282),
        lead_in_radius_m=20.0,
        lead_out_radius_m=20.0,
        altitude_agl=99.0,
        altitude_wgs84_m=359.0,
    )
    area_b = _make_provided_area(
        "B",
        ring_origin_lng=0.018,
        ring_origin_lat=0.0,
        start_point=(0.0190, 0.0030),
        end_point=(0.0195, 0.0030),
        lead_in_center=(0.0190, 0.00282),
        lead_out_center=(0.0195, 0.00282),
        lead_in_radius_m=20.0,
        lead_out_radius_m=20.0,
        altitude_agl=99.0,
        altitude_wgs84_m=191.0,
    )

    from_option, _ = build_area_traversal_options(0, area_a, dem, altitude_mode="legacy", min_clearance_m=60.0)
    to_option, _ = build_area_traversal_options(1, area_b, dem, altitude_mode="legacy", min_clearance_m=60.0)
    connection = build_connection_candidate(from_option, to_option, dem, max_height_above_ground_m=139.0)

    assert connection.model.connectionMode == "stepped-fallback"
    assert len(connection.model.loiterSteps) == 2
    assert connection.model.loiterSteps[0].direction == "climb"
    assert connection.model.loiterSteps[1].direction == "descent"
    assert connection.model.loiterSteps[0].targetAltitudeWgs84M < 383.5
    assert connection.model.loiterSteps[1].targetAltitudeWgs84M == to_option.start_altitude_wgs84_m


def test_connection_builder_emits_direct_fallback_when_terrain_samples_are_unavailable() -> None:
    class NanDem:
        def sample_mercator(self, x: float, y: float) -> float:
            return math.nan

    dem = NanDem()
    area_a = _make_provided_area(
        "A",
        ring_origin_lng=0.0,
        ring_origin_lat=0.0,
        start_point=(0.0005, 0.0010),
        end_point=(0.0010, 0.0010),
        lead_in_center=(0.0005, 0.00082),
        lead_out_center=(0.0010, 0.00082),
        lead_in_radius_m=20.0,
        lead_out_radius_m=20.0,
    )
    area_b = _make_provided_area(
        "B",
        ring_origin_lng=0.003,
        ring_origin_lat=0.0,
        start_point=(0.0022, 0.0010),
        end_point=(0.0027, 0.0010),
        lead_in_center=(0.0022, 0.00082),
        lead_out_center=(0.0027, 0.00082),
        lead_in_radius_m=20.0,
        lead_out_radius_m=20.0,
    )

    from_option, _ = build_area_traversal_options(0, area_a, dem, altitude_mode="legacy", min_clearance_m=60.0)
    to_option, _ = build_area_traversal_options(1, area_b, dem, altitude_mode="legacy", min_clearance_m=60.0)
    connection = build_connection_candidate(from_option, to_option, dem, max_height_above_ground_m=600.0)

    assert connection.model.connectionMode == "direct-fallback"
    assert connection.model.line == [from_option.end_point, to_option.start_point]
    assert connection.model.trajectory == [from_option.end_point, to_option.start_point]
    assert connection.model.loiterSteps == []


def test_build_area_traversal_options_handles_empty_scanline_fallback(monkeypatch) -> None:
    dem = FakeDem()
    area = _make_area("A", 0.0000, 0.0000)

    monkeypatch.setattr(mission_optimizer, "_scanline_intervals", lambda *args, **kwargs: [])

    forward, flipped = build_area_traversal_options(0, area, dem, altitude_mode="legacy", min_clearance_m=60.0)

    assert forward.flipped is False
    assert flipped.flipped is True
    assert forward.start_point == flipped.start_point
    assert forward.end_point == flipped.end_point
    assert math.isclose(forward.start_altitude_wgs84_m, flipped.start_altitude_wgs84_m)


def test_build_area_traversal_options_prefers_provided_traversals(monkeypatch) -> None:
    dem = FakeDem()
    area = MissionAreaRequest(
        polygonId="A",
        ring=_rectangle(0.0, 0.0, 0.0018, 0.0012),
        bearingDeg=25.0,
        payloadKind="camera",
        params=CAMERA_PARAMS.model_copy(deep=True),
        forwardTraversal=MissionAreaTraversalRequestModel(
            altitudeAGL=80.0,
            startPoint=(7.0, 47.0),
            endPoint=(7.1, 47.1),
            startTerrainElevationWgs84M=430.0,
            endTerrainElevationWgs84M=435.0,
            startAltitudeWgs84M=510.0,
            endAltitudeWgs84M=515.0,
            leadIn=_loiter((7.0, 47.0)),
            leadOut=_loiter((7.1, 47.1)),
        ),
        flippedTraversal=MissionAreaTraversalRequestModel(
            altitudeAGL=80.0,
            startPoint=(7.1, 47.1),
            endPoint=(7.0, 47.0),
            startTerrainElevationWgs84M=435.0,
            endTerrainElevationWgs84M=430.0,
            startAltitudeWgs84M=515.0,
            endAltitudeWgs84M=510.0,
            leadIn=_loiter((7.1, 47.1), direction="counterclockwise"),
            leadOut=_loiter((7.0, 47.0), direction="counterclockwise"),
        ),
    )

    def _unexpected_scanline(*_args, **_kwargs):
        raise AssertionError("scanline geometry should not be used when traversal descriptors are provided")

    monkeypatch.setattr(mission_optimizer, "_scanline_edges", _unexpected_scanline)

    forward, flipped = build_area_traversal_options(0, area, dem, altitude_mode="legacy", min_clearance_m=60.0)

    assert forward.start_point == (7.0, 47.0)
    assert forward.end_point == (7.1, 47.1)
    assert forward.start_altitude_wgs84_m == 510.0
    assert flipped.start_point == (7.1, 47.1)
    assert flipped.end_point == (7.0, 47.0)
    assert flipped.end_altitude_wgs84_m == 510.0


def test_mission_area_request_requires_both_traversal_descriptors() -> None:
    with pytest.raises(ValueError, match="provided together"):
        MissionAreaRequest(
            polygonId="A",
            ring=_rectangle(0.0, 0.0, 0.0018, 0.0012),
            bearingDeg=25.0,
            payloadKind="camera",
            params=CAMERA_PARAMS.model_copy(deep=True),
            forwardTraversal=MissionAreaTraversalRequestModel(
                altitudeAGL=80.0,
                startPoint=(7.0, 47.0),
                endPoint=(7.1, 47.1),
                startTerrainElevationWgs84M=430.0,
                endTerrainElevationWgs84M=435.0,
                startAltitudeWgs84M=510.0,
                endAltitudeWgs84M=515.0,
                leadIn=_loiter((7.0, 47.0)),
                leadOut=_loiter((7.1, 47.1)),
            ),
        )


def test_single_area_request_allows_exact_limit_of_one() -> None:
    dem = FakeDem()
    request = MissionOptimizeAreaSequenceRequest(
        areas=[_make_area("solo", 0.0, 0.0)],
        altitudeMode="legacy",
        minClearanceM=60.0,
        maxHeightAboveGroundM=120.0,
        exactSearchMaxAreas=1,
    )

    response = optimize_area_sequence(request, dem, request_id="req-single")

    assert response.solveMode == "exact-dp"
    assert response.solvedExactly is True
    assert [area.polygonId for area in response.areas] == ["solo"]
    assert response.connections == []
    assert response.totalTransferDistanceM == 0.0
    assert response.totalTransferTimeSec == 0.0
    assert response.totalTransferCost == 0.0


def test_optimizer_uses_provided_traversal_descriptors_for_costs() -> None:
    dem = FlatDem()
    request = MissionOptimizeAreaSequenceRequest(
        areas=[
            MissionAreaRequest(
                polygonId="A",
                ring=_rectangle(0.0, 0.0, 0.0018, 0.0012),
                bearingDeg=0.0,
                payloadKind="camera",
                params=CAMERA_PARAMS.model_copy(deep=True),
                forwardTraversal=MissionAreaTraversalRequestModel(
                    altitudeAGL=80.0,
                    startPoint=(7.0000, 47.0000),
                    endPoint=(7.0100, 47.0000),
                    startTerrainElevationWgs84M=100.0,
                    endTerrainElevationWgs84M=100.0,
                    startAltitudeWgs84M=180.0,
                    endAltitudeWgs84M=180.0,
                    leadIn=_loiter((7.0000, 47.0000)),
                    leadOut=_loiter((7.0100, 47.0000)),
                ),
                flippedTraversal=MissionAreaTraversalRequestModel(
                    altitudeAGL=80.0,
                    startPoint=(7.0100, 47.0000),
                    endPoint=(7.0000, 47.0000),
                    startTerrainElevationWgs84M=100.0,
                    endTerrainElevationWgs84M=100.0,
                    startAltitudeWgs84M=180.0,
                    endAltitudeWgs84M=180.0,
                    leadIn=_loiter((7.0100, 47.0000), direction="counterclockwise"),
                    leadOut=_loiter((7.0000, 47.0000), direction="counterclockwise"),
                ),
            ),
            MissionAreaRequest(
                polygonId="B",
                ring=_rectangle(0.02, 0.0, 0.0018, 0.0012),
                bearingDeg=0.0,
                payloadKind="camera",
                params=CAMERA_PARAMS.model_copy(deep=True),
                forwardTraversal=MissionAreaTraversalRequestModel(
                    altitudeAGL=80.0,
                    startPoint=(7.0103, 47.0000),
                    endPoint=(7.0200, 47.0000),
                    startTerrainElevationWgs84M=100.0,
                    endTerrainElevationWgs84M=100.0,
                    startAltitudeWgs84M=180.0,
                    endAltitudeWgs84M=180.0,
                    leadIn=_loiter((7.0103, 47.0000)),
                    leadOut=_loiter((7.0200, 47.0000)),
                ),
                flippedTraversal=MissionAreaTraversalRequestModel(
                    altitudeAGL=80.0,
                    startPoint=(7.0200, 47.0000),
                    endPoint=(7.0103, 47.0000),
                    startTerrainElevationWgs84M=100.0,
                    endTerrainElevationWgs84M=100.0,
                    startAltitudeWgs84M=180.0,
                    endAltitudeWgs84M=180.0,
                    leadIn=_loiter((7.0200, 47.0000), direction="counterclockwise"),
                    leadOut=_loiter((7.0103, 47.0000), direction="counterclockwise"),
                ),
            ),
        ],
        altitudeMode="legacy",
        minClearanceM=60.0,
        maxHeightAboveGroundM=120.0,
        exactSearchMaxAreas=12,
    )

    response = optimize_area_sequence(request, dem, request_id="req-provided")
    first_area, second_area = response.areas
    connection = response.connections[0]

    assert [(area.polygonId, area.flipped) for area in response.areas] in (
        [("A", False), ("B", False)],
        [("B", True), ("A", True)],
    )
    assert len(response.connections) == 1
    assert connection.fromPolygonId == first_area.polygonId
    assert connection.toPolygonId == second_area.polygonId
    assert connection.fromFlipped == first_area.flipped
    assert connection.toFlipped == second_area.flipped
    assert connection.line[0] == first_area.endPoint
    assert connection.line[-1] == second_area.startPoint
    assert connection.trajectory[0] == first_area.endPoint
    assert connection.trajectory[-1] == second_area.startPoint
    assert connection.loiterSteps == []
    assert connection.transferDistanceM < 50.0


def test_solver_falls_back_to_greedy_when_exact_limit_is_low() -> None:
    dem = FakeDem()
    request = MissionOptimizeAreaSequenceRequest(
        areas=[
            _make_area("A", 0.0000, 0.0000),
            _make_area("B", 0.0030, 0.0000),
            _make_area("C", 0.0060, 0.0000),
        ],
        altitudeMode="legacy",
        minClearanceM=60.0,
        maxHeightAboveGroundM=120.0,
        exactSearchMaxAreas=2,
    )

    response = optimize_area_sequence(request, dem, request_id="req-greedy")

    assert response.solveMode == "greedy-fallback"
    assert response.solvedExactly is False
    assert sorted(area.polygonId for area in response.areas) == ["A", "B", "C"]


def test_single_area_solver_accounts_for_optional_start_and_end_endpoints() -> None:
    dem = FlatDem()
    area = _make_provided_area(
        "solo",
        ring_origin_lng=0.0,
        ring_origin_lat=0.0,
        start_point=(7.0000, 47.0000),
        end_point=(7.0100, 47.0000),
        lead_in_center=(7.0000, 46.9997),
        lead_out_center=(7.0100, 46.9997),
        lead_in_radius_m=25.0,
        lead_out_radius_m=25.0,
        altitude_agl=80.0,
        altitude_wgs84_m=180.0,
    )
    request = MissionOptimizeAreaSequenceRequest(
        areas=[area],
        altitudeMode="legacy",
        minClearanceM=60.0,
        maxHeightAboveGroundM=120.0,
        exactSearchMaxAreas=1,
        startEndpoint=_endpoint((7.0000, 47.0000), altitude_wgs84_m=180.0, heading_deg=90.0),
        endEndpoint=_endpoint((7.0100, 47.0000), altitude_wgs84_m=180.0, heading_deg=90.0),
    )

    response = optimize_area_sequence(request, dem, request_id="req-single-endpoints")

    options = build_area_traversal_options(0, area, dem, altitude_mode="legacy", min_clearance_m=60.0)
    start_costs = {
        flipped: mission_optimizer._build_endpoint_connection_candidate(
            request.startEndpoint,
            options[1 if flipped else 0],
            dem,
            role="start",
            max_height_above_ground_m=120.0,
        ).objective_cost
        for flipped in (False, True)
    }
    end_costs = {
        flipped: mission_optimizer._build_endpoint_connection_candidate(
            request.endEndpoint,
            options[1 if flipped else 0],
            dem,
            role="end",
            max_height_above_ground_m=120.0,
        ).objective_cost
        for flipped in (False, True)
    }
    expected_flipped = min((False, True), key=lambda flipped: start_costs[flipped] + end_costs[flipped])

    assert response.solveMode == "exact-dp"
    assert response.solvedExactly is True
    assert len(response.areas) == 1
    assert response.areas[0].flipped is expected_flipped
    assert response.startConnection is not None
    assert response.endConnection is not None
    assert math.isclose(
        response.totalTransferCost,
        response.startConnection.transferCost + response.endConnection.transferCost,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_exact_solver_accepts_one_sided_endpoint_constraints() -> None:
    dem = FlatDem()
    areas = [
        _make_area("A", 0.0000, 0.0000),
        _make_area("B", 0.0030, 0.0000),
    ]
    request = MissionOptimizeAreaSequenceRequest(
        areas=areas,
        altitudeMode="legacy",
        minClearanceM=60.0,
        maxHeightAboveGroundM=120.0,
        exactSearchMaxAreas=12,
        startEndpoint=_endpoint((6.9990, 47.0000), altitude_wgs84_m=180.0, heading_deg=90.0),
    )

    response = optimize_area_sequence(request, dem, request_id="req-start-only")

    assert response.startConnection is not None
    assert response.endConnection is None
    assert response.startConnection.fromPolygonId == "__depot_start__"
    assert response.totalTransferCost >= response.startConnection.transferCost


def test_connection_builder_handles_same_direction_overlap_with_auxiliary_arc() -> None:
    dem = FlatDem()
    area_a = _make_provided_area(
        "A",
        ring_origin_lng=0.0,
        ring_origin_lat=0.0,
        start_point=(0.00055, 0.0010),
        end_point=(0.00075, 0.0010),
        lead_in_center=(0.0010, 0.0010),
        lead_out_center=(0.0010, 0.0010),
        lead_in_radius_m=30.0,
        lead_out_radius_m=30.0,
    )
    area_b = _make_provided_area(
        "B",
        ring_origin_lng=0.003,
        ring_origin_lat=0.0,
        start_point=(0.00155, 0.0010),
        end_point=(0.00175, 0.0010),
        lead_in_center=(0.00112, 0.0010),
        lead_out_center=(0.00112, 0.0010),
        lead_in_radius_m=50.0,
        lead_out_radius_m=50.0,
    )

    from_option, _ = build_area_traversal_options(0, area_a, dem, altitude_mode="legacy", min_clearance_m=60.0)
    to_option, _ = build_area_traversal_options(1, area_b, dem, altitude_mode="legacy", min_clearance_m=60.0)
    connection = build_connection_candidate(from_option, to_option, dem, max_height_above_ground_m=600.0)

    assert connection.model.loiterSteps == []
    assert len(connection.model.line) > 4


def test_connection_builder_handles_opposite_direction_overlap_with_auxiliary_arc() -> None:
    dem = FlatDem()
    area_a = _make_provided_area(
        "A",
        ring_origin_lng=0.0,
        ring_origin_lat=0.0,
        start_point=(0.00055, 0.0010),
        end_point=(0.00075, 0.0010),
        lead_in_center=(0.0010, 0.0010),
        lead_out_center=(0.0010, 0.0010),
        lead_in_radius_m=35.0,
        lead_out_radius_m=35.0,
    )
    area_b = _make_provided_area(
        "B",
        ring_origin_lng=0.003,
        ring_origin_lat=0.0,
        start_point=(0.00145, 0.0010),
        end_point=(0.00165, 0.0010),
        lead_in_center=(0.00118, 0.0010),
        lead_out_center=(0.00118, 0.0010),
        lead_in_radius_m=25.0,
        lead_out_radius_m=25.0,
        lead_in_direction="counterclockwise",
        lead_out_direction="counterclockwise",
    )

    from_option, _ = build_area_traversal_options(0, area_a, dem, altitude_mode="legacy", min_clearance_m=60.0)
    to_option, _ = build_area_traversal_options(1, area_b, dem, altitude_mode="legacy", min_clearance_m=60.0)
    connection = build_connection_candidate(from_option, to_option, dem, max_height_above_ground_m=600.0)

    assert connection.model.loiterSteps == []
    assert len(connection.model.line) > 4
