from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from terrain_splitter.costs import _estimate_region_flight_time_geos_deprecated, estimate_region_flight_time
from terrain_splitter.schemas import FlightParamsModel

CAMERA_PARAMS = FlightParamsModel(
    payloadKind="camera",
    altitudeAGL=120.0,
    frontOverlap=70.0,
    sideOverlap=70.0,
    cameraKey="SONY_RX1R2",
    speedMps=12.0,
)


def _assert_outputs_close(reference: dict[str, float], candidate: dict[str, float]) -> None:
    assert set(reference) == set(candidate)
    for key, reference_value in reference.items():
        candidate_value = candidate[key]
        assert math.isclose(reference_value, candidate_value, rel_tol=1e-12, abs_tol=1e-8), key


def _assert_shared_scanline_metrics(reference: dict[str, float], candidate: dict[str, float]) -> None:
    assert set(reference) == set(candidate)
    for key in (
        "line_spacing_m",
        "line_count",
        "fragmented_line_count",
        "fragmented_line_fraction",
        "inter_segment_gap_length_m",
        "turn_count",
        "cruise_speed_mps",
    ):
        assert math.isclose(reference[key], candidate[key], rel_tol=1e-12, abs_tol=1e-8), key


@pytest.mark.parametrize(
    ("polygon", "bearings"),
    [
        (
            Polygon([(0.0, 0.0), (1800.0, 0.0), (1800.0, 900.0), (0.0, 900.0), (0.0, 0.0)]),
            [0.0, 37.5, 90.0],
        ),
        (
            Polygon(
                [
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
            ),
            [0.0, 45.0, 90.0],
        ),
        (
            Polygon(
                [
                    (0.0, 0.0),
                    (2200.0, 0.0),
                    (2200.0, 2200.0),
                    (1425.0, 2200.0),
                    (1100.0, 900.0),
                    (775.0, 2200.0),
                    (0.0, 2200.0),
                    (0.0, 0.0),
                ]
            ),
            [0.0, 45.0, 90.0],
        ),
    ],
)
def test_scanline_flight_time_uses_connected_path_surrogate(polygon: Polygon, bearings: list[float]) -> None:
    for bearing_deg in bearings:
        reference = _estimate_region_flight_time_geos_deprecated(polygon, bearing_deg, CAMERA_PARAMS)
        candidate = estimate_region_flight_time(polygon, bearing_deg, CAMERA_PARAMS)
        _assert_shared_scanline_metrics(reference, candidate)

        assert 0.0 <= candidate["overflight_transit_fraction"] <= 1.0
        assert candidate["total_mission_time_sec"] >= 25.0

        if reference["total_flight_line_length_m"] <= 1e-9:
            assert candidate["total_flight_line_length_m"] > 0.0
            assert candidate["mean_line_length_m"] > 0.0
            assert candidate["median_line_length_m"] > 0.0
            assert candidate["short_line_fraction"] < 1.0
            assert candidate["total_mission_time_sec"] > reference["total_mission_time_sec"]
            continue

        assert candidate["total_flight_line_length_m"] >= reference["total_flight_line_length_m"]
        assert candidate["mean_line_length_m"] >= reference["mean_line_length_m"]
        assert candidate["median_line_length_m"] >= reference["median_line_length_m"]
        assert candidate["short_line_fraction"] <= reference["short_line_fraction"] + 1e-12

        if math.isclose(reference["total_flight_line_length_m"], candidate["total_flight_line_length_m"], rel_tol=1e-12, abs_tol=1e-8):
            assert math.isclose(
                reference["overflight_transit_fraction"],
                candidate["overflight_transit_fraction"],
                rel_tol=1e-12,
                abs_tol=1e-8,
            )
            assert math.isclose(
                reference["total_mission_time_sec"],
                candidate["total_mission_time_sec"],
                rel_tol=1e-12,
                abs_tol=1e-8,
            )
        else:
            assert candidate["overflight_transit_fraction"] <= reference["overflight_transit_fraction"] + 1e-12
            assert candidate["total_mission_time_sec"] >= reference["total_mission_time_sec"]
