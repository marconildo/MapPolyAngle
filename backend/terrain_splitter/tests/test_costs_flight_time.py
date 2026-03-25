from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon

from terrain_splitter.costs import (
    _estimate_region_flight_time_geos_deprecated,
    estimate_region_flight_time,
)
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
def test_scanline_flight_time_matches_deprecated_geos(polygon: Polygon, bearings: list[float]) -> None:
    for bearing_deg in bearings:
        reference = _estimate_region_flight_time_geos_deprecated(polygon, bearing_deg, CAMERA_PARAMS)
        candidate = estimate_region_flight_time(polygon, bearing_deg, CAMERA_PARAMS)
        _assert_outputs_close(reference, candidate)
