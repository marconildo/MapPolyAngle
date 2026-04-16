from __future__ import annotations

from fastapi.testclient import TestClient

from terrain_splitter import app as app_module


class FlatDem:
    def sample_mercator(self, x: float, y: float) -> float:
        return 100.0


def _loiter(center_point: tuple[float, float], *, radius_m: float = 1.0, direction: str = "clockwise") -> dict[str, object]:
    return {
        "centerPoint": [center_point[0], center_point[1]],
        "radiusM": radius_m,
        "direction": direction,
    }


def test_optimize_area_sequence_endpoint_returns_order_and_connections(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_rings", lambda *_args, **_kwargs: (FlatDem(), 14))

    request_payload = {
        "areas": [
            {
                "polygonId": "area-a",
                "ring": [[7.0, 47.0], [7.0018, 47.0], [7.0018, 47.0012], [7.0, 47.0012], [7.0, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
            },
            {
                "polygonId": "area-b",
                "ring": [[7.004, 47.0], [7.0058, 47.0], [7.0058, 47.0012], [7.004, 47.0012], [7.004, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
            },
        ],
        "terrainSource": {"mode": "mapbox"},
        "altitudeMode": "legacy",
        "minClearanceM": 60,
        "maxHeightAboveGroundM": 120,
    }

    with TestClient(app_module.app) as client:
        response = client.post("/v1/mission/optimize-area-sequence", json=request_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["solveMode"] == "exact-dp"
    assert payload["solvedExactly"] is True
    assert len(payload["areas"]) == 2
    assert len(payload["connections"]) == 1
    assert payload["connections"][0]["requestedMaxHeightAboveGroundM"] == 120
    assert payload["connections"][0]["transferMinClearanceM"] == 80
    assert payload["connections"][0]["startAltitudeWgs84M"] == 180
    assert payload["connections"][0]["endAltitudeWgs84M"] == 180
    assert payload["connections"][0]["transferCost"] >= 0
    assert payload["totalTransferCost"] >= 0


def test_optimize_area_sequence_endpoint_accepts_provided_traversals(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_rings", lambda *_args, **_kwargs: (FlatDem(), 14))

    request_payload = {
        "areas": [
            {
                "polygonId": "area-a",
                "ring": [[7.0, 47.0], [7.0018, 47.0], [7.0018, 47.0012], [7.0, 47.0012], [7.0, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
                "forwardTraversal": {
                    "altitudeAGL": 80,
                    "startPoint": [7.0, 47.0],
                    "endPoint": [7.01, 47.0],
                    "startTerrainElevationWgs84M": 100,
                    "endTerrainElevationWgs84M": 100,
                    "startAltitudeWgs84M": 180,
                    "endAltitudeWgs84M": 180,
                    "leadIn": _loiter((7.0, 47.0)),
                    "leadOut": _loiter((7.01, 47.0)),
                },
                "flippedTraversal": {
                    "altitudeAGL": 80,
                    "startPoint": [7.01, 47.0],
                    "endPoint": [7.0, 47.0],
                    "startTerrainElevationWgs84M": 100,
                    "endTerrainElevationWgs84M": 100,
                    "startAltitudeWgs84M": 180,
                    "endAltitudeWgs84M": 180,
                    "leadIn": _loiter((7.01, 47.0), direction="counterclockwise"),
                    "leadOut": _loiter((7.0, 47.0), direction="counterclockwise"),
                },
            },
            {
                "polygonId": "area-b",
                "ring": [[7.02, 47.0], [7.0218, 47.0], [7.0218, 47.0012], [7.02, 47.0012], [7.02, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
                "forwardTraversal": {
                    "altitudeAGL": 80,
                    "startPoint": [7.0103, 47.0],
                    "endPoint": [7.02, 47.0],
                    "startTerrainElevationWgs84M": 100,
                    "endTerrainElevationWgs84M": 100,
                    "startAltitudeWgs84M": 180,
                    "endAltitudeWgs84M": 180,
                    "leadIn": _loiter((7.0103, 47.0)),
                    "leadOut": _loiter((7.02, 47.0)),
                },
                "flippedTraversal": {
                    "altitudeAGL": 80,
                    "startPoint": [7.02, 47.0],
                    "endPoint": [7.0103, 47.0],
                    "startTerrainElevationWgs84M": 100,
                    "endTerrainElevationWgs84M": 100,
                    "startAltitudeWgs84M": 180,
                    "endAltitudeWgs84M": 180,
                    "leadIn": _loiter((7.02, 47.0), direction="counterclockwise"),
                    "leadOut": _loiter((7.0103, 47.0), direction="counterclockwise"),
                },
            },
        ],
        "terrainSource": {"mode": "mapbox"},
        "altitudeMode": "legacy",
        "minClearanceM": 60,
        "maxHeightAboveGroundM": 120,
    }

    with TestClient(app_module.app) as client:
        response = client.post("/v1/mission/optimize-area-sequence", json=request_payload)

    assert response.status_code == 200
    payload = response.json()
    assert [(area["polygonId"], area["flipped"]) for area in payload["areas"]] in (
        [("area-a", False), ("area-b", False)],
        [("area-b", True), ("area-a", True)],
    )
    first_area = payload["areas"][0]
    second_area = payload["areas"][1]
    connection = payload["connections"][0]
    assert connection["fromPolygonId"] == first_area["polygonId"]
    assert connection["toPolygonId"] == second_area["polygonId"]
    assert connection["fromFlipped"] == first_area["flipped"]
    assert connection["toFlipped"] == second_area["flipped"]
    assert connection["line"][0] == first_area["endPoint"]
    assert connection["line"][-1] == second_area["startPoint"]
    assert connection["trajectory"][0] == first_area["endPoint"]
    assert connection["trajectory"][-1] == second_area["startPoint"]
    assert connection["loiterSteps"] == []
    assert connection["transferDistanceM"] < 50.0


def test_optimize_area_sequence_endpoint_accepts_transfer_cost_overrides(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_rings", lambda *_args, **_kwargs: (FlatDem(), 14))

    request_payload = {
        "areas": [
            {
                "polygonId": "area-a",
                "ring": [[7.0, 47.0], [7.0018, 47.0], [7.0018, 47.0012], [7.0, 47.0012], [7.0, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
            },
            {
                "polygonId": "area-b",
                "ring": [[7.004, 47.0], [7.0058, 47.0], [7.0058, 47.0012], [7.004, 47.0012], [7.004, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
            },
        ],
        "terrainSource": {"mode": "mapbox"},
        "altitudeMode": "legacy",
        "minClearanceM": 60,
        "maxHeightAboveGroundM": 120,
        "transferCost": {
            "horizontalSpeedMps": 18,
            "climbRateMps": 3,
            "descentRateMps": 7,
            "horizontalEnergyRate": 1.1,
            "climbEnergyRate": 3.2,
            "descentEnergyRate": 0.4,
        },
    }

    with TestClient(app_module.app) as client:
        response = client.post("/v1/mission/optimize-area-sequence", json=request_payload)

    assert response.status_code == 200
    payload = response.json()
    connection = payload["connections"][0]
    assert connection["transferHorizontalSpeedMps"] == 18


def test_optimize_area_sequence_endpoint_accepts_optional_start_and_end_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_rings", lambda *_args, **_kwargs: (FlatDem(), 14))

    request_payload = {
        "areas": [
            {
                "polygonId": "area-a",
                "ring": [[7.0, 47.0], [7.0018, 47.0], [7.0018, 47.0012], [7.0, 47.0012], [7.0, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
            },
            {
                "polygonId": "area-b",
                "ring": [[7.004, 47.0], [7.0058, 47.0], [7.0058, 47.0012], [7.004, 47.0012], [7.004, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
            },
        ],
        "terrainSource": {"mode": "mapbox"},
        "altitudeMode": "legacy",
        "minClearanceM": 60,
        "maxHeightAboveGroundM": 120,
        "startEndpoint": {
            "point": [6.9995, 47.0002],
            "altitudeWgs84M": 180,
            "headingDeg": 90,
            "loiterRadiusM": 60,
        },
        "endEndpoint": {
            "point": [7.0063, 47.0002],
            "altitudeWgs84M": 180,
            "headingDeg": 90,
            "loiterRadiusM": 60,
        },
    }

    with TestClient(app_module.app) as client:
        response = client.post("/v1/mission/optimize-area-sequence", json=request_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["startConnection"]["fromPolygonId"] == "__depot_start__"
    assert payload["endConnection"]["toPolygonId"] == "__depot_end__"
    assert payload["totalTransferCost"] >= payload["startConnection"]["transferCost"] + payload["endConnection"]["transferCost"]


def test_optimize_area_sequence_endpoint_prefetches_area_rings_and_uses_lazy_dem_loading(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_fetch_dem_for_rings(rings, *_args, **kwargs):
        captured["rings"] = rings
        captured["lazy_load_missing"] = kwargs.get("lazy_load_missing")
        captured["grid_step_m"] = kwargs.get("grid_step_m")
        return FlatDem(), 14

    monkeypatch.setattr(app_module, "fetch_dem_for_rings", _fake_fetch_dem_for_rings)

    request_payload = {
        "areas": [
            {
                "polygonId": "area-a",
                "ring": [[7.0, 47.0], [7.0018, 47.0], [7.0018, 47.0012], [7.0, 47.0012], [7.0, 47.0]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
            },
            {
                "polygonId": "area-b",
                "ring": [[7.5, 47.5], [7.5018, 47.5], [7.5018, 47.5012], [7.5, 47.5012], [7.5, 47.5]],
                "bearingDeg": 0,
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 80,
                    "frontOverlap": 70,
                    "sideOverlap": 70,
                    "cameraKey": "SONY_RX1R2",
                    "speedMps": 12,
                },
            },
        ],
        "terrainSource": {"mode": "mapbox"},
        "altitudeMode": "legacy",
        "minClearanceM": 60,
        "maxHeightAboveGroundM": 120,
    }

    with TestClient(app_module.app) as client:
        response = client.post("/v1/mission/optimize-area-sequence", json=request_payload)

    assert response.status_code == 200
    assert captured["rings"] == [
        [tuple(point) for point in area["ring"]]
        for area in request_payload["areas"]
    ]
    assert captured["lazy_load_missing"] is True
