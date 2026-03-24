from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_FLIGHTPLAN = Path(__file__).resolve().parents[1] / "escrow_convex_hull.flightplan"


def _build_request_from_flightplan(flightplan_path: Path) -> dict[str, Any]:
    payload = json.loads(flightplan_path.read_text())
    flight_plan = payload["flightPlan"]
    item = flight_plan["items"][0]
    grid = item["grid"]
    camera = item["camera"]
    ring = [(lng, lat) for lat, lng in item["polygon"]]
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    return {
        "polygonId": "escrow-convex-hull",
        "ring": ring,
        "payloadKind": "lidar",
        "params": {
            "payloadKind": "lidar",
            "altitudeAGL": float(grid["altitude"]),
            "frontOverlap": float(camera["imageFrontalOverlap"]),
            "sideOverlap": float(camera["imageSideOverlap"]),
            "lidarKey": "WINGTRA_LIDAR_XT32M2X",
            "speedMps": float(flight_plan["cruiseSpeed"]),
            "lidarReturnMode": "single",
            "mappingFovDeg": 90.0,
            "maxLidarRangeM": 200.0,
            "pointDensityPtsM2": float(camera["pointDensity"]),
            "triggerDistanceM": float(camera["cameraTriggerDistance"]),
        },
        "altitudeMode": "legacy",
        "minClearanceM": 60.0,
        "turnExtendM": float(grid["turnAroundDistance"]),
        "debug": False,
    }


def _solution_summary(response_json: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for solution in response_json.get("solutions", []):
        summaries.append(
            {
                "regionCount": solution["regionCount"],
                "quality": round(solution["normalizedQualityCost"], 6),
                "timeSec": round(solution["totalMissionTimeSec"], 3),
                "largestRegionFraction": round(solution["largestRegionFraction"], 6),
                "isFirstPracticalSplit": solution["isFirstPracticalSplit"],
                "regions": [
                    {
                        "areaM2": round(region["areaM2"], 3),
                        "bearingDeg": round(region["bearingDeg"], 4),
                    }
                    for region in solution["regions"]
                ],
            }
        )
    return summaries


def _call_backend(url: str, request_json: dict[str, Any], timeout_sec: float) -> tuple[float, dict[str, Any]]:
    started_at = time.perf_counter()
    with httpx.Client(timeout=timeout_sec) as client:
        response = client.post(f"{url.rstrip('/')}/v1/partition/solve", json=request_json)
        response.raise_for_status()
        payload = response.json()
    return time.perf_counter() - started_at, payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark terrain splitter backends against a flightplan request.")
    parser.add_argument(
        "--flightplan",
        type=Path,
        default=DEFAULT_FLIGHTPLAN,
        help="Flightplan to convert into a partition solve request.",
    )
    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="Backend base URL. Repeat to compare multiple backends.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=900.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--dump-request",
        type=Path,
        help="Optional path to write the generated request JSON.",
    )
    args = parser.parse_args()

    request_json = _build_request_from_flightplan(args.flightplan)
    if args.dump_request is not None:
        args.dump_request.write_text(json.dumps(request_json, indent=2))

    results: list[tuple[str, float, dict[str, Any]]] = []
    for url in args.url:
        elapsed_sec, response_json = _call_backend(url, request_json, args.timeout_sec)
        results.append((url, elapsed_sec, response_json))
        print(f"\n=== {url} ===")
        print(f"Elapsed: {elapsed_sec:.3f}s")
        print(f"Request ID: {response_json.get('requestId')}")
        print(json.dumps(_solution_summary(response_json), indent=2))

    if len(results) > 1:
        canonical = json.dumps(_solution_summary(results[0][2]), sort_keys=True)
        identical = all(json.dumps(_solution_summary(result[2]), sort_keys=True) == canonical for result in results[1:])
        print(f"\nOutputs identical across backends: {identical}")
        if not identical:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
