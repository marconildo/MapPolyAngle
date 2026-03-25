from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import threading
from base64 import b64encode
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from fastapi.testclient import TestClient
from PIL import Image

import main as main_module
from terrain_splitter import app as app_module
from terrain_splitter import dsm_store as dsm_store_module
from terrain_splitter.dsm_store import (
    DsmDatasetStore,
    S3BackedDsmDatasetStore,
    create_dsm_dataset_store,
)
from terrain_splitter.exact_bridge import LocalExactRuntimeSidecarBridge
from terrain_splitter.mapbox_tiles import TerrainTile, fetch_dem_for_ring
from terrain_splitter.schemas import (
    DsmSourceDescriptorModel,
    PartitionSolutionPreviewModel,
    TerrainBatchRequestModel,
    TerrainBatchResponseModel,
    TerrainSourceModel,
)


def _descriptor() -> DsmSourceDescriptorModel:
    return DsmSourceDescriptorModel.model_validate(
        {
            "id": "frontend-random-id",
            "name": "test-dsm.tiff",
            "fileSizeBytes": 0,
            "width": 4,
            "height": 4,
            "sourceBounds": {"minX": 0, "minY": 0, "maxX": 4, "maxY": 4},
            "footprint3857": {"minX": 0, "minY": 0, "maxX": 4, "maxY": 4},
            "footprintLngLat": {"minLng": 0, "minLat": 0, "maxLng": 0.0001, "maxLat": 0.0001},
            "footprintRingLngLat": [[0, 0], [0.0001, 0], [0.0001, 0.0001], [0, 0.0001], [0, 0]],
            "sourceCrsCode": "EPSG:3857",
            "sourceCrsLabel": "EPSG:3857",
            "sourceProj4": "EPSG:3857",
            "horizontalUnits": "metre",
            "verticalScaleToMeters": 1,
            "noDataValue": None,
            "loadedAtIso": "2026-03-21T00:00:00Z",
        }
    )


def _write_test_tiff(path: Path, raster: np.ndarray) -> bytes:
    tifffile.imwrite(path, raster.astype(np.float32), compression="lzw")
    return path.read_bytes()


def _write_alpha_masked_tiff(path: Path, raster: np.ndarray, alpha: np.ndarray) -> bytes:
    stacked = np.stack([raster.astype(np.float32), alpha.astype(np.uint8)], axis=-1)
    tifffile.imwrite(
        path,
        stacked,
        compression="lzw",
        metadata=None,
        photometric="minisblack",
        extrasamples=["unassalpha"],
    )
    return path.read_bytes()


def _write_geotiff(path: Path, raster: np.ndarray) -> bytes:
    pixel_scale = (1.0, 1.0, 0.0)
    tiepoint = (0.0, 0.0, 0.0, 0.0, float(raster.shape[0]), 0.0)
    geo_key_directory = (
        1,
        1,
        0,
        3,
        1024,
        0,
        1,
        1,
        1025,
        0,
        1,
        1,
        3072,
        0,
        1,
        3857,
    )
    tifffile.imwrite(
        path,
        raster.astype(np.float32),
        compression="lzw",
        extratags=[
            (33550, "d", 3, pixel_scale, False),
            (33922, "d", 6, tiepoint, False),
            (34735, "H", len(geo_key_directory), geo_key_directory, False),
        ],
    )
    return path.read_bytes()


def _write_rgba_geotiff(path: Path, raster: np.ndarray) -> bytes:
    pixel_scale = (1.0, 1.0, 0.0)
    tiepoint = (0.0, 0.0, 0.0, 0.0, float(raster.shape[0]), 0.0)
    geo_key_directory = (
        1,
        1,
        0,
        3,
        1024,
        0,
        1,
        1,
        1025,
        0,
        1,
        1,
        3072,
        0,
        1,
        3857,
    )
    tifffile.imwrite(
        path,
        raster.astype(np.uint8),
        compression="lzw",
        metadata=None,
        photometric="rgb",
        extrasamples=["unassalpha"],
        extratags=[
            (33550, "d", 3, pixel_scale, False),
            (33922, "d", 6, tiepoint, False),
            (34735, "H", len(geo_key_directory), geo_key_directory, False),
        ],
    )
    return path.read_bytes()


def _write_geographic_geotiff(path: Path, raster: np.ndarray) -> bytes:
    pixel_scale = (0.01, 0.01, 0.0)
    tiepoint = (0.0, 0.0, 0.0, 7.0, 47.0, 0.0)
    geo_key_directory = (
        1,
        1,
        0,
        3,
        1024,
        0,
        1,
        2,
        1025,
        0,
        1,
        1,
        2048,
        0,
        1,
        4326,
    )
    tifffile.imwrite(
        path,
        raster.astype(np.float32),
        compression="lzw",
        extratags=[
            (33550, "d", 3, pixel_scale, False),
            (33922, "d", 6, tiepoint, False),
            (34735, "H", len(geo_key_directory), geo_key_directory, False),
        ],
    )
    return path.read_bytes()


def _write_transform_geotiff(path: Path, raster: np.ndarray) -> bytes:
    transform = (
        2.0, 0.0, 0.0, 100.0,
        0.0, -3.0, 0.0, 200.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    geo_key_directory = (
        1,
        1,
        0,
        3,
        1024,
        0,
        1,
        1,
        1025,
        0,
        1,
        1,
        3072,
        0,
        1,
        3857,
    )
    tifffile.imwrite(
        path,
        raster.astype(np.float32),
        compression="lzw",
        extratags=[
            (34264, "d", len(transform), transform, False),
            (34735, "H", len(geo_key_directory), geo_key_directory, False),
        ],
    )
    return path.read_bytes()


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _real_dsm_fixture_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "example_dsm_23_5_cmpx_crop_512.tiff"


def _ring_inside_descriptor(descriptor: DsmSourceDescriptorModel, inset_fraction: float = 0.2) -> list[list[float]]:
    bounds = descriptor.footprintLngLat
    span_lng = bounds.maxLng - bounds.minLng
    span_lat = bounds.maxLat - bounds.minLat
    inset_lng = span_lng * inset_fraction
    inset_lat = span_lat * inset_fraction
    min_lng = bounds.minLng + inset_lng
    max_lng = bounds.maxLng - inset_lng
    min_lat = bounds.minLat + inset_lat
    max_lat = bounds.maxLat - inset_lat
    return [
        [min_lng, min_lat],
        [max_lng, min_lat],
        [max_lng, max_lat],
        [min_lng, max_lat],
        [min_lng, min_lat],
    ]


def _tile_ref_for_descriptor(descriptor: DsmSourceDescriptorModel, zoom: int = 14) -> tuple[int, int, int]:
    bounds = descriptor.footprintLngLat
    lng = (bounds.minLng + bounds.maxLng) / 2
    lat = (bounds.minLat + bounds.maxLat) / 2
    n = 2 ** zoom
    tile_x = int((lng + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    tile_y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    tile_x = max(0, min(n - 1, tile_x))
    tile_y = max(0, min(n - 1, tile_y))
    return zoom, tile_x, tile_y


def _seed_legacy_dataset_entry(store: DsmDatasetStore, dataset_id: str, *, name: str, file_path: Path) -> None:
    descriptor = _descriptor().model_copy(
        update={
            "id": dataset_id,
            "name": name,
            "fileSizeBytes": int(file_path.stat().st_size),
            "loadedAtIso": "2026-03-24T00:00:00Z",
        }
    )
    index = store._load_index()
    index["datasets"][dataset_id] = {
        "descriptor": descriptor.model_dump(mode="json"),
        "filePath": str(file_path),
        "pyramidPath": str(file_path.with_suffix(".npz")),
        "sourceCrs": descriptor.sourceCrsCode,
    }
    store._save_index(index)


def _seed_legacy_dataset_from_file(
    store: DsmDatasetStore,
    file_path: Path,
    *,
    original_name: str,
    source_descriptor: DsmSourceDescriptorModel | None = None,
) -> DsmSourceDescriptorModel:
    dataset_id = _sha256_hex(file_path.read_bytes())
    source_descriptor = source_descriptor or dsm_store_module.derive_descriptor_from_path(file_path, original_name)
    destination_path = store._dataset_file_path(dataset_id, Path(original_name).suffix or ".tiff")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.resolve() != destination_path.resolve():
        destination_path.write_bytes(file_path.read_bytes())
    normalized, valid_mask = dsm_store_module._read_raster_and_valid_mask_from_path(destination_path, source_descriptor)
    bounds = source_descriptor.sourceBounds
    pixel_size_x = (bounds.maxX - bounds.minX) / source_descriptor.width
    pixel_size_y = (bounds.maxY - bounds.minY) / source_descriptor.height
    levels = dsm_store_module._build_pyramid(normalized, valid_mask, pixel_size_x, pixel_size_y)
    pyramid_path = store._dataset_pyramid_path(dataset_id)
    dsm_store_module._serialize_pyramid(levels, pyramid_path)
    descriptor = store._build_descriptor(dataset_id, original_name, int(destination_path.stat().st_size), source_descriptor, valid_mask)
    index = store._load_index()
    index["datasets"][dataset_id] = {
        "descriptor": descriptor.model_dump(mode="json"),
        "filePath": str(destination_path),
        "pyramidPath": str(pyramid_path),
        "sourceCrs": dsm_store_module._resolve_source_crs(descriptor),
    }
    store._save_index(index)
    return descriptor


class _FakePaginator:
    def __init__(self, client: "_FakeS3Client"):
        self.client = client

    def paginate(self, Bucket: str, Prefix: str):  # noqa: N803
        contents = [{"Key": key} for (bucket, key), _ in sorted(self.client.objects.items()) if bucket == Bucket and key.startswith(Prefix)]
        yield {"Contents": contents}


class _FakeS3Client:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[tuple[str, str]] = []
        self.presigned_requests: list[dict] = []
        self.get_object_requests: list[tuple[str, str]] = []

    def put_object(self, *, Bucket: str, Key: str, Body, **kwargs):  # noqa: N803
        if isinstance(Body, bytes):
            payload = Body
        elif hasattr(Body, "read"):
            payload = Body.read()
        else:
            payload = bytes(Body)
        self.objects[(Bucket, Key)] = payload
        return {}

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
        self.get_object_requests.append((Bucket, Key))
        payload = self.objects.get((Bucket, Key))
        if payload is None:
            raise FileNotFoundError(Key)
        return {"Body": io.BytesIO(payload)}

    def delete_object(self, *, Bucket: str, Key: str):  # noqa: N803
        self.objects.pop((Bucket, Key), None)
        self.deleted.append((Bucket, Key))
        return {}

    def generate_presigned_url(self, ClientMethod: str, Params: dict, ExpiresIn: int, HttpMethod: str):  # noqa: N803
        self.presigned_requests.append(
            {
                "ClientMethod": ClientMethod,
                "Params": Params,
                "ExpiresIn": ExpiresIn,
                "HttpMethod": HttpMethod,
            }
        )
        return f"https://presigned-upload.test/{Params['Bucket']}/{Params['Key']}?expires={ExpiresIn}"

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _FakePaginator(self)


@contextmanager
def _override_app_dsm_state(*, dsm_dir: Path, store, staging_dir: Path):
    original_dir = app_module.DSM_DIR
    original_store = app_module.DSM_DATASET_STORE
    original_staging_dir = app_module.DSM_UPLOAD_STAGING_DIR
    app_module.DSM_DIR = dsm_dir
    app_module.DSM_DATASET_STORE = store
    app_module.DSM_UPLOAD_STAGING_DIR = staging_dir
    try:
        yield
    finally:
        app_module.DSM_DIR = original_dir
        app_module.DSM_DATASET_STORE = original_store
        app_module.DSM_UPLOAD_STAGING_DIR = original_staging_dir


class _FakeExactBridge:
    def __init__(
        self,
        optimize_response: dict | None = None,
        rerank_response: dict | None = None,
        *,
        evaluate_solution_responses: dict[str, dict] | None = None,
        evaluate_solution_errors: dict[str, Exception] | None = None,
        supports_candidate_fanout: bool = False,
        candidate_max_inflight: int = 1,
    ):
        self.optimize_response = optimize_response or {}
        self.rerank_response = rerank_response or {}
        self.evaluate_solution_responses = evaluate_solution_responses or {}
        self.evaluate_solution_errors = evaluate_solution_errors or {}
        self._supports_candidate_fanout = supports_candidate_fanout
        self._candidate_max_inflight = candidate_max_inflight
        self.optimize_requests: list[dict] = []
        self.evaluate_solution_requests: list[dict] = []
        self.rerank_requests: list[dict] = []
        self.begin_candidate_batch_calls = 0
        self.end_candidate_batch_calls = 0

    def supports_candidate_fanout(self) -> bool:
        return self._supports_candidate_fanout

    def candidate_max_inflight(self) -> int:
        return self._candidate_max_inflight

    def begin_candidate_batch(self):
        self.begin_candidate_batch_calls += 1
        return object()

    def end_candidate_batch(self, _batch_handle) -> None:
        self.end_candidate_batch_calls += 1

    def optimize_bearing(self, request: dict) -> dict:
        self.optimize_requests.append(request)
        return self.optimize_response

    def evaluate_solution(self, request: dict, *, batch_handle=None) -> dict:
        self.evaluate_solution_requests.append(request)
        signature = request["solution"]["signature"]
        if signature in self.evaluate_solution_errors:
            raise self.evaluate_solution_errors[signature]
        if signature in self.evaluate_solution_responses:
            return self.evaluate_solution_responses[signature]
        return {
            "solution": request["solution"],
            "preview": {
                "metricKind": "gsd",
                "stats": {"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0},
                "regionStats": [{"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0}],
                "regionCount": 1,
                "sampleCount": 1,
                "sampleLabel": "Images",
            },
        }

    def rerank_solutions(self, request: dict) -> dict:
        self.rerank_requests.append(request)
        return self.rerank_response


def _read_debug_artifacts(debug_dir: Path, request_id: str) -> dict[str, dict]:
    request_dir = debug_dir / request_id
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(request_dir.glob("*.json"))
    }


def _fake_exact_debug_trace(signature: str, *, exact_score: float = 4.5) -> dict:
    return {
        "signature": signature,
        "polygonId": "poly-1",
        "rankingSource": "backend-exact",
        "exactOptimizeZoom": 14,
        "timeWeight": 0.1,
        "qualityWeight": 0.9,
        "fastestMissionTimeSec": 120.0,
        "partitionScoreBreakdown": {
            "modelVersion": "camera-partition-v1",
            "total": exact_score,
            "signals": {"worstRegionQ90Cm": exact_score / 2.6},
            "weights": {"worstRegionQ90Cm": 2.6},
            "contributions": {"worstRegionQ90Cm": exact_score},
        },
        "preview": {
            "solution": {
                "signature": signature,
                "tradeoff": 0.5,
                "regionCount": 1,
                "totalMissionTimeSec": 120.0,
                "normalizedQualityCost": 0.2,
                "weightedMeanMismatchDeg": 0.0,
                "hierarchyLevel": 0,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 1.0,
                "isFirstPracticalSplit": signature == "surrogate-a",
                "regions": [],
            },
            "metricKind": "gsd",
            "stats": {
                "mean": 1.0,
                "q25": 1.0,
                "q75": 1.0,
                "q90": 1.0,
                "max": 1.0,
                "count": 1,
                "totalAreaM2": 1.0,
                "histogram": [],
            },
            "regionStats": [
                {
                    "mean": 1.0,
                    "q25": 1.0,
                    "q75": 1.0,
                    "q90": 1.0,
                    "max": 1.0,
                    "count": 1,
                    "totalAreaM2": 1.0,
                    "histogram": [],
                }
            ],
            "regionCount": 1,
            "sampleCount": 1,
            "sampleLabel": "Images",
        },
        "timings": {
            "totalElapsedMs": 12.0,
            "previewElapsedMs": 3.0,
            "regionSearchElapsedMs": [9.0],
        },
        "regions": [
            {
                "regionIndex": 0,
                "originalBearingDeg": 35.0,
                "seedBearingDeg": 35.0,
                "chosenBearingDeg": 90.0,
                "chosenExactCost": 1.5,
                "searchMode": "local",
                "halfWindowDeg": 30,
                "lineSpacingM": 50.0,
                "elapsedMs": 9.0,
                "evaluatedBearings": [
                    {
                        "bearingDeg": 90.0,
                        "exactCost": 1.5,
                        "qualityCost": 0.5,
                        "missionTimeSec": 120.0,
                        "normalizedTimeCost": 0.6666666667,
                        "metricKind": "gsd",
                        "stats": {
                            "mean": 1.0,
                            "q25": 1.0,
                            "q75": 1.0,
                            "q90": 1.0,
                            "max": 1.0,
                            "count": 1,
                            "totalAreaM2": 1.0,
                            "histogram": [],
                        },
                        "diagnostics": {
                            "qualityCost": 0.5,
                            "missionTimeSec": 120.0,
                            "normalizedTimeCost": 0.6666666667,
                            "targetGsdM": 0.02,
                            "overTargetAreaFraction": 0.0,
                            "q75": 0.02,
                            "q90": 0.03,
                        },
                        "qualityBreakdown": {
                            "modelVersion": "camera-region-v1",
                            "total": 0.5,
                            "signals": {"q90Overshoot": 0.27027027027},
                            "weights": {"q90Overshoot": 1.85},
                            "contributions": {"q90Overshoot": 0.5},
                        },
                        "costBreakdown": {
                            "modelVersion": "camera-region-v1",
                            "total": 1.5,
                            "signals": {"qualityCost": 0.5, "normalizedTimeCost": 0.6666666667},
                            "weights": {"qualityCost": 0.9, "normalizedTimeCost": 0.1},
                            "contributions": {"qualityCost": 1.43333333333, "normalizedTimeCost": 0.06666666667},
                        },
                        "missionBreakdown": {
                            "totalLengthM": 1440.0,
                            "speedMps": 12.0,
                            "lineCount": 4,
                            "sampleCount": 25,
                            "sampleLabel": "Images",
                        },
                    }
                ],
            }
        ],
    }


def _debug_grid_stub():
    return SimpleNamespace(grid_step_m=40.0, cells=[], edges=[])


def _debug_feature_field_stub():
    return SimpleNamespace(dominant_preferred_bearing_deg=0.0, cells=[])


def _encode_png_bytes(size: int, rgba_value: int = 128) -> bytes:
    rgba = np.full((size, size, 4), rgba_value, dtype=np.uint8)
    rgba[:, :, 3] = 255
    out = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(out, format="PNG")
    return out.getvalue()


def _encode_terrain_png_bytes(size: int) -> bytes:
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    for row in range(size):
        for col in range(size):
            elevation = 220.0 + row * 0.4 + col * 0.3
            encoded = max(0, min(16777215, round((elevation + 10000.0) * 10.0)))
            rgba[row, col, 0] = (encoded >> 16) & 255
            rgba[row, col, 1] = (encoded >> 8) & 255
            rgba[row, col, 2] = encoded & 255
            rgba[row, col, 3] = 255
    out = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(out, format="PNG")
    return out.getvalue()


class _TerrainBatchStubServer:
    def __init__(self, png_payload: bytes):
        self.png_payload = png_payload
        self.requests: list[dict] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None

    def __enter__(self):
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                if self.path != "/v1/internal/terrain-batch":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.requests.append(payload)
                tiles = []
                for tile in payload.get("tiles", []):
                    tiles.append(
                        {
                            "z": tile["z"],
                            "x": tile["x"],
                            "y": tile["y"],
                            "size": 64,
                            "pngBase64": b64encode(outer.png_payload).decode("ascii"),
                            "demPngBase64": None,
                            "demSize": None,
                            "demPadTiles": tile.get("padTiles"),
                        }
                    )
                body = json.dumps({"operation": "terrain-batch", "tiles": tiles}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):  # noqa: A003
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        host, port = self._server.server_address
        self.base_url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


class _LocalTerrainBatchServer:
    def __init__(self, store, png_payload: bytes):
        self.store = store
        self.png_payload = png_payload
        self.requests: list[dict] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None
        self._original_store = None
        self._original_cache = None
        self._original_token = None

    def __enter__(self):
        outer = self
        self._original_store = app_module.DSM_DATASET_STORE
        self._original_cache = app_module.TerrainTileCache
        self._original_token = app_module.mapbox_token
        app_module.DSM_DATASET_STORE = self.store

        class _FakeTerrainTileCache:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_or_fetch(self, *_args, **_kwargs):
                return outer.png_payload

        app_module.TerrainTileCache = _FakeTerrainTileCache
        app_module.mapbox_token = lambda: "test-token"

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                if self.path != "/v1/internal/terrain-batch":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                outer.requests.append(payload)
                response_model = app_module._build_terrain_batch_response(TerrainBatchRequestModel.model_validate(payload))
                body = response_model.model_dump_json().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):  # noqa: A003
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        host, port = self._server.server_address
        self.base_url = f"http://{host}:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._original_store is not None:
            app_module.DSM_DATASET_STORE = self._original_store
        if self._original_cache is not None:
            app_module.TerrainTileCache = self._original_cache
        if self._original_token is not None:
            app_module.mapbox_token = self._original_token


def _exact_request_payload() -> dict:
    return {
        "polygonId": "poly-1",
        "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0045], [0.0, 0.0]],
        "payloadKind": "camera",
        "params": {
            "payloadKind": "camera",
            "altitudeAGL": 110,
            "frontOverlap": 75,
            "sideOverlap": 70,
        },
        "terrainSource": {"mode": "mapbox"},
        "altitudeMode": "legacy",
        "minClearanceM": 0,
        "turnExtendM": 0,
        "seedBearingDeg": 17,
        "mode": "global",
        "halfWindowDeg": 90,
    }


def _route_paths(module) -> set[str]:
    return {route.path for route in module.app.routes}


def test_dataset_store_overlays_terrain_tile_with_explicit_source(tmp_path: Path) -> None:
    store = DsmDatasetStore(tmp_path)
    descriptor = _descriptor()
    payload = _write_test_tiff(tmp_path / "dataset.tiff", np.full((4, 4), 123.0, dtype=np.float32))
    stored_descriptor, reused = store.ingest_dataset(payload, "dataset.tiff", descriptor)
    assert reused is False

    tile = TerrainTile(
        z=0,
        x=0,
        y=0,
        elevation=np.zeros((4, 4), dtype=np.float32),
        min_x=0,
        min_y=0,
        max_x=4,
        max_y=4,
    )
    changed = store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=stored_descriptor.id),
        tile,
    )
    assert changed is True
    assert np.allclose(tile.elevation, 123.0, atol=1e-3)


def test_dataset_store_falls_back_to_existing_terrain_for_nan_pixels(tmp_path: Path) -> None:
    store = DsmDatasetStore(tmp_path)
    descriptor = _descriptor()
    raster = np.full((4, 4), 123.0, dtype=np.float32)
    raster[0, 0] = np.nan
    payload = _write_test_tiff(tmp_path / "dataset-nan.tiff", raster)
    stored_descriptor, _ = store.ingest_dataset(payload, "dataset-nan.tiff", descriptor)

    tile = TerrainTile(
        z=0,
        x=0,
        y=0,
        elevation=np.full((4, 4), 10.0, dtype=np.float32),
        min_x=0,
        min_y=0,
        max_x=4,
        max_y=4,
    )
    changed = store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=stored_descriptor.id),
        tile,
    )
    assert changed is True
    assert tile.elevation[0, 0] == 10.0
    assert np.any(np.isclose(tile.elevation, 123.0, atol=1e-3))
    assert np.all(np.isfinite(tile.elevation))


def test_dataset_store_falls_back_to_existing_terrain_for_numeric_nodata(tmp_path: Path) -> None:
    store = DsmDatasetStore(tmp_path)
    descriptor = _descriptor().model_copy(update={"noDataValue": -9999.0})
    raster = np.full((4, 4), 123.0, dtype=np.float32)
    raster[1, 1] = -9999.0
    payload = _write_test_tiff(tmp_path / "dataset-nodata.tiff", raster)
    stored_descriptor, _ = store.ingest_dataset(payload, "dataset-nodata.tiff", descriptor)

    tile = TerrainTile(
        z=0,
        x=0,
        y=0,
        elevation=np.full((4, 4), 10.0, dtype=np.float32),
        min_x=0,
        min_y=0,
        max_x=4,
        max_y=4,
    )
    changed = store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=stored_descriptor.id),
        tile,
    )
    assert changed is True
    assert tile.elevation[1, 1] == 10.0
    assert np.any(np.isclose(tile.elevation, 123.0, atol=1e-3))


def test_dataset_store_uses_alpha_mask_for_valid_coverage_and_sampling(tmp_path: Path) -> None:
    store = DsmDatasetStore(tmp_path)
    descriptor = _descriptor()
    raster = np.full((4, 4), 123.0, dtype=np.float32)
    alpha = np.full((4, 4), 255, dtype=np.uint8)
    alpha[0, 1] = 0
    payload = _write_alpha_masked_tiff(tmp_path / "dataset-alpha.tiff", raster, alpha)
    stored_descriptor, _ = store.ingest_dataset(payload, "dataset-alpha.tiff", descriptor)

    assert stored_descriptor.validCoverageRatio is not None
    assert stored_descriptor.validCoverageRatio < 1.0

    tile = TerrainTile(
        z=0,
        x=0,
        y=0,
        elevation=np.full((4, 4), 10.0, dtype=np.float32),
        min_x=0,
        min_y=0,
        max_x=4,
        max_y=4,
    )
    changed = store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=stored_descriptor.id),
        tile,
    )
    assert changed is True
    assert tile.elevation[0, 1] == 10.0
    assert np.any(np.isclose(tile.elevation, 123.0, atol=1e-3))


def test_dataset_store_reuses_existing_dataset_for_duplicate_upload(tmp_path: Path) -> None:
    store = DsmDatasetStore(tmp_path)
    descriptor = _descriptor()
    payload = _write_test_tiff(tmp_path / "dedupe.tiff", np.full((4, 4), 42.0, dtype=np.float32))
    first_descriptor, first_reused = store.ingest_dataset(payload, "dedupe-a.tiff", descriptor)
    second_descriptor, second_reused = store.ingest_dataset(payload, "dedupe-b.tiff", descriptor)

    assert first_reused is False
    assert second_reused is True
    assert first_descriptor.id == second_descriptor.id


def test_dataset_store_creates_new_dataset_for_same_filename_with_different_bytes(tmp_path: Path) -> None:
    store = DsmDatasetStore(tmp_path)
    descriptor = _descriptor()
    first_payload = _write_test_tiff(tmp_path / "same-name-a.tiff", np.full((4, 4), 42.0, dtype=np.float32))
    second_payload = _write_test_tiff(tmp_path / "same-name-b.tiff", np.full((4, 4), 84.0, dtype=np.float32))

    first_descriptor, first_reused = store.ingest_dataset(first_payload, "same-name.tiff", descriptor)
    second_descriptor, second_reused = store.ingest_dataset(second_payload, "same-name.tiff", descriptor)

    assert first_reused is False
    assert second_reused is False
    assert first_descriptor.id != second_descriptor.id


def test_local_dataset_store_persists_dataset_across_restart(tmp_path: Path) -> None:
    descriptor = _descriptor()
    payload = _write_test_tiff(tmp_path / "persist-local.tiff", np.full((4, 4), 55.0, dtype=np.float32))

    first_store = DsmDatasetStore(tmp_path / "local-store")
    stored_descriptor, reused = first_store.ingest_dataset(payload, "persist-local-a.tiff", descriptor)
    assert reused is False

    second_store = DsmDatasetStore(tmp_path / "local-store")
    listed = second_store.list_datasets()
    assert len(listed) == 1
    assert listed[0].id == stored_descriptor.id

    tile = TerrainTile(
        z=0,
        x=0,
        y=0,
        elevation=np.zeros((4, 4), dtype=np.float32),
        min_x=0,
        min_y=0,
        max_x=4,
        max_y=4,
    )
    changed = second_store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=stored_descriptor.id),
        tile,
    )
    assert changed is True
    assert np.allclose(tile.elevation, 55.0, atol=1e-3)


def test_sharded_local_dataset_matches_legacy_pyramid_for_tile_and_rgba_overlay(tmp_path: Path) -> None:
    raster = np.asarray(
        [
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
            [40.0, 41.0, 42.0, 43.0],
        ],
        dtype=np.float32,
    )
    payload_path = tmp_path / "parity-local.tiff"
    payload = _write_geotiff(payload_path, raster)

    sharded_store = DsmDatasetStore(tmp_path / "sharded-store")
    sharded_descriptor, reused = sharded_store.ingest_dataset(payload, "parity-local.tiff", _descriptor())
    assert reused is False

    legacy_store = DsmDatasetStore(tmp_path / "legacy-store")
    legacy_descriptor = _seed_legacy_dataset_from_file(
        legacy_store,
        payload_path,
        original_name="parity-local.tiff",
        source_descriptor=dsm_store_module.derive_descriptor_from_path(payload_path, "parity-local.tiff"),
    )

    sharded_tile = TerrainTile(z=0, x=0, y=0, elevation=np.zeros((4, 4), dtype=np.float32), min_x=0, min_y=0, max_x=4, max_y=4)
    legacy_tile = TerrainTile(z=0, x=0, y=0, elevation=np.zeros((4, 4), dtype=np.float32), min_x=0, min_y=0, max_x=4, max_y=4)

    assert sharded_store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=sharded_descriptor.id),
        sharded_tile,
    )
    assert legacy_store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=legacy_descriptor.id),
        legacy_tile,
    )
    assert np.allclose(sharded_tile.elevation, legacy_tile.elevation, atol=1e-6)

    sharded_rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    legacy_rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    zoom = 22
    tile_x = 2 ** (zoom - 1)
    tile_y = tile_x - 1
    assert sharded_store.apply_terrain_source_to_rgba_tile(
        TerrainSourceModel(mode="blended", datasetId=sharded_descriptor.id),
        zoom,
        tile_x,
        tile_y,
        sharded_rgba,
    )
    assert legacy_store.apply_terrain_source_to_rgba_tile(
        TerrainSourceModel(mode="blended", datasetId=legacy_descriptor.id),
        zoom,
        tile_x,
        tile_y,
        legacy_rgba,
    )
    assert np.array_equal(sharded_rgba, legacy_rgba)


def test_fetch_dem_for_ring_matches_legacy_pyramid_for_sharded_dataset(tmp_path: Path, monkeypatch) -> None:
    raster = np.arange(64, dtype=np.float32).reshape(8, 8)
    payload_path = tmp_path / "parity-geographic.tiff"
    _write_geographic_geotiff(payload_path, raster)
    source_descriptor = dsm_store_module.derive_descriptor_from_path(payload_path, "parity-geographic.tiff")

    sharded_store = DsmDatasetStore(tmp_path / "sharded-store")
    sharded_descriptor, _ = sharded_store.ingest_dataset_file(payload_path, "parity-geographic.tiff", source_descriptor=source_descriptor)

    legacy_store = DsmDatasetStore(tmp_path / "legacy-store")
    legacy_descriptor = _seed_legacy_dataset_from_file(
        legacy_store,
        payload_path,
        original_name="parity-geographic.tiff",
        source_descriptor=source_descriptor,
    )

    png_payload = _encode_terrain_png_bytes(64)

    class _FakeTerrainTileCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_fetch(self, *_args, **_kwargs):
            return png_payload

    monkeypatch.setattr(dsm_store_module, "mapbox_token", lambda: "test-token", raising=False)
    monkeypatch.setattr("terrain_splitter.mapbox_tiles.TerrainTileCache", _FakeTerrainTileCache)
    monkeypatch.setattr("terrain_splitter.mapbox_tiles.mapbox_token", lambda: "test-token")

    ring = [(7.01, 46.99), (7.03, 46.99), (7.03, 46.97), (7.01, 46.97), (7.01, 46.99)]
    sharded_dem, sharded_zoom = fetch_dem_for_ring(
        ring,
        tmp_path / "cache-sharded",
        grid_step_m=40,
        terrain_source=TerrainSourceModel(mode="blended", datasetId=sharded_descriptor.id),
        dsm_store=sharded_store,
    )
    legacy_dem, legacy_zoom = fetch_dem_for_ring(
        ring,
        tmp_path / "cache-legacy",
        grid_step_m=40,
        terrain_source=TerrainSourceModel(mode="blended", datasetId=legacy_descriptor.id),
        dsm_store=legacy_store,
    )

    assert sharded_zoom == legacy_zoom
    assert sharded_dem.tiles.keys() == legacy_dem.tiles.keys()
    for key in sharded_dem.tiles:
        assert np.allclose(sharded_dem.tiles[key].elevation, legacy_dem.tiles[key].elevation, atol=1e-6)


def test_internal_terrain_batch_matches_legacy_pyramid_for_sharded_dataset(tmp_path: Path, monkeypatch) -> None:
    raster = np.arange(64, dtype=np.float32).reshape(8, 8)
    payload_path = tmp_path / "parity-batch.tiff"
    _write_geotiff(payload_path, raster)
    source_descriptor = dsm_store_module.derive_descriptor_from_path(payload_path, "parity-batch.tiff")

    sharded_store = DsmDatasetStore(tmp_path / "sharded-store")
    sharded_descriptor, _ = sharded_store.ingest_dataset_file(payload_path, "parity-batch.tiff", source_descriptor=source_descriptor)

    legacy_store = DsmDatasetStore(tmp_path / "legacy-store")
    legacy_descriptor = _seed_legacy_dataset_from_file(
        legacy_store,
        payload_path,
        original_name="parity-batch.tiff",
        source_descriptor=source_descriptor,
    )

    png_payload = _encode_terrain_png_bytes(64)

    class _FakeTerrainTileCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_fetch(self, *_args, **_kwargs):
            return png_payload

    monkeypatch.setattr(app_module, "TerrainTileCache", _FakeTerrainTileCache)
    monkeypatch.setattr(app_module, "mapbox_token", lambda: "test-token")

    request = TerrainBatchRequestModel.model_validate(
        {
            "terrainSource": {"mode": "blended", "datasetId": sharded_descriptor.id},
            "tiles": [{"z": 0, "x": 0, "y": 0, "padTiles": 1}],
        }
    )
    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets-a", store=sharded_store, staging_dir=tmp_path / "staging-a"):
        sharded_response = app_module._build_terrain_batch_response(request).model_dump(mode="json")
    request = TerrainBatchRequestModel.model_validate(
        {
            "terrainSource": {"mode": "blended", "datasetId": legacy_descriptor.id},
            "tiles": [{"z": 0, "x": 0, "y": 0, "padTiles": 1}],
        }
    )
    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets-b", store=legacy_store, staging_dir=tmp_path / "staging-b"):
        legacy_response = app_module._build_terrain_batch_response(request).model_dump(mode="json")
    assert sharded_response == legacy_response


def test_s3_backed_store_reuses_existing_dataset_across_cold_start(tmp_path: Path) -> None:
    fake_s3 = _FakeS3Client()
    descriptor = _descriptor()
    payload = _write_test_tiff(tmp_path / "s3-dedupe.tiff", np.full((4, 4), 42.0, dtype=np.float32))

    first_store = S3BackedDsmDatasetStore(tmp_path / "cache-a", bucket="bucket", prefix="stage", client=fake_s3)
    first_descriptor, first_reused = first_store.ingest_dataset(payload, "s3-dedupe-a.tiff", descriptor)

    second_store = S3BackedDsmDatasetStore(tmp_path / "cache-b", bucket="bucket", prefix="stage", client=fake_s3)
    second_descriptor, second_reused = second_store.ingest_dataset(payload, "s3-dedupe-b.tiff", descriptor)

    assert first_reused is False
    assert second_reused is True
    assert first_descriptor.id == second_descriptor.id
    assert len(second_store.list_datasets()) == 1


def test_s3_backed_store_loads_remote_dataset_for_tile_sampling(tmp_path: Path) -> None:
    fake_s3 = _FakeS3Client()
    descriptor = _descriptor()
    payload = _write_test_tiff(tmp_path / "s3-remote.tiff", np.full((4, 4), 88.0, dtype=np.float32))

    upload_store = S3BackedDsmDatasetStore(tmp_path / "cache-upload", bucket="bucket", prefix="stage", client=fake_s3)
    stored_descriptor, _ = upload_store.ingest_dataset(payload, "s3-remote.tiff", descriptor)

    cold_store = S3BackedDsmDatasetStore(tmp_path / "cache-cold", bucket="bucket", prefix="stage", client=fake_s3)
    tile = TerrainTile(
        z=0,
        x=0,
        y=0,
        elevation=np.zeros((4, 4), dtype=np.float32),
        min_x=0,
        min_y=0,
        max_x=4,
        max_y=4,
    )
    changed = cold_store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=stored_descriptor.id),
        tile,
    )
    assert changed is True
    assert np.allclose(tile.elevation, 88.0, atol=1e-3)


def test_create_dsm_dataset_store_selects_s3_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TERRAIN_SPLITTER_DSM_STORE_MODE", "s3")
    monkeypatch.setenv("TERRAIN_SPLITTER_DSM_S3_BUCKET", "bucket")
    monkeypatch.setenv("TERRAIN_SPLITTER_DSM_S3_PREFIX", "stage")
    store = create_dsm_dataset_store(tmp_path)
    assert isinstance(store, S3BackedDsmDatasetStore)


def test_prepare_upload_returns_existing_dataset_immediately(tmp_path: Path) -> None:
    payload = _write_geotiff(tmp_path / "existing-upload.tiff", np.full((4, 4), 42.0, dtype=np.float32))
    store = DsmDatasetStore(tmp_path / "datasets")
    staging_dir = tmp_path / "staging"
    descriptor = _descriptor()
    stored_descriptor, reused_existing = store.ingest_dataset(payload, "existing-upload.tiff", descriptor)
    assert reused_existing is False

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/dsm/prepare-upload",
                json={
                    "sha256": _sha256_hex(payload),
                    "fileSizeBytes": len(payload),
                    "originalName": "existing-upload.tiff",
                    "contentType": "image/tiff",
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "existing"
    assert body["dataset"]["datasetId"] == stored_descriptor.id
    assert body["dataset"]["reusedExisting"] is True


def test_local_prepare_upload_put_finalize_flow(tmp_path: Path) -> None:
    payload = _write_geotiff(tmp_path / "local-flow.tiff", np.full((4, 4), 37.0, dtype=np.float32))
    store = DsmDatasetStore(tmp_path / "datasets")
    staging_dir = tmp_path / "staging"

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            prepare = client.post(
                "/v1/dsm/prepare-upload",
                json={
                    "sha256": _sha256_hex(payload),
                    "fileSizeBytes": len(payload),
                    "originalName": "local-flow.tiff",
                    "contentType": "image/tiff",
                },
            )
            assert prepare.status_code == 200
            prepared = prepare.json()
            assert prepared["status"] == "upload-required"
            upload_id = prepared["uploadId"]
            upload_path = f"/v1/dsm/upload-sessions/{upload_id}"

            upload = client.put(upload_path, content=payload, headers={"Content-Type": "image/tiff"})
            assert upload.status_code == 204

            duplicate_upload = client.put(upload_path, content=payload, headers={"Content-Type": "image/tiff"})
            assert duplicate_upload.status_code == 409

            finalize = client.post("/v1/dsm/finalize-upload", json={"uploadId": upload_id})

    assert finalize.status_code == 200
    body = finalize.json()
    assert body["processingStatus"] == "ready"
    assert body["reusedExisting"] is False
    assert body["descriptor"]["id"] == _sha256_hex(payload)
    assert not (staging_dir / upload_id).exists()


def test_local_finalize_rejects_hash_mismatch_and_cleans_up(tmp_path: Path) -> None:
    payload = _write_geotiff(tmp_path / "bad-hash.tiff", np.full((4, 4), 19.0, dtype=np.float32))
    store = DsmDatasetStore(tmp_path / "datasets")
    staging_dir = tmp_path / "staging"

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            prepare = client.post(
                "/v1/dsm/prepare-upload",
                json={
                    "sha256": "0" * 64,
                    "fileSizeBytes": len(payload),
                    "originalName": "bad-hash.tiff",
                    "contentType": "image/tiff",
                },
            )
            upload_id = prepare.json()["uploadId"]
            upload = client.put(f"/v1/dsm/upload-sessions/{upload_id}", content=payload, headers={"Content-Type": "image/tiff"})
            assert upload.status_code == 204
            finalize = client.post("/v1/dsm/finalize-upload", json={"uploadId": upload_id})

    assert finalize.status_code == 400
    assert "hash" in finalize.text.lower()
    assert not (staging_dir / upload_id).exists()


def test_local_finalize_rejects_multiband_rgba_geotiff(tmp_path: Path) -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 120
    rgba[..., 1] = 80
    rgba[..., 2] = 40
    rgba[..., 3] = 255
    payload = _write_rgba_geotiff(tmp_path / "ortho-rgba.tiff", rgba)
    store = DsmDatasetStore(tmp_path / "datasets")
    staging_dir = tmp_path / "staging"

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            prepare = client.post(
                "/v1/dsm/prepare-upload",
                json={
                    "sha256": _sha256_hex(payload),
                    "fileSizeBytes": len(payload),
                    "originalName": "ortho-rgba.tiff",
                    "contentType": "image/tiff",
                },
            )
            upload_id = prepare.json()["uploadId"]
            upload = client.put(f"/v1/dsm/upload-sessions/{upload_id}", content=payload, headers={"Content-Type": "image/tiff"})
            assert upload.status_code == 204
            finalize = client.post("/v1/dsm/finalize-upload", json={"uploadId": upload_id})

    assert finalize.status_code == 400
    assert "single-band elevation geotiff" in finalize.text.lower()
    assert "rgb/rgba" in finalize.text.lower()
    assert not (staging_dir / upload_id).exists()
    assert store.get_dataset_descriptor(_sha256_hex(payload)) is None


def test_invalid_legacy_rgba_dataset_is_not_returned_by_detail_endpoint(tmp_path: Path) -> None:
    store = DsmDatasetStore(tmp_path / "datasets")
    staging_dir = tmp_path / "staging"
    dataset_id = "legacy-invalid-rgba"
    payload_path = tmp_path / "legacy-invalid-rgba.tiff"
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 120
    rgba[..., 1] = 80
    rgba[..., 2] = 40
    rgba[..., 3] = 255
    _write_rgba_geotiff(payload_path, rgba)
    _seed_legacy_dataset_entry(store, dataset_id, name="legacy-invalid-rgba.tiff", file_path=payload_path)

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            detail = client.get(f"/v1/dsm/datasets/{dataset_id}")

    assert detail.status_code == 404
    assert store.get_dataset_descriptor(dataset_id) is None


def test_invalid_legacy_rgba_dataset_does_not_modify_terrain_tiles(tmp_path: Path, monkeypatch) -> None:
    png_payload = _encode_png_bytes(4, 64)

    class _FakeTerrainTileCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_fetch(self, *_args, **_kwargs):
            return png_payload

    monkeypatch.setattr(app_module, "TerrainTileCache", _FakeTerrainTileCache)
    monkeypatch.setattr(app_module, "mapbox_token", lambda: "test-token")

    store = DsmDatasetStore(tmp_path / "datasets")
    staging_dir = tmp_path / "staging"
    dataset_id = "legacy-invalid-rgba"
    payload_path = tmp_path / "legacy-invalid-rgba.tiff"
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 120
    rgba[..., 1] = 80
    rgba[..., 2] = 40
    rgba[..., 3] = 255
    _write_rgba_geotiff(payload_path, rgba)
    _seed_legacy_dataset_entry(store, dataset_id, name="legacy-invalid-rgba.tiff", file_path=payload_path)

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            response = client.get(f"/v1/terrain-rgb/0/0/0.png?mode=blended&datasetId={dataset_id}")

    assert response.status_code == 200
    assert response.content == png_payload
    assert store.get_dataset_descriptor(dataset_id) is None


def test_local_finalize_rejects_missing_staged_payload_and_cleans_up(tmp_path: Path) -> None:
    payload = _write_geotiff(tmp_path / "missing-payload.tiff", np.full((4, 4), 17.0, dtype=np.float32))
    store = DsmDatasetStore(tmp_path / "datasets")
    staging_dir = tmp_path / "staging"

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            prepare = client.post(
                "/v1/dsm/prepare-upload",
                json={
                    "sha256": _sha256_hex(payload),
                    "fileSizeBytes": len(payload),
                    "originalName": "missing-payload.tiff",
                    "contentType": "image/tiff",
                },
            )
            upload_id = prepare.json()["uploadId"]
            finalize = client.post("/v1/dsm/finalize-upload", json={"uploadId": upload_id})

    assert finalize.status_code == 400
    assert "not found" in finalize.text.lower()
    assert not (staging_dir / upload_id).exists()


def test_local_finalize_rejects_expired_upload_session(tmp_path: Path) -> None:
    payload = _write_geotiff(tmp_path / "expired.tiff", np.full((4, 4), 23.0, dtype=np.float32))
    store = DsmDatasetStore(tmp_path / "datasets")
    staging_dir = tmp_path / "staging"

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            prepare = client.post(
                "/v1/dsm/prepare-upload",
                json={
                    "sha256": _sha256_hex(payload),
                    "fileSizeBytes": len(payload),
                    "originalName": "expired.tiff",
                    "contentType": "image/tiff",
                },
            )
            upload_id = prepare.json()["uploadId"]
            upload = client.put(f"/v1/dsm/upload-sessions/{upload_id}", content=payload, headers={"Content-Type": "image/tiff"})
            assert upload.status_code == 204

            manifest_path = staging_dir / upload_id / "session.json"
            session_payload = json.loads(manifest_path.read_text())
            session_payload["expiresAtIso"] = "2000-01-01T00:00:00Z"
            manifest_path.write_text(json.dumps(session_payload))

            finalize = client.post("/v1/dsm/finalize-upload", json={"uploadId": upload_id})

    assert finalize.status_code == 410
    assert not (staging_dir / upload_id).exists()


def test_s3_prepare_upload_returns_presigned_target(tmp_path: Path, monkeypatch) -> None:
    payload = _write_geotiff(tmp_path / "s3-prepare.tiff", np.full((4, 4), 42.0, dtype=np.float32))
    fake_s3 = _FakeS3Client()
    store = S3BackedDsmDatasetStore(tmp_path / "s3-cache", bucket="bucket", prefix="stage", client=fake_s3)
    staging_dir = tmp_path / "staging"
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "terrain-splitter")

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/dsm/prepare-upload",
                json={
                    "sha256": _sha256_hex(payload),
                    "fileSizeBytes": len(payload),
                    "originalName": "s3-prepare.tiff",
                    "contentType": "image/tiff",
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "upload-required"
    assert body["uploadId"]
    assert body["uploadTarget"]["method"] == "PUT"
    assert body["uploadTarget"]["url"].startswith("https://presigned-upload.test/")
    assert fake_s3.presigned_requests
    request = fake_s3.presigned_requests[0]
    assert request["ClientMethod"] == "put_object"
    assert request["Params"]["Bucket"] == "bucket"
    assert f"stage/uploads/{body['uploadId']}/" in request["Params"]["Key"]
    assert ("bucket", f"stage/uploads/{body['uploadId']}/session.json") in fake_s3.objects


def test_s3_finalize_upload_ingests_staged_object_and_cleans_up(tmp_path: Path, monkeypatch) -> None:
    payload = _write_geotiff(tmp_path / "s3-finalize.tiff", np.full((4, 4), 52.0, dtype=np.float32))
    fake_s3 = _FakeS3Client()
    store = S3BackedDsmDatasetStore(tmp_path / "s3-cache", bucket="bucket", prefix="stage", client=fake_s3)
    staging_dir = tmp_path / "staging"
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "terrain-splitter")

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets", store=store, staging_dir=staging_dir):
        with TestClient(app_module.app) as client:
            prepare = client.post(
                "/v1/dsm/prepare-upload",
                json={
                    "sha256": _sha256_hex(payload),
                    "fileSizeBytes": len(payload),
                    "originalName": "s3-finalize.tiff",
                    "contentType": "image/tiff",
                },
            )
            prepared = prepare.json()
            upload_id = prepared["uploadId"]
            staged_key = fake_s3.presigned_requests[-1]["Params"]["Key"]
            fake_s3.objects[("bucket", staged_key)] = payload

            finalize = client.post("/v1/dsm/finalize-upload", json={"uploadId": upload_id})

    assert finalize.status_code == 200
    body = finalize.json()
    dataset_id = body["datasetId"]
    assert dataset_id == _sha256_hex(payload)
    assert body["reusedExisting"] is False
    assert ("bucket", staged_key) not in fake_s3.objects
    assert ("bucket", f"stage/uploads/{upload_id}/session.json") not in fake_s3.objects
    remote_keys = {key for (bucket, key) in fake_s3.objects if bucket == "bucket"}
    assert f"stage/datasets/{dataset_id}/descriptor.json" in remote_keys
    assert f"stage/datasets/{dataset_id}/analysis/manifest.json" in remote_keys
    assert any(key == f"stage/datasets/{dataset_id}/analysis/level-0.npz" for key in remote_keys)
    assert f"stage/datasets/{dataset_id}/pyramid.npz" not in remote_keys


def test_dsm_upload_and_dataset_detail_endpoint(tmp_path: Path) -> None:
    payload = _write_geotiff(tmp_path / "upload.tiff", np.full((4, 4), 42.0, dtype=np.float32))
    original_dir = app_module.DSM_DIR
    original_store = app_module.DSM_DATASET_STORE
    app_module.DSM_DIR = tmp_path
    app_module.DSM_DATASET_STORE = DsmDatasetStore(tmp_path)

    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/dsm/upload",
                files={"file": ("upload.tiff", payload, "image/tiff")},
            )
            assert response.status_code == 200
            upload_payload = response.json()
            assert upload_payload["datasetId"]
            assert upload_payload["descriptor"]["id"] == upload_payload["datasetId"]
            assert upload_payload["processingStatus"] == "ready"
            assert upload_payload["terrainTileUrlTemplate"].endswith(
                f"/v1/terrain-rgb/{{z}}/{{x}}/{{y}}.png?mode=blended&datasetId={upload_payload['datasetId']}"
            )

            detail = client.get(f"/v1/dsm/datasets/{upload_payload['datasetId']}")
            assert detail.status_code == 200
            assert detail.json()["datasetId"] == upload_payload["datasetId"]
    finally:
        app_module.DSM_DIR = original_dir
        app_module.DSM_DATASET_STORE = original_store


def test_dsm_upload_and_dataset_detail_endpoint_with_s3_store(tmp_path: Path) -> None:
    payload = _write_geotiff(tmp_path / "upload-s3.tiff", np.full((4, 4), 42.0, dtype=np.float32))
    fake_s3 = _FakeS3Client()
    original_dir = app_module.DSM_DIR
    original_store = app_module.DSM_DATASET_STORE
    app_module.DSM_DIR = tmp_path
    app_module.DSM_DATASET_STORE = S3BackedDsmDatasetStore(tmp_path / "s3-cache", bucket="bucket", prefix="stage", client=fake_s3)

    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/dsm/upload",
                files={"file": ("upload-s3.tiff", payload, "image/tiff")},
            )
            assert response.status_code == 200
            upload_payload = response.json()
            dataset_id = upload_payload["datasetId"]
            assert dataset_id
            assert upload_payload["descriptor"]["id"] == dataset_id

            detail = client.get(f"/v1/dsm/datasets/{dataset_id}")
            assert detail.status_code == 200
            assert detail.json()["descriptor"]["id"] == dataset_id

            remote_keys = {key for (bucket, key) in fake_s3.objects if bucket == "bucket"}
            assert any(key.endswith("/descriptor.json") for key in remote_keys)
            assert any(key.endswith("/analysis/manifest.json") for key in remote_keys)
            assert any("/analysis/level-" in key and key.endswith(".npz") for key in remote_keys)
            assert not any(key.endswith("/pyramid.npz") for key in remote_keys)
    finally:
        app_module.DSM_DIR = original_dir
        app_module.DSM_DATASET_STORE = original_store


def test_s3_descriptor_lookup_does_not_download_source_or_analysis_artifacts(tmp_path: Path) -> None:
    payload = _write_geotiff(tmp_path / "s3-descriptor-only.tiff", np.full((4, 4), 22.0, dtype=np.float32))
    fake_s3 = _FakeS3Client()
    upload_store = S3BackedDsmDatasetStore(tmp_path / "cache-upload", bucket="bucket", prefix="stage", client=fake_s3)
    stored_descriptor, _ = upload_store.ingest_dataset(payload, "s3-descriptor-only.tiff", _descriptor())

    cold_store = S3BackedDsmDatasetStore(tmp_path / "cache-cold", bucket="bucket", prefix="stage", client=fake_s3)
    fake_s3.get_object_requests.clear()
    descriptor = cold_store.get_dataset_descriptor(stored_descriptor.id)

    assert descriptor is not None
    requested_keys = [key for (bucket, key) in fake_s3.get_object_requests if bucket == "bucket"]
    assert requested_keys == [f"stage/datasets/{stored_descriptor.id}/descriptor.json"]


def test_s3_cold_tile_sampling_downloads_only_selected_analysis_level(tmp_path: Path) -> None:
    raster = np.arange(600 * 600, dtype=np.float32).reshape(600, 600)
    payload = _write_geotiff(tmp_path / "s3-level-select.tiff", raster)
    descriptor = dsm_store_module.derive_descriptor_from_payload(payload, "s3-level-select.tiff")
    fake_s3 = _FakeS3Client()
    upload_store = S3BackedDsmDatasetStore(tmp_path / "cache-upload", bucket="bucket", prefix="stage", client=fake_s3)
    stored_descriptor, _ = upload_store.ingest_dataset(payload, "s3-level-select.tiff", descriptor)

    cold_store = S3BackedDsmDatasetStore(tmp_path / "cache-cold", bucket="bucket", prefix="stage", client=fake_s3)
    fake_s3.get_object_requests.clear()
    tile = TerrainTile(
        z=0,
        x=0,
        y=0,
        elevation=np.zeros((64, 64), dtype=np.float32),
        min_x=0,
        min_y=0,
        max_x=600,
        max_y=600,
    )
    changed = cold_store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=stored_descriptor.id),
        tile,
    )
    assert changed is True
    requested_keys = [key for (bucket, key) in fake_s3.get_object_requests if bucket == "bucket"]
    assert f"stage/datasets/{stored_descriptor.id}/{stored_descriptor.id}.tiff" not in requested_keys
    assert f"stage/datasets/{stored_descriptor.id}/analysis/manifest.json" in requested_keys
    level_keys = [key for key in requested_keys if f"stage/datasets/{stored_descriptor.id}/analysis/level-" in key]
    assert len(level_keys) == 1


def test_s3_legacy_remote_dataset_still_serves_descriptor_tile_and_terrain_batch(tmp_path: Path, monkeypatch) -> None:
    fixture_path = _real_dsm_fixture_path()
    local_legacy_store = DsmDatasetStore(tmp_path / "legacy-local")
    descriptor = _seed_legacy_dataset_from_file(
        local_legacy_store,
        fixture_path,
        original_name=fixture_path.name,
    )
    dataset_id = descriptor.id
    local_entry = local_legacy_store._load_index()["datasets"][dataset_id]
    fake_s3 = _FakeS3Client()
    fake_s3.objects[("bucket", f"stage/datasets/{dataset_id}/descriptor.json")] = json.dumps(
        local_entry["descriptor"]
    ).encode("utf-8")
    fake_s3.objects[("bucket", f"stage/datasets/{dataset_id}/pyramid.npz")] = Path(local_entry["pyramidPath"]).read_bytes()
    fake_s3.objects[("bucket", f"stage/datasets/{dataset_id}/{dataset_id}{fixture_path.suffix}") ] = Path(local_entry["filePath"]).read_bytes()

    cold_store = S3BackedDsmDatasetStore(tmp_path / "legacy-cold", bucket="bucket", prefix="stage", client=fake_s3)
    descriptor_out = cold_store.get_dataset_descriptor(dataset_id)
    assert descriptor_out is not None
    assert descriptor_out.id == dataset_id

    tile = TerrainTile(
        z=14,
        x=4288,
        y=2870,
        elevation=np.zeros((64, 64), dtype=np.float32),
        min_x=descriptor.footprint3857.minX,
        min_y=descriptor.footprint3857.minY,
        max_x=descriptor.footprint3857.maxX,
        max_y=descriptor.footprint3857.maxY,
    )
    changed = cold_store.apply_terrain_source_to_tile(
        TerrainSourceModel(mode="blended", datasetId=dataset_id),
        tile,
    )
    assert changed is True
    assert np.isfinite(tile.elevation).all()

    png_payload = _encode_terrain_png_bytes(64)
    with _LocalTerrainBatchServer(cold_store, png_payload):
        request = TerrainBatchRequestModel.model_validate(
            {
                "terrainSource": {"mode": "blended", "datasetId": dataset_id},
                "tiles": [{"z": 14, "x": 4288, "y": 2870, "padTiles": 1}],
            }
        )
        response = app_module._build_terrain_batch_response(request)
    assert response.tiles
    assert response.tiles[0].pngBase64
    assert response.tiles[0].demPngBase64


def test_public_terrain_rgb_route_matches_between_sharded_and_legacy_dsm(tmp_path: Path, monkeypatch) -> None:
    fixture_path = _real_dsm_fixture_path()
    source_descriptor = dsm_store_module.derive_descriptor_from_path(fixture_path, fixture_path.name)
    sharded_store = DsmDatasetStore(tmp_path / "sharded-store")
    sharded_descriptor, _ = sharded_store.ingest_dataset_file(fixture_path, fixture_path.name, source_descriptor=source_descriptor)
    legacy_store = DsmDatasetStore(tmp_path / "legacy-store")
    legacy_descriptor = _seed_legacy_dataset_from_file(
        legacy_store,
        fixture_path,
        original_name=fixture_path.name,
        source_descriptor=source_descriptor,
    )
    z, x, y = _tile_ref_for_descriptor(source_descriptor, zoom=14)
    png_payload = _encode_terrain_png_bytes(64)

    class _FakeTerrainTileCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_fetch(self, *_args, **_kwargs):
            return png_payload

    monkeypatch.setattr(app_module, "TerrainTileCache", _FakeTerrainTileCache)
    monkeypatch.setattr(app_module, "mapbox_token", lambda: "test-token")

    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets-a", store=sharded_store, staging_dir=tmp_path / "staging-a"):
        with TestClient(app_module.app) as client:
            sharded_response = client.get(f"/v1/terrain-rgb/{z}/{x}/{y}.png?mode=blended&datasetId={sharded_descriptor.id}")
    with _override_app_dsm_state(dsm_dir=tmp_path / "datasets-b", store=legacy_store, staging_dir=tmp_path / "staging-b"):
        with TestClient(app_module.app) as client:
            legacy_response = client.get(f"/v1/terrain-rgb/{z}/{x}/{y}.png?mode=blended&datasetId={legacy_descriptor.id}")

    assert sharded_response.status_code == 200
    assert legacy_response.status_code == 200
    assert sharded_response.content == legacy_response.content


def test_dsm_upload_rejects_ungeoreferenced_tiff(tmp_path: Path) -> None:
    raster = np.full((4, 4), 42.0, dtype=np.float32)
    payload_path = tmp_path / "plain.tiff"
    tifffile.imwrite(payload_path, raster, compression="lzw")
    payload = payload_path.read_bytes()
    original_dir = app_module.DSM_DIR
    original_store = app_module.DSM_DATASET_STORE
    app_module.DSM_DIR = tmp_path
    app_module.DSM_DATASET_STORE = DsmDatasetStore(tmp_path)

    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/dsm/upload",
                files={"file": ("plain.tiff", payload, "image/tiff")},
            )
            assert response.status_code == 400
            assert "georeferenced" in response.text.lower() or "geotiff" in response.text.lower()
    finally:
        app_module.DSM_DIR = original_dir
        app_module.DSM_DATASET_STORE = original_store


def test_dsm_upload_accepts_geographic_geotiff(tmp_path: Path) -> None:
    payload = _write_geographic_geotiff(tmp_path / "geographic.tiff", np.full((4, 4), 7.0, dtype=np.float32))
    original_dir = app_module.DSM_DIR
    original_store = app_module.DSM_DATASET_STORE
    app_module.DSM_DIR = tmp_path
    app_module.DSM_DATASET_STORE = DsmDatasetStore(tmp_path)

    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/dsm/upload",
                files={"file": ("geographic.tiff", payload, "image/tiff")},
            )
            assert response.status_code == 200
            descriptor = response.json()["descriptor"]
            assert descriptor["sourceCrsCode"] == "EPSG:4326"
            assert descriptor["footprintLngLat"]["minLng"] < descriptor["footprintLngLat"]["maxLng"]
            assert descriptor["footprintLngLat"]["minLat"] < descriptor["footprintLngLat"]["maxLat"]
    finally:
        app_module.DSM_DIR = original_dir
        app_module.DSM_DATASET_STORE = original_store


def test_dsm_upload_accepts_model_transformation_geotiff(tmp_path: Path) -> None:
    payload = _write_transform_geotiff(tmp_path / "transform.tiff", np.full((4, 4), 9.0, dtype=np.float32))
    original_dir = app_module.DSM_DIR
    original_store = app_module.DSM_DATASET_STORE
    app_module.DSM_DIR = tmp_path
    app_module.DSM_DATASET_STORE = DsmDatasetStore(tmp_path)

    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/dsm/upload",
                files={"file": ("transform.tiff", payload, "image/tiff")},
            )
            assert response.status_code == 200
            descriptor = response.json()["descriptor"]
            assert descriptor["sourceBounds"]["minX"] == 100.0
            assert descriptor["sourceBounds"]["maxX"] == 108.0
            assert descriptor["sourceBounds"]["minY"] == 188.0
            assert descriptor["sourceBounds"]["maxY"] == 200.0
    finally:
        app_module.DSM_DIR = original_dir
        app_module.DSM_DATASET_STORE = original_store


def test_internal_terrain_batch_returns_png_payloads(monkeypatch) -> None:
    png_payload = _encode_png_bytes(4, 64)

    class _FakeTerrainTileCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_fetch(self, *_args, **_kwargs):
            return png_payload

    monkeypatch.setattr(app_module, "TerrainTileCache", _FakeTerrainTileCache)
    monkeypatch.setattr(app_module, "mapbox_token", lambda: "test-token")
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    try:
        app_module.EXACT_RUNTIME_BRIDGE = None
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/internal/terrain-batch",
                json={
                    "terrainSource": {"mode": "mapbox"},
                    "tiles": [{"z": 0, "x": 0, "y": 0, "padTiles": 1}],
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["operation"] == "terrain-batch"
            assert len(payload["tiles"]) == 1
            tile = payload["tiles"][0]
            assert tile["pngBase64"]
            assert tile["demPngBase64"]
            assert tile["demSize"] == 12
            assert tile["demPadTiles"] == 1
    finally:
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_internal_terrain_batch_http_route_only_mounts_in_local_mode(monkeypatch) -> None:
    try:
        with monkeypatch.context() as local_ctx:
            local_ctx.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
            local_ctx.delenv("TERRAIN_SPLITTER_ENABLE_INTERNAL_HTTP", raising=False)
            local_ctx.setenv("TERRAIN_SPLITTER_DISABLE_EXACT_POSTPROCESS", "true")
            local_module = importlib.reload(app_module)
            assert "/v1/internal/terrain-batch" in _route_paths(local_module)

        with monkeypatch.context() as lambda_ctx:
            lambda_ctx.setenv("AWS_LAMBDA_FUNCTION_NAME", "terrain-splitter")
            lambda_ctx.delenv("TERRAIN_SPLITTER_ENABLE_INTERNAL_HTTP", raising=False)
            lambda_ctx.setenv("TERRAIN_SPLITTER_DISABLE_EXACT_POSTPROCESS", "true")
            lambda_module = importlib.reload(app_module)
            assert "/v1/internal/terrain-batch" not in _route_paths(lambda_module)
            with TestClient(lambda_module.app) as client:
                response = client.post(
                    "/v1/internal/terrain-batch",
                    json={"terrainSource": {"mode": "mapbox"}, "tiles": [{"z": 0, "x": 0, "y": 0}]},
                )
                assert response.status_code == 404
    finally:
        monkeypatch.delenv("AWS_LAMBDA_FUNCTION_NAME", raising=False)
        monkeypatch.delenv("TERRAIN_SPLITTER_ENABLE_INTERNAL_HTTP", raising=False)
        monkeypatch.delenv("TERRAIN_SPLITTER_DISABLE_EXACT_POSTPROCESS", raising=False)
        importlib.reload(app_module)


def test_runtime_dir_defaults_to_repo_root_and_migrates_legacy_local_state(tmp_path: Path, monkeypatch) -> None:
    legacy_base_dir = tmp_path / "backend" / "terrain_splitter"
    legacy_base_dir.mkdir(parents=True)
    legacy_dir = legacy_base_dir / ".dsm"
    legacy_dir.mkdir()
    legacy_file = legacy_dir / "index.json"
    legacy_file.write_text('{"datasets":{}}', encoding="utf-8")

    monkeypatch.delenv("TERRAIN_SPLITTER_DSM_DIR", raising=False)
    monkeypatch.setattr(app_module, "BASE_DIR", legacy_base_dir)
    monkeypatch.setattr(app_module, "LOCAL_RUNTIME_ROOT", tmp_path / ".terrain-splitter-runtime")

    migrated_dir = app_module._runtime_dir("TERRAIN_SPLITTER_DSM_DIR", ".dsm")

    assert migrated_dir == tmp_path / ".terrain-splitter-runtime" / "dsm"
    assert (migrated_dir / "index.json").read_text(encoding="utf-8") == '{"datasets":{}}'
    assert not legacy_dir.exists()


def test_lambda_internal_terrain_batch_event_still_works(monkeypatch) -> None:
    captured = {}

    def _fake_build_response(request):
        captured["request"] = request
        return TerrainBatchResponseModel.model_validate(
            {
                "operation": "terrain-batch",
                "tiles": [
                    {
                        "z": 0,
                        "x": 0,
                        "y": 0,
                        "size": 4,
                        "pngBase64": "Zm9v",
                        "demPngBase64": None,
                        "demSize": None,
                        "demPadTiles": None,
                    }
                ],
            }
        )

    monkeypatch.setattr(main_module, "_build_terrain_batch_response", _fake_build_response)
    response = main_module.lambda_handler(
        {
            "terrainSplitterInternal": "terrain-batch",
            "payload": {
                "terrainSource": {"mode": "mapbox"},
                "tiles": [{"z": 0, "x": 0, "y": 0}],
            },
        },
        None,
    )
    assert response["operation"] == "terrain-batch"
    assert captured["request"].terrainSource.mode == "mapbox"
    assert captured["request"].tiles[0].z == 0


def test_exact_optimize_endpoint_uses_bridge() -> None:
    bridge = _FakeExactBridge(
        optimize_response={
            "best": {
                "bearingDeg": 27.5,
                "exactCost": 1.2,
                "qualityCost": 0.8,
                "missionTimeSec": 90.0,
                "normalizedTimeCost": 0.5,
                "metricKind": "gsd",
                "diagnostics": {"q90": 0.03},
            },
            "seedBearingDeg": 10.0,
            "lineSpacingM": 34.0,
        }
    )
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    app_module.EXACT_RUNTIME_BRIDGE = bridge
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/exact/optimize-bearing",
                json={
                    "polygonId": "poly-1",
                    "ring": [[7.0, 47.0], [7.01, 47.0], [7.01, 47.01], [7.0, 47.01], [7.0, 47.0]],
                    "payloadKind": "camera",
                    "params": {
                        "payloadKind": "camera",
                        "altitudeAGL": 100,
                        "frontOverlap": 75,
                        "sideOverlap": 70,
                    },
                    "terrainSource": {"mode": "mapbox"},
                    "seedBearingDeg": 10,
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["bearingDeg"] == 27.5
            assert payload["metricKind"] == "gsd"
            assert payload["diagnostics"]["q90"] == 0.03
            assert bridge.optimize_requests and bridge.optimize_requests[0]["polygonId"] == "poly-1"
    finally:
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_exact_optimize_endpoint_matches_local_exact_runtime(monkeypatch) -> None:
    png_payload = _encode_terrain_png_bytes(64)
    repo_root = Path(__file__).resolve().parents[3]
    with _TerrainBatchStubServer(png_payload) as terrain_server:
        monkeypatch.setenv("TERRAIN_SPLITTER_INTERNAL_BASE_URL", terrain_server.base_url)
        bridge = LocalExactRuntimeSidecarBridge(repo_root)
        original_bridge = app_module.EXACT_RUNTIME_BRIDGE
        app_module.EXACT_RUNTIME_BRIDGE = bridge
        request_payload = _exact_request_payload()
        try:
            expected = bridge.optimize_bearing(request_payload)
            with TestClient(app_module.app) as client:
                response = client.post("/v1/exact/optimize-bearing", json=request_payload)
            assert response.status_code == 200
            payload = response.json()
            assert payload["bearingDeg"] == expected["best"]["bearingDeg"]
            assert payload["exactScore"] == expected["best"]["exactCost"]
            assert payload["qualityCost"] == expected["best"]["qualityCost"]
            assert payload["missionTimeSec"] == expected["best"]["missionTimeSec"]
            assert payload["metricKind"] == expected["best"]["metricKind"]
            assert payload["seedBearingDeg"] == expected["seedBearingDeg"]
            assert terrain_server.requests, "exact runtime should request terrain batches through the stub server"
        finally:
            bridge.close()
            app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_exact_optimize_endpoint_matches_between_sharded_and_legacy_dsm(monkeypatch, tmp_path: Path) -> None:
    fixture_path = _real_dsm_fixture_path()
    source_descriptor = dsm_store_module.derive_descriptor_from_path(fixture_path, fixture_path.name)
    sharded_store = DsmDatasetStore(tmp_path / "sharded-store")
    sharded_descriptor, _ = sharded_store.ingest_dataset_file(fixture_path, fixture_path.name, source_descriptor=source_descriptor)
    legacy_store = DsmDatasetStore(tmp_path / "legacy-store")
    legacy_descriptor = _seed_legacy_dataset_from_file(
        legacy_store,
        fixture_path,
        original_name=fixture_path.name,
        source_descriptor=source_descriptor,
    )
    ring = _ring_inside_descriptor(source_descriptor, inset_fraction=0.25)
    request_template = {
        "polygonId": "fixture-poly",
        "ring": ring,
        "payloadKind": "camera",
        "params": {
            "payloadKind": "camera",
            "altitudeAGL": 110,
            "frontOverlap": 75,
            "sideOverlap": 70,
        },
        "altitudeMode": "legacy",
        "minClearanceM": 0,
        "turnExtendM": 0,
        "seedBearingDeg": 17,
    }
    png_payload = _encode_terrain_png_bytes(64)
    repo_root = Path(__file__).resolve().parents[3]

    def _run(store, descriptor):
        with _LocalTerrainBatchServer(store, png_payload) as terrain_server:
            monkeypatch.setenv("TERRAIN_SPLITTER_INTERNAL_BASE_URL", terrain_server.base_url)
            bridge = LocalExactRuntimeSidecarBridge(repo_root)
            original_bridge = app_module.EXACT_RUNTIME_BRIDGE
            app_module.EXACT_RUNTIME_BRIDGE = bridge
            try:
                with TestClient(app_module.app) as client:
                    response = client.post(
                        "/v1/exact/optimize-bearing",
                        json={**request_template, "terrainSource": {"mode": "blended", "datasetId": descriptor.id}},
                    )
                assert response.status_code == 200
                assert terrain_server.requests
                return response.json()
            finally:
                bridge.close()
                app_module.EXACT_RUNTIME_BRIDGE = original_bridge

    sharded_response = _run(sharded_store, sharded_descriptor)
    legacy_response = _run(legacy_store, legacy_descriptor)
    assert sharded_response["bearingDeg"] == legacy_response["bearingDeg"]
    assert sharded_response["exactScore"] == legacy_response["exactScore"]
    assert sharded_response["qualityCost"] == legacy_response["qualityCost"]
    assert sharded_response["missionTimeSec"] == legacy_response["missionTimeSec"]
    assert sharded_response["metricKind"] == legacy_response["metricKind"]
    assert sharded_response["seedBearingDeg"] == legacy_response["seedBearingDeg"]


def test_partition_solve_returns_backend_exact_reranked_solutions(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4), dtype=np.float32), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())

    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.5,
                "regionCount": 2,
                "totalMissionTimeSec": 120.0,
                "normalizedQualityCost": 0.4,
                "weightedMeanMismatchDeg": 4.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 0.6,
                "meanConvexity": 0.9,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 20.0, "atomCount": 2, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "convexity": 1.0, "compactness": 0.8},
                    {"areaM2": 12.0, "bearingDeg": 25.0, "atomCount": 2, "ring": [[1, 0], [2, 0], [2, 1], [1, 0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 2,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 0.6,
                "meanConvexity": 0.9,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "convexity": 1.0, "compactness": 0.8},
                    {"areaM2": 12.0, "bearingDeg": 45.0, "atomCount": 2, "ring": [[1, 0], [2, 0], [2, 1], [1, 0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)
    original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
    app_module.EXACT_POSTPROCESS_TOP_K = 1

    bridge = _FakeExactBridge(
        rerank_response={
            "solutions": [
                {
                    **surrogate_solutions[1].model_dump(mode="json"),
                    "rankingSource": "backend-exact",
                    "exactScore": 0.2,
                    "exactMetricKind": "density",
                    "regions": [
                        {**surrogate_solutions[1].regions[0].model_dump(mode="json"), "bearingDeg": 41.0, "exactScore": 0.1, "exactSeedBearingDeg": 40.0},
                        {**surrogate_solutions[1].regions[1].model_dump(mode="json"), "bearingDeg": 46.0, "exactScore": 0.1, "exactSeedBearingDeg": 45.0},
                    ],
                }
            ]
        }
    )
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    app_module.EXACT_RUNTIME_BRIDGE = bridge
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[7.0, 47.0], [7.01, 47.0], [7.01, 47.01], [7.0, 47.01], [7.0, 47.0]],
                    "payloadKind": "lidar",
                    "params": {
                        "payloadKind": "lidar",
                        "altitudeAGL": 120,
                        "frontOverlap": 0,
                        "sideOverlap": 40,
                    },
                    "terrainSource": {"mode": "mapbox"},
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert len(payload["solutions"]) == 1
            assert payload["solutions"][0]["rankingSource"] == "backend-exact"
            assert payload["solutions"][0]["regions"][0]["bearingDeg"] == 41.0
            assert bridge.rerank_requests and bridge.rerank_requests[0]["polygonId"] == "poly-1"
            assert len(bridge.rerank_requests[0]["solutions"]) == 1
    finally:
        app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_partition_solve_preserves_surrogate_only_behavior_when_exact_disabled(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4), dtype=np.float32), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())

    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.5,
                "regionCount": 2,
                "totalMissionTimeSec": 120.0,
                "normalizedQualityCost": 0.4,
                "weightedMeanMismatchDeg": 4.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 0.6,
                "meanConvexity": 0.9,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 20.0, "atomCount": 2, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "convexity": 1.0, "compactness": 0.8},
                    {"areaM2": 12.0, "bearingDeg": 25.0, "atomCount": 2, "ring": [[1, 0], [2, 0], [2, 1], [1, 0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 2,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 0.6,
                "meanConvexity": 0.9,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "convexity": 1.0, "compactness": 0.8},
                    {"areaM2": 12.0, "bearingDeg": 45.0, "atomCount": 2, "ring": [[1, 0], [2, 0], [2, 1], [1, 0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)

    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    app_module.EXACT_RUNTIME_BRIDGE = None
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[7.0, 47.0], [7.01, 47.0], [7.01, 47.01], [7.0, 47.01], [7.0, 47.0]],
                    "payloadKind": "camera",
                    "params": {
                        "payloadKind": "camera",
                        "altitudeAGL": 120,
                        "frontOverlap": 75,
                        "sideOverlap": 70,
                    },
                    "terrainSource": {"mode": "mapbox"},
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert [solution["signature"] for solution in payload["solutions"]] == ["surrogate-a", "surrogate-b"]
        assert payload["solutions"][0]["regions"][0]["bearingDeg"] == 20.0
        assert payload["solutions"][1]["regions"][0]["bearingDeg"] == 40.0
    finally:
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_partition_solve_debug_false_writes_no_debug_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4), dtype=np.float32), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        app_module,
        "solve_partition_hierarchy",
        lambda *_args, **_kwargs: [
            PartitionSolutionPreviewModel.model_validate(
                {
                    "signature": "surrogate-a",
                    "tradeoff": 0.5,
                    "regionCount": 1,
                    "totalMissionTimeSec": 120.0,
                    "normalizedQualityCost": 0.4,
                    "weightedMeanMismatchDeg": 4.0,
                    "hierarchyLevel": 1,
                    "largestRegionFraction": 1.0,
                    "meanConvexity": 1.0,
                    "boundaryBreakAlignment": 0.7,
                    "isFirstPracticalSplit": True,
                    "regions": [
                        {"areaM2": 10.0, "bearingDeg": 20.0, "atomCount": 2, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "convexity": 1.0, "compactness": 0.8},
                    ],
                }
            ),
            PartitionSolutionPreviewModel.model_validate(
                {
                    "signature": "surrogate-b",
                    "tradeoff": 0.6,
                    "regionCount": 1,
                    "totalMissionTimeSec": 130.0,
                    "normalizedQualityCost": 0.5,
                    "weightedMeanMismatchDeg": 5.0,
                    "hierarchyLevel": 1,
                    "largestRegionFraction": 1.0,
                    "meanConvexity": 1.0,
                    "boundaryBreakAlignment": 0.7,
                    "isFirstPracticalSplit": False,
                    "regions": [
                        {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "convexity": 1.0, "compactness": 0.8},
                    ],
                }
            ),
        ],
    )
    debug_dir = tmp_path / "debug-artifacts"
    debug_dir.mkdir()
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    original_debug_dir = app_module.DEBUG_DIR
    app_module.EXACT_RUNTIME_BRIDGE = None
    app_module.DEBUG_DIR = debug_dir
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[7.0, 47.0], [7.01, 47.0], [7.01, 47.01], [7.0, 47.01], [7.0, 47.0]],
                    "payloadKind": "camera",
                    "params": {"payloadKind": "camera", "altitudeAGL": 120, "frontOverlap": 75, "sideOverlap": 70},
                    "terrainSource": {"mode": "mapbox"},
                },
            )
        assert response.status_code == 200
        assert list(debug_dir.iterdir()) == []
    finally:
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge
        app_module.DEBUG_DIR = original_debug_dir


def test_partition_solve_debug_true_writes_skipped_exact_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4), dtype=np.float32), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: _debug_grid_stub())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: _debug_feature_field_stub())
    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.5,
                "regionCount": 1,
                "totalMissionTimeSec": 120.0,
                "normalizedQualityCost": 0.4,
                "weightedMeanMismatchDeg": 4.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 20.0, "atomCount": 2, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0, 0], [1, 0], [1, 1], [0, 0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)
    debug_dir = tmp_path / "debug-artifacts"
    debug_dir.mkdir()
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    original_debug_dir = app_module.DEBUG_DIR
    app_module.EXACT_RUNTIME_BRIDGE = None
    app_module.DEBUG_DIR = debug_dir
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[7.0, 47.0], [7.01, 47.0], [7.01, 47.01], [7.0, 47.01], [7.0, 47.0]],
                    "payloadKind": "camera",
                    "params": {"payloadKind": "camera", "altitudeAGL": 120, "frontOverlap": 75, "sideOverlap": 70},
                    "terrainSource": {"mode": "mapbox"},
                    "debug": True,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        artifacts = _read_debug_artifacts(debug_dir, payload["requestId"])
        assert "exact_rerank_summary" in artifacts
        summary = artifacts["exact_rerank_summary"]
        assert summary["exactStatus"] == "skipped"
        assert summary["skipReason"] == "no-exact-bridge"
        assert [candidate["notEvaluatedReason"] for candidate in summary["candidateSolutions"]] == [
            "no-exact-bridge",
            "no-exact-bridge",
        ]
    finally:
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge
        app_module.DEBUG_DIR = original_debug_dir


def test_partition_solve_debug_true_writes_exact_rerank_candidate_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4), dtype=np.float32), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: _debug_grid_stub())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: _debug_feature_field_stub())
    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.4,
                "regionCount": 1,
                "totalMissionTimeSec": 125.0,
                "normalizedQualityCost": 0.3,
                "weightedMeanMismatchDeg": 3.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.8,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 35.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-c",
                "tradeoff": 0.8,
                "regionCount": 1,
                "totalMissionTimeSec": 150.0,
                "normalizedQualityCost": 0.7,
                "weightedMeanMismatchDeg": 7.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.6,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 55.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)
    bridge = _FakeExactBridge(
        evaluate_solution_responses={
            "surrogate-a": {
                "solution": {
                    **surrogate_solutions[0].model_dump(mode="json"),
                    "rankingSource": "backend-exact",
                    "exactScore": 9.2,
                    "exactQualityCost": 1.1,
                    "exactMissionTimeSec": 125.0,
                    "exactMetricKind": "gsd",
                    "regions": [
                        {
                            **surrogate_solutions[0].regions[0].model_dump(mode="json"),
                            "bearingDeg": 90.0,
                            "exactScore": 3.2,
                            "exactSeedBearingDeg": 35.0,
                        }
                    ],
                },
                "preview": {
                    "metricKind": "gsd",
                    "stats": {"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0},
                    "regionStats": [{"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0}],
                    "regionCount": 1,
                    "sampleCount": 1,
                    "sampleLabel": "Images",
                },
                "debugTrace": _fake_exact_debug_trace("surrogate-a", exact_score=9.2),
            },
            "surrogate-b": {
                "solution": {
                    **surrogate_solutions[1].model_dump(mode="json"),
                    "rankingSource": "backend-exact",
                    "exactScore": 4.5,
                    "exactQualityCost": 0.7,
                    "exactMissionTimeSec": 130.0,
                    "exactMetricKind": "gsd",
                    "regions": [
                        {
                            **surrogate_solutions[1].regions[0].model_dump(mode="json"),
                            "bearingDeg": 12.0,
                            "exactScore": 2.0,
                            "exactSeedBearingDeg": 40.0,
                        }
                    ],
                },
                "preview": {
                    "metricKind": "gsd",
                    "stats": {"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0},
                    "regionStats": [{"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0}],
                    "regionCount": 1,
                    "sampleCount": 1,
                    "sampleLabel": "Images",
                },
                "debugTrace": _fake_exact_debug_trace("surrogate-b", exact_score=4.5),
            },
        },
        supports_candidate_fanout=True,
        candidate_max_inflight=3,
    )
    debug_dir = tmp_path / "debug-artifacts"
    debug_dir.mkdir()
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
    original_debug_dir = app_module.DEBUG_DIR
    app_module.EXACT_RUNTIME_BRIDGE = bridge
    app_module.EXACT_POSTPROCESS_TOP_K = 2
    app_module.DEBUG_DIR = debug_dir
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0045], [0.0, 0.0]],
                    "payloadKind": "camera",
                    "params": {"payloadKind": "camera", "altitudeAGL": 110, "frontOverlap": 75, "sideOverlap": 70},
                    "terrainSource": {"mode": "mapbox"},
                    "altitudeMode": "legacy",
                    "minClearanceM": 0,
                    "turnExtendM": 0,
                    "debug": True,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert [solution["signature"] for solution in payload["solutions"]] == ["surrogate-b", "surrogate-a"]
        artifacts = _read_debug_artifacts(debug_dir, payload["requestId"])
        summary = artifacts["exact_rerank_summary"]
        assert summary["exactStatus"] == "executed"
        assert summary["rerankMode"] == "fanout"
        assert [item["signature"] for item in summary["exactRankingOrder"]] == ["surrogate-b", "surrogate-a"]
        assert summary["candidateSolutions"][2]["signature"] == "surrogate-c"
        assert summary["candidateSolutions"][2]["notEvaluatedReason"] == "outside-top-k"
        assert summary["performanceSummary"]["candidateCount"] == 2
        assert summary["performanceSummary"]["topCandidatesByBridgeElapsedMs"]
        assert "exact_candidate_1_surrogate-a" in artifacts
        assert "exact_candidate_2_surrogate-b" in artifacts
        assert "exact_candidate_3_surrogate-c" not in artifacts
        candidate_artifact = artifacts["exact_candidate_1_surrogate-a"]
        assert candidate_artifact["surrogateSolution"]["signature"] == "surrogate-a"
        assert candidate_artifact["partitionScoreBreakdown"]["modelVersion"] == "camera-partition-v1"
        assert candidate_artifact["performanceSummary"]["regionSearchTotalMs"] == 9.0
        assert candidate_artifact["performanceSummary"]["longestRegion"]["regionIndex"] == 0
        assert candidate_artifact["regions"][0]["evaluatedBearings"][0]["qualityBreakdown"]["modelVersion"] == "camera-region-v1"
        assert artifacts["timing"]["performanceSummary"]["requestPhases"]["topStages"]
        assert artifacts["timing"]["performanceSummary"]["exactRerank"]["candidateCount"] == 2
        assert payload["solutions"][0]["debug"]["artifactPaths"]
    finally:
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge
        app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
        app_module.DEBUG_DIR = original_debug_dir


def test_build_solver_performance_summary_includes_surrogate_breakdown() -> None:
    summary = app_module._build_solver_performance_summary(
        {
            "solverSummary": {
                "performance": {
                    "objective_ms": 100.0,
                    "flight_time_ms": 25.0,
                    "line_lift_ms": 15.0,
                    "node_cost_ms": 35.0,
                    "shape_metric_ms": 5.0,
                    "build_region_ms": 220.0,
                    "split_candidate_enumeration_ms": 40.0,
                    "recursive_subsolve_ms": 120.0,
                    "plan_combine_ms": 18.0,
                    "frontier_prune_ms": 6.0,
                    "exact_geometry_reeval_ms": 60.0,
                    "exact_geometry_region_reeval_ms": 48.0,
                    "exact_geometry_reconstruct_ms": 12.0,
                    "exact_region_objective_ms": 44.0,
                    "exact_geometry_plan_count": 5.0,
                    "root_parallel_ms": 0.0,
                    "nested_parallel_ms": 0.0,
                    "root_parallel_workers_used": 0.0,
                    "nested_parallel_workers_used_max": 0.0,
                    "region_cache_hits": 4.0,
                    "region_cache_misses": 6.0,
                    "region_static_hits": 2.0,
                    "region_static_misses": 8.0,
                    "region_bearing_hits": 3.0,
                    "region_bearing_misses": 9.0,
                    "build_region_calls": 10.0,
                    "objective_calls": 12.0,
                    "split_attempts": 20.0,
                    "split_candidates_returned": 4.0,
                    "frontier_plan_count": 7.0,
                }
            }
        },
        solve_ms=400.0,
    )

    assert summary is not None
    assert summary["flightTimeAnalysis"]["shareOfObjectiveMs"] == 0.25
    assert summary["flightTimeAnalysis"]["shareOfSolveMs"] == 0.0625
    assert summary["objectiveComponentBreakdown"]["topStages"][0]["stage"] == "nodeCostMs"
    assert any(stage["stage"] == "splitCandidateEnumerationMs" for stage in summary["searchStageBreakdown"]["topStages"])
    assert summary["exactGeometryBreakdown"]["planCount"] == 5
    assert summary["exactGeometryBreakdown"]["topStages"][0]["stage"] == "regionReevalMs"


def test_partition_solve_debug_true_records_exact_failure_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4), dtype=np.float32), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: _debug_grid_stub())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: _debug_feature_field_stub())
    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.4,
                "regionCount": 1,
                "totalMissionTimeSec": 125.0,
                "normalizedQualityCost": 0.3,
                "weightedMeanMismatchDeg": 3.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.8,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 35.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)
    bridge = _FakeExactBridge(
        evaluate_solution_responses={
            "surrogate-a": {
                "solution": {
                    **surrogate_solutions[0].model_dump(mode="json"),
                    "rankingSource": "backend-exact",
                    "exactScore": 9.2,
                    "exactQualityCost": 1.1,
                    "exactMissionTimeSec": 125.0,
                    "exactMetricKind": "gsd",
                },
                "preview": {
                    "metricKind": "gsd",
                    "stats": {"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0},
                    "regionStats": [{"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0}],
                    "regionCount": 1,
                    "sampleCount": 1,
                    "sampleLabel": "Images",
                },
                "debugTrace": _fake_exact_debug_trace("surrogate-a", exact_score=9.2),
            },
        },
        evaluate_solution_errors={"surrogate-b": RuntimeError("boom")},
        supports_candidate_fanout=True,
        candidate_max_inflight=3,
    )
    debug_dir = tmp_path / "debug-artifacts"
    debug_dir.mkdir()
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
    original_debug_dir = app_module.DEBUG_DIR
    app_module.EXACT_RUNTIME_BRIDGE = bridge
    app_module.EXACT_POSTPROCESS_TOP_K = 2
    app_module.DEBUG_DIR = debug_dir
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0045], [0.0, 0.0]],
                    "payloadKind": "camera",
                    "params": {"payloadKind": "camera", "altitudeAGL": 110, "frontOverlap": 75, "sideOverlap": 70},
                    "terrainSource": {"mode": "mapbox"},
                    "altitudeMode": "legacy",
                    "minClearanceM": 0,
                    "turnExtendM": 0,
                    "debug": True,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert [solution["signature"] for solution in payload["solutions"]] == ["surrogate-a", "surrogate-b"]
        artifacts = _read_debug_artifacts(debug_dir, payload["requestId"])
        summary = artifacts["exact_rerank_summary"]
        assert summary["exactStatus"] == "failed"
        assert "boom" in summary["error"]
        candidate_map = {candidate["signature"]: candidate for candidate in summary["candidateSolutions"]}
        assert candidate_map["surrogate-b"]["notEvaluatedReason"] == "candidate-failed"
        assert candidate_map["surrogate-b"]["candidateError"] == "boom"
        assert "exact_candidate_1_surrogate-a" in artifacts
        assert "exact_candidate_2_surrogate-b" not in artifacts
    finally:
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge
        app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
        app_module.DEBUG_DIR = original_debug_dir


def test_partition_solve_exact_rerank_matches_local_exact_runtime(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4), dtype=np.float32), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())

    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.5,
                "regionCount": 1,
                "totalMissionTimeSec": 120.0,
                "normalizedQualityCost": 0.4,
                "weightedMeanMismatchDeg": 4.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 90.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 0.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)

    png_payload = _encode_terrain_png_bytes(64)
    repo_root = Path(__file__).resolve().parents[3]
    with _TerrainBatchStubServer(png_payload) as terrain_server:
        monkeypatch.setenv("TERRAIN_SPLITTER_INTERNAL_BASE_URL", terrain_server.base_url)
        bridge = LocalExactRuntimeSidecarBridge(repo_root)
        original_bridge = app_module.EXACT_RUNTIME_BRIDGE
        original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
        app_module.EXACT_RUNTIME_BRIDGE = bridge
        app_module.EXACT_POSTPROCESS_TOP_K = 2
        try:
            request_json = {
                "polygonId": "poly-1",
                "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0045], [0.0, 0.0]],
                "payloadKind": "camera",
                "params": {
                    "payloadKind": "camera",
                    "altitudeAGL": 110,
                    "frontOverlap": 75,
                    "sideOverlap": 70,
                },
                "terrainSource": {"mode": "mapbox"},
                "altitudeMode": "legacy",
                "minClearanceM": 0,
                "turnExtendM": 0,
            }
            expected = bridge.rerank_solutions(
                {
                    "polygonId": "poly-1",
                    "payloadKind": "camera",
                    "terrainSource": {"mode": "mapbox"},
                    "params": request_json["params"],
                    "ring": request_json["ring"],
                    "altitudeMode": "legacy",
                    "minClearanceM": 0,
                    "turnExtendM": 0,
                    "solutions": [solution.model_dump(mode="json") for solution in surrogate_solutions],
                    "rankingSource": "backend-exact",
                }
            )
            with TestClient(app_module.app) as client:
                response = client.post("/v1/partition/solve", json=request_json)
            assert response.status_code == 200
            payload = response.json()
            assert [solution["signature"] for solution in payload["solutions"]] == [
                solution["signature"] for solution in expected["solutions"]
            ]
            assert payload["solutions"][0]["rankingSource"] == "backend-exact"
            assert payload["solutions"][0]["regions"][0]["bearingDeg"] == expected["solutions"][0]["regions"][0]["bearingDeg"]
            assert payload["solutions"][1]["regions"][0]["bearingDeg"] == expected["solutions"][1]["regions"][0]["bearingDeg"]
            assert payload["solutions"][0]["exactScore"] == expected["solutions"][0]["exactScore"]
            assert payload["solutions"][1]["exactScore"] == expected["solutions"][1]["exactScore"]
            assert terrain_server.requests, "exact rerank should fetch terrain through the stub server"
        finally:
            bridge.close()
            app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
            app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_partition_solve_endpoint_uses_lambda_exact_candidate_fanout(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4)), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())
    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.4,
                "regionCount": 1,
                "totalMissionTimeSec": 125.0,
                "normalizedQualityCost": 0.3,
                "weightedMeanMismatchDeg": 3.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.8,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 35.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)
    preview_payload = {
        "metricKind": "gsd",
        "stats": {"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0},
        "regionStats": [{"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0}],
        "regionCount": 1,
        "sampleCount": 1,
        "sampleLabel": "Images",
    }
    bridge = _FakeExactBridge(
        evaluate_solution_responses={
            "surrogate-a": {
                "solution": {
                    **surrogate_solutions[0].model_dump(mode="json"),
                    "rankingSource": "backend-exact",
                    "exactScore": 9.2,
                    "exactQualityCost": 1.1,
                    "exactMissionTimeSec": 125.0,
                    "exactMetricKind": "gsd",
                    "regions": [
                        {
                            **surrogate_solutions[0].regions[0].model_dump(mode="json"),
                            "bearingDeg": 90.0,
                            "exactScore": 3.2,
                            "exactSeedBearingDeg": 35.0,
                        }
                    ],
                },
                "preview": preview_payload,
            },
            "surrogate-b": {
                "solution": {
                    **surrogate_solutions[1].model_dump(mode="json"),
                    "rankingSource": "backend-exact",
                    "exactScore": 4.5,
                    "exactQualityCost": 0.7,
                    "exactMissionTimeSec": 130.0,
                    "exactMetricKind": "gsd",
                    "regions": [
                        {
                            **surrogate_solutions[1].regions[0].model_dump(mode="json"),
                            "bearingDeg": 12.0,
                            "exactScore": 2.0,
                            "exactSeedBearingDeg": 40.0,
                        }
                    ],
                },
                "preview": preview_payload,
            },
        },
        supports_candidate_fanout=True,
        candidate_max_inflight=3,
    )
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
    app_module.EXACT_RUNTIME_BRIDGE = bridge
    app_module.EXACT_POSTPROCESS_TOP_K = 2
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0045], [0.0, 0.0]],
                    "payloadKind": "camera",
                    "params": {"payloadKind": "camera", "altitudeAGL": 110, "frontOverlap": 75, "sideOverlap": 70},
                    "terrainSource": {"mode": "mapbox"},
                    "altitudeMode": "legacy",
                    "minClearanceM": 0,
                    "turnExtendM": 0,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert [solution["signature"] for solution in payload["solutions"]] == ["surrogate-b", "surrogate-a"]
        assert [solution["rankingSource"] for solution in payload["solutions"]] == ["backend-exact", "backend-exact"]
        assert bridge.rerank_requests == []
        assert [request["solution"]["signature"] for request in bridge.evaluate_solution_requests] == ["surrogate-a", "surrogate-b"]
        assert bridge.begin_candidate_batch_calls == 1
        assert bridge.end_candidate_batch_calls == 1
    finally:
        app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_partition_solve_endpoint_fanout_failure_falls_back_to_surrogate(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4)), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())
    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.4,
                "regionCount": 1,
                "totalMissionTimeSec": 125.0,
                "normalizedQualityCost": 0.3,
                "weightedMeanMismatchDeg": 3.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.8,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 35.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)
    bridge = _FakeExactBridge(
        evaluate_solution_errors={"surrogate-b": RuntimeError("boom")},
        supports_candidate_fanout=True,
        candidate_max_inflight=3,
    )
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
    app_module.EXACT_RUNTIME_BRIDGE = bridge
    app_module.EXACT_POSTPROCESS_TOP_K = 2
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0045], [0.0, 0.0]],
                    "payloadKind": "camera",
                    "params": {"payloadKind": "camera", "altitudeAGL": 110, "frontOverlap": 75, "sideOverlap": 70},
                    "terrainSource": {"mode": "mapbox"},
                    "altitudeMode": "legacy",
                    "minClearanceM": 0,
                    "turnExtendM": 0,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert [solution["signature"] for solution in payload["solutions"]] == ["surrogate-a", "surrogate-b"]
        assert [solution["rankingSource"] for solution in payload["solutions"]] == [None, None]
        assert bridge.rerank_requests == []
        assert len(bridge.evaluate_solution_requests) == 2
        assert bridge.begin_candidate_batch_calls == 1
        assert bridge.end_candidate_batch_calls == 1
    finally:
        app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_partition_solve_endpoint_fanout_preserves_surrogate_order_on_tie(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "fetch_dem_for_ring", lambda *_args, **_kwargs: (np.zeros((4, 4)), 14))
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())
    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.4,
                "regionCount": 1,
                "totalMissionTimeSec": 125.0,
                "normalizedQualityCost": 0.3,
                "weightedMeanMismatchDeg": 3.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.8,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 35.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 40.0, "atomCount": 2, "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0]], "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)
    preview_payload = {
        "metricKind": "gsd",
        "stats": {"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0},
        "regionStats": [{"mean": 1.0, "q25": 1.0, "q75": 1.0, "q90": 1.0, "max": 1.0, "count": 1, "totalAreaM2": 1.0}],
        "regionCount": 1,
        "sampleCount": 1,
        "sampleLabel": "Images",
    }
    bridge = _FakeExactBridge(
        evaluate_solution_responses={
            "surrogate-a": {
                "solution": {
                    **surrogate_solutions[0].model_dump(mode="json"),
                    "rankingSource": "backend-exact",
                    "exactScore": 5.0,
                    "exactQualityCost": 0.5,
                    "exactMissionTimeSec": 125.0,
                    "exactMetricKind": "gsd",
                },
                "preview": preview_payload,
            },
            "surrogate-b": {
                "solution": {
                    **surrogate_solutions[1].model_dump(mode="json"),
                    "rankingSource": "backend-exact",
                    "exactScore": 5.0,
                    "exactQualityCost": 0.5,
                    "exactMissionTimeSec": 130.0,
                    "exactMetricKind": "gsd",
                },
                "preview": preview_payload,
            },
        },
        supports_candidate_fanout=True,
        candidate_max_inflight=3,
    )
    original_bridge = app_module.EXACT_RUNTIME_BRIDGE
    original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
    app_module.EXACT_RUNTIME_BRIDGE = bridge
    app_module.EXACT_POSTPROCESS_TOP_K = 2
    try:
        with TestClient(app_module.app) as client:
            response = client.post(
                "/v1/partition/solve",
                json={
                    "polygonId": "poly-1",
                    "ring": [[0.0, 0.0], [0.018, 0.0], [0.018, 0.0045], [0.0, 0.0045], [0.0, 0.0]],
                    "payloadKind": "camera",
                    "params": {"payloadKind": "camera", "altitudeAGL": 110, "frontOverlap": 75, "sideOverlap": 70},
                    "terrainSource": {"mode": "mapbox"},
                    "altitudeMode": "legacy",
                    "minClearanceM": 0,
                    "turnExtendM": 0,
                },
            )
        assert response.status_code == 200
        payload = response.json()
        assert [solution["signature"] for solution in payload["solutions"]] == ["surrogate-a", "surrogate-b"]
    finally:
        app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
        app_module.EXACT_RUNTIME_BRIDGE = original_bridge


def test_partition_solve_endpoint_matches_between_sharded_and_legacy_dsm(monkeypatch, tmp_path: Path) -> None:
    fixture_path = _real_dsm_fixture_path()
    source_descriptor = dsm_store_module.derive_descriptor_from_path(fixture_path, fixture_path.name)
    sharded_store = DsmDatasetStore(tmp_path / "sharded-store")
    sharded_descriptor, _ = sharded_store.ingest_dataset_file(fixture_path, fixture_path.name, source_descriptor=source_descriptor)
    legacy_store = DsmDatasetStore(tmp_path / "legacy-store")
    legacy_descriptor = _seed_legacy_dataset_from_file(
        legacy_store,
        fixture_path,
        original_name=fixture_path.name,
        source_descriptor=source_descriptor,
    )

    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.5,
                "regionCount": 1,
                "totalMissionTimeSec": 120.0,
                "normalizedQualityCost": 0.4,
                "weightedMeanMismatchDeg": 4.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 90.0, "atomCount": 2, "ring": _ring_inside_descriptor(source_descriptor, 0.2), "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 0.0, "atomCount": 2, "ring": _ring_inside_descriptor(source_descriptor, 0.3), "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)

    request_json = {
        "polygonId": "fixture-poly",
        "ring": _ring_inside_descriptor(source_descriptor, inset_fraction=0.15),
        "payloadKind": "camera",
        "params": {
            "payloadKind": "camera",
            "altitudeAGL": 110,
            "frontOverlap": 75,
            "sideOverlap": 70,
        },
        "altitudeMode": "legacy",
        "minClearanceM": 0,
        "turnExtendM": 0,
    }
    png_payload = _encode_terrain_png_bytes(64)
    repo_root = Path(__file__).resolve().parents[3]

    class _FakeTerrainTileCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_fetch(self, *_args, **_kwargs):
            return png_payload

    monkeypatch.setattr("terrain_splitter.mapbox_tiles.TerrainTileCache", _FakeTerrainTileCache)
    monkeypatch.setattr("terrain_splitter.mapbox_tiles.mapbox_token", lambda: "test-token")

    def _run(store, descriptor):
        with _LocalTerrainBatchServer(store, png_payload) as terrain_server:
            monkeypatch.setenv("TERRAIN_SPLITTER_INTERNAL_BASE_URL", terrain_server.base_url)
            bridge = LocalExactRuntimeSidecarBridge(repo_root)
            original_bridge = app_module.EXACT_RUNTIME_BRIDGE
            original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
            original_store = app_module.DSM_DATASET_STORE
            app_module.EXACT_RUNTIME_BRIDGE = bridge
            app_module.EXACT_POSTPROCESS_TOP_K = 2
            app_module.DSM_DATASET_STORE = store
            try:
                with TestClient(app_module.app) as client:
                    response = client.post(
                        "/v1/partition/solve",
                        json={**request_json, "terrainSource": {"mode": "blended", "datasetId": descriptor.id}},
                    )
                assert response.status_code == 200
                assert terrain_server.requests
                return response.json()
            finally:
                bridge.close()
                app_module.EXACT_RUNTIME_BRIDGE = original_bridge
                app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
                app_module.DSM_DATASET_STORE = original_store

    sharded_response = _run(sharded_store, sharded_descriptor)
    legacy_response = _run(legacy_store, legacy_descriptor)
    assert [solution["signature"] for solution in sharded_response["solutions"]] == [solution["signature"] for solution in legacy_response["solutions"]]
    assert sharded_response["solutions"][0]["rankingSource"] == legacy_response["solutions"][0]["rankingSource"]
    assert sharded_response["solutions"][0]["exactScore"] == legacy_response["solutions"][0]["exactScore"]
    assert sharded_response["solutions"][1]["exactScore"] == legacy_response["solutions"][1]["exactScore"]
    assert sharded_response["solutions"][0]["regions"][0]["bearingDeg"] == legacy_response["solutions"][0]["regions"][0]["bearingDeg"]
    assert sharded_response["solutions"][1]["regions"][0]["bearingDeg"] == legacy_response["solutions"][1]["regions"][0]["bearingDeg"]


def test_partition_solve_lidar_endpoint_matches_between_sharded_and_legacy_dsm(monkeypatch, tmp_path: Path) -> None:
    fixture_path = _real_dsm_fixture_path()
    source_descriptor = dsm_store_module.derive_descriptor_from_path(fixture_path, fixture_path.name)
    sharded_store = DsmDatasetStore(tmp_path / "sharded-store")
    sharded_descriptor, _ = sharded_store.ingest_dataset_file(fixture_path, fixture_path.name, source_descriptor=source_descriptor)
    legacy_store = DsmDatasetStore(tmp_path / "legacy-store")
    legacy_descriptor = _seed_legacy_dataset_from_file(
        legacy_store,
        fixture_path,
        original_name=fixture_path.name,
        source_descriptor=source_descriptor,
    )

    surrogate_solutions = [
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-a",
                "tradeoff": 0.5,
                "regionCount": 1,
                "totalMissionTimeSec": 120.0,
                "normalizedQualityCost": 0.4,
                "weightedMeanMismatchDeg": 4.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": True,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 90.0, "atomCount": 2, "ring": _ring_inside_descriptor(source_descriptor, 0.2), "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
        PartitionSolutionPreviewModel.model_validate(
            {
                "signature": "surrogate-b",
                "tradeoff": 0.6,
                "regionCount": 1,
                "totalMissionTimeSec": 130.0,
                "normalizedQualityCost": 0.5,
                "weightedMeanMismatchDeg": 5.0,
                "hierarchyLevel": 1,
                "largestRegionFraction": 1.0,
                "meanConvexity": 1.0,
                "boundaryBreakAlignment": 0.7,
                "isFirstPracticalSplit": False,
                "regions": [
                    {"areaM2": 10.0, "bearingDeg": 0.0, "atomCount": 2, "ring": _ring_inside_descriptor(source_descriptor, 0.3), "convexity": 1.0, "compactness": 0.8},
                ],
            }
        ),
    ]
    monkeypatch.setattr(app_module, "build_grid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "compute_feature_field", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(app_module, "solve_partition_hierarchy", lambda *_args, **_kwargs: surrogate_solutions)

    request_json = {
        "polygonId": "fixture-poly-lidar",
        "ring": _ring_inside_descriptor(source_descriptor, inset_fraction=0.15),
        "payloadKind": "lidar",
        "params": {
            "payloadKind": "lidar",
            "altitudeAGL": 120,
            "frontOverlap": 0,
            "sideOverlap": 40,
            "lidarKey": "WINGTRA_LIDAR_XT32M2X",
            "speedMps": 16,
            "mappingFovDeg": 90,
            "lidarReturnMode": "single",
            "pointDensityPtsM2": 50,
            "maxLidarRangeM": 200,
        },
        "altitudeMode": "legacy",
        "minClearanceM": 0,
        "turnExtendM": 0,
    }
    png_payload = _encode_terrain_png_bytes(64)
    repo_root = Path(__file__).resolve().parents[3]

    class _FakeTerrainTileCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_fetch(self, *_args, **_kwargs):
            return png_payload

    monkeypatch.setattr("terrain_splitter.mapbox_tiles.TerrainTileCache", _FakeTerrainTileCache)
    monkeypatch.setattr("terrain_splitter.mapbox_tiles.mapbox_token", lambda: "test-token")

    def _run(store, descriptor):
        with _LocalTerrainBatchServer(store, png_payload) as terrain_server:
            monkeypatch.setenv("TERRAIN_SPLITTER_INTERNAL_BASE_URL", terrain_server.base_url)
            bridge = LocalExactRuntimeSidecarBridge(repo_root)
            original_bridge = app_module.EXACT_RUNTIME_BRIDGE
            original_top_k = app_module.EXACT_POSTPROCESS_TOP_K
            original_store = app_module.DSM_DATASET_STORE
            app_module.EXACT_RUNTIME_BRIDGE = bridge
            app_module.EXACT_POSTPROCESS_TOP_K = 2
            app_module.DSM_DATASET_STORE = store
            try:
                with TestClient(app_module.app) as client:
                    response = client.post(
                        "/v1/partition/solve",
                        json={**request_json, "terrainSource": {"mode": "blended", "datasetId": descriptor.id}},
                    )
                assert response.status_code == 200
                assert terrain_server.requests
                return response.json()
            finally:
                bridge.close()
                app_module.EXACT_RUNTIME_BRIDGE = original_bridge
                app_module.EXACT_POSTPROCESS_TOP_K = original_top_k
                app_module.DSM_DATASET_STORE = original_store

    sharded_response = _run(sharded_store, sharded_descriptor)
    legacy_response = _run(legacy_store, legacy_descriptor)
    assert [solution["signature"] for solution in sharded_response["solutions"]] == [solution["signature"] for solution in legacy_response["solutions"]]
    assert sharded_response["solutions"][0]["rankingSource"] == legacy_response["solutions"][0]["rankingSource"]
    assert sharded_response["solutions"][0]["exactScore"] == legacy_response["solutions"][0]["exactScore"]
    assert sharded_response["solutions"][1]["exactScore"] == legacy_response["solutions"][1]["exactScore"]
    assert sharded_response["solutions"][0]["regions"][0]["bearingDeg"] == legacy_response["solutions"][0]["regions"][0]["bearingDeg"]
    assert sharded_response["solutions"][1]["regions"][0]["bearingDeg"] == legacy_response["solutions"][1]["regions"][0]["bearingDeg"]


def test_s3_cold_exact_optimize_uses_selected_level_without_source_download(monkeypatch, tmp_path: Path) -> None:
    fixture_path = _real_dsm_fixture_path()
    source_descriptor = dsm_store_module.derive_descriptor_from_path(fixture_path, fixture_path.name)
    fake_s3 = _FakeS3Client()
    upload_store = S3BackedDsmDatasetStore(tmp_path / "cache-upload", bucket="bucket", prefix="stage", client=fake_s3)
    stored_descriptor, _ = upload_store.ingest_dataset_file(fixture_path, fixture_path.name, source_descriptor=source_descriptor)
    cold_store = S3BackedDsmDatasetStore(tmp_path / "cache-cold", bucket="bucket", prefix="stage", client=fake_s3)
    ring = _ring_inside_descriptor(source_descriptor, inset_fraction=0.25)
    png_payload = _encode_terrain_png_bytes(64)
    repo_root = Path(__file__).resolve().parents[3]

    with _LocalTerrainBatchServer(cold_store, png_payload) as terrain_server:
        monkeypatch.setenv("TERRAIN_SPLITTER_INTERNAL_BASE_URL", terrain_server.base_url)
        bridge = LocalExactRuntimeSidecarBridge(repo_root)
        try:
            fake_s3.get_object_requests.clear()
            response = bridge.optimize_bearing(
                {
                    "polygonId": "fixture-poly",
                    "ring": ring,
                    "payloadKind": "camera",
                    "params": {
                        "payloadKind": "camera",
                        "altitudeAGL": 110,
                        "frontOverlap": 75,
                        "sideOverlap": 70,
                    },
                    "terrainSource": {"mode": "blended", "datasetId": stored_descriptor.id},
                    "altitudeMode": "legacy",
                    "minClearanceM": 0,
                    "turnExtendM": 0,
                    "seedBearingDeg": 17,
                    "mode": "global",
                    "halfWindowDeg": 90,
                }
            )
            assert response["best"]["bearingDeg"] is not None
            requested_keys = [key for (bucket, key) in fake_s3.get_object_requests if bucket == "bucket"]
            assert f"stage/datasets/{stored_descriptor.id}/{stored_descriptor.id}{fixture_path.suffix}" not in requested_keys
            assert f"stage/datasets/{stored_descriptor.id}/analysis/manifest.json" in requested_keys
            level_keys = [key for key in requested_keys if f"stage/datasets/{stored_descriptor.id}/analysis/level-" in key]
            assert len(level_keys) >= 1
            assert len(set(level_keys)) <= 2
        finally:
            bridge.close()
