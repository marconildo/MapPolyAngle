from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import tifffile
from fastapi.testclient import TestClient

from terrain_splitter import app as app_module
from terrain_splitter.dsm_store import DsmDatasetStore, S3BackedDsmDatasetStore, create_dsm_dataset_store
from terrain_splitter.mapbox_tiles import TerrainTile
from terrain_splitter.schemas import DsmSourceDescriptorModel, TerrainSourceModel


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


class _FakePaginator:
    def __init__(self, client: "_FakeS3Client"):
        self.client = client

    def paginate(self, Bucket: str, Prefix: str):  # noqa: N803
        contents = [{"Key": key} for (bucket, key), _ in sorted(self.client.objects.items()) if bucket == Bucket and key.startswith(Prefix)]
        yield {"Contents": contents}


class _FakeS3Client:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

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
        payload = self.objects.get((Bucket, Key))
        if payload is None:
            raise FileNotFoundError(Key)
        return {"Body": io.BytesIO(payload)}

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return _FakePaginator(self)


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


def test_dsm_upload_and_dataset_endpoints(tmp_path: Path) -> None:
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

            listing = client.get("/v1/dsm/datasets")
            assert listing.status_code == 200
            datasets = listing.json()["datasets"]
            assert len(datasets) == 1
            assert datasets[0]["datasetId"] == upload_payload["datasetId"]

            detail = client.get(f"/v1/dsm/datasets/{upload_payload['datasetId']}")
            assert detail.status_code == 200
            assert detail.json()["datasetId"] == upload_payload["datasetId"]
    finally:
        app_module.DSM_DIR = original_dir
        app_module.DSM_DATASET_STORE = original_store


def test_dsm_upload_and_dataset_endpoints_with_s3_store(tmp_path: Path) -> None:
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

            listing = client.get("/v1/dsm/datasets")
            assert listing.status_code == 200
            datasets = listing.json()["datasets"]
            assert len(datasets) == 1
            assert datasets[0]["datasetId"] == dataset_id

            detail = client.get(f"/v1/dsm/datasets/{dataset_id}")
            assert detail.status_code == 200
            assert detail.json()["descriptor"]["id"] == dataset_id

            remote_keys = {key for (bucket, key) in fake_s3.objects if bucket == "bucket"}
            assert any(key.endswith("/descriptor.json") for key in remote_keys)
            assert any(key.endswith("/pyramid.npz") for key in remote_keys)
    finally:
        app_module.DSM_DIR = original_dir
        app_module.DSM_DATASET_STORE = original_store


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
