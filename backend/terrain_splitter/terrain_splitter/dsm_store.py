from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import tifffile
from pyproj import CRS, Transformer

from .mapbox_tiles import TerrainDEM, TerrainTile, tile_bounds_mercator
from .schemas import DsmSourceDescriptorModel, TerrainSourceModel


@dataclass(slots=True)
class DsmDatasetLevel:
    raster: np.ndarray
    valid_mask: np.ndarray
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float


@dataclass(slots=True)
class DsmDataset:
    descriptor: DsmSourceDescriptorModel
    file_path: Path
    source_crs: str
    to_source: Transformer
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    levels: list[DsmDatasetLevel]


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_source_crs(descriptor: DsmSourceDescriptorModel) -> str:
    if descriptor.sourceCrsCode and descriptor.sourceCrsCode.strip():
        return descriptor.sourceCrsCode.strip()
    proj = descriptor.sourceProj4.strip()
    if not proj:
        raise ValueError("DSM descriptor does not include a usable source CRS.")
    return proj.replace("+axis=en ", "").replace(" +axis=en", "")


def _intersects_bounds(
    a_min_x: float,
    a_min_y: float,
    a_max_x: float,
    a_max_y: float,
    b_min_x: float,
    b_min_y: float,
    b_max_x: float,
    b_max_y: float,
) -> bool:
    return not (a_max_x <= b_min_x or a_min_x >= b_max_x or a_max_y <= b_min_y or a_min_y >= b_max_y)


def _normalize_raster_shape(raster: np.ndarray, width: int, height: int) -> np.ndarray:
    if raster.ndim == 2:
        normalized = raster
    elif raster.ndim == 3:
        if raster.shape[0] == height and raster.shape[1] == width:
            normalized = raster[:, :, 0]
        elif raster.shape[1] == height and raster.shape[2] == width:
            normalized = raster[0, :, :]
        else:
            raise ValueError(f"Unsupported DSM raster shape {tuple(raster.shape)} for declared size {width}x{height}.")
    else:
        raise ValueError(f"Unsupported DSM raster rank {raster.ndim}.")

    if normalized.shape[0] != height or normalized.shape[1] != width:
        raise ValueError(
            f"DSM raster shape {tuple(normalized.shape)} does not match descriptor size {width}x{height}."
        )
    return np.asarray(normalized, dtype=np.float32)


def _normalize_raster_samples(raster: np.ndarray, width: int, height: int) -> np.ndarray:
    if raster.ndim == 2:
        normalized = raster[:, :, np.newaxis]
    elif raster.ndim == 3:
        if raster.shape[0] == height and raster.shape[1] == width:
            normalized = raster
        elif raster.shape[1] == height and raster.shape[2] == width:
            normalized = np.moveaxis(raster, 0, -1)
        else:
            raise ValueError(f"Unsupported DSM raster shape {tuple(raster.shape)} for declared size {width}x{height}.")
    else:
        raise ValueError(f"Unsupported DSM raster rank {raster.ndim}.")

    if normalized.shape[0] != height or normalized.shape[1] != width:
        raise ValueError(
            f"DSM raster shape {tuple(normalized.shape)} does not match descriptor size {width}x{height}."
    )
    return np.asarray(normalized)


def _is_alpha_or_mask_extra_sample(extra_sample: Any) -> bool:
    extra_name = getattr(extra_sample, "name", str(extra_sample)).lower()
    return "alpha" in extra_name or "mask" in extra_name


def _validate_supported_dsm_sample_layout(sample_count: int, extrasamples: tuple[Any, ...]) -> None:
    alpha_or_mask_extra_count = sum(1 for extra_sample in extrasamples if _is_alpha_or_mask_extra_sample(extra_sample))
    unsupported_extra_count = len(extrasamples) - alpha_or_mask_extra_count
    base_sample_count = sample_count - alpha_or_mask_extra_count
    if base_sample_count != 1 or unsupported_extra_count > 0:
        raise ValueError(
            "DSM must be a single-band elevation GeoTIFF with at most alpha/mask extra samples; "
            "RGB/RGBA ortho imagery is not supported."
        )


def _read_raster_and_valid_mask_from_tiff(tf: tifffile.TiffFile, source_descriptor: DsmSourceDescriptorModel) -> tuple[np.ndarray, np.ndarray]:
    page = tf.pages[0]
    raster = page.asarray()
    samples = _normalize_raster_samples(raster, source_descriptor.width, source_descriptor.height)
    extrasamples = tuple(getattr(page, "extrasamples", ()) or ())
    _validate_supported_dsm_sample_layout(samples.shape[2], extrasamples)
    normalized = np.asarray(samples[:, :, 0], dtype=np.float32)
    valid_mask = np.isfinite(normalized)

    if source_descriptor.noDataValue is not None:
        valid_mask &= ~np.isclose(normalized, float(source_descriptor.noDataValue))

    if extrasamples:
        base_sample_count = max(samples.shape[2] - len(extrasamples), 1)
        for extra_index, extra_sample in enumerate(extrasamples):
            if not _is_alpha_or_mask_extra_sample(extra_sample):
                continue
            sample_index = base_sample_count + extra_index
            if sample_index >= samples.shape[2]:
                continue
            alpha = np.asarray(samples[:, :, sample_index], dtype=np.float32)
            valid_mask &= np.isfinite(alpha) & (alpha > 0)

    return normalized, valid_mask


def _read_raster_and_valid_mask(payload: bytes, source_descriptor: DsmSourceDescriptorModel) -> tuple[np.ndarray, np.ndarray]:
    with tifffile.TiffFile(io.BytesIO(payload)) as tf:
        return _read_raster_and_valid_mask_from_tiff(tf, source_descriptor)


def _read_raster_and_valid_mask_from_path(path: Path, source_descriptor: DsmSourceDescriptorModel) -> tuple[np.ndarray, np.ndarray]:
    with tifffile.TiffFile(str(path)) as tf:
        return _read_raster_and_valid_mask_from_tiff(tf, source_descriptor)


def _downsample_once(raster: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = raster.shape
    pad_h = height % 2
    pad_w = width % 2
    if pad_h or pad_w:
        raster = np.pad(raster, ((0, pad_h), (0, pad_w)), mode="edge")
        valid_mask = np.pad(valid_mask, ((0, pad_h), (0, pad_w)), mode="edge")

    reshaped_raster = raster.reshape(raster.shape[0] // 2, 2, raster.shape[1] // 2, 2)
    reshaped_mask = valid_mask.reshape(valid_mask.shape[0] // 2, 2, valid_mask.shape[1] // 2, 2)
    counts = reshaped_mask.sum(axis=(1, 3)).astype(np.float32)
    sums = np.where(reshaped_mask, reshaped_raster, 0.0).sum(axis=(1, 3), dtype=np.float32)
    down_mask = counts > 0
    down_raster = np.divide(sums, np.maximum(counts, 1.0), dtype=np.float32)
    down_raster[~down_mask] = 0.0
    return down_raster.astype(np.float32), down_mask.astype(bool)


def _build_pyramid(
    raster: np.ndarray,
    valid_mask: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
) -> list[DsmDatasetLevel]:
    levels = [
        DsmDatasetLevel(
            raster=np.asarray(raster, dtype=np.float32),
            valid_mask=np.asarray(valid_mask, dtype=bool),
            width=int(raster.shape[1]),
            height=int(raster.shape[0]),
            pixel_size_x=float(pixel_size_x),
            pixel_size_y=float(pixel_size_y),
        )
    ]

    current_raster = raster
    current_mask = valid_mask
    current_pixel_x = float(pixel_size_x)
    current_pixel_y = float(pixel_size_y)

    while min(current_raster.shape[0], current_raster.shape[1]) > 256:
        current_raster, current_mask = _downsample_once(current_raster, current_mask)
        current_pixel_x *= 2.0
        current_pixel_y *= 2.0
        levels.append(
            DsmDatasetLevel(
                raster=np.asarray(current_raster, dtype=np.float32),
                valid_mask=np.asarray(current_mask, dtype=bool),
                width=int(current_raster.shape[1]),
                height=int(current_raster.shape[0]),
                pixel_size_x=float(current_pixel_x),
                pixel_size_y=float(current_pixel_y),
            )
        )
        if min(current_raster.shape[0], current_raster.shape[1]) <= 64:
            break

    return levels


def _serialize_pyramid(levels: list[DsmDatasetLevel], path: Path) -> None:
    payload: dict[str, Any] = {"level_count": np.asarray([len(levels)], dtype=np.int32)}
    for index, level in enumerate(levels):
        payload[f"raster_{index}"] = level.raster.astype(np.float32)
        payload[f"valid_{index}"] = level.valid_mask.astype(np.uint8)
        payload[f"shape_{index}"] = np.asarray([level.height, level.width], dtype=np.int32)
        payload[f"pixel_{index}"] = np.asarray([level.pixel_size_x, level.pixel_size_y], dtype=np.float64)
    np.savez_compressed(path, **payload)


def _serialize_pyramid_bytes(levels: list[DsmDatasetLevel]) -> bytes:
    buffer = io.BytesIO()
    payload: dict[str, Any] = {"level_count": np.asarray([len(levels)], dtype=np.int32)}
    for index, level in enumerate(levels):
        payload[f"raster_{index}"] = level.raster.astype(np.float32)
        payload[f"valid_{index}"] = level.valid_mask.astype(np.uint8)
        payload[f"shape_{index}"] = np.asarray([level.height, level.width], dtype=np.int32)
        payload[f"pixel_{index}"] = np.asarray([level.pixel_size_x, level.pixel_size_y], dtype=np.float64)
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def _deserialize_pyramid(path: Path) -> list[DsmDatasetLevel]:
    with np.load(path) as payload:
        return _deserialize_pyramid_payload(payload)


def _deserialize_pyramid_bytes(payload: bytes) -> list[DsmDatasetLevel]:
    with np.load(io.BytesIO(payload)) as archive:
        return _deserialize_pyramid_payload(archive)


def _deserialize_pyramid_payload(payload: Any) -> list[DsmDatasetLevel]:
    level_count = int(payload["level_count"][0])
    levels: list[DsmDatasetLevel] = []
    for index in range(level_count):
        shape = payload[f"shape_{index}"]
        pixel = payload[f"pixel_{index}"]
        raster = np.asarray(payload[f"raster_{index}"], dtype=np.float32).reshape(int(shape[0]), int(shape[1]))
        valid_mask = np.asarray(payload[f"valid_{index}"], dtype=np.uint8).reshape(int(shape[0]), int(shape[1])) > 0
        levels.append(
            DsmDatasetLevel(
                raster=raster,
                valid_mask=valid_mask,
                width=int(shape[1]),
                height=int(shape[0]),
                pixel_size_x=float(pixel[0]),
                pixel_size_y=float(pixel[1]),
            )
        )
    return levels


def _encode_terrain_rgb(elevation_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded = np.clip(np.rint((elevation_m + 10000.0) * 10.0), 0, 256 * 256 * 256 - 1).astype(np.uint32)
    return (
        ((encoded >> 16) & 255).astype(np.uint8),
        ((encoded >> 8) & 255).astype(np.uint8),
        (encoded & 255).astype(np.uint8),
    )


def _bounds_from_points(points: list[tuple[float, float]]) -> dict[str, float]:
    return {
        "minX": min(point[0] for point in points),
        "minY": min(point[1] for point in points),
        "maxX": max(point[0] for point in points),
        "maxY": max(point[1] for point in points),
    }


def _lnglat_bounds_from_points(points: list[tuple[float, float]]) -> dict[str, float]:
    return {
        "minLng": min(point[0] for point in points),
        "minLat": min(point[1] for point in points),
        "maxLng": max(point[0] for point in points),
        "maxLat": max(point[1] for point in points),
    }


def _derive_source_bounds_from_tags(page: tifffile.TiffPage, width: int, height: int) -> dict[str, float]:
    metadata = page.geotiff_tags or page.parent.geotiff_metadata or {}
    model_pixel_scale = metadata.get("ModelPixelScale")
    model_tiepoint = metadata.get("ModelTiepoint")
    if model_pixel_scale and len(model_pixel_scale) >= 2 and model_tiepoint and len(model_tiepoint) >= 6:
        scale_x = float(model_pixel_scale[0])
        scale_y = float(model_pixel_scale[1])
        tie_x = float(model_tiepoint[3])
        tie_y = float(model_tiepoint[4])
        if not math.isfinite(scale_x) or not math.isfinite(scale_y) or scale_x <= 0 or scale_y <= 0:
            raise ValueError("GeoTIFF ModelPixelScale values are invalid.")
        return {
            "minX": tie_x,
            "minY": tie_y - height * scale_y,
            "maxX": tie_x + width * scale_x,
            "maxY": tie_y,
        }

    transform_tag = page.tags.get("ModelTransformationTag") or page.tags.get(34264)
    if transform_tag is not None:
        values = np.asarray(transform_tag.value, dtype=np.float64).reshape(4, 4)

        def transform(col: float, row: float) -> tuple[float, float]:
            vector = values @ np.asarray([col, row, 0.0, 1.0], dtype=np.float64)
            if not np.isfinite(vector[0]) or not np.isfinite(vector[1]):
                raise ValueError("GeoTIFF ModelTransformationTag produced non-finite coordinates.")
            return float(vector[0]), float(vector[1])

        corners = [
            transform(0.0, 0.0),
            transform(float(width), 0.0),
            transform(float(width), float(height)),
            transform(0.0, float(height)),
        ]
        return _bounds_from_points(corners)

    raise ValueError("GeoTIFF is missing ModelPixelScale/ModelTiepoint or ModelTransformationTag metadata.")


def _derive_descriptor_from_tiff(
    tf: tifffile.TiffFile,
    *,
    original_name: str,
    file_size_bytes: int,
) -> DsmSourceDescriptorModel:
    page = tf.pages[0]
    metadata = page.geotiff_tags or tf.geotiff_metadata
    if not metadata:
        raise ValueError("GeoTIFF metadata is missing; uploaded TIFF is not georeferenced.")

    extrasamples = tuple(getattr(page, "extrasamples", ()) or ())
    sample_count = int(getattr(page, "samplesperpixel", 1) or 1)
    _validate_supported_dsm_sample_layout(sample_count, extrasamples)

    width = int(page.imagewidth)
    height = int(page.imagelength)
    if width <= 0 or height <= 0:
        raise ValueError("GeoTIFF dimensions are invalid.")

    projected_code = metadata.get("ProjectedCSTypeGeoKey")
    geographic_code = metadata.get("GeographicTypeGeoKey")
    source_code = None
    if projected_code:
        source_code = f"EPSG:{int(projected_code)}"
    elif geographic_code:
        source_code = f"EPSG:{int(geographic_code)}"

    if not source_code:
        raise ValueError("GeoTIFF does not declare a supported projected or geographic CRS.")

    try:
        source_crs = CRS.from_user_input(source_code)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not resolve GeoTIFF CRS {source_code}.") from exc

    source_proj4 = source_crs.to_proj4()
    if not source_proj4:
        raise ValueError("GeoTIFF CRS could not be converted to a usable PROJ string.")

    source_bounds = _derive_source_bounds_from_tags(page, width, height)

    source_corners = [
        (source_bounds["minX"], source_bounds["maxY"]),
        (source_bounds["maxX"], source_bounds["maxY"]),
        (source_bounds["maxX"], source_bounds["minY"]),
        (source_bounds["minX"], source_bounds["minY"]),
    ]
    to_3857 = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
    to_4326 = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    footprint_3857_points = [tuple(map(float, to_3857.transform(x, y))) for x, y in source_corners]
    footprint_lnglat_points = [tuple(map(float, to_4326.transform(x, y))) for x, y in source_corners]
    footprint_ring_lnglat = [(lng, lat) for lng, lat in footprint_lnglat_points]
    footprint_ring_lnglat.append(footprint_ring_lnglat[0])

    nodata_tag = page.tags.get("GDAL_NODATA")
    nodata_value: float | None = None
    if nodata_tag is not None:
        try:
            parsed = float(str(nodata_tag.value).strip())
            nodata_value = parsed if math.isfinite(parsed) else None
        except Exception:  # noqa: BLE001
            nodata_value = None

    horizontal_units = source_crs.axis_info[0].unit_name if source_crs.axis_info else None

    vertical_scale_to_meters = 1.0
    vertical_code = metadata.get("VerticalCSTypeGeoKey")
    if vertical_code:
        try:
            vertical_crs = CRS.from_epsg(int(vertical_code))
            if vertical_crs.axis_info and vertical_crs.axis_info[0].unit_conversion_factor:
                vertical_scale_to_meters = float(vertical_crs.axis_info[0].unit_conversion_factor)
        except Exception:  # noqa: BLE001
            vertical_scale_to_meters = 1.0

    return DsmSourceDescriptorModel.model_validate(
        {
            "id": "pending-ingest",
            "name": original_name,
            "fileSizeBytes": file_size_bytes,
            "width": width,
            "height": height,
            "sourceBounds": source_bounds,
            "footprint3857": _bounds_from_points(footprint_3857_points),
            "footprintLngLat": _lnglat_bounds_from_points(footprint_lnglat_points),
            "footprintRingLngLat": footprint_ring_lnglat,
            "sourceCrsCode": source_code,
            "sourceCrsLabel": str(metadata.get("GTCitationGeoKey") or source_crs.name or source_code),
            "sourceProj4": source_proj4,
            "horizontalUnits": horizontal_units,
            "verticalScaleToMeters": vertical_scale_to_meters,
            "noDataValue": nodata_value,
            "loadedAtIso": _utc_now_iso(),
        }
    )


def derive_descriptor_from_payload(payload: bytes, original_name: str) -> DsmSourceDescriptorModel:
    with tifffile.TiffFile(io.BytesIO(payload)) as tf:
        return _derive_descriptor_from_tiff(tf, original_name=original_name, file_size_bytes=len(payload))


def derive_descriptor_from_path(path: Path, original_name: str) -> DsmSourceDescriptorModel:
    with tifffile.TiffFile(str(path)) as tf:
        return _derive_descriptor_from_tiff(tf, original_name=original_name, file_size_bytes=int(path.stat().st_size))


def _sha256_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class DsmDatasetStore:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir = self.root_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_dir / "index.json"
        self._lock = RLock()
        self._dataset_cache: dict[str, DsmDataset] = {}

    def _load_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"version": 1, "datasets": {}}
        payload = json.loads(self.index_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("DSM dataset index is invalid.")
        payload.setdefault("version", 1)
        payload.setdefault("datasets", {})
        return payload

    def _save_index(self, payload: dict[str, Any]) -> None:
        temp_path = self.index_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temp_path.replace(self.index_path)

    def _dataset_dir(self, dataset_id: str) -> Path:
        path = self.datasets_dir / dataset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _dataset_file_path(self, dataset_id: str, suffix: str) -> Path:
        suffix = suffix or ".tiff"
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return self._dataset_dir(dataset_id) / f"{dataset_id}{suffix}"

    def _dataset_pyramid_path(self, dataset_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "pyramid.npz"

    def _evict_dataset_entry(self, index: dict[str, Any], dataset_id: str) -> None:
        datasets = index.get("datasets", {})
        if isinstance(datasets, dict) and dataset_id in datasets:
            datasets.pop(dataset_id, None)
            self._save_index(index)
        self._dataset_cache.pop(dataset_id, None)

    def _validate_existing_source_file(
        self,
        dataset_id: str,
        descriptor: DsmSourceDescriptorModel,
        file_path: Path,
        *,
        index: dict[str, Any] | None = None,
    ) -> bool:
        if not file_path.exists():
            return True
        try:
            derive_descriptor_from_path(file_path, descriptor.name or f"{dataset_id}.tiff")
            return True
        except Exception:  # noqa: BLE001
            if index is not None:
                self._evict_dataset_entry(index, dataset_id)
            return False

    def _build_descriptor(
        self,
        dataset_id: str,
        original_name: str,
        file_size_bytes: int,
        source_descriptor: DsmSourceDescriptorModel,
        valid_mask: np.ndarray,
    ) -> DsmSourceDescriptorModel:
        footprint = source_descriptor.footprint3857
        native_resolution_x = abs((footprint.maxX - footprint.minX) / max(source_descriptor.width, 1))
        native_resolution_y = abs((footprint.maxY - footprint.minY) / max(source_descriptor.height, 1))
        return source_descriptor.model_copy(
            update={
                "id": dataset_id,
                "name": original_name or source_descriptor.name,
                "fileSizeBytes": file_size_bytes,
                "loadedAtIso": _utc_now_iso(),
                "nativeResolutionXM": native_resolution_x if native_resolution_x > 0 else None,
                "nativeResolutionYM": native_resolution_y if native_resolution_y > 0 else None,
                "validCoverageRatio": float(valid_mask.mean()) if valid_mask.size else 0.0,
            }
        )

    def ingest_dataset(
        self,
        payload: bytes,
        original_name: str,
        source_descriptor: DsmSourceDescriptorModel,
    ) -> tuple[DsmSourceDescriptorModel, bool]:
        temp_file_path = self.root_dir / f".ingest-{hashlib.sha256(payload).hexdigest()}{Path(original_name or source_descriptor.name or 'surface.tiff').suffix or '.tiff'}"
        temp_file_path.write_bytes(payload)
        try:
            return self.ingest_dataset_file(
                temp_file_path,
                original_name,
                source_descriptor=source_descriptor,
            )
        finally:
            temp_file_path.unlink(missing_ok=True)

    def ingest_dataset_file(
        self,
        file_path: Path,
        original_name: str,
        *,
        source_descriptor: DsmSourceDescriptorModel | None = None,
        verified_sha256: str | None = None,
    ) -> tuple[DsmSourceDescriptorModel, bool]:
        dataset_id = verified_sha256 or _sha256_for_path(file_path)
        suffix = Path(
            original_name or (source_descriptor.name if source_descriptor is not None else "surface.tiff")
        ).suffix or ".tiff"
        source_descriptor = source_descriptor or derive_descriptor_from_path(file_path, original_name)
        with self._lock:
            index = self._load_index()
            entry = index["datasets"].get(dataset_id)
            if isinstance(entry, dict):
                existing_file_path = Path(entry.get("filePath", ""))
                pyramid_path = Path(entry.get("pyramidPath", ""))
                if existing_file_path.exists() and pyramid_path.exists():
                    descriptor = DsmSourceDescriptorModel.model_validate(entry["descriptor"])
                    return descriptor, True

            destination_file_path = self._dataset_file_path(dataset_id, suffix)
            if file_path.resolve() != destination_file_path.resolve():
                shutil.copyfile(file_path, destination_file_path)
            else:
                destination_file_path = file_path

            normalized, valid_mask = _read_raster_and_valid_mask_from_path(destination_file_path, source_descriptor)

            bounds = source_descriptor.sourceBounds
            pixel_size_x = (bounds.maxX - bounds.minX) / source_descriptor.width
            pixel_size_y = (bounds.maxY - bounds.minY) / source_descriptor.height
            if pixel_size_x <= 0 or pixel_size_y <= 0:
                raise ValueError("DSM descriptor bounds are invalid.")

            descriptor = self._build_descriptor(dataset_id, original_name, int(destination_file_path.stat().st_size), source_descriptor, valid_mask)
            levels = _build_pyramid(normalized, valid_mask, pixel_size_x, pixel_size_y)
            pyramid_path = self._dataset_pyramid_path(dataset_id)
            _serialize_pyramid(levels, pyramid_path)

            index["datasets"][dataset_id] = {
                "descriptor": descriptor.model_dump(mode="json"),
                "filePath": str(destination_file_path),
                "pyramidPath": str(pyramid_path),
                "sourceCrs": _resolve_source_crs(descriptor),
            }
            self._save_index(index)
            self._dataset_cache.pop(dataset_id, None)
            return descriptor, False

    def list_datasets(self) -> list[DsmSourceDescriptorModel]:
        with self._lock:
            index = self._load_index()
            datasets = [
                DsmSourceDescriptorModel.model_validate(entry["descriptor"])
                for entry in index.get("datasets", {}).values()
                if isinstance(entry, dict) and "descriptor" in entry
            ]
        return sorted(datasets, key=lambda descriptor: descriptor.loadedAtIso, reverse=True)

    def get_dataset_descriptor(self, dataset_id: str) -> DsmSourceDescriptorModel | None:
        with self._lock:
            index = self._load_index()
            entry = index.get("datasets", {}).get(dataset_id)
            if not isinstance(entry, dict):
                return None
            descriptor = DsmSourceDescriptorModel.model_validate(entry["descriptor"])
            file_path = Path(entry.get("filePath", ""))
            if not self._validate_existing_source_file(dataset_id, descriptor, file_path, index=index):
                return None
            return descriptor

    def _load_dataset(self, dataset_id: str) -> DsmDataset:
        cached = self._dataset_cache.get(dataset_id)
        if cached is not None:
            return cached

        with self._lock:
            index = self._load_index()
            entry = index.get("datasets", {}).get(dataset_id)
            if not isinstance(entry, dict):
                raise KeyError(f"DSM dataset {dataset_id} was not found.")

            descriptor = DsmSourceDescriptorModel.model_validate(entry["descriptor"])
            file_path = Path(entry["filePath"])
            if not self._validate_existing_source_file(dataset_id, descriptor, file_path, index=index):
                raise KeyError(f"DSM dataset {dataset_id} is invalid.")
            pyramid_path = Path(entry["pyramidPath"])
            if not file_path.exists() or not pyramid_path.exists():
                raise FileNotFoundError(f"DSM dataset {dataset_id} is missing required files.")

            dataset = DsmDataset(
                descriptor=descriptor,
                file_path=file_path,
                source_crs=str(entry.get("sourceCrs") or _resolve_source_crs(descriptor)),
                to_source=Transformer.from_crs("EPSG:3857", str(entry.get("sourceCrs") or _resolve_source_crs(descriptor)), always_xy=True),
                min_x=descriptor.sourceBounds.minX,
                min_y=descriptor.sourceBounds.minY,
                max_x=descriptor.sourceBounds.maxX,
                max_y=descriptor.sourceBounds.maxY,
                levels=_deserialize_pyramid(pyramid_path),
            )
            self._dataset_cache[dataset_id] = dataset
            return dataset

    def _select_level(self, dataset: DsmDataset, bounds: tuple[float, float, float, float], size: int) -> DsmDatasetLevel:
        min_x, min_y, max_x, max_y = bounds
        corners3857 = [
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y),
        ]
        source_corners = [dataset.to_source.transform(x, y) for x, y in corners3857]
        source_span_x = abs(max(corner[0] for corner in source_corners) - min(corner[0] for corner in source_corners))
        source_span_y = abs(max(corner[1] for corner in source_corners) - min(corner[1] for corner in source_corners))
        base = dataset.levels[0]
        raw_cols = source_span_x / max(base.pixel_size_x, 1e-9)
        raw_rows = source_span_y / max(base.pixel_size_y, 1e-9)
        oversampling = max(raw_cols / max(size, 1), raw_rows / max(size, 1), 1.0)
        level_index = min(len(dataset.levels) - 1, max(0, int(math.floor(math.log2(oversampling)))))
        return dataset.levels[level_index]

    def _sample_dataset_grid(
        self,
        dataset: DsmDataset,
        bounds: tuple[float, float, float, float],
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        min_x, min_y, max_x, max_y = bounds
        footprint = dataset.descriptor.footprint3857
        if not _intersects_bounds(
            min_x,
            min_y,
            max_x,
            max_y,
            footprint.minX,
            footprint.minY,
            footprint.maxX,
            footprint.maxY,
        ):
            empty = np.zeros((size, size), dtype=np.float32)
            return empty, np.zeros((size, size), dtype=bool)

        level = self._select_level(dataset, bounds, size)
        xs = min_x + ((np.arange(size, dtype=np.float64) + 0.5) / size) * (max_x - min_x)
        ys = max_y - ((np.arange(size, dtype=np.float64) + 0.5) / size) * (max_y - min_y)
        grid_x, grid_y = np.meshgrid(xs, ys)

        src_x, src_y = dataset.to_source.transform(grid_x, grid_y)
        src_x = np.asarray(src_x, dtype=np.float64)
        src_y = np.asarray(src_y, dtype=np.float64)

        cols = (src_x - (dataset.min_x + 0.5 * level.pixel_size_x)) / level.pixel_size_x
        rows = ((dataset.max_y - 0.5 * level.pixel_size_y) - src_y) / level.pixel_size_y

        valid = (
            np.isfinite(cols)
            & np.isfinite(rows)
            & (cols >= 0.0)
            & (rows >= 0.0)
            & (cols <= max(level.width - 1, 0))
            & (rows <= max(level.height - 1, 0))
        )
        if not np.any(valid):
            empty = np.zeros((size, size), dtype=np.float32)
            return empty, np.zeros((size, size), dtype=bool)

        col0 = np.floor(cols).astype(np.int32)
        row0 = np.floor(rows).astype(np.int32)
        col0 = np.clip(col0, 0, max(level.width - 1, 0))
        row0 = np.clip(row0, 0, max(level.height - 1, 0))
        col1 = np.clip(col0 + 1, 0, level.width - 1)
        row1 = np.clip(row0 + 1, 0, level.height - 1)

        fx = np.clip(cols - col0, 0.0, 1.0).astype(np.float32)
        fy = np.clip(rows - row0, 0.0, 1.0).astype(np.float32)

        z00 = level.raster[row0, col0]
        z10 = level.raster[row0, col1]
        z01 = level.raster[row1, col0]
        z11 = level.raster[row1, col1]

        valid &= (
            level.valid_mask[row0, col0]
            & level.valid_mask[row0, col1]
            & level.valid_mask[row1, col0]
            & level.valid_mask[row1, col1]
        )

        sampled = (
            z00 * (1.0 - fx) * (1.0 - fy)
            + z10 * fx * (1.0 - fy)
            + z01 * (1.0 - fx) * fy
            + z11 * fx * fy
        ).astype(np.float32)
        sampled *= float(dataset.descriptor.verticalScaleToMeters)
        sampled[~valid] = 0.0
        return sampled, valid

    def _dataset_for_source(self, terrain_source: TerrainSourceModel | None) -> DsmDataset | None:
        if terrain_source is None or terrain_source.mode != "blended" or not terrain_source.datasetId:
            return None
        try:
            return self._load_dataset(terrain_source.datasetId)
        except (KeyError, FileNotFoundError, ValueError):
            return None

    def apply_terrain_source_to_tile(self, terrain_source: TerrainSourceModel | None, tile: TerrainTile) -> bool:
        dataset = self._dataset_for_source(terrain_source)
        if dataset is None:
            return False
        size = int(tile.elevation.shape[0])
        sampled, valid = self._sample_dataset_grid(dataset, (tile.min_x, tile.min_y, tile.max_x, tile.max_y), size)
        if not np.any(valid):
            return False
        tile.elevation = tile.elevation.copy()
        tile.elevation[valid] = sampled[valid]
        return True

    def apply_terrain_source_to_dem(self, terrain_source: TerrainSourceModel | None, dem: TerrainDEM) -> bool:
        changed = False
        for tile in dem.tiles.values():
            changed = self.apply_terrain_source_to_tile(terrain_source, tile) or changed
        return changed

    def apply_terrain_source_to_rgba_tile(
        self,
        terrain_source: TerrainSourceModel | None,
        z: int,
        x: int,
        y: int,
        rgba: np.ndarray,
    ) -> bool:
        dataset = self._dataset_for_source(terrain_source)
        if dataset is None:
            return False
        height, width = rgba.shape[:2]
        if height <= 0 or width <= 0 or height != width:
            return False

        sampled, valid = self._sample_dataset_grid(dataset, tile_bounds_mercator(z, x, y), height)
        if not np.any(valid):
            return False

        red, green, blue = _encode_terrain_rgb(sampled)
        rgba[..., 0][valid] = red[valid]
        rgba[..., 1][valid] = green[valid]
        rgba[..., 2][valid] = blue[valid]
        rgba[..., 3][valid] = 255
        return True


def _is_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict) and str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}:
            return True
    return isinstance(exc, FileNotFoundError)


class S3BackedDsmDatasetStore(DsmDatasetStore):
    def __init__(
        self,
        root_dir: Path,
        bucket: str,
        prefix: str = "",
        client: Any | None = None,
    ):
        super().__init__(root_dir)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    def _s3_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _key(self, suffix: str) -> str:
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _dataset_key_prefix(self, dataset_id: str) -> str:
        return self._key(f"datasets/{dataset_id}")

    def _descriptor_key(self, dataset_id: str) -> str:
        return f"{self._dataset_key_prefix(dataset_id)}/descriptor.json"

    def _pyramid_key(self, dataset_id: str) -> str:
        return f"{self._dataset_key_prefix(dataset_id)}/pyramid.npz"

    def _source_key(self, dataset_id: str, suffix: str) -> str:
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return f"{self._dataset_key_prefix(dataset_id)}/{dataset_id}{suffix}"

    def _upsert_local_index_entry(
        self,
        dataset_id: str,
        descriptor: DsmSourceDescriptorModel,
        suffix: str,
        *,
        source_crs: str | None = None,
    ) -> dict[str, Any]:
        suffix = suffix or ".tiff"
        file_path = self._dataset_file_path(dataset_id, suffix)
        pyramid_path = self._dataset_pyramid_path(dataset_id)
        entry = {
            "descriptor": descriptor.model_dump(mode="json"),
            "filePath": str(file_path),
            "pyramidPath": str(pyramid_path),
            "sourceCrs": source_crs or _resolve_source_crs(descriptor),
            "remoteDescriptorKey": self._descriptor_key(dataset_id),
            "remotePyramidKey": self._pyramid_key(dataset_id),
            "remoteSourceKey": self._source_key(dataset_id, suffix),
        }
        index = self._load_index()
        index["datasets"][dataset_id] = entry
        self._save_index(index)
        return entry

    def _load_remote_descriptor(self, dataset_id: str) -> DsmSourceDescriptorModel | None:
        client = self._s3_client()
        try:
            response = client.get_object(Bucket=self.bucket, Key=self._descriptor_key(dataset_id))
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                return None
            raise
        body = response["Body"].read()
        payload = json.loads(body.decode("utf-8"))
        descriptor = DsmSourceDescriptorModel.model_validate(payload)
        suffix = Path(descriptor.name or "surface.tiff").suffix or ".tiff"
        self._upsert_local_index_entry(dataset_id, descriptor, suffix)
        return descriptor

    def _ensure_local_dataset_files(self, dataset_id: str) -> None:
        index = self._load_index()
        entry = index.get("datasets", {}).get(dataset_id)
        if not isinstance(entry, dict):
            descriptor = self._load_remote_descriptor(dataset_id)
            if descriptor is None:
                raise KeyError(f"DSM dataset {dataset_id} was not found.")
            index = self._load_index()
            entry = index.get("datasets", {}).get(dataset_id)
        if not isinstance(entry, dict):
            raise KeyError(f"DSM dataset {dataset_id} was not found.")

        file_path = Path(entry["filePath"])
        pyramid_path = Path(entry["pyramidPath"])
        if file_path.exists() and pyramid_path.exists():
            return

        client = self._s3_client()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        pyramid_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_payload = client.get_object(Bucket=self.bucket, Key=entry["remoteSourceKey"])["Body"].read()
            file_path.write_bytes(file_payload)
        if not pyramid_path.exists():
            pyramid_payload = client.get_object(Bucket=self.bucket, Key=entry["remotePyramidKey"])["Body"].read()
            pyramid_path.write_bytes(pyramid_payload)

    def ingest_dataset(
        self,
        payload: bytes,
        original_name: str,
        source_descriptor: DsmSourceDescriptorModel,
    ) -> tuple[DsmSourceDescriptorModel, bool]:
        temp_file_path = self.root_dir / f".ingest-{hashlib.sha256(payload).hexdigest()}{Path(original_name or source_descriptor.name or 'surface.tiff').suffix or '.tiff'}"
        temp_file_path.write_bytes(payload)
        try:
            return self.ingest_dataset_file(
                temp_file_path,
                original_name,
                source_descriptor=source_descriptor,
            )
        finally:
            temp_file_path.unlink(missing_ok=True)

    def ingest_dataset_file(
        self,
        file_path: Path,
        original_name: str,
        *,
        source_descriptor: DsmSourceDescriptorModel | None = None,
        verified_sha256: str | None = None,
    ) -> tuple[DsmSourceDescriptorModel, bool]:
        digest = verified_sha256 or _sha256_for_path(file_path)
        source_descriptor = source_descriptor or derive_descriptor_from_path(file_path, original_name)
        suffix = Path(original_name or source_descriptor.name or "surface.tiff").suffix or ".tiff"
        with self._lock:
            existing_descriptor = self._load_remote_descriptor(digest)
            if existing_descriptor is not None:
                return existing_descriptor, True

            stored_descriptor, reused_existing = super().ingest_dataset_file(
                file_path,
                original_name,
                source_descriptor=source_descriptor,
                verified_sha256=digest,
            )
            if reused_existing:
                return stored_descriptor, True

            index = self._load_index()
            entry = index["datasets"][digest]
            file_path = Path(entry["filePath"])
            pyramid_path = Path(entry["pyramidPath"])
            client = self._s3_client()
            with file_path.open("rb") as source_handle:
                client.put_object(
                    Bucket=self.bucket,
                    Key=self._source_key(digest, suffix),
                    Body=source_handle,
                    ContentType="image/tiff",
                )
            with pyramid_path.open("rb") as pyramid_handle:
                client.put_object(
                    Bucket=self.bucket,
                    Key=self._pyramid_key(digest),
                    Body=pyramid_handle,
                    ContentType="application/octet-stream",
                )
            client.put_object(
                Bucket=self.bucket,
                Key=self._descriptor_key(digest),
                Body=stored_descriptor.model_dump_json().encode("utf-8"),
                ContentType="application/json",
            )
            self._upsert_local_index_entry(digest, stored_descriptor, suffix)
            return stored_descriptor, False

    def list_datasets(self) -> list[DsmSourceDescriptorModel]:
        client = self._s3_client()
        paginator = client.get_paginator("list_objects_v2")
        descriptors: list[DsmSourceDescriptorModel] = []
        prefix = self._key("datasets/")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []) or []:
                key = item.get("Key")
                if not isinstance(key, str) or not key.endswith("/descriptor.json"):
                    continue
                dataset_id = key.split("/")[-2]
                descriptor = self._load_remote_descriptor(dataset_id)
                if descriptor is not None:
                    descriptors.append(descriptor)
        return sorted(descriptors, key=lambda descriptor: descriptor.loadedAtIso, reverse=True)

    def get_dataset_descriptor(self, dataset_id: str) -> DsmSourceDescriptorModel | None:
        try:
            self._ensure_local_dataset_files(dataset_id)
        except KeyError:
            return None
        return super().get_dataset_descriptor(dataset_id)

    def _load_dataset(self, dataset_id: str) -> DsmDataset:
        cached = self._dataset_cache.get(dataset_id)
        if cached is not None:
            return cached
        with self._lock:
            self._ensure_local_dataset_files(dataset_id)
            return super()._load_dataset(dataset_id)


def create_dsm_dataset_store(root_dir: Path) -> DsmDatasetStore:
    mode = (os.environ.get("TERRAIN_SPLITTER_DSM_STORE_MODE") or "local").strip().lower()
    if mode == "s3":
        bucket = (os.environ.get("TERRAIN_SPLITTER_DSM_S3_BUCKET") or "").strip()
        if not bucket:
            raise RuntimeError("TERRAIN_SPLITTER_DSM_S3_BUCKET is required when TERRAIN_SPLITTER_DSM_STORE_MODE=s3.")
        prefix = os.environ.get("TERRAIN_SPLITTER_DSM_S3_PREFIX") or ""
        return S3BackedDsmDatasetStore(root_dir, bucket=bucket, prefix=prefix)
    return DsmDatasetStore(root_dir)
