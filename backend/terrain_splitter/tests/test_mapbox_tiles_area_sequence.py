from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image

from terrain_splitter.geometry import lnglat_to_mercator
from terrain_splitter.mapbox_tiles import (
    fetch_dem_for_rings,
    mercator_bounds_to_tile_range,
)


def _encode_terrain_png_bytes(value: int = 96) -> bytes:
    rgba = np.full((256, 256, 3), value, dtype=np.uint8)
    image = Image.fromarray(rgba)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _rectangle(min_lng: float, min_lat: float, width_deg: float, height_deg: float) -> list[tuple[float, float]]:
    return [
        (min_lng, min_lat),
        (min_lng + width_deg, min_lat),
        (min_lng + width_deg, min_lat + height_deg),
        (min_lng, min_lat + height_deg),
        (min_lng, min_lat),
    ]


def test_fetch_dem_for_rings_preloads_per_area_tiles_and_lazy_loads_gap(monkeypatch, tmp_path: Path) -> None:
    png_payload = _encode_terrain_png_bytes()

    class _FakeTerrainTileCache:
        requests: list[tuple[int, int, int]] = []

        def __init__(self, *_args, **_kwargs):
            pass

        def get_or_fetch(self, _client, _token: str, z: int, x: int, y: int) -> bytes:
            self.__class__.requests.append((z, x, y))
            return png_payload

    monkeypatch.setattr("terrain_splitter.mapbox_tiles.TerrainTileCache", _FakeTerrainTileCache)
    monkeypatch.setattr("terrain_splitter.mapbox_tiles.mapbox_token", lambda: "test-token")

    ring_a = _rectangle(7.0000, 47.0000, 0.0018, 0.0012)
    ring_b = _rectangle(7.5000, 47.5000, 0.0018, 0.0012)
    grid_step_m = 40.0
    padding_m = max(200.0, grid_step_m * 2.0)

    dem, zoom = fetch_dem_for_rings(
        [ring_a, ring_b],
        tmp_path / "cache",
        grid_step_m=grid_step_m,
        lazy_load_missing=True,
    )

    def _tile_coords_for_ring(ring: list[tuple[float, float]]) -> set[tuple[int, int]]:
        mercator = [lnglat_to_mercator(lng, lat) for lng, lat in ring]
        xs = [coord[0] for coord in mercator]
        ys = [coord[1] for coord in mercator]
        xs_range, ys_range = mercator_bounds_to_tile_range(
            min(xs) - padding_m,
            min(ys) - padding_m,
            max(xs) + padding_m,
            max(ys) + padding_m,
            zoom,
        )
        return {(x, y) for x in xs_range for y in ys_range}

    union_tile_coords = _tile_coords_for_ring(ring_a) | _tile_coords_for_ring(ring_b)
    combined_mercator = [lnglat_to_mercator(lng, lat) for ring in (ring_a, ring_b) for lng, lat in ring]
    xs = [coord[0] for coord in combined_mercator]
    ys = [coord[1] for coord in combined_mercator]
    combined_xs_range, combined_ys_range = mercator_bounds_to_tile_range(
        min(xs) - padding_m,
        min(ys) - padding_m,
        max(xs) + padding_m,
        max(ys) + padding_m,
        zoom,
    )
    combined_bbox_tile_count = len(combined_xs_range) * len(combined_ys_range)

    assert set(dem.tiles.keys()) == union_tile_coords
    assert len(dem.tiles) < combined_bbox_tile_count

    before_lazy_count = len(dem.tiles)
    midpoint_x, midpoint_y = lnglat_to_mercator(7.25, 47.25)
    sampled = dem.sample_mercator(midpoint_x, midpoint_y)

    assert math.isfinite(sampled)
    assert len(dem.tiles) > before_lazy_count
