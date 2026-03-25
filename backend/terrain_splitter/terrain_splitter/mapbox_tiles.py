from __future__ import annotations

import io
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .geometry import clamp, lnglat_to_mercator
from .schemas import TerrainSourceModel

logger = logging.getLogger("uvicorn.error")


def _terrain_rgb_to_elevation(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    return -10000.0 + (rgb[:, :, 0] * 256.0 * 256.0 + rgb[:, :, 1] * 256.0 + rgb[:, :, 2]) * 0.1


def tile_bounds_mercator(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    world = 2.0 * math.pi * 6378137.0
    tile_size = world / (2**z)
    min_x = -world / 2.0 + x * tile_size
    max_x = min_x + tile_size
    max_y = world / 2.0 - y * tile_size
    min_y = max_y - tile_size
    return min_x, min_y, max_x, max_y


def choose_grid_step_m(area_m2: float) -> float:
    return clamp(math.sqrt(max(area_m2, 1.0)) / 28.0, 24.0, 120.0)


def choose_terrain_zoom(grid_step_m: float) -> int:
    if grid_step_m <= 24:
        return 15
    if grid_step_m <= 40:
        return 14
    if grid_step_m <= 65:
        return 13
    return 12


def mercator_bounds_to_tile_range(min_x: float, min_y: float, max_x: float, max_y: float, z: int) -> tuple[range, range]:
    world = 2.0 * math.pi * 6378137.0
    tiles_per_axis = 2**z
    tile_span = world / tiles_per_axis
    x0 = int(math.floor((min_x + world / 2.0) / tile_span))
    x1 = int(math.floor((max_x + world / 2.0) / tile_span))
    y0 = int(math.floor((world / 2.0 - max_y) / tile_span))
    y1 = int(math.floor((world / 2.0 - min_y) / tile_span))
    x0 = max(0, min(tiles_per_axis - 1, x0))
    x1 = max(0, min(tiles_per_axis - 1, x1))
    y0 = max(0, min(tiles_per_axis - 1, y0))
    y1 = max(0, min(tiles_per_axis - 1, y1))
    return range(x0, x1 + 1), range(y0, y1 + 1)


def mapbox_token() -> str:
    token = os.environ.get("MAPBOX_TOKEN") or os.environ.get("VITE_MAPBOX_TOKEN") or os.environ.get("VITE_MAPBOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MAPBOX_TOKEN is required for the terrain splitter backend.")
    token = token.strip()
    if not token:
        raise RuntimeError("MAPBOX_TOKEN is required for the terrain splitter backend.")
    lowered = token.lower()
    if (
        "your_mapbox_token" in lowered
        or "your mapbox token" in lowered
        or token == "..."
        or not token.startswith(("pk.", "sk."))
    ):
        raise RuntimeError(
            "Terrain splitter backend is using a placeholder or invalid Mapbox token. "
            "Set MAPBOX_TOKEN (or VITE_MAPBOX_TOKEN) in the backend shell to a real Mapbox token before starting uvicorn."
        )
    return token


@dataclass(slots=True)
class TerrainTile:
    z: int
    x: int
    y: int
    elevation: np.ndarray
    min_x: float
    min_y: float
    max_x: float
    max_y: float


class TerrainDEM:
    def __init__(self, z: int, tiles: dict[tuple[int, int], TerrainTile]):
        self.z = z
        self.tiles = tiles

    def sample_mercator(self, x: float, y: float) -> float:
        world = 2.0 * math.pi * 6378137.0
        tiles_per_axis = 2**self.z
        tile_span = world / tiles_per_axis
        tx = int(math.floor((x + world / 2.0) / tile_span))
        ty = int(math.floor((world / 2.0 - y) / tile_span))
        tile = self.tiles.get((tx, ty))
        if tile is None:
            return float("nan")
        size = tile.elevation.shape[0]
        px = (x - tile.min_x) / max(1e-9, tile.max_x - tile.min_x) * (size - 1)
        py = (tile.max_y - y) / max(1e-9, tile.max_y - tile.min_y) * (size - 1)
        ix = int(math.floor(px))
        iy = int(math.floor(py))
        if ix < 0 or iy < 0 or ix >= size - 1 or iy >= size - 1:
            ix = max(0, min(size - 2, ix))
            iy = max(0, min(size - 2, iy))
        fx = px - ix
        fy = py - iy
        z00 = tile.elevation[iy, ix]
        z10 = tile.elevation[iy, ix + 1]
        z01 = tile.elevation[iy + 1, ix]
        z11 = tile.elevation[iy + 1, ix + 1]
        return float(
            z00 * (1.0 - fx) * (1.0 - fy)
            + z10 * fx * (1.0 - fy)
            + z01 * (1.0 - fx) * fy
            + z11 * fx * fy
        )


class TerrainTileCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, z: int, x: int, y: int) -> Path:
        return self.cache_dir / str(z) / str(x) / f"{y}.pngraw"

    def get_or_fetch(self, client: httpx.Client, token: str, z: int, x: int, y: int) -> bytes:
        path = self.path_for(z, x, y)
        if path.exists():
            return path.read_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw"
        response = client.get(url, params={"access_token": token}, timeout=30.0)
        response.raise_for_status()
        payload = response.content
        path.write_bytes(payload)
        return payload


def fetch_dem_for_ring(
    ring: list[tuple[float, float]],
    cache_dir: Path,
    grid_step_m: float | None = None,
    terrain_source: TerrainSourceModel | None = None,
    dsm_store: object | None = None,
) -> tuple[TerrainDEM, int]:
    mercator = [lnglat_to_mercator(lng, lat) for lng, lat in ring]
    xs = [coord[0] for coord in mercator]
    ys = [coord[1] for coord in mercator]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    padding = max(200.0, (grid_step_m or 40.0) * 2.0)
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding
    zoom = choose_terrain_zoom(grid_step_m or 40.0)
    xs_range, ys_range = mercator_bounds_to_tile_range(min_x, min_y, max_x, max_y, zoom)
    terrain_source_label = (
        f"{terrain_source.mode}:{terrain_source.datasetId}"
        if terrain_source is not None and terrain_source.datasetId
        else terrain_source.mode
        if terrain_source is not None
        else "mapbox"
    )
    logger.info(
        "[terrain-split-backend] dem fetch choose zoom=%d gridStepM=%.1f paddingM=%.1f tileCount=%d xTiles=%d..%d yTiles=%d..%d terrainSource=%s",
        zoom,
        float(grid_step_m or 40.0),
        padding,
        len(xs_range) * len(ys_range),
        xs_range.start,
        xs_range.stop - 1,
        ys_range.start,
        ys_range.stop - 1,
        terrain_source_label,
    )
    token = mapbox_token()
    cache = TerrainTileCache(cache_dir)
    tiles: dict[tuple[int, int], TerrainTile] = {}
    with httpx.Client(follow_redirects=True) as client:
        for x in xs_range:
            for y in ys_range:
                payload = cache.get_or_fetch(client, token, zoom, x, y)
                image = Image.open(io.BytesIO(payload))
                elevation = _terrain_rgb_to_elevation(image)
                bounds = tile_bounds_mercator(zoom, x, y)
                tiles[(x, y)] = TerrainTile(
                    z=zoom,
                    x=x,
                    y=y,
                    elevation=elevation,
                    min_x=bounds[0],
                    min_y=bounds[1],
                    max_x=bounds[2],
                    max_y=bounds[3],
                )
    dem = TerrainDEM(zoom, tiles)
    if dsm_store is not None:
        apply_fn = getattr(dsm_store, "apply_terrain_source_to_dem", None)
        if callable(apply_fn):
            apply_fn(terrain_source, dem)
    return dem, zoom
