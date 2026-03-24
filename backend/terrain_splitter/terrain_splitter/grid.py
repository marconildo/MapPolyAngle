from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon, box

from .geometry import mercator_to_lnglat, normalize_ring, ring_to_polygon_mercator
from .mapbox_tiles import TerrainDEM, choose_grid_step_m


@dataclass(slots=True)
class GridCell:
    index: int
    row: int
    col: int
    x: float
    y: float
    lng: float
    lat: float
    area_m2: float
    terrain_z: float
    polygon: Polygon


@dataclass(slots=True)
class GridEdge:
    a: int
    b: int
    shared_boundary_m: float


@dataclass(slots=True)
class GridData:
    ring: list[tuple[float, float]]
    polygon_mercator: Polygon
    cells: list[GridCell]
    edges: list[GridEdge]
    grid_step_m: float
    area_m2: float


def build_grid(
    ring: list[tuple[float, float]],
    dem: TerrainDEM,
    grid_step_m: float | None = None,
) -> GridData:
    normalized = normalize_ring(ring)
    polygon = ring_to_polygon_mercator(normalized)
    area_m2 = float(polygon.area)
    step = grid_step_m or choose_grid_step_m(area_m2)
    min_x, min_y, max_x, max_y = polygon.bounds
    cells: list[GridCell] = []
    cell_lookup: dict[tuple[int, int], int] = {}
    row = 0
    y = min_y + step * 0.5
    while y <= max_y:
        col = 0
        x = min_x + step * 0.5
        while x <= max_x:
            cell_box = box(x - step * 0.5, y - step * 0.5, x + step * 0.5, y + step * 0.5)
            clipped = cell_box.intersection(polygon)
            if not clipped.is_empty and clipped.area >= step * step * 0.12:
                lng, lat = mercator_to_lnglat(x, y)
                terrain_z = dem.sample_mercator(x, y)
                if np.isfinite(terrain_z):
                    cell = GridCell(
                        index=len(cells),
                        row=row,
                        col=col,
                        x=x,
                        y=y,
                        lng=lng,
                        lat=lat,
                        area_m2=float(clipped.area),
                        terrain_z=float(terrain_z),
                        polygon=clipped,
                    )
                    cell_lookup[(row, col)] = cell.index
                    cells.append(cell)
            x += step
            col += 1
        y += step
        row += 1

    edges: list[GridEdge] = []
    for cell in cells:
        for neighbor_key in ((cell.row + 1, cell.col), (cell.row, cell.col + 1)):
            neighbor_idx = cell_lookup.get(neighbor_key)
            if neighbor_idx is None:
                continue
            edges.append(GridEdge(a=cell.index, b=neighbor_idx, shared_boundary_m=step))

    return GridData(
        ring=normalized,
        polygon_mercator=polygon,
        cells=cells,
        edges=edges,
        grid_step_m=step,
        area_m2=area_m2,
    )
