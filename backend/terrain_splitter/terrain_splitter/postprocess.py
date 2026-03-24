from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .geometry import polygon_to_lnglat_ring
from .grid import GridData


@dataclass(slots=True)
class RegionGeometry:
    region_id: int
    label_idx: int
    cell_ids: list[int]
    polygon: Polygon
    ring: list[tuple[float, float]]


def connected_components_by_label(grid: GridData, labels: list[int]) -> list[tuple[int, list[int]]]:
    neighbors: dict[int, list[int]] = defaultdict(list)
    for edge in grid.edges:
        neighbors[edge.a].append(edge.b)
        neighbors[edge.b].append(edge.a)

    visited: set[int] = set()
    components: list[tuple[int, list[int]]] = []
    for cell in grid.cells:
        if cell.index in visited:
            continue
        label_idx = labels[cell.index]
        queue = deque([cell.index])
        visited.add(cell.index)
        component: list[int] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in neighbors.get(current, []):
                if neighbor in visited or labels[neighbor] != label_idx:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        components.append((label_idx, component))
    return components


def region_polygon_from_cells(grid: GridData, cell_ids: list[int]) -> Polygon:
    polygons = [grid.cells[cell_id].polygon for cell_id in cell_ids]
    merged = unary_union(polygons).buffer(0)
    if merged.geom_type == "Polygon":
        return merged
    if merged.geom_type == "MultiPolygon":
        largest = max(merged.geoms, key=lambda geom: geom.area)
        return largest.buffer(0)
    raise ValueError("Unable to build region polygon from grid cells.")


def build_region_geometries(grid: GridData, labels: list[int]) -> tuple[list[RegionGeometry], dict[int, int]]:
    components = connected_components_by_label(grid, labels)
    regions: list[RegionGeometry] = []
    cell_to_region: dict[int, int] = {}
    for region_id, (label_idx, cell_ids) in enumerate(components):
        polygon = region_polygon_from_cells(grid, cell_ids)
        ring = polygon_to_lnglat_ring(polygon)
        regions.append(
            RegionGeometry(
                region_id=region_id,
                label_idx=label_idx,
                cell_ids=cell_ids,
                polygon=polygon,
                ring=ring,
            )
        )
        for cell_id in cell_ids:
            cell_to_region[cell_id] = region_id
    return regions, cell_to_region


def compute_region_adjacency(
    grid: GridData,
    cell_to_region: dict[int, int],
    break_strength_by_cell: dict[int, float],
) -> dict[tuple[int, int], dict[str, float]]:
    adjacency: dict[tuple[int, int], dict[str, float]] = {}
    for edge in grid.edges:
        ra = cell_to_region.get(edge.a)
        rb = cell_to_region.get(edge.b)
        if ra is None or rb is None or ra == rb:
            continue
        key = (ra, rb) if ra < rb else (rb, ra)
        stats = adjacency.setdefault(key, {"sharedBoundaryM": 0.0, "breakWeight": 0.0})
        stats["sharedBoundaryM"] += edge.shared_boundary_m
        stats["breakWeight"] += edge.shared_boundary_m * (
            break_strength_by_cell.get(edge.a, 0.0) + break_strength_by_cell.get(edge.b, 0.0)
        ) * 0.5
    return adjacency
