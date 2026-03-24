from __future__ import annotations

from dataclasses import replace

from shapely.geometry import Polygon, box

import terrain_splitter.solver_frontier as solver_frontier_module
from terrain_splitter.costs import RegionObjective
from terrain_splitter.features import CellFeatures, FeatureField
from terrain_splitter.geometry import simplify_polygon_coverage
from terrain_splitter.grid import GridCell, GridData, GridEdge
from terrain_splitter.schemas import FlightParamsModel
from terrain_splitter.solver_frontier import (
    DEFAULT_DEPTH_LARGE,
    BoundaryStats,
    EvaluatedRegion,
    RootSplitTask,
    SolverContext,
    SubtreeSolveTask,
    _cell_lookup,
    _deserialize_partition_plan,
    _feature_lookup,
    _neighbor_lookup,
    _pareto_frontier,
    _plan_from_regions,
    _region_gate_diagnostics,
    _region_practical,
    _relaxed_region_failure_sets,
    _resolve_lambda_invoke_read_timeout_sec,
    _resolve_lambda_parallel_invocations,
    _resolve_nested_lambda_max_inflight,
    _resolve_nested_lambda_min_cells,
    _resolve_root_parallel_max_inflight,
    _score_relaxed_fallback_candidate,
    _serialize_partition_plan,
    _serialize_solver_context,
    _solve_region_recursive,
    _solve_root_split_branch_with_context,
    _solve_subtree_task_with_context,
    solve_root_split_branch_event,
    solve_subtree_task_event,
)
from terrain_splitter.solver_graphcut import solve_partition_hierarchy


def _toy_grid() -> GridData:
    cells = [
        GridCell(index=0, row=0, col=0, x=0, y=100, lng=0, lat=0, area_m2=10_000, terrain_z=100, polygon=box(0, 100, 100, 200)),
        GridCell(index=1, row=0, col=1, x=100, y=100, lng=0, lat=0, area_m2=10_000, terrain_z=105, polygon=box(100, 100, 200, 200)),
        GridCell(index=2, row=1, col=0, x=0, y=0, lng=0, lat=0, area_m2=10_000, terrain_z=120, polygon=box(0, 0, 100, 100)),
        GridCell(index=3, row=1, col=1, x=100, y=0, lng=0, lat=0, area_m2=10_000, terrain_z=125, polygon=box(100, 0, 200, 100)),
    ]
    edges = [
        GridEdge(a=0, b=1, shared_boundary_m=100),
        GridEdge(a=0, b=2, shared_boundary_m=100),
        GridEdge(a=1, b=3, shared_boundary_m=100),
        GridEdge(a=2, b=3, shared_boundary_m=100),
    ]
    return GridData(
        ring=[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)],
        polygon_mercator=box(0, 0, 200, 200),
        cells=cells,
        edges=edges,
        grid_step_m=100,
        area_m2=40_000,
    )


def _multimodal_grid() -> GridData:
    cell_size = 100.0
    cols = 4
    rows = 2
    cells: list[GridCell] = []
    edges: list[GridEdge] = []

    def cell_index(row: int, col: int) -> int:
        return row * cols + col

    for row in range(rows):
        for col in range(cols):
            x = col * cell_size
            y = (rows - 1 - row) * cell_size
            cells.append(
                GridCell(
                    index=cell_index(row, col),
                    row=row,
                    col=col,
                    x=x,
                    y=y,
                    lng=0,
                    lat=0,
                    area_m2=cell_size * cell_size,
                    terrain_z=100 + cell_index(row, col),
                    polygon=box(x, y, x + cell_size, y + cell_size),
                )
            )

    for row in range(rows):
        for col in range(cols):
            if col + 1 < cols:
                edges.append(GridEdge(a=cell_index(row, col), b=cell_index(row, col + 1), shared_boundary_m=cell_size))
            if row + 1 < rows:
                edges.append(GridEdge(a=cell_index(row, col), b=cell_index(row + 1, col), shared_boundary_m=cell_size))

    return GridData(
        ring=[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)],
        polygon_mercator=box(0, 0, cols * cell_size, rows * cell_size),
        cells=cells,
        edges=edges,
        grid_step_m=cell_size,
        area_m2=cols * rows * cell_size * cell_size,
    )


def test_solver_returns_coarse_to_fine_options_for_multimodal_grid() -> None:
    grid = _multimodal_grid()
    feature_field = FeatureField(
        cells=[
            CellFeatures(index=index, preferred_bearing_deg=(0 if (index % 4) < 2 else 90), slope_magnitude=0.25, break_strength=18, confidence=0.9, aspect_deg=(270 if (index % 4) < 2 else 0))
            for index in range(8)
        ],
        dominant_preferred_bearing_deg=45,
    )
    params = FlightParamsModel(payloadKind="camera", altitudeAGL=120, frontOverlap=70, sideOverlap=70, cameraKey="MAP61_17MM")
    solutions = solve_partition_hierarchy(grid, feature_field, params)
    assert solutions
    assert solutions[0].regionCount >= 2
    assert any(solution.isFirstPracticalSplit for solution in solutions)


def test_solver_frontier_contains_face_aligned_quality_solution() -> None:
    grid = _multimodal_grid()
    feature_field = FeatureField(
        cells=[
            CellFeatures(index=index, preferred_bearing_deg=(0 if (index % 4) < 2 else 90), slope_magnitude=0.25, break_strength=18, confidence=0.9, aspect_deg=(270 if (index % 4) < 2 else 0))
            for index in range(8)
        ],
        dominant_preferred_bearing_deg=45,
    )
    params = FlightParamsModel(payloadKind="camera", altitudeAGL=120, frontOverlap=70, sideOverlap=70, cameraKey="MAP61_17MM")
    solutions = solve_partition_hierarchy(grid, feature_field, params)
    assert any(solution.normalizedQualityCost < 0.1 for solution in solutions)
    assert any(
        sorted(round(region.bearingDeg) % 180 for region in solution.regions) == [0, 90]
        for solution in solutions
    )


def test_solver_parallel_root_path_matches_serial_frontier() -> None:
    grid = _toy_grid()
    feature_field = FeatureField(
        cells=[
            CellFeatures(index=0, preferred_bearing_deg=0, slope_magnitude=0.3, break_strength=18, confidence=0.9, aspect_deg=270),
            CellFeatures(index=1, preferred_bearing_deg=0, slope_magnitude=0.25, break_strength=18, confidence=0.85, aspect_deg=270),
            CellFeatures(index=2, preferred_bearing_deg=90, slope_magnitude=0.28, break_strength=18, confidence=0.88, aspect_deg=0),
            CellFeatures(index=3, preferred_bearing_deg=90, slope_magnitude=0.26, break_strength=18, confidence=0.86, aspect_deg=0),
        ],
        dominant_preferred_bearing_deg=45,
    )
    params = FlightParamsModel(payloadKind="camera", altitudeAGL=120, frontOverlap=70, sideOverlap=70, cameraKey="MAP61_17MM")

    serial = solve_partition_hierarchy(grid, feature_field, params, root_parallel_workers=0)
    parallel = solve_partition_hierarchy(grid, feature_field, params, root_parallel_workers=2)
    subtree_parallel = solve_partition_hierarchy(
        grid,
        feature_field,
        params,
        root_parallel_workers=2,
        root_parallel_granularity="subtree",
    )

    def summarize(solution_list):
        return [
            (
                solution.regionCount,
                round(solution.totalMissionTimeSec, 6),
                round(solution.normalizedQualityCost, 6),
                tuple(
                    (
                        round(region.areaM2, 6),
                        round(region.bearingDeg, 6),
                        tuple((round(lng, 8), round(lat, 8)) for lng, lat in region.ring),
                    )
                    for region in solution.regions
                ),
            )
            for solution in solution_list
        ]

    assert summarize(serial) == summarize(parallel)
    assert summarize(serial) == summarize(subtree_parallel)


def test_lambda_parallel_invocations_respects_explicit_max_inflight() -> None:
    assert _resolve_lambda_parallel_invocations(16, 8, 12) == 12
    assert _resolve_lambda_parallel_invocations(16, 8, 0) == 16
    assert _resolve_lambda_parallel_invocations(8, 4, None) == 4


def test_root_parallel_max_inflight_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("TERRAIN_SPLITTER_ROOT_PARALLEL_MAX_INFLIGHT", "24")
    assert _resolve_root_parallel_max_inflight(None) == 24


def test_lambda_invoke_read_timeout_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("TERRAIN_SPLITTER_LAMBDA_INVOKE_READ_TIMEOUT_SEC", "300")
    assert _resolve_lambda_invoke_read_timeout_sec(None) == 300


def test_nested_lambda_controls_read_environment(monkeypatch) -> None:
    monkeypatch.setenv("TERRAIN_SPLITTER_NESTED_LAMBDA_MIN_CELLS", "72")
    monkeypatch.setenv("TERRAIN_SPLITTER_NESTED_LAMBDA_MAX_INFLIGHT", "5")
    assert _resolve_nested_lambda_min_cells(None) == 72
    assert _resolve_nested_lambda_max_inflight(None) == 5


def test_subtree_worker_nested_lambda_matches_serial_frontier(monkeypatch) -> None:
    grid = _multimodal_grid()
    feature_field = FeatureField(
        cells=[
            CellFeatures(index=index, preferred_bearing_deg=(0 if (index % 4) < 2 else 90), slope_magnitude=0.25, break_strength=18, confidence=0.9, aspect_deg=(270 if (index % 4) < 2 else 0))
            for index in range(8)
        ],
        dominant_preferred_bearing_deg=45,
    )
    params = FlightParamsModel(payloadKind="camera", altitudeAGL=120, frontOverlap=70, sideOverlap=70, cameraKey="MAP61_17MM")
    context = SolverContext(
        grid=grid,
        feature_field=feature_field,
        params=params,
        root_area_m2=max(1.0, grid.area_m2),
        feature_lookup=_feature_lookup(feature_field),
        cell_lookup=_cell_lookup(grid),
        neighbors=_neighbor_lookup(grid),
        basic_line_length_scale=solver_frontier_module.DEFAULT_BASIC_LINE_LENGTH_SCALE,
        practical_line_length_scale=solver_frontier_module.DEFAULT_PRACTICAL_LINE_LENGTH_SCALE,
    )
    task = SubtreeSolveTask(cell_ids=tuple(sorted(context.cell_lookup)), depth=2)

    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "terrain-splitter-test")
    monkeypatch.setenv("TERRAIN_SPLITTER_ROOT_PARALLEL_MODE", "lambda")
    monkeypatch.setenv("TERRAIN_SPLITTER_ROOT_PARALLEL_GRANULARITY", "subtree")
    monkeypatch.setenv("TERRAIN_SPLITTER_ROOT_PARALLEL_WORKERS", "4")
    monkeypatch.setenv("TERRAIN_SPLITTER_NESTED_LAMBDA_MIN_CELLS", "1")
    monkeypatch.setenv("TERRAIN_SPLITTER_NESTED_LAMBDA_MAX_INFLIGHT", "4")

    nested_calls: dict[str, int] = {}

    def fake_solve_subtrees_via_lambda(tasks, context_arg, max_workers):
        nested_calls["task_count"] = len(tasks)
        nested_calls["max_workers"] = max_workers
        results: dict[int, dict[str, list]] = {}
        for index, side, subtask in tasks:
            caches = solver_frontier_module._make_solver_caches()
            perf = solver_frontier_module._make_perf()
            frontier = _solve_region_recursive(
                subtask.cell_ids,
                subtask.depth,
                context_arg,
                caches,
                perf,
                allow_nested_lambda_fanout=False,
            )
            results.setdefault(index, {})[side] = frontier
        return results, {"fake_nested_parallel": 1.0}

    monkeypatch.setattr(solver_frontier_module, "_solve_subtrees_via_lambda", fake_solve_subtrees_via_lambda)

    nested_frontier, _ = _solve_subtree_task_with_context(task, context)
    serial_frontier = _solve_region_recursive(
        task.cell_ids,
        task.depth,
        context,
        solver_frontier_module._make_solver_caches(),
        solver_frontier_module._make_perf(),
        allow_nested_lambda_fanout=False,
    )

    assert nested_calls["task_count"] > 0
    assert nested_calls["max_workers"] == 4
    assert [_serialize_partition_plan(plan) for plan in nested_frontier] == [
        _serialize_partition_plan(plan) for plan in serial_frontier
    ]


def test_lambda_root_split_worker_event_matches_direct_branch_solver() -> None:
    grid = _toy_grid()
    feature_field = FeatureField(
        cells=[
            CellFeatures(index=0, preferred_bearing_deg=0, slope_magnitude=0.3, break_strength=18, confidence=0.9, aspect_deg=270),
            CellFeatures(index=1, preferred_bearing_deg=0, slope_magnitude=0.25, break_strength=18, confidence=0.85, aspect_deg=270),
            CellFeatures(index=2, preferred_bearing_deg=90, slope_magnitude=0.28, break_strength=18, confidence=0.88, aspect_deg=0),
            CellFeatures(index=3, preferred_bearing_deg=90, slope_magnitude=0.26, break_strength=18, confidence=0.86, aspect_deg=0),
        ],
        dominant_preferred_bearing_deg=45,
        dominant_aspect_deg=315,
    )
    params = FlightParamsModel(payloadKind="camera", altitudeAGL=120, frontOverlap=70, sideOverlap=70, cameraKey="MAP61_17MM")
    context = SolverContext(
        grid=grid,
        feature_field=feature_field,
        params=params,
        root_area_m2=max(1.0, grid.area_m2),
        feature_lookup=_feature_lookup(feature_field),
        cell_lookup=_cell_lookup(grid),
        neighbors=_neighbor_lookup(grid),
        basic_line_length_scale=0.35,
        practical_line_length_scale=0.4,
    )
    task = RootSplitTask(
        left_ids=(0, 1),
        right_ids=(2, 3),
        boundary=BoundaryStats(shared_boundary_m=10.0, break_weight_sum=25.0),
        depth=1,
    )

    direct_frontier, _ = _solve_root_split_branch_with_context(task, context)
    worker_payload = {
        "context": _serialize_solver_context(context),
        "task": {
            "leftIds": list(task.left_ids),
            "rightIds": list(task.right_ids),
            "boundary": {"sharedBoundaryM": task.boundary.shared_boundary_m, "breakWeightSum": task.boundary.break_weight_sum},
            "depth": task.depth,
        },
    }
    worker_result = solve_root_split_branch_event(worker_payload)
    worker_frontier = [_deserialize_partition_plan(plan) for plan in worker_result["plans"]]

    assert [_serialize_partition_plan(plan) for plan in direct_frontier] == [
        _serialize_partition_plan(plan) for plan in worker_frontier
    ]


def test_lambda_subtree_worker_event_matches_direct_subtree_solver() -> None:
    grid = _toy_grid()
    feature_field = FeatureField(
        cells=[
            CellFeatures(index=0, preferred_bearing_deg=0, slope_magnitude=0.3, break_strength=18, confidence=0.9, aspect_deg=270),
            CellFeatures(index=1, preferred_bearing_deg=0, slope_magnitude=0.25, break_strength=18, confidence=0.85, aspect_deg=270),
            CellFeatures(index=2, preferred_bearing_deg=90, slope_magnitude=0.28, break_strength=18, confidence=0.88, aspect_deg=0),
            CellFeatures(index=3, preferred_bearing_deg=90, slope_magnitude=0.26, break_strength=18, confidence=0.86, aspect_deg=0),
        ],
        dominant_preferred_bearing_deg=45,
        dominant_aspect_deg=315,
    )
    params = FlightParamsModel(payloadKind="camera", altitudeAGL=120, frontOverlap=70, sideOverlap=70, cameraKey="MAP61_17MM")
    context = SolverContext(
        grid=grid,
        feature_field=feature_field,
        params=params,
        root_area_m2=max(1.0, grid.area_m2),
        feature_lookup=_feature_lookup(feature_field),
        cell_lookup=_cell_lookup(grid),
        neighbors=_neighbor_lookup(grid),
        basic_line_length_scale=0.35,
        practical_line_length_scale=0.4,
    )
    task = SubtreeSolveTask(cell_ids=(0, 1), depth=1)

    direct_frontier, _ = _solve_subtree_task_with_context(task, context)
    worker_payload = {
        "context": _serialize_solver_context(context),
        "task": {
            "cellIds": list(task.cell_ids),
            "depth": task.depth,
        },
    }
    worker_result = solve_subtree_task_event(worker_payload)
    worker_frontier = [_deserialize_partition_plan(plan) for plan in worker_result["plans"]]

    assert [_serialize_partition_plan(plan) for plan in direct_frontier] == [
        _serialize_partition_plan(plan) for plan in worker_frontier
    ]


def _mock_region(area_m2: float, mean_line_length_m: float, *, region_id: int, bearing_deg: float = 0.0) -> EvaluatedRegion:
    objective = RegionObjective(
        bearing_deg=bearing_deg,
        normalized_quality_cost=1.0,
        total_mission_time_sec=100.0,
        weighted_mean_mismatch_deg=5.0,
        area_m2=area_m2,
        convexity=0.92,
        compactness=1.4,
        boundary_break_alignment=0.0,
        flight_line_count=12,
        line_spacing_m=72.0,
        along_track_length_m=500.0,
        cross_track_width_m=300.0,
        fragmented_line_fraction=0.05,
        overflight_transit_fraction=0.0,
        short_line_fraction=0.1,
        mean_line_length_m=mean_line_length_m,
        median_line_length_m=max(mean_line_length_m, 260.0),
        mean_line_lift_m=40.0,
        p90_line_lift_m=80.0,
        max_line_lift_m=120.0,
        elevated_area_fraction=0.2,
        severe_lift_area_fraction=0.05,
    )
    return EvaluatedRegion(
        cell_ids=(region_id,),
        polygon=box(0, 0, 10, 10),
        ring=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
        objective=objective,
        score=objective.normalized_quality_cost,
        hard_invalid=False,
    )


def test_pareto_frontier_preserves_practical_plan_against_non_practical_dominator() -> None:
    root_area_m2 = 1_000.0
    baseline = _plan_from_regions((_mock_region(1_000.0, 600.0, region_id=1),), 0.0, 0.0)

    practical = _plan_from_regions(
        (
            _mock_region(500.0, 260.0, region_id=2, bearing_deg=0.0),
            _mock_region(500.0, 255.0, region_id=3, bearing_deg=90.0),
        ),
        0.0,
        0.0,
    )
    non_practical = _plan_from_regions(
        (
            _mock_region(500.0, 260.0, region_id=4, bearing_deg=0.0),
            _mock_region(500.0, 120.0, region_id=5, bearing_deg=90.0),
        ),
        0.0,
        0.0,
    )
    practical = practical.__class__(
        regions=practical.regions,
        quality_cost=5.0,
        mission_time_sec=1_000.0,
        weighted_mean_mismatch_deg=practical.weighted_mean_mismatch_deg,
        internal_boundary_m=practical.internal_boundary_m,
        break_weight_sum=practical.break_weight_sum,
        largest_region_fraction=practical.largest_region_fraction,
        mean_convexity=practical.mean_convexity,
        region_count=practical.region_count,
    )
    non_practical = non_practical.__class__(
        regions=non_practical.regions,
        quality_cost=4.0,
        mission_time_sec=900.0,
        weighted_mean_mismatch_deg=non_practical.weighted_mean_mismatch_deg,
        internal_boundary_m=non_practical.internal_boundary_m,
        break_weight_sum=non_practical.break_weight_sum,
        largest_region_fraction=non_practical.largest_region_fraction,
        mean_convexity=non_practical.mean_convexity,
        region_count=non_practical.region_count,
    )

    frontier = _pareto_frontier([baseline, practical, non_practical], root_area_m2, 0.4)
    assert any(plan.region_count == 2 and abs(plan.quality_cost - 5.0) < 1e-6 for plan in frontier)


def test_coverage_simplify_reduces_vertices_while_preserving_shared_boundary() -> None:
    shared = [
        (4, 0),
        (4, 1),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 3),
        (3, 3),
        (3, 4),
        (4, 4),
        (4, 5),
    ]
    left = Polygon([(0, 0)] + shared + [(0, 5), (0, 0)])
    right = Polygon(shared + [(8, 5), (8, 0), (4, 0)])

    simplified = simplify_polygon_coverage([left, right], 0.8)

    assert len(simplified) == 2
    assert len(list(simplified[0].exterior.coords)) < len(list(left.exterior.coords))
    assert len(list(simplified[1].exterior.coords)) < len(list(right.exterior.coords))

    shared_edge = simplified[0].boundary.intersection(simplified[1].boundary)
    assert shared_edge.length > 0
    assert shared_edge.geom_type in {"LineString", "MultiLineString"}


def test_relaxed_region_failure_sets_keep_convexity_and_20m_floor_hard() -> None:
    region = _mock_region(500.0, 260.0, region_id=9)
    objective = replace(
        region.objective,
        mean_line_length_m=18.0,
        convexity=0.5,
    )
    adjusted_region = region.__class__(
        cell_ids=region.cell_ids,
        polygon=region.polygon,
        ring=region.ring,
        objective=objective,
        score=region.score,
        hard_invalid=region.hard_invalid,
    )

    diagnostics = _region_gate_diagnostics(adjusted_region, 1_000.0, practical=False, line_length_scale=0.37)
    hard_failures, soft_failures = _relaxed_region_failure_sets(diagnostics, practical=False)

    assert "convexity" in hard_failures
    assert "mean_line_length_hard_floor_m" in hard_failures
    assert "mean_line_length_m" not in soft_failures


def test_relaxed_region_failure_sets_keep_practical_convexity_hard() -> None:
    region = _mock_region(500.0, 260.0, region_id=11)
    objective = replace(
        region.objective,
        convexity=0.6,
        mean_line_length_m=40.0,
    )
    adjusted_region = region.__class__(
        cell_ids=region.cell_ids,
        polygon=region.polygon,
        ring=region.ring,
        objective=objective,
        score=region.score,
        hard_invalid=region.hard_invalid,
    )

    diagnostics = _region_gate_diagnostics(adjusted_region, 1_000.0, practical=True, line_length_scale=0.45)
    hard_failures, soft_failures = _relaxed_region_failure_sets(diagnostics, practical=True)

    assert "convexity" in hard_failures
    assert "convexity" not in soft_failures


def test_relaxed_region_failure_sets_treat_scaled_line_length_as_soft() -> None:
    region = _mock_region(500.0, 260.0, region_id=10)
    objective = replace(
        region.objective,
        along_track_length_m=500.0,
        cross_track_width_m=1_200.0,
        mean_line_length_m=300.0,
    )
    adjusted_region = region.__class__(
        cell_ids=region.cell_ids,
        polygon=region.polygon,
        ring=region.ring,
        objective=objective,
        score=region.score,
        hard_invalid=region.hard_invalid,
    )

    diagnostics = _region_gate_diagnostics(adjusted_region, 1_000.0, practical=False, line_length_scale=0.37)
    hard_failures, soft_failures = _relaxed_region_failure_sets(diagnostics, practical=False)

    assert "mean_line_length_hard_floor_m" not in hard_failures
    assert "mean_line_length_m" in soft_failures


def test_region_practical_accepts_small_line_length_near_miss_when_other_gates_pass() -> None:
    region = _mock_region(500.0, 260.0, region_id=13)
    objective = replace(
        region.objective,
        cross_track_width_m=700.0,
        mean_line_length_m=260.0,
    )
    adjusted_region = region.__class__(
        cell_ids=region.cell_ids,
        polygon=region.polygon,
        ring=region.ring,
        objective=objective,
        score=region.score,
        hard_invalid=region.hard_invalid,
    )

    diagnostics = _region_gate_diagnostics(adjusted_region, 1_000.0, practical=True, line_length_scale=0.4)

    assert diagnostics["line_length_shortfall_m"] == 20.0
    assert diagnostics["line_length_near_miss_allowed"] is True
    assert _region_practical(adjusted_region, 1_000.0, 0.4)


def test_region_practical_rejects_large_line_length_miss_even_without_other_gates() -> None:
    region = _mock_region(500.0, 230.0, region_id=14)
    objective = replace(
        region.objective,
        cross_track_width_m=700.0,
        mean_line_length_m=230.0,
    )
    adjusted_region = region.__class__(
        cell_ids=region.cell_ids,
        polygon=region.polygon,
        ring=region.ring,
        objective=objective,
        score=region.score,
        hard_invalid=region.hard_invalid,
    )

    diagnostics = _region_gate_diagnostics(adjusted_region, 1_000.0, practical=True, line_length_scale=0.4)

    assert diagnostics["line_length_shortfall_m"] == 50.0
    assert diagnostics["line_length_near_miss_allowed"] is False
    assert not _region_practical(adjusted_region, 1_000.0, 0.4)


def test_solver_populates_debug_output_with_returned_plan_snapshots() -> None:
    grid = _multimodal_grid()
    feature_field = FeatureField(
        cells=[
            CellFeatures(index=index, preferred_bearing_deg=(0 if (index % 4) < 2 else 90), slope_magnitude=0.25, break_strength=18, confidence=0.9, aspect_deg=(270 if (index % 4) < 2 else 0))
            for index in range(8)
        ],
        dominant_preferred_bearing_deg=45,
    )
    params = FlightParamsModel(payloadKind="camera", altitudeAGL=120, frontOverlap=70, sideOverlap=70, cameraKey="MAP61_17MM")
    debug_output: dict[str, object] = {}

    solutions = solve_partition_hierarchy(grid, feature_field, params, debug_output=debug_output)

    assert solutions
    assert debug_output["solverSummary"]
    assert debug_output["rejectionDiagnostics"]
    assert debug_output["returnedPlans"]
    assert len(debug_output["returnedPlans"]) == len(solutions)


def test_large_root_default_depth_is_three() -> None:
    assert DEFAULT_DEPTH_LARGE == 3


def test_relaxed_fallback_score_penalizes_quality_and_time_regression() -> None:
    baseline = _plan_from_regions((_mock_region(1_000.0, 600.0, region_id=12),), 0.0, 0.0)
    same_quality_slower = replace(
        baseline,
        mission_time_sec=baseline.mission_time_sec + 500.0,
    )
    worse_quality_same_time = replace(
        baseline,
        quality_cost=baseline.quality_cost + 0.5,
    )

    baseline_score = _score_relaxed_fallback_candidate(
        baseline,
        baseline,
        soft_total_margin=0.0,
        soft_max_margin=0.0,
    )
    slower_score = _score_relaxed_fallback_candidate(
        same_quality_slower,
        baseline,
        soft_total_margin=0.0,
        soft_max_margin=0.0,
    )
    worse_quality_score = _score_relaxed_fallback_candidate(
        worse_quality_same_time,
        baseline,
        soft_total_margin=0.0,
        soft_max_margin=0.0,
    )

    assert slower_score > baseline_score
    assert worse_quality_score > baseline_score
