import assert from "node:assert/strict";

import { build3DFlightPath, mergeContiguous3DPathSegmentsForRender, sampleCameraPositionsOnFlightPath, sampleCameraPositionsOnPlannedFlightGeometry } from "../components/MapFlightDirection/utils/geometry.ts";
import { generateFlightLinesForPolygon } from "../components/MapFlightDirection/utils/mapbox-layers.ts";
import { generatePlannedFlightGeometryForPolygon } from "../flight/plannedGeometry.ts";
import { haversineDistance } from "../flight/geometry.ts";
import type { TerrainTile } from "../domain/types.ts";

type Line2D = [number, number][];
type Line3D = [number, number, number][];

function stripAltitude(line: Line3D): Line2D {
  return line.map(([lng, lat]) => [lng, lat]);
}

function headingDeg(start: [number, number], end: [number, number]) {
  return (Math.atan2(end[1] - start[1], end[0] - start[0]) * 180) / Math.PI;
}

function headingDeltaDeg(aDeg: number, bDeg: number) {
  return ((bDeg - aDeg + 180) % 360 + 360) % 360 - 180;
}

function toLocalPoint(point: [number, number], reference: [number, number]) {
  const dLat = ((point[1] - reference[1]) * Math.PI) / 180;
  const dLng = ((point[0] - reference[0]) * Math.PI) / 180;
  const cosLat = Math.cos((reference[1] * Math.PI) / 180);
  return {
    north: dLat * 6378137,
    east: dLng * 6378137 * cosLat,
  };
}

function projectOntoSweepFrame(
  point: [number, number],
  sweepStart: [number, number],
  nextSweepPoint: [number, number],
) {
  const pointLocal = toLocalPoint(point, sweepStart);
  const nextLocal = toLocalPoint(nextSweepPoint, sweepStart);
  const sweepLength = Math.hypot(nextLocal.north, nextLocal.east);
  const forward = {
    north: nextLocal.north / sweepLength,
    east: nextLocal.east / sweepLength,
  };
  const right = {
    north: -forward.east,
    east: forward.north,
  };
  return {
    along: pointLocal.north * forward.north + pointLocal.east * forward.east,
    cross: pointLocal.north * right.north + pointLocal.east * right.east,
  };
}

function pointInRing(lng: number, lat: number, ring: [number, number][]) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersect = ((yi > lat) !== (yj > lat)) &&
      (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function destination(start: [number, number], bearingDeg: number, distanceM: number): [number, number] {
  const bearingRad = (bearingDeg * Math.PI) / 180;
  const radiusM = 6371000;
  const angularDistance = distanceM / radiusM;
  const lat1 = (start[1] * Math.PI) / 180;
  const lng1 = (start[0] * Math.PI) / 180;

  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(angularDistance) +
      Math.cos(lat1) * Math.sin(angularDistance) * Math.cos(bearingRad),
  );
  const lng2 =
    lng1 +
    Math.atan2(
      Math.sin(bearingRad) * Math.sin(angularDistance) * Math.cos(lat1),
      Math.cos(angularDistance) - Math.sin(lat1) * Math.sin(lat2),
    );

  return [(lng2 * 180) / Math.PI, (lat2 * 180) / Math.PI];
}

function rectangleRing(
  center: [number, number],
  widthM: number,
  heightM: number,
  bearingDeg: number,
): [number, number][] {
  const halfHeightM = heightM / 2;
  const halfWidthM = widthM / 2;
  const topCenter = destination(center, bearingDeg, halfHeightM);
  const bottomCenter = destination(center, bearingDeg + 180, halfHeightM);
  const topLeft = destination(topCenter, bearingDeg - 90, halfWidthM);
  const topRight = destination(topCenter, bearingDeg + 90, halfWidthM);
  const bottomRight = destination(bottomCenter, bearingDeg + 90, halfWidthM);
  const bottomLeft = destination(bottomCenter, bearingDeg - 90, halfWidthM);
  return [topLeft, topRight, bottomRight, bottomLeft, topLeft];
}

function straightSegments(path: Line3D[]): Line3D[] {
  return path.filter((_, index) => index % 2 === 0);
}

function encodeTerrainRgb(size: number, elevationForPixel: (row: number, col: number) => number) {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const elevationM = elevationForPixel(row, col);
      const encoded = Math.max(0, Math.min(16777215, Math.round((elevationM + 10000) * 10)));
      const offset = (row * size + col) * 4;
      out[offset] = (encoded >> 16) & 255;
      out[offset + 1] = (encoded >> 8) & 255;
      out[offset + 2] = encoded & 255;
      out[offset + 3] = 255;
    }
  }
  return out;
}

function lngLatToTile(lng: number, lat: number, zoom: number) {
  const tilesPerAxis = 2 ** zoom;
  const x = Math.floor(((lng + 180) / 360) * tilesPerAxis);
  const latRad = (lat * Math.PI) / 180;
  const y = Math.floor(
    ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * tilesPerAxis,
  );
  return { x, y };
}

function createSteepFixtureTile(): TerrainTile {
  const zoom = 14;
  const size = 256;
  const { x, y } = lngLatToTile(0.01, 0.003, zoom);
  return {
    z: zoom,
    x,
    y,
    width: size,
    height: size,
    data: encodeTerrainRgb(size, (_row, col) => 120 + col * 10),
  };
}

function createUniformFixtureTile(elevationM: number): TerrainTile {
  const zoom = 14;
  const size = 256;
  const { x, y } = lngLatToTile(0.01, 0.003, zoom);
  return {
    z: zoom,
    x,
    y,
    width: size,
    height: size,
    data: encodeTerrainRgb(size, () => elevationM),
  };
}

function runConcaveSweepGenerationCase() {
  const concaveRing: [number, number][] = [
    [0, 0],
    [0.02, 0],
    [0.02, 0.02],
    [0.012, 0.02],
    [0.012, 0.008],
    [0.008, 0.008],
    [0.008, 0.02],
    [0, 0.02],
    [0, 0],
  ];

  const { flightLines, sweepIndices } = generateFlightLinesForPolygon(concaveRing, 90, 300);
  assert.ok(flightLines.length > 0, "concave sweep case should generate clipped flight lines");
  assert.ok(
    sweepIndices.some((value, index) => index > 0 && sweepIndices[index - 1] === value),
    "concave polygon should preserve repeated sweep ids for split sweep fragments",
  );
}

function runExplicitSweepOrderingCase() {
  const lines: number[][][] = [
    [[0.0, 0.0], [0.005, 0.0]],
    [[0.010, 0.0002], [0.015, 0.0002]],
    [[0.010, 0.0008], [0.015, 0.0008]],
  ];
  const sweepIndices = [0, 0, 1];
  const lineSpacing = 40;

  const withSweepIds = straightSegments(
    build3DFlightPath(lines, [], lineSpacing, { altitudeAGL: 100 }, sweepIndices),
  );
  assert.equal(withSweepIds.length, 2, "same-sweep fragments should merge into a single flown sweep");
  assert.deepEqual(
    stripAltitude(withSweepIds[0]),
    [
      ...lines[0],
      lines[1][0],
      lines[1][1],
    ] as Line2D,
    "same-sweep fragments should stay in one forward traversal instead of alternating between them",
  );
  assert.deepEqual(
    stripAltitude(withSweepIds[1]),
    [...lines[2]].reverse() as Line2D,
    "the next distinct sweep should still alternate direction",
  );
}

function runPlannedGeometryContractCase() {
  const squareRing: [number, number][] = [
    [8.0, 47.0],
    [8.004, 47.0],
    [8.004, 47.004],
    [8.0, 47.004],
    [8.0, 47.0],
  ];

  const geometry = generatePlannedFlightGeometryForPolygon(squareRing, 0, 40, {
    payloadKind: "camera",
    altitudeAGL: 80,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "ILX_LR1_INSPECT_85MM",
  });

  assert.ok(geometry.flightLines.length > 0, "planned geometry should retain raw clipped fragments");
  assert.ok(geometry.sweepLines.length > 1, "planned geometry should expose merged per-sweep lines");
  assert.equal(
    geometry.gridPoints.length,
    geometry.sweepLines.length * 4,
    "planned geometry should expose the mission-style 4-point grid cadence per sweep",
  );
  assert.equal(
    geometry.connectedLines.length,
    geometry.sweepLines.length * 2 - 1,
    "planned geometry should alternate sweep and turnaround segments in flown order",
  );
  assert.equal(
    geometry.turnBlocks.length,
    geometry.sweepLines.length - 1,
    "planned geometry should carry one explicit turn block per turnaround",
  );
}

function runDynamicTurnRegressionCase() {
  const diagonalRing: [number, number][] = [
    [8.0, 47.0],
    [8.006, 47.0],
    [8.006, 47.004],
    [8.004, 47.005],
    [8.001, 47.004],
    [8.0, 47.0],
  ];

  const geometry = generatePlannedFlightGeometryForPolygon(diagonalRing, 310, 38.4, {
    payloadKind: "camera",
    altitudeAGL: 80,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "ILX_LR1_INSPECT_85MM",
  });

  const turnaroundLengths = geometry.connectedLines
    .filter((_, index) => index % 2 === 1)
    .map((line) => line.slice(1).reduce((sum, point, index) => sum + haversineDistance(line[index] as [number, number], point as [number, number]), 0));

  assert.ok(turnaroundLengths.length > 0, "diagonal survey case should generate turnaround segments");
  assert.ok(
    turnaroundLengths.every((length) => length < 350),
    "turnaround preview should complete one turn block instead of orbiting into repeated circles",
  );

  for (let index = 1; index < geometry.connectedLines.length - 1; index += 2) {
    const turnaroundEnd = geometry.connectedLines[index][geometry.connectedLines[index].length - 1];
    const nextSweepStart = geometry.connectedLines[index + 1][0];
    assert.deepEqual(
      turnaroundEnd,
      nextSweepStart,
      "completed turnaround blocks should reconnect exactly onto the next sweep",
    );

    const turnaround = geometry.connectedLines[index];
    const nextSweep = geometry.connectedLines[index + 1];
    const turnaroundHeadingDeg = headingDeg(
      turnaround[turnaround.length - 2] as [number, number],
      turnaround[turnaround.length - 1] as [number, number],
    );
    const sweepHeadingDeg = headingDeg(nextSweep[0] as [number, number], nextSweep[1] as [number, number]);
    assert.ok(
      Math.abs(headingDeltaDeg(turnaroundHeadingDeg, sweepHeadingDeg)) < 12,
      "completed turnaround blocks should settle onto the next sweep without a visible heading notch",
    );
  }
}

function runSteepTurnaroundClimbRegressionCase() {
  const steepTile = createSteepFixtureTile();
  const squareRing: [number, number][] = [
    [0.001, 0.001],
    [0.019, 0.001],
    [0.019, 0.005],
    [0.001, 0.005],
    [0.001, 0.001],
  ];

  const geometry = generatePlannedFlightGeometryForPolygon(squareRing, 0, 80, {
    payloadKind: "camera",
    altitudeAGL: 80,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "ILX_LR1_INSPECT_85MM",
  });
  const path3D = build3DFlightPath(
    geometry,
    [steepTile],
    geometry.lineSpacing,
    { altitudeAGL: 80, mode: "min-clearance", minClearance: 60, preconnected: true },
  );

  const connectedTurnarounds = geometry.connectedLines.filter((_, index) => index % 2 === 1);
  const liftedTurnarounds = path3D.filter((_, index) => index % 2 === 1);

  assert.equal(
    liftedTurnarounds.length,
    connectedTurnarounds.length,
    "lifted path should preserve turnaround segment count",
  );
  assert.ok(
    liftedTurnarounds.some((segment, index) => segment.length > connectedTurnarounds[index].length + 30),
    "steep sweep-to-sweep altitude deltas should insert extra turnaround coil points in 3D",
  );
}

function runLargeAreaTurnaroundDensityRegressionCase() {
  const largeRing = rectangleRing([16.37, 48.21], 8000, 25000, 15);
  const geometry = generatePlannedFlightGeometryForPolygon(largeRing, 28, 45, {
    payloadKind: "camera",
    altitudeAGL: 120,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "MAP61_17MM",
  });

  const turnaroundPointCounts = geometry.connectedLines
    .filter((_, index) => index % 2 === 1)
    .map((line) => line.length);

  assert.ok(turnaroundPointCounts.length > 0, "large area regression case should generate turnaround segments");
  assert.ok(
    turnaroundPointCounts.every((count) => count < 1200),
    "turnaround previews should not replay the full sweep and explode point counts on large areas",
  );
}

function runFinalSweepReconnectRegressionCase() {
  const concaveRing: [number, number][] = [
    [8.0, 47.0],
    [8.012, 47.0],
    [8.012, 47.012],
    [8.008, 47.012],
    [8.008, 47.004],
    [8.004, 47.004],
    [8.004, 47.012],
    [8.0, 47.012],
    [8.0, 47.0],
  ];

  const geometry = generatePlannedFlightGeometryForPolygon(concaveRing, 5, 55, {
    payloadKind: "camera",
    altitudeAGL: 90,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "ILX_LR1_INSPECT_85MM",
  });

  const lastTurnaroundIndex = geometry.connectedLines.length - 2;
  const lastTurnaround = geometry.connectedLines[lastTurnaroundIndex];
  const lastSweep = geometry.connectedLines[lastTurnaroundIndex + 1];
  assert.ok(lastTurnaroundIndex >= 1 && lastTurnaroundIndex % 2 === 1, "regression case should include a final turnaround before the last sweep");
  assert.ok(lastTurnaround && lastSweep, "regression case should produce both the final turnaround and last sweep");
  assert.ok(
    haversineDistance(
      lastTurnaround[lastTurnaround.length - 1] as [number, number],
      lastSweep[0] as [number, number],
    ) < 1e-6,
    "the final turnaround should reconnect exactly onto the stitched last sweep",
  );
}

function runSweepRunInPreservationRegressionCase() {
  const largeRing = rectangleRing([16.37, 48.21], 12000, 40000, 15);
  const geometry = generatePlannedFlightGeometryForPolygon(largeRing, 28, 45, {
    payloadKind: "camera",
    altitudeAGL: 120,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "MAP61_17MM",
  });

  const maxStartTrimM = geometry.sweepLines.reduce((maxTrimM, sweepLine, sweepIndex) => {
    const connectedSweep = geometry.connectedLines[sweepIndex * 2];
    if (!connectedSweep) return maxTrimM;
    return Math.max(
      maxTrimM,
      haversineDistance(
        sweepLine[0] as [number, number],
        connectedSweep[0] as [number, number],
      ),
    );
  }, 0);

  assert.ok(
    maxStartTrimM < 6,
    "connecting turnarounds should not trim away long sweep run-ins before image triggering starts",
  );
}

function runConnectedSegmentContinuityRegressionCase() {
  const largeRing = rectangleRing([16.37, 48.21], 12000, 40000, 15);
  const geometry = generatePlannedFlightGeometryForPolygon(largeRing, 28, 45, {
    payloadKind: "camera",
    altitudeAGL: 120,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "MAP61_17MM",
  });

  for (let lineIndex = 0; lineIndex < geometry.connectedLines.length - 1; lineIndex += 1) {
    const currentLine = geometry.connectedLines[lineIndex];
    const nextLine = geometry.connectedLines[lineIndex + 1];
    assert.ok(
      haversineDistance(
        currentLine[currentLine.length - 1] as [number, number],
        nextLine[0] as [number, number],
      ) < 1e-6,
      "adjacent connected path segments should share exact boundary points so run-ins are not dropped",
    );
  }
}

function runTurnaroundTailMonotonicRegressionCase() {
  const ring = rectangleRing([16.37, 48.21], 1500, 2500, 20);
  const geometry = generatePlannedFlightGeometryForPolygon(ring, 287.6, 60, {
    payloadKind: "camera",
    altitudeAGL: 120,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "MAP61_17MM",
  });

  for (let lineIndex = 1; lineIndex < geometry.connectedLines.length - 1; lineIndex += 2) {
    const turnaround = geometry.connectedLines[lineIndex];
    const nextSweep = geometry.connectedLines[lineIndex + 1];
    assert.ok(nextSweep.length >= 2, "regression case should provide the next sweep direction");

    const tailFrames = turnaround
      .slice(-4)
      .map((point) => projectOntoSweepFrame(point as [number, number], nextSweep[0] as [number, number], nextSweep[1] as [number, number]));

    for (let frameIndex = 1; frameIndex < tailFrames.length; frameIndex += 1) {
      assert.ok(
        tailFrames[frameIndex].along >= tailFrames[frameIndex - 1].along - 1e-6,
        "turnaround tail should not step backward along the next sweep axis before rejoining the line",
      );
    }
  }
}

function runTurnaroundTailCrossTrackBlendRegressionCase() {
  const ring = rectangleRing([16.37, 48.21], 1500, 2500, 20);
  const geometry = generatePlannedFlightGeometryForPolygon(ring, 283.2, 60, {
    payloadKind: "camera",
    altitudeAGL: 120,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "MAP61_17MM",
  });

  for (let lineIndex = 1; lineIndex < geometry.connectedLines.length - 1; lineIndex += 2) {
    const turnaround = geometry.connectedLines[lineIndex];
    const nextSweep = geometry.connectedLines[lineIndex + 1];
    const tailFrames = turnaround
      .slice(-8)
      .map((point) => projectOntoSweepFrame(point as [number, number], nextSweep[0] as [number, number], nextSweep[1] as [number, number]));

    for (let frameIndex = 1; frameIndex < tailFrames.length; frameIndex += 1) {
      assert.ok(
        Math.abs(tailFrames[frameIndex].cross) <= Math.abs(tailFrames[frameIndex - 1].cross) + 1e-6,
        "turnaround tail cross-track offset should decay monotonically into the next sweep",
      );
    }

    assert.ok(
      Math.abs(tailFrames[tailFrames.length - 2].cross) < 0.1,
      "the penultimate turnaround point should already be essentially on the next sweep corridor",
    );
  }
}

function runRenderPathMergeRegressionCase() {
  const concaveRing: [number, number][] = [
    [8.0, 47.0],
    [8.012, 47.0],
    [8.012, 47.012],
    [8.008, 47.012],
    [8.008, 47.004],
    [8.004, 47.004],
    [8.004, 47.012],
    [8.0, 47.012],
    [8.0, 47.0],
  ];

  const geometry = generatePlannedFlightGeometryForPolygon(concaveRing, 5, 55, {
    payloadKind: "camera",
    altitudeAGL: 90,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "ILX_LR1_INSPECT_85MM",
  });
  const path3D = build3DFlightPath(
    geometry,
    [],
    geometry.lineSpacing,
    { altitudeAGL: 90, mode: "legacy", preconnected: true },
  );
  const mergedRenderPaths = mergeContiguous3DPathSegmentsForRender(path3D);

  assert.equal(
    mergedRenderPaths.length,
    1,
    "render-time path merging should collapse contiguous sweep/turn segments into a continuous 3D path",
  );
  assert.deepEqual(mergedRenderPaths[0][0], path3D[0][0], "merged render path should preserve the mission start point");
  assert.deepEqual(
    mergedRenderPaths[0][mergedRenderPaths[0].length - 1],
    path3D[path3D.length - 1][path3D[path3D.length - 1].length - 1],
    "merged render path should preserve the mission end point",
  );
}

function runSweepFragmentSamplingEquivalenceCase() {
  const concaveRing: [number, number][] = [
    [8.0, 47.0],
    [8.012, 47.0],
    [8.012, 47.012],
    [8.008, 47.012],
    [8.008, 47.004],
    [8.004, 47.004],
    [8.004, 47.012],
    [8.0, 47.012],
    [8.0, 47.0],
  ];

  const geometry = generatePlannedFlightGeometryForPolygon(concaveRing, 90, 55, {
    payloadKind: "camera",
    altitudeAGL: 90,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "ILX_LR1_INSPECT_85MM",
  });
  const path3D = build3DFlightPath(
    geometry,
    [],
    geometry.lineSpacing,
    { altitudeAGL: 90, mode: "legacy", preconnected: true },
  );

  const fragmentAware = sampleCameraPositionsOnPlannedFlightGeometry(geometry, path3D, 40)
    .filter(([lng, lat]) => pointInRing(lng, lat, concaveRing));
  const filteredPreviousBehavior = sampleCameraPositionsOnFlightPath(path3D, 40, { includeTurns: false })
    .filter(([lng, lat]) => pointInRing(lng, lat, concaveRing));

  assert.equal(
    fragmentAware.length,
    filteredPreviousBehavior.length,
    "fragment-aware sweep sampling should emit the same number of in-polygon camera positions as sample-then-filter",
  );

  filteredPreviousBehavior.forEach((expected, index) => {
    const actual = fragmentAware[index];
    assert.ok(actual, "fragment-aware sweep sampling should preserve camera position ordering");
    assert.ok(Math.abs(actual[0] - expected[0]) < 1e-10, "camera longitude should match previous sample/filter behavior");
    assert.ok(Math.abs(actual[1] - expected[1]) < 1e-10, "camera latitude should match previous sample/filter behavior");
    assert.ok(Math.abs(actual[2] - expected[2]) < 1e-6, "camera altitude should match previous sample/filter behavior");
    assert.ok(Math.abs(headingDeltaDeg(actual[3], expected[3])) < 1e-6, "camera yaw should match previous sample/filter behavior");
  });
}

function runPreconnectedPathCacheTerrainInvalidationCase() {
  const ring = rectangleRing([0.01, 0.003], 180, 120, 18);
  const geometry = generatePlannedFlightGeometryForPolygon(ring, 18, 35, {
    payloadKind: "camera",
    altitudeAGL: 100,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "ILX_LR1_INSPECT_85MM",
  });

  const lowTerrainPath = build3DFlightPath(
    geometry,
    [createUniformFixtureTile(120)],
    geometry.lineSpacing,
    { altitudeAGL: 100, mode: "legacy", preconnected: true },
  );
  const highTerrainPath = build3DFlightPath(
    geometry,
    [createUniformFixtureTile(620)],
    geometry.lineSpacing,
    { altitudeAGL: 100, mode: "legacy", preconnected: true },
  );

  const lowAltitude = lowTerrainPath[0]?.[0]?.[2];
  const highAltitude = highTerrainPath[0]?.[0]?.[2];
  assert.ok(Number.isFinite(lowAltitude), "low-terrain path should produce a finite altitude");
  assert.ok(Number.isFinite(highAltitude), "high-terrain path should produce a finite altitude");
  assert.notEqual(lowTerrainPath, highTerrainPath, "terrain changes should invalidate the cached preconnected path");
  assert.ok(
    (highAltitude as number) - (lowAltitude as number) > 400,
    "identical z/x/y tiles with different terrain payloads should recompute flight altitudes",
  );
}

runConcaveSweepGenerationCase();
runExplicitSweepOrderingCase();
runPlannedGeometryContractCase();
runDynamicTurnRegressionCase();
runSteepTurnaroundClimbRegressionCase();
runLargeAreaTurnaroundDensityRegressionCase();
runFinalSweepReconnectRegressionCase();
runSweepRunInPreservationRegressionCase();
runConnectedSegmentContinuityRegressionCase();
runTurnaroundTailMonotonicRegressionCase();
runTurnaroundTailCrossTrackBlendRegressionCase();
runRenderPathMergeRegressionCase();
runSweepFragmentSamplingEquivalenceCase();
runPreconnectedPathCacheTerrainInvalidationCase();

console.log("flight_path_sweeps.test.ts: all assertions passed");
