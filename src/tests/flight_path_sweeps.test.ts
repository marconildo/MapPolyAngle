import assert from "node:assert/strict";

import { build3DFlightPath } from "../components/MapFlightDirection/utils/geometry.ts";
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

runConcaveSweepGenerationCase();
runExplicitSweepOrderingCase();
runPlannedGeometryContractCase();
runDynamicTurnRegressionCase();
runSteepTurnaroundClimbRegressionCase();

console.log("flight_path_sweeps.test.ts: all assertions passed");
