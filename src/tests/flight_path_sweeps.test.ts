import assert from "node:assert/strict";

import { build3DFlightPath } from "../components/MapFlightDirection/utils/geometry.ts";
import { generateFlightLinesForPolygon } from "../components/MapFlightDirection/utils/mapbox-layers.ts";

type Line2D = [number, number][];
type Line3D = [number, number, number][];

function stripAltitude(line: Line3D): Line2D {
  return line.map(([lng, lat]) => [lng, lat]);
}

function straightSegments(path: Line3D[]): Line3D[] {
  return path.filter((_, index) => index % 2 === 0);
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

runConcaveSweepGenerationCase();
runExplicitSweepOrderingCase();

console.log("flight_path_sweeps.test.ts: all assertions passed");
