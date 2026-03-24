import assert from "node:assert/strict";

import { DJI_ZENMUSE_P1_24MM, SONY_RX1R2, calculateGSD } from "../domain/camera.ts";
import { DEFAULT_LIDAR, getLidarMappingFovDeg, getLidarModel, lidarDeliverableDensity } from "../domain/lidar.ts";
import type { FlightParams } from "../domain/types.ts";
import { evaluateCameraTileExact, evaluateLidarTileExact, scoreExactCameraStats, scoreExactLidarStats } from "../overlap/exact-core/index.ts";
import { tileMetersBounds } from "../overlap/mercator.ts";
import { runLidarWorkerMessage } from "../overlap/lidar-worker.ts";
import type { GSDStats, LidarWorkerOut, WorkerOut } from "../overlap/types.ts";
import { runCameraWorkerMessage } from "../overlap/worker.ts";

const WEB_MERCATOR_RADIUS_M = 6378137;

function encodeTerrainRgb(size: number, elevationForPixel: (row: number, col: number) => number) {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
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

function worldMetersToLngLat(x: number, y: number): [number, number] {
  const lng = (x / WEB_MERCATOR_RADIUS_M) * (180 / Math.PI);
  const lat = Math.atan(Math.sinh(y / WEB_MERCATOR_RADIUS_M)) * (180 / Math.PI);
  return [lng, lat];
}

function pixelVertexToLngLat(z: number, x: number, y: number, size: number, col: number, row: number): [number, number] {
  const bounds = tileMetersBounds(z, x, y);
  const pixelSize = (bounds.maxX - bounds.minX) / size;
  return worldMetersToLngLat(bounds.minX + col * pixelSize, bounds.maxY - row * pixelSize);
}

function pixelCenterToMeters(z: number, x: number, y: number, size: number, col: number, row: number) {
  const bounds = tileMetersBounds(z, x, y);
  const pixelSize = (bounds.maxX - bounds.minX) / size;
  return {
    x: bounds.minX + (col + 0.5) * pixelSize,
    y: bounds.maxY - (row + 0.5) * pixelSize,
  };
}

function polygonFromPixelVertices(
  z: number,
  x: number,
  y: number,
  size: number,
  vertices: Array<[number, number]>,
) {
  return {
    id: "poly-1",
    ring: vertices.map(([col, row]) => pixelVertexToLngLat(z, x, y, size, col, row)),
  };
}

function serializeStats(stats?: GSDStats) {
  if (!stats) return undefined;
  return {
    ...stats,
    histogram: stats.histogram.map((bin) => ({ ...bin })),
  };
}

function serializeTileResult(result: WorkerOut | LidarWorkerOut) {
  return {
    ...result,
    overlap: Array.from(result.overlap),
    gsdMin: Array.from(result.gsdMin),
    density: result.density ? Array.from(result.density) : undefined,
    gsdStats: serializeStats(result.gsdStats),
    densityStats: serializeStats(result.densityStats),
    perPolygon: (result.perPolygon ?? []).map((entry) => ({
      ...entry,
      gsdStats: serializeStats(entry.gsdStats),
      densityStats: serializeStats(entry.densityStats),
      hitPoseIds: entry.hitPoseIds ? Array.from(entry.hitPoseIds) : undefined,
      hitLineIds: entry.hitLineIds ? Array.from(entry.hitLineIds) : undefined,
    })),
  };
}

function histogramEdges(stats: GSDStats) {
  const bins = [...stats.histogram].filter((bin) => (bin.areaM2 || 0) > 0).sort((a, b) => a.bin - b.bin);
  if (bins.length === 0) return { bins, edges: [] as number[] };
  if (bins.length === 1) {
    const only = bins[0].bin;
    const lo = Number.isFinite(stats.min) ? Math.min(stats.min, only) : only;
    const hi = Number.isFinite(stats.max) ? Math.max(stats.max, only) : only;
    return { bins, edges: [lo, hi > lo ? hi : lo + 1] };
  }
  const centers = bins.map((bin) => bin.bin);
  const edges = new Array<number>(bins.length + 1);
  edges[0] = centers[0] - (centers[1] - centers[0]) * 0.5;
  for (let index = 1; index < centers.length; index++) {
    edges[index] = (centers[index - 1] + centers[index]) * 0.5;
  }
  edges[bins.length] = centers[bins.length - 1] + (centers[bins.length - 1] - centers[bins.length - 2]) * 0.5;
  return { bins, edges };
}

function expectedAreaBelow(stats: GSDStats, threshold: number) {
  const { bins, edges } = histogramEdges(stats);
  let area = 0;
  for (let index = 0; index < bins.length; index++) {
    const binArea = bins[index].areaM2 || 0;
    if (!(binArea > 0)) continue;
    const lo = edges[index];
    const hi = edges[index + 1];
    if (threshold >= hi) {
      area += binArea;
      continue;
    }
    if (threshold <= lo) continue;
    area += binArea * Math.max(0, Math.min(1, (threshold - lo) / Math.max(1e-9, hi - lo)));
  }
  return area;
}

function expectedQuantile(stats: GSDStats, q: number) {
  const { bins, edges } = histogramEdges(stats);
  const totalArea = (stats.totalAreaM2 && stats.totalAreaM2 > 0)
    ? stats.totalAreaM2
    : stats.histogram.reduce((sum, bin) => sum + (bin.areaM2 || 0), 0);
  if (!(totalArea > 0) || bins.length === 0) return 0;
  const target = Math.max(0, Math.min(1, q)) * totalArea;
  let cumulative = 0;
  for (let index = 0; index < bins.length; index++) {
    const area = bins[index].areaM2 || 0;
    cumulative += area;
    if (cumulative >= target) {
      const previous = cumulative - area;
      const fraction = area > 0 ? Math.max(0, Math.min(1, (target - previous) / area)) : 0;
      return edges[index] + fraction * (edges[index + 1] - edges[index]);
    }
  }
  return edges[edges.length - 1] ?? bins[bins.length - 1]?.bin ?? 0;
}

function expectedCameraScore(stats: GSDStats, params: FlightParams) {
  const targetGsdM = calculateGSD(
    params.cameraKey === "DJI_ZENMUSE_P1_24MM" ? DJI_ZENMUSE_P1_24MM : SONY_RX1R2,
    params.altitudeAGL,
  );
  const totalAreaM2 = Math.max(1, stats.totalAreaM2 || 0);
  const q75 = expectedQuantile(stats, 0.75);
  const q90 = expectedQuantile(stats, 0.9);
  const overTargetAreaFraction = Math.max(0, totalAreaM2 - expectedAreaBelow(stats, targetGsdM)) / totalAreaM2;
  const meanOvershoot = Math.max(0, stats.mean / Math.max(1e-6, targetGsdM) - 1);
  const q75Overshoot = Math.max(0, q75 / Math.max(1e-6, targetGsdM) - 1);
  const q90Overshoot = Math.max(0, q90 / Math.max(1e-6, targetGsdM) - 1);
  const maxOvershoot = Math.max(0, stats.max / Math.max(1e-6, targetGsdM) - 1);
  return {
    qualityCost:
      1.85 * q90Overshoot +
      1.25 * overTargetAreaFraction +
      0.95 * meanOvershoot +
      0.55 * q75Overshoot +
      0.2 * maxOvershoot,
    targetGsdM,
    overTargetAreaFraction,
    q75,
    q90,
  };
}

function expectedLidarScore(stats: GSDStats, params: FlightParams) {
  const model = getLidarModel(params.lidarKey);
  const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
  const speedMps = params.speedMps ?? model.defaultSpeedMps;
  const returnMode = params.lidarReturnMode ?? "single";
  const targetDensityPtsM2 = params.pointDensityPtsM2
    ?? lidarDeliverableDensity(model, params.altitudeAGL, params.sideOverlap, speedMps, returnMode, mappingFovDeg);
  const totalAreaM2 = Math.max(1, stats.totalAreaM2 || 0);
  const holeThreshold = Math.max(5, targetDensityPtsM2 * 0.2);
  const weakThreshold = Math.max(holeThreshold + 1e-6, targetDensityPtsM2 * 0.7);
  const q10 = expectedQuantile(stats, 0.1);
  const q25 = expectedQuantile(stats, 0.25);
  const holeFraction = expectedAreaBelow(stats, holeThreshold) / totalAreaM2;
  const lowFraction = expectedAreaBelow(stats, weakThreshold) / totalAreaM2;
  const q10Deficit = Math.max(0, 1 - q10 / Math.max(1e-6, targetDensityPtsM2));
  const q25Deficit = Math.max(0, 1 - q25 / Math.max(1e-6, targetDensityPtsM2));
  const meanDeficit = Math.max(0, 1 - stats.mean / Math.max(1e-6, targetDensityPtsM2));
  return {
    qualityCost:
      4.2 * holeFraction +
      2.4 * lowFraction +
      1.9 * q10Deficit +
      1.2 * q25Deficit +
      0.8 * meanDeficit,
    targetDensityPtsM2,
    holeFraction,
    lowFraction,
    q10,
    q25,
  };
}

function approxEqual(actual: number, expected: number, tolerance = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `expected ${expected}, got ${actual}`);
}

function makeCameraFixture() {
  const z = 16;
  const x = 34120;
  const y = 22980;
  const size = 32;
  const tile = {
    z,
    x,
    y,
    size,
    data: encodeTerrainRgb(size, (row, col) => 420 + row * 1.2 + col * 0.8),
  };
  const polygon = polygonFromPixelVertices(z, x, y, size, [
    [4, 5],
    [27, 5],
    [27, 12],
    [18, 12],
    [18, 25],
    [4, 25],
    [4, 5],
  ]);
  const posePixels: Array<[number, number]> = [
    [9, 10],
    [13, 10],
    [17, 10],
    [21, 10],
    [25, 10],
  ];
  const poses = posePixels.map(([col, row], index) => {
    const position = pixelCenterToMeters(z, x, y, size, col, row);
    return {
      id: `pose-${index + 1}`,
      x: position.x,
      y: position.y,
      z: 820,
      omega_deg: 0,
      phi_deg: 0,
      kappa_deg: 0,
      polygonId: polygon.id,
    };
  });
  return {
    tile,
    polygons: [polygon],
    poses,
    camera: SONY_RX1R2,
    options: {
      maxOverlapNeeded: 8,
      minOverlapForGsd: 1,
      gridSize: 8,
      clipInnerBufferM: 0,
    },
  };
}

function makeLidarFixture() {
  const z = 16;
  const x = 34121;
  const y = 22980;
  const size = 32;
  const demSize = 96;
  const padTiles = 1;
  const tile = {
    z,
    x,
    y,
    size,
    data: encodeTerrainRgb(size, (row, col) => 380 + row * 0.6 + col * 0.3),
  };
  const demTile = {
    size: demSize,
    padTiles,
    data: encodeTerrainRgb(demSize, (row, col) => 380 + row * 0.2 + col * 0.15),
  };
  const polygon = polygonFromPixelVertices(z, x, y, size, [
    [3, 3],
    [29, 3],
    [29, 29],
    [3, 29],
    [3, 3],
  ]);
  const start = pixelCenterToMeters(z, x, y, size, 5, 16);
  const end = pixelCenterToMeters(z, x, y, size, 27, 16);
  const strip = {
    id: "strip-1",
    polygonId: polygon.id,
    x1: start.x,
    y1: start.y,
    z1: 760,
    x2: end.x,
    y2: end.y,
    z2: 760,
    halfWidthM: 70,
    densityPerPass: 18,
    speedMps: DEFAULT_LIDAR.defaultSpeedMps,
    effectivePointRate: DEFAULT_LIDAR.effectivePointRates.single,
    maxRangeM: 900,
    passIndex: 0,
    frameRateHz: DEFAULT_LIDAR.defaultFrameRateHz,
    mappingFovDeg: DEFAULT_LIDAR.mappingHorizontalFovDeg,
    verticalAnglesDeg: [0],
    azimuthSectorCenterDeg: DEFAULT_LIDAR.defaultAzimuthSectorCenterDeg ?? 0,
    comparisonMode: "all-returns" as const,
  };
  return {
    tile,
    demTile,
    polygons: [polygon],
    strips: [strip],
    options: {
      clipInnerBufferM: 0,
    },
  };
}

function testCameraExactCoreMatchesWorkerWrapper() {
  assert.equal(typeof (globalThis as { window?: unknown }).window, "undefined");
  const input = makeCameraFixture();
  const core = evaluateCameraTileExact(input);
  const wrapped = runCameraWorkerMessage(input);

  assert.deepEqual(serializeTileResult(wrapped), serializeTileResult(core));
  assert.ok(core.perPolygon && core.perPolygon.length === 1, "camera exact core should report per-polygon stats");
  assert.ok((core.perPolygon?.[0].activePixelCount ?? 0) > 0, "camera exact core should produce active pixels");
  assert.ok(core.maxOverlap > 0, "camera exact core should accumulate overlap");
  assert.ok(Number.isFinite(core.minGsd) && core.minGsd > 0, "camera exact core should compute a finite min GSD");
}

function testLidarExactCoreMatchesWorkerWrapperAndPreservesHoleBucket() {
  assert.equal(typeof (globalThis as { self?: unknown }).self, "undefined");
  const input = makeLidarFixture();
  const core = evaluateLidarTileExact(input);
  const wrapped = runLidarWorkerMessage(input);

  assert.deepEqual(serializeTileResult(wrapped), serializeTileResult(core));
  assert.ok(core.perPolygon && core.perPolygon.length === 1, "lidar exact core should report per-polygon stats");
  assert.ok((core.densityStats?.histogram.length ?? 0) > 0, "lidar exact core should emit density histogram bins");
  assert.equal(core.densityStats?.histogram[0]?.bin, 0, "lidar exact core should keep an exact zero-density hole bucket");
  assert.ok((core.densityStats?.histogram[0]?.areaM2 ?? 0) > 0, "hole bucket should carry non-zero area");
  assert.ok((core.maxDensity ?? 0) > 0, "lidar exact core should accumulate density");
}

function testExactScoringHelpersMatchPreviousFormulas() {
  const cameraStats: GSDStats = {
    min: 0.018,
    max: 0.064,
    mean: 0.036,
    count: 100,
    totalAreaM2: 1000,
    histogram: [
      { bin: 0.02, count: 25, areaM2: 250 },
      { bin: 0.035, count: 50, areaM2: 500 },
      { bin: 0.06, count: 25, areaM2: 250 },
    ],
  };
  const cameraParams: FlightParams = {
    payloadKind: "camera",
    altitudeAGL: 120,
    frontOverlap: 80,
    sideOverlap: 70,
    cameraKey: "DJI_ZENMUSE_P1_24MM",
  };
  const expectedCamera = expectedCameraScore(cameraStats, cameraParams);
  const actualCamera = scoreExactCameraStats(cameraStats, cameraParams);
  approxEqual(actualCamera.qualityCost, expectedCamera.qualityCost);
  approxEqual(actualCamera.targetGsdM, expectedCamera.targetGsdM);
  approxEqual(actualCamera.overTargetAreaFraction, expectedCamera.overTargetAreaFraction);
  approxEqual(actualCamera.q75, expectedCamera.q75);
  approxEqual(actualCamera.q90, expectedCamera.q90);

  const lidarStats: GSDStats = {
    min: 0,
    max: 145,
    mean: 42,
    count: 120,
    totalAreaM2: 1200,
    histogram: [
      { bin: 0, count: 30, areaM2: 300 },
      { bin: 25, count: 35, areaM2: 350 },
      { bin: 70, count: 35, areaM2: 350 },
      { bin: 130, count: 20, areaM2: 200 },
    ],
  };
  const lidarParams: FlightParams = {
    payloadKind: "lidar",
    altitudeAGL: 120,
    frontOverlap: 0,
    sideOverlap: 60,
    lidarKey: DEFAULT_LIDAR.key,
    speedMps: DEFAULT_LIDAR.defaultSpeedMps,
    lidarReturnMode: "single",
    mappingFovDeg: DEFAULT_LIDAR.mappingHorizontalFovDeg,
  };
  const expectedLidar = expectedLidarScore(lidarStats, lidarParams);
  const actualLidar = scoreExactLidarStats(lidarStats, lidarParams);
  approxEqual(actualLidar.qualityCost, expectedLidar.qualityCost);
  approxEqual(actualLidar.targetDensityPtsM2, expectedLidar.targetDensityPtsM2);
  approxEqual(actualLidar.holeFraction, expectedLidar.holeFraction);
  approxEqual(actualLidar.lowFraction, expectedLidar.lowFraction);
  approxEqual(actualLidar.q10, expectedLidar.q10);
  approxEqual(actualLidar.q25, expectedLidar.q25);
}

testCameraExactCoreMatchesWorkerWrapper();
testLidarExactCoreMatchesWorkerWrapperAndPreservesHoleBucket();
testExactScoringHelpersMatchPreviousFormulas();

console.log("exact_core.test.ts passed");
