import assert from "node:assert/strict";

import { evaluateCameraTileExact, evaluateLidarTileExact } from "../overlap/exact-core/index.ts";
import {
  evaluateRegionBearingExact,
  evaluatePartitionSolutionCandidateExact,
  optimizeBearingExact,
  rerankPartitionSolutionsExact,
  type ExactRegionRuntime,
  type ExactTileRef,
  type ExactTileWithHalo,
} from "../overlap/exact-region/index.ts";
import type { FlightParams, TerrainTile } from "../domain/types.ts";
import type { PaddedDemTileRGBA, TileRGBA } from "../overlap/types.ts";
import type { TerrainPartitionSolutionPreview } from "../terrain-partition/types.ts";

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

function repeatTileToDem(tile: TileRGBA, padTiles: number): PaddedDemTileRGBA {
  if (padTiles <= 0) {
    return {
      size: tile.size,
      padTiles: 0,
      data: new Uint8ClampedArray(tile.data),
    };
  }
  const span = padTiles * 2 + 1;
  const demSize = tile.size * span;
  const demData = new Uint8ClampedArray(demSize * demSize * 4);
  for (let dy = 0; dy < span; dy++) {
    for (let dx = 0; dx < span; dx++) {
      const offsetX = dx * tile.size;
      const offsetY = dy * tile.size;
      for (let row = 0; row < tile.size; row++) {
        const srcStart = row * tile.size * 4;
        const dstStart = ((offsetY + row) * demSize + offsetX) * 4;
        demData.set(tile.data.subarray(srcStart, srcStart + tile.size * 4), dstStart);
      }
    }
  }
  return {
    size: demSize,
    padTiles,
    data: demData,
  };
}

function createFixtureTile(size = 64): { tile: TileRGBA; geometryTile: TerrainTile } {
  const tile: TileRGBA = {
    z: 0,
    x: 0,
    y: 0,
    size,
    data: encodeTerrainRgb(size, (row, col) => 180 + row * 0.8 + col * 0.5),
  };
  return {
    tile,
    geometryTile: {
      z: tile.z,
      x: tile.x,
      y: tile.y,
      width: tile.size,
      height: tile.size,
      data: tile.data,
    },
  };
}

function createRuntime(tile: TileRGBA, options?: { candidateConcurrency?: number }): ExactRegionRuntime {
  const exactTileEvaluator = {
    evaluateCameraTile: async (input: Parameters<typeof evaluateCameraTileExact>[0]) => evaluateCameraTileExact(input),
    evaluateLidarTile: async (input: Parameters<typeof evaluateLidarTileExact>[0]) => evaluateLidarTileExact(input),
  };
  const terrainProvider = {
    async getTerrainTiles(tileRefs: ExactTileRef[]) {
      return new Map(tileRefs.map((tileRef) => [
        `${tileRef.z}/${tileRef.x}/${tileRef.y}`,
        {
          z: tileRef.z,
          x: tileRef.x,
          y: tileRef.y,
          size: tile.size,
          data: new Uint8ClampedArray(tile.data),
        },
      ]));
    },
    async getTerrainTilesWithHalo(tileRefs: ExactTileRef[], padTiles: number) {
      const demTile = repeatTileToDem(tile, padTiles);
      return new Map<string, ExactTileWithHalo>(tileRefs.map((tileRef) => [
        `${tileRef.z}/${tileRef.x}/${tileRef.y}`,
        {
          tile: {
            z: tileRef.z,
            x: tileRef.x,
            y: tileRef.y,
            size: tile.size,
            data: new Uint8ClampedArray(tile.data),
          },
          demTile: {
            size: demTile.size,
            padTiles: demTile.padTiles,
            data: new Uint8ClampedArray(demTile.data),
          },
        },
      ]));
    },
  };
  return {
    terrainProvider,
    tileEvaluator: exactTileEvaluator,
    yieldToEventLoop: async () => undefined,
    candidateConcurrency: options?.candidateConcurrency,
  };
}

function makeBaseArgs(geometryTile: TerrainTile) {
  return {
    ring: [
      [0, 0],
      [0.018, 0],
      [0.018, 0.0045],
      [0, 0.0045],
      [0, 0],
    ] as [number, number][],
    altitudeMode: "legacy" as const,
    minClearanceM: 0,
    turnExtendM: 0,
    exactOptimizeZoom: 14,
    minOverlapForGsd: 1,
    geometryTiles: [geometryTile],
  };
}

function makeCameraParams(): FlightParams {
  return {
    payloadKind: "camera",
    altitudeAGL: 110,
    frontOverlap: 75,
    sideOverlap: 70,
  };
}

function makeLidarParams(): FlightParams {
  return {
    payloadKind: "lidar",
    altitudeAGL: 120,
    frontOverlap: 0,
    sideOverlap: 40,
  };
}

function makeSolution(signature: string, bearingDeg: number, ring: [number, number][]): TerrainPartitionSolutionPreview {
  return {
    signature,
    tradeoff: 0.5,
    regionCount: 1,
    totalMissionTimeSec: 120,
    normalizedQualityCost: 0.4,
    weightedMeanMismatchDeg: 5,
    hierarchyLevel: 1,
    largestRegionFraction: 1,
    meanConvexity: 1,
    boundaryBreakAlignment: 1,
    isFirstPracticalSplit: signature === "good",
    regions: [
      {
        ring,
        areaM2: 1000,
        bearingDeg,
        atomCount: 1,
        convexity: 1,
        compactness: 1,
      },
    ],
  };
}

function axialDistanceDeg(left: number, right: number) {
  const diff = Math.abs((((left - right) % 180) + 180) % 180);
  return Math.min(diff, 180 - diff);
}

function approxEqual(actual: number, expected: number, tolerance = 5e-4) {
  const relativeTolerance = Math.abs(expected) * 1e-3;
  assert.ok(Math.abs(actual - expected) <= Math.max(tolerance, relativeTolerance), `expected ${expected}, got ${actual}`);
}

function assertBreakdownTotal(actualTotal: number, breakdown: { total: number; contributions: Record<string, number> }) {
  approxEqual(actualTotal, breakdown.total, 1e-12);
  approxEqual(
    Object.values(breakdown.contributions).reduce((sum, value) => sum + value, 0),
    actualTotal,
    1e-12,
  );
}

async function main() {
  const { tile, geometryTile } = createFixtureTile();
  const runtime = createRuntime(tile);
  const baseArgs = makeBaseArgs(geometryTile);

  const cameraCandidate = await evaluateRegionBearingExact(runtime, {
    ...baseArgs,
    scopeId: "camera-region",
    params: makeCameraParams(),
    bearingDeg: 90,
  });
  assert.ok(cameraCandidate, "camera exact region evaluation should return a candidate");
  assert.equal(cameraCandidate.metricKind, "gsd");
  approxEqual(cameraCandidate.bearingDeg, 90);
  approxEqual(cameraCandidate.exactCost, 1.3259692230118365);
  approxEqual(cameraCandidate.qualityCost, 0.008119844657775368);
  approxEqual(cameraCandidate.missionTimeSec, 2373.59045307571);
  approxEqual(cameraCandidate.stats.mean, 0.012879512189621372);
  assert.equal(cameraCandidate.stats.count, 728);
  assert.equal(cameraCandidate.stats.histogram.length, 1);
  assert.equal(cameraCandidate.qualityBreakdown.modelVersion, "camera-region-v1");
  assert.equal(cameraCandidate.costBreakdown.modelVersion, "camera-region-v1");
  assertBreakdownTotal(cameraCandidate.qualityCost, cameraCandidate.qualityBreakdown);
  assertBreakdownTotal(cameraCandidate.exactCost, cameraCandidate.costBreakdown);
  approxEqual(cameraCandidate.missionBreakdown.totalLengthM / cameraCandidate.missionBreakdown.speedMps, cameraCandidate.missionTimeSec);

  const lidarCandidate = await evaluateRegionBearingExact(runtime, {
    ...baseArgs,
    scopeId: "lidar-region",
    params: makeLidarParams(),
    bearingDeg: 90,
  });
  assert.ok(lidarCandidate, "lidar exact region evaluation should return a candidate");
  assert.equal(lidarCandidate.metricKind, "density");
  approxEqual(lidarCandidate.bearingDeg, 90);
  approxEqual(lidarCandidate.exactCost, 5.641167085388158);
  approxEqual(lidarCandidate.qualityCost, 6.016771435812384);
  approxEqual(lidarCandidate.missionTimeSec, 406.93102768262224);
  approxEqual(lidarCandidate.stats.mean, 26.667487393142114);
  assert.equal(lidarCandidate.stats.count, 728);
  assert.equal(lidarCandidate.stats.histogram[0]?.bin, 0);
  assert.equal(lidarCandidate.qualityBreakdown.modelVersion, "lidar-region-v1");
  assert.equal(lidarCandidate.costBreakdown.modelVersion, "lidar-region-v1");
  assertBreakdownTotal(lidarCandidate.qualityCost, lidarCandidate.qualityBreakdown);
  assertBreakdownTotal(lidarCandidate.exactCost, lidarCandidate.costBreakdown);
  approxEqual(lidarCandidate.missionBreakdown.totalLengthM / lidarCandidate.missionBreakdown.speedMps, lidarCandidate.missionTimeSec);

  const globalOptimize = await optimizeBearingExact(runtime, {
    ...baseArgs,
    scopeId: "global-opt",
    params: makeCameraParams(),
    seedBearingDeg: 17,
    mode: "global",
  });
  assert.ok(globalOptimize.best, "global optimize should select a best candidate");
  assert.ok(globalOptimize.evaluated.length > 0);
  const globalBest = globalOptimize.evaluated.reduce((best, candidate) =>
    candidate.exactCost < best.exactCost ? candidate : best,
  );
  assert.equal(globalOptimize.best?.bearingDeg, globalBest.bearingDeg);
  assert.equal(globalOptimize.best?.exactCost, globalBest.exactCost);
  approxEqual(globalOptimize.best!.bearingDeg, 90);
  approxEqual(globalOptimize.best!.exactCost, 1.3259692230118365);

  const localOptimize = await optimizeBearingExact(runtime, {
    ...baseArgs,
    scopeId: "local-opt",
    params: makeCameraParams(),
    seedBearingDeg: 42,
    mode: "local",
    halfWindowDeg: 30,
  });
  assert.ok(localOptimize.best, "local optimize should select a best candidate");
  assert.equal(localOptimize.evaluated.length, 10);
  approxEqual(localOptimize.best.bearingDeg, 72);
  approxEqual(localOptimize.best.exactCost, 1.6235215595657575);
  approxEqual(localOptimize.best.qualityCost, 0.005820561760900072);
  for (const candidate of localOptimize.evaluated) {
    assert.ok(axialDistanceDeg(candidate.bearingDeg, localOptimize.seedBearingDeg) <= 30.0001);
  }

  const parallelRuntime = createRuntime(tile, { candidateConcurrency: 4 });
  const parallelGlobalOptimize = await optimizeBearingExact(parallelRuntime, {
    ...baseArgs,
    scopeId: "global-opt-parallel",
    params: makeCameraParams(),
    seedBearingDeg: 17,
    mode: "global",
  });
  assert.deepEqual(
    parallelGlobalOptimize.evaluated.map((candidate) => ({
      bearingDeg: candidate.bearingDeg,
      exactCost: candidate.exactCost,
      qualityCost: candidate.qualityCost,
      missionTimeSec: candidate.missionTimeSec,
    })),
    globalOptimize.evaluated.map((candidate) => ({
      bearingDeg: candidate.bearingDeg,
      exactCost: candidate.exactCost,
      qualityCost: candidate.qualityCost,
      missionTimeSec: candidate.missionTimeSec,
    })),
  );
  assert.deepEqual(
    parallelGlobalOptimize.best && {
      bearingDeg: parallelGlobalOptimize.best.bearingDeg,
      exactCost: parallelGlobalOptimize.best.exactCost,
      qualityCost: parallelGlobalOptimize.best.qualityCost,
      missionTimeSec: parallelGlobalOptimize.best.missionTimeSec,
    },
    globalOptimize.best && {
      bearingDeg: globalOptimize.best.bearingDeg,
      exactCost: globalOptimize.best.exactCost,
      qualityCost: globalOptimize.best.qualityCost,
      missionTimeSec: globalOptimize.best.missionTimeSec,
    },
  );

  const parallelLocalOptimize = await optimizeBearingExact(parallelRuntime, {
    ...baseArgs,
    scopeId: "local-opt-parallel",
    params: makeCameraParams(),
    seedBearingDeg: 42,
    mode: "local",
    halfWindowDeg: 30,
  });
  assert.deepEqual(
    parallelLocalOptimize.evaluated.map((candidate) => ({
      bearingDeg: candidate.bearingDeg,
      exactCost: candidate.exactCost,
      qualityCost: candidate.qualityCost,
      missionTimeSec: candidate.missionTimeSec,
    })),
    localOptimize.evaluated.map((candidate) => ({
      bearingDeg: candidate.bearingDeg,
      exactCost: candidate.exactCost,
      qualityCost: candidate.qualityCost,
      missionTimeSec: candidate.missionTimeSec,
    })),
  );
  assert.deepEqual(
    parallelLocalOptimize.best && {
      bearingDeg: parallelLocalOptimize.best.bearingDeg,
      exactCost: parallelLocalOptimize.best.exactCost,
      qualityCost: parallelLocalOptimize.best.qualityCost,
      missionTimeSec: parallelLocalOptimize.best.missionTimeSec,
    },
    localOptimize.best && {
      bearingDeg: localOptimize.best.bearingDeg,
      exactCost: localOptimize.best.exactCost,
      qualityCost: localOptimize.best.qualityCost,
      missionTimeSec: localOptimize.best.missionTimeSec,
    },
  );

  const cameraAt0 = await evaluateRegionBearingExact(runtime, {
    ...baseArgs,
    scopeId: "rerank-0",
    params: makeCameraParams(),
    bearingDeg: 0,
  });
  const cameraAt90 = await evaluateRegionBearingExact(runtime, {
    ...baseArgs,
    scopeId: "rerank-90",
    params: makeCameraParams(),
    bearingDeg: 90,
  });
  assert.ok(cameraAt0 && cameraAt90);
  const goodSeed = cameraAt0.exactCost <= cameraAt90.exactCost ? 0 : 90;
  const badSeed = goodSeed === 0 ? 90 : 0;
  const reranked = await rerankPartitionSolutionsExact(runtime, {
    ...baseArgs,
    scopeId: "partition",
    polygonId: "partition",
    params: makeCameraParams(),
    solutions: [
      makeSolution("good", goodSeed, baseArgs.ring),
      makeSolution("bad", badSeed, baseArgs.ring),
    ],
    rankingSource: "frontend-exact",
    debugTrace: true,
  });
  assert.equal(reranked.bestIndex, 0);
  assert.equal(reranked.solutions.length, 2);
  assert.equal(reranked.solutions[0].rankingSource, "frontend-exact");
  assert.equal(reranked.solutions[0].signature, "good");
  assert.equal(reranked.solutions[1].signature, "bad");
  approxEqual(reranked.solutions[0].regions[0].bearingDeg, 90);
  approxEqual(reranked.solutions[1].regions[0].bearingDeg, 179);
  approxEqual(reranked.solutions[0].regions[0].exactScore ?? Number.NaN, 1.3259692230118365);
  approxEqual(reranked.solutions[1].regions[0].exactScore ?? Number.NaN, 1.8233536741405434);
  approxEqual(reranked.solutions[0].exactScore ?? Number.NaN, 11.664341671458846);
  approxEqual(reranked.solutions[1].exactScore ?? Number.NaN, 11.766320052080799);
  assert.equal(reranked.solutions[0].regions[0].exactSeedBearingDeg, goodSeed);
  assert.equal(reranked.solutions[1].regions[0].exactSeedBearingDeg, badSeed);
  approxEqual(reranked.previewsBySignature.good.stats.mean, 0.012879512189621372);
  approxEqual(reranked.previewsBySignature.bad.stats.mean, 0.0129949159492933);
  assert.ok(reranked.previewsBySignature.good);
  assert.ok(reranked.previewsBySignature.bad);
  assert.ok(reranked.debugBySignature?.good);
  assert.ok(reranked.debugBySignature?.bad);
  assert.equal(reranked.debugBySignature?.good.partitionScoreBreakdown.modelVersion, "camera-partition-v1");
  assert.equal(reranked.debugBySignature?.good.regions.length, 1);
  assert.equal(reranked.debugBySignature?.good.regions[0].evaluatedBearings[0].qualityBreakdown.modelVersion, "camera-region-v1");
  assertBreakdownTotal(
    reranked.solutions[0].exactScore ?? Number.NaN,
    reranked.debugBySignature!.good.partitionScoreBreakdown,
  );

  const candidateSolutions = [
    makeSolution("good", goodSeed, baseArgs.ring),
    makeSolution("bad", badSeed, baseArgs.ring),
  ];
  const fastestMissionTimeSec = Math.min(...candidateSolutions.map((solution) => solution.totalMissionTimeSec));
  const individuallyEvaluated = await Promise.all(
    candidateSolutions.map((solution) =>
      evaluatePartitionSolutionCandidateExact(runtime, {
        ...baseArgs,
        scopeId: "partition",
        polygonId: "partition",
        params: makeCameraParams(),
        solution,
        fastestMissionTimeSec,
        rankingSource: "frontend-exact",
        debugTrace: true,
      })
    ),
  );
  const rerankedBySignature = new Map(
    reranked.solutions.map((solution) => [solution.signature, solution] as const),
  );
  individuallyEvaluated.forEach((item) => {
    const rerankedSolution = rerankedBySignature.get(item.solution.signature);
    assert.ok(rerankedSolution);
    approxEqual(item.solution.exactScore ?? Number.NaN, rerankedSolution.exactScore ?? Number.NaN);
    approxEqual(item.solution.exactQualityCost ?? Number.NaN, rerankedSolution.exactQualityCost ?? Number.NaN);
    approxEqual(item.solution.regions[0].bearingDeg, rerankedSolution.regions[0].bearingDeg);
    assert.ok(item.debugTrace);
    assertBreakdownTotal(item.solution.exactScore ?? Number.NaN, item.debugTrace!.partitionScoreBreakdown);
  });

  console.log("exact_region tests passed");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
