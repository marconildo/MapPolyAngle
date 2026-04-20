import { lngLatToMeters } from "@/overlap/mercator";
import type { TerrainTile } from "@/domain/types";
import { queryElevationAtPointWGS84 } from "@/components/MapFlightDirection/utils/geometry";
import type {
  MissionProfileCoreSample,
  MissionProfileData,
  MissionProfileSegmentKind,
  MissionProfileSamplingOptions,
  MissionProfileSnapshot,
} from "./missionProfileWorker.types";

type TerrainQueryFn = (lng: number, lat: number, terrainTiles: TerrainTile[]) => number;

type EvaluatedPoint = MissionProfileCoreSample & {
  terrainAltitudeM: number | null;
  clearanceM: number | null;
};

export type MissionProfileSegmentDistanceRange = {
  key: string;
  segmentLabel: string;
  segmentKind: MissionProfileSegmentKind;
  index: number;
  startDistanceM: number;
  endDistanceM: number;
};

const OVERVIEW_MAX_SAMPLES = 800;
const OVERVIEW_TERRAIN_TOLERANCE_M = 8;
const OVERVIEW_CLEARANCE_TOLERANCE_M = 8;
const DETAIL_TERRAIN_TOLERANCE_M = 2;
const DETAIL_CLEARANCE_TOLERANCE_M = 2;

export function horizontalDistanceMeters3D(
  start: [number, number, number],
  end: [number, number, number],
) {
  const [x1, y1] = lngLatToMeters(start[0], start[1]);
  const [x2, y2] = lngLatToMeters(end[0], end[1]);
  return Math.hypot(x2 - x1, y2 - y1);
}

function interpolatePoint3D(
  start: [number, number, number],
  end: [number, number, number],
  t: number,
): [number, number, number] {
  return [
    start[0] + (end[0] - start[0]) * t,
    start[1] + (end[1] - start[1]) * t,
    start[2] + (end[2] - start[2]) * t,
  ];
}

function pointsNearlyMatchSample(left: MissionProfileCoreSample, right: MissionProfileCoreSample) {
  return Math.abs(left.distanceM - right.distanceM) <= 0.05
    && horizontalDistanceMeters3D(
      [left.lng, left.lat, left.droneAltitudeM],
      [right.lng, right.lat, right.droneAltitudeM],
    ) <= 0.05
    && Math.abs(left.droneAltitudeM - right.droneAltitudeM) <= 0.05;
}

function summarizeMissionProfile(samples: MissionProfileCoreSample[]) {
  const finiteClearances = samples
    .map((sample) => sample.clearanceM)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return {
    totalDistanceM: samples[samples.length - 1]?.distanceM ?? 0,
    minClearanceM: finiteClearances.length > 0 ? Math.min(...finiteClearances) : null,
    meanClearanceM: finiteClearances.length > 0
      ? finiteClearances.reduce((sum, value) => sum + value, 0) / finiteClearances.length
      : null,
    maxClearanceM: finiteClearances.length > 0 ? Math.max(...finiteClearances) : null,
    sampleCount: samples.length,
  };
}

function decimateMissionProfilePreservingExtrema(
  samples: MissionProfileCoreSample[],
  maxSamples: number,
): MissionProfileCoreSample[] {
  if (samples.length <= maxSamples) return samples;
  const totalDistanceM = samples[samples.length - 1]?.distanceM ?? 0;
  if (!(totalDistanceM > 0)) return samples.slice(0, maxSamples);
  const bucketCount = Math.max(1, Math.floor(maxSamples / 4));
  const bucketSizeM = totalDistanceM / bucketCount;
  const retained = new Set<number>([0, samples.length - 1]);

  for (let bucketIndex = 0; bucketIndex < bucketCount; bucketIndex += 1) {
    const bucketStart = bucketIndex * bucketSizeM;
    const bucketEnd = bucketStart + bucketSizeM;
    let firstIndex = -1;
    let lastIndex = -1;
    let minTerrainIndex = -1;
    let maxTerrainIndex = -1;
    let minClearanceIndex = -1;
    let maxClearanceIndex = -1;
    let minTerrainValue = Number.POSITIVE_INFINITY;
    let maxTerrainValue = Number.NEGATIVE_INFINITY;
    let minClearanceValue = Number.POSITIVE_INFINITY;
    let maxClearanceValue = Number.NEGATIVE_INFINITY;

    for (let index = 0; index < samples.length; index += 1) {
      const sample = samples[index]!;
      if (sample.distanceM < bucketStart || sample.distanceM > bucketEnd) continue;
      if (firstIndex === -1) firstIndex = index;
      lastIndex = index;
      if (typeof sample.terrainAltitudeM === "number") {
        if (sample.terrainAltitudeM < minTerrainValue) {
          minTerrainValue = sample.terrainAltitudeM;
          minTerrainIndex = index;
        }
        if (sample.terrainAltitudeM > maxTerrainValue) {
          maxTerrainValue = sample.terrainAltitudeM;
          maxTerrainIndex = index;
        }
      }
      if (typeof sample.clearanceM === "number") {
        if (sample.clearanceM < minClearanceValue) {
          minClearanceValue = sample.clearanceM;
          minClearanceIndex = index;
        }
        if (sample.clearanceM > maxClearanceValue) {
          maxClearanceValue = sample.clearanceM;
          maxClearanceIndex = index;
        }
      }
    }

    [firstIndex, lastIndex, minTerrainIndex, maxTerrainIndex, minClearanceIndex, maxClearanceIndex]
      .filter((index) => index >= 0)
      .forEach((index) => retained.add(index));
  }

  const decimated = [...retained]
    .sort((left, right) => left - right)
    .map((index) => samples[index]!)
    .filter((sample, index, array) => index === 0 || !pointsNearlyMatchSample(sample, array[index - 1]!));

  if (decimated.length <= maxSamples) return decimated;

  const step = (decimated.length - 1) / Math.max(maxSamples - 1, 1);
  const evenlySpaced: MissionProfileCoreSample[] = [];
  let lastIndex = -1;
  for (let bucket = 0; bucket < maxSamples; bucket += 1) {
    const index = Math.round(bucket * step);
    if (index === lastIndex) continue;
    const sample = decimated[index];
    if (sample) {
      evenlySpaced.push(sample);
      lastIndex = index;
    }
  }
  const finalSample = decimated[decimated.length - 1];
  if (finalSample && evenlySpaced[evenlySpaced.length - 1] !== finalSample) {
    evenlySpaced.push(finalSample);
  }
  return evenlySpaced;
}

function buildSampleEvaluator(
  terrainQuery: TerrainQueryFn,
  terrainQueryCache: Map<string, number | null>,
  segmentLabel: string,
  segmentKind: "area" | "connector",
  terrainTiles: TerrainTile[],
) {
  return (point: [number, number, number], distanceM: number): EvaluatedPoint => {
    let terrainAltitudeM: number | null = null;
    if (terrainTiles.length > 0) {
      const cacheKey = `${point[0].toFixed(6)},${point[1].toFixed(6)}|${terrainTiles.length}`;
      const cached = terrainQueryCache.get(cacheKey);
      if (cached !== undefined) {
        terrainAltitudeM = cached;
      } else {
        const queried = terrainQuery(point[0], point[1], terrainTiles);
        terrainAltitudeM = Number.isFinite(queried) ? queried : null;
        terrainQueryCache.set(cacheKey, terrainAltitudeM);
      }
    }
    return {
      distanceM,
      lng: point[0],
      lat: point[1],
      droneAltitudeM: point[2],
      terrainAltitudeM,
      clearanceM: terrainAltitudeM !== null ? point[2] - terrainAltitudeM : null,
      segmentLabel,
      segmentKind,
    };
  };
}

function shouldSplitAdaptiveSegment(
  start: EvaluatedPoint,
  mid: EvaluatedPoint,
  end: EvaluatedPoint,
  options: MissionProfileSamplingOptions,
) {
  const spanM = end.distanceM - start.distanceM;
  if (spanM > options.targetSpacingM) return true;
  if (typeof start.terrainAltitudeM === "number"
    && typeof mid.terrainAltitudeM === "number"
    && typeof end.terrainAltitudeM === "number") {
    const terrainLerp = start.terrainAltitudeM + (end.terrainAltitudeM - start.terrainAltitudeM) * 0.5;
    if (Math.abs(mid.terrainAltitudeM - terrainLerp) > options.terrainToleranceM) return true;
  }
  if (typeof start.clearanceM === "number"
    && typeof mid.clearanceM === "number"
    && typeof end.clearanceM === "number") {
    const clearanceLerp = start.clearanceM + (end.clearanceM - start.clearanceM) * 0.5;
    if (Math.abs(mid.clearanceM - clearanceLerp) > options.clearanceToleranceM) return true;
  }
  return false;
}

function appendUniqueSample(target: MissionProfileCoreSample[], sample: MissionProfileCoreSample) {
  const previous = target[target.length - 1];
  if (!previous || !pointsNearlyMatchSample(previous, sample)) {
    target.push(sample);
  }
}

function sampleAdaptiveSubsegment(
  startPoint: [number, number, number],
  endPoint: [number, number, number],
  startDistanceM: number,
  endDistanceM: number,
  evaluatePoint: (point: [number, number, number], distanceM: number) => EvaluatedPoint,
  options: MissionProfileSamplingOptions,
  samples: MissionProfileCoreSample[],
  depth = 0,
) {
  const startEval = evaluatePoint(startPoint, startDistanceM);
  const endEval = evaluatePoint(endPoint, endDistanceM);
  appendUniqueSample(samples, startEval);

  const visit = (
    leftPoint: [number, number, number],
    rightPoint: [number, number, number],
    leftEval: EvaluatedPoint,
    rightEval: EvaluatedPoint,
    currentDepth: number,
  ) => {
    const spanM = rightEval.distanceM - leftEval.distanceM;
    if (!(spanM > 0)) {
      appendUniqueSample(samples, rightEval);
      return;
    }
    if (currentDepth >= options.maxDepth || spanM <= options.targetSpacingM * 0.5) {
      appendUniqueSample(samples, rightEval);
      return;
    }

    const midpoint = interpolatePoint3D(leftPoint, rightPoint, 0.5);
    const midDistanceM = leftEval.distanceM + spanM * 0.5;
    const midEval = evaluatePoint(midpoint, midDistanceM);
    if (!shouldSplitAdaptiveSegment(leftEval, midEval, rightEval, options)) {
      appendUniqueSample(samples, rightEval);
      return;
    }

    visit(leftPoint, midpoint, leftEval, midEval, currentDepth + 1);
    visit(midpoint, rightPoint, midEval, rightEval, currentDepth + 1);
  };

  visit(startPoint, endPoint, startEval, endEval, depth);
}

export function computeMissionTotalDistanceM(snapshot: Pick<MissionProfileSnapshot, "segments">) {
  let totalDistanceM = 0;
  for (const segment of snapshot.segments) {
    for (const path of segment.path3D) {
      for (let index = 1; index < path.length; index += 1) {
        totalDistanceM += horizontalDistanceMeters3D(path[index - 1]!, path[index]!);
      }
    }
  }
  return totalDistanceM;
}

export function computeMissionSegmentDistanceRanges(
  snapshot: Pick<MissionProfileSnapshot, "segments">,
): MissionProfileSegmentDistanceRange[] {
  const ranges: MissionProfileSegmentDistanceRange[] = [];
  let cumulativeDistanceM = 0;

  for (let segmentIndex = 0; segmentIndex < snapshot.segments.length; segmentIndex += 1) {
    const segment = snapshot.segments[segmentIndex]!;
    const segmentStartM = cumulativeDistanceM;

    for (const path of segment.path3D) {
      for (let pointIndex = 1; pointIndex < path.length; pointIndex += 1) {
        cumulativeDistanceM += horizontalDistanceMeters3D(path[pointIndex - 1]!, path[pointIndex]!);
      }
    }

    if (cumulativeDistanceM <= segmentStartM) continue;
    ranges.push({
      key: segment.key,
      segmentLabel: segment.segmentLabel,
      segmentKind: segment.segmentKind,
      index: segmentIndex,
      startDistanceM: segmentStartM,
      endDistanceM: cumulativeDistanceM,
    });
  }

  return ranges;
}

export function buildMissionProfileOverview(
  snapshot: MissionProfileSnapshot,
  terrainQuery: TerrainQueryFn = queryElevationAtPointWGS84,
  terrainQueryCache = new Map<string, number | null>(),
): MissionProfileData | null {
  const totalDistanceM = snapshot.totalDistanceM || computeMissionTotalDistanceM(snapshot);
  if (!(totalDistanceM > 0)) return null;
  return sampleMissionProfile(snapshot, {
    rangeStartM: 0,
    rangeEndM: totalDistanceM,
    targetSpacingM: Math.max(totalDistanceM / OVERVIEW_MAX_SAMPLES, 40),
    maxSamples: OVERVIEW_MAX_SAMPLES,
    terrainToleranceM: OVERVIEW_TERRAIN_TOLERANCE_M,
    clearanceToleranceM: OVERVIEW_CLEARANCE_TOLERANCE_M,
    maxDepth: 8,
  }, terrainQuery, terrainQueryCache);
}

export function sampleMissionProfile(
  snapshot: MissionProfileSnapshot,
  options: MissionProfileSamplingOptions,
  terrainQuery: TerrainQueryFn = queryElevationAtPointWGS84,
  terrainQueryCache = new Map<string, number | null>(),
): MissionProfileData | null {
  const totalDistanceM = snapshot.totalDistanceM || computeMissionTotalDistanceM(snapshot);
  if (!(totalDistanceM > 0)) return null;

  const rangeStartM = Math.max(0, Math.min(options.rangeStartM ?? 0, totalDistanceM));
  const rangeEndM = Math.max(rangeStartM, Math.min(options.rangeEndM ?? totalDistanceM, totalDistanceM));
  if (!(rangeEndM > rangeStartM)) return null;

  const collectedSamples: MissionProfileCoreSample[] = [];
  let cumulativeDistanceM = 0;

  for (const segment of snapshot.segments) {
    for (const path of segment.path3D) {
      if (path.length < 2) continue;
      const evaluatePoint = buildSampleEvaluator(
        terrainQuery,
        terrainQueryCache,
        segment.segmentLabel,
        segment.segmentKind,
        segment.terrainTiles,
      );

      for (let index = 1; index < path.length; index += 1) {
        const start = path[index - 1]!;
        const end = path[index]!;
        const segmentDistanceM = horizontalDistanceMeters3D(start, end);
        const segmentStartM = cumulativeDistanceM;
        const segmentEndM = cumulativeDistanceM + segmentDistanceM;
        cumulativeDistanceM = segmentEndM;

        if (!(segmentDistanceM > 0)) continue;
        if (segmentEndM < rangeStartM || segmentStartM > rangeEndM) continue;

        const overlapStartM = Math.max(segmentStartM, rangeStartM);
        const overlapEndM = Math.min(segmentEndM, rangeEndM);
        if (!(overlapEndM > overlapStartM)) continue;

        const overlapStartT = (overlapStartM - segmentStartM) / segmentDistanceM;
        const overlapEndT = (overlapEndM - segmentStartM) / segmentDistanceM;
        const clippedStart = interpolatePoint3D(start, end, overlapStartT);
        const clippedEnd = interpolatePoint3D(start, end, overlapEndT);

        sampleAdaptiveSubsegment(
          clippedStart,
          clippedEnd,
          overlapStartM,
          overlapEndM,
          evaluatePoint,
          options,
          collectedSamples,
        );
      }
    }
  }

  if (collectedSamples.length < 2) return null;
  const decimatedSamples = decimateMissionProfilePreservingExtrema(collectedSamples, options.maxSamples);
  return {
    samples: decimatedSamples,
    summary: summarizeMissionProfile(decimatedSamples),
  };
}

export function buildMissionProfileDetail(
  snapshot: MissionProfileSnapshot,
  rangeStartM: number,
  rangeEndM: number,
  spacingBucketM: number,
  maxSamples: number,
  terrainQuery: TerrainQueryFn = queryElevationAtPointWGS84,
  terrainQueryCache = new Map<string, number | null>(),
): MissionProfileData | null {
  return sampleMissionProfile(snapshot, {
    rangeStartM,
    rangeEndM,
    targetSpacingM: Math.max(spacingBucketM, 1),
    maxSamples,
    terrainToleranceM: DETAIL_TERRAIN_TOLERANCE_M,
    clearanceToleranceM: DETAIL_CLEARANCE_TOLERANCE_M,
    maxDepth: 12,
  }, terrainQuery, terrainQueryCache);
}

export function quantizeMissionProfileSpacingBucket(spacingM: number) {
  if (!(spacingM > 0)) return 1;
  const exponent = Math.floor(Math.log10(spacingM));
  const base = 10 ** exponent;
  const normalized = spacingM / base;
  const step = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return step * base;
}

export function buildMissionProfileDetailRange(
  viewportStartM: number,
  viewportEndM: number,
  totalDistanceM: number,
  spacingBucketM: number,
) {
  const visibleSpanM = Math.max(viewportEndM - viewportStartM, 1);
  const paddedSpanM = visibleSpanM * 1.25;
  const paddedStartM = Math.max(0, viewportStartM - visibleSpanM * 0.25);
  const paddedEndM = Math.min(totalDistanceM, viewportEndM + visibleSpanM * 0.25);
  const bucketSpanM = Math.max(
    spacingBucketM * 64,
    quantizeMissionProfileSpacingBucket(paddedSpanM),
  );
  const quantizedStartM = Math.max(0, Math.floor(paddedStartM / bucketSpanM) * bucketSpanM);
  const quantizedEndM = Math.min(totalDistanceM, Math.ceil(paddedEndM / bucketSpanM) * bucketSpanM);
  return {
    requestStartM: quantizedStartM,
    requestEndM: Math.max(quantizedStartM, quantizedEndM),
    visibleSpanM,
  };
}

export function clipMissionProfileToRange(
  profile: MissionProfileData | null,
  rangeStartM: number,
  rangeEndM: number,
): MissionProfileData | null {
  if (!profile) return null;
  const visibleSamples = profile.samples.filter(
    (sample) => sample.distanceM >= rangeStartM && sample.distanceM <= rangeEndM,
  );
  const clippedSamples = [...visibleSamples];
  if (clippedSamples.length < 2) {
    const before = [...profile.samples]
      .reverse()
      .find((sample) => sample.distanceM < rangeStartM) ?? null;
    const after = profile.samples.find((sample) => sample.distanceM > rangeEndM) ?? null;
    if (before) clippedSamples.unshift(before);
    if (after) clippedSamples.push(after);
  }
  if (clippedSamples.length < 2) return null;
  return {
    samples: clippedSamples,
    summary: visibleSamples.length > 0
      ? summarizeMissionProfile(visibleSamples)
      : {
          totalDistanceM: 0,
          minClearanceM: null,
          meanClearanceM: null,
          maxClearanceM: null,
          sampleCount: 0,
        },
  };
}
