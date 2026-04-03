/***********************************************************************
 * flight/geometry.ts
 *
 * Pure geometric and geographic utility functions shared by frontend UI,
 * exact-region evaluation, and backend exact runtime orchestration.
 ***********************************************************************/

import {
  queryElevationAtPoint,
  destination as geoDestination,
  calculateBearing as geoBearing,
  TerrainTile,
} from '@/utils/terrainAspectHybrid';
import * as egm96 from 'egm96-universal';
import type { CameraModel, PlannedFlightGeometry, PlannedTurnBlock } from '@/domain/types';

const TURN_LIFT_ALTITUDE_MARGIN_M = 20;
const TURN_LIFT_EASING_POWER = 2;
const TURN_LIFT_ELEVATION_PER_METER = 0.25;
const TURN_LIFT_LOOP_DEGREES = 360;
const TURN_LIFT_LOOP_POINT_SPACING_M = 8;
const TURN_LIFT_CIRCLE_TOLERANCE_M = 5;

function convertElevationToWGS84(lat: number, lng: number, elevationEGM96: number): number {
  return egm96.egm96ToEllipsoid(lat, lng, elevationEGM96);
}

function queryMaxElevationAlongLineWGS84(
  startLng: number,
  startLat: number,
  endLng: number,
  endLat: number,
  tiles: TerrainTile[],
  samples: number = 20,
): number {
  let maxElevationWGS84 = -Infinity;

  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const lng = startLng + t * (endLng - startLng);
    const lat = startLat + t * (endLat - startLat);

    const elevationEGM96 = queryElevationAtPoint(lng, lat, tiles);
    if (Number.isFinite(elevationEGM96)) {
      const elevationWGS84 = convertElevationToWGS84(lat, lng, elevationEGM96);
      maxElevationWGS84 = Math.max(maxElevationWGS84, elevationWGS84);
    }
  }

  return Number.isFinite(maxElevationWGS84) ? maxElevationWGS84 : -Infinity;
}

export function queryMinMaxElevationAlongPolylineWGS84(
  line: [number, number][],
  tiles: TerrainTile[],
  samplesPerSegment: number = 20,
): { min: number; max: number } {
  let minElev = +Infinity;
  let maxElev = -Infinity;
  if (!Array.isArray(line) || line.length === 0) return { min: minElev, max: maxElev };

  for (let i = 0; i < line.length - 1; i++) {
    const [startLng, startLat] = line[i];
    const [endLng, endLat] = line[i + 1];
    for (let s = 0; s <= samplesPerSegment; s++) {
      const t = s / samplesPerSegment;
      const lng = startLng + t * (endLng - startLng);
      const lat = startLat + t * (endLat - startLat);
      const elevEGM96 = queryElevationAtPoint(lng, lat, tiles);
      if (!Number.isFinite(elevEGM96)) continue;
      const elevW = convertElevationToWGS84(lat, lng, elevEGM96);
      if (elevW < minElev) minElev = elevW;
      if (elevW > maxElev) maxElev = elevW;
    }
  }

  for (const [lng, lat] of line) {
    const elevEGM96 = queryElevationAtPoint(lng, lat, tiles);
    if (!Number.isFinite(elevEGM96)) continue;
    const elevW = convertElevationToWGS84(lat, lng, elevEGM96);
    if (elevW < minElev) minElev = elevW;
    if (elevW > maxElev) maxElev = elevW;
  }

  return { min: minElev, max: maxElev };
}

function normaliseDegrees(angleDeg: number): number {
  const wrapped = angleDeg % 360;
  return wrapped >= 0 ? wrapped : wrapped + 360;
}

function lineLengthMeters(line: [number, number][]): number {
  let totalDistanceM = 0;
  for (let index = 1; index < line.length; index += 1) {
    totalDistanceM += haversineDistance(line[index - 1], line[index]);
  }
  return totalDistanceM;
}

function dedupePolyline2D(line: [number, number][]): [number, number][] {
  return line.filter((point, index) => {
    if (index === 0) return true;
    const previous = line[index - 1];
    return point[0] !== previous[0] || point[1] !== previous[1];
  });
}

function getLastLoiterAnchorIndex(
  line: [number, number][],
  turnBlock: PlannedTurnBlock,
): number {
  const toleranceM = Math.max(TURN_LIFT_CIRCLE_TOLERANCE_M, turnBlock.loiterRadiusM * 0.15);
  let anchorIndex = -1;

  for (let index = 0; index < line.length - 1; index += 1) {
    const distanceToCenterM = haversineDistance(line[index], turnBlock.loiterCenter);
    if (Math.abs(distanceToCenterM - turnBlock.loiterRadiusM) <= toleranceM) {
      anchorIndex = index;
    }
  }

  return anchorIndex;
}

function getNearestPointIndex(
  line: [number, number][],
  targetPoint: [number, number],
): number {
  let bestIndex = -1;
  let bestDistanceM = Infinity;
  line.forEach((point, index) => {
    const distanceM = haversineDistance(point, targetPoint);
    if (distanceM < bestDistanceM) {
      bestDistanceM = distanceM;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function generateCoilLoopPoints(
  turnBlock: PlannedTurnBlock,
  anchorPoint: [number, number],
  loopCount: number,
): [number, number][] {
  if (loopCount <= 0) return [anchorPoint];

  const startBearingDeg = geoBearing(turnBlock.loiterCenter, anchorPoint);
  const loopCircumferenceM = 2 * Math.PI * turnBlock.loiterRadiusM;
  const pointsPerLoop = Math.max(36, Math.ceil(loopCircumferenceM / TURN_LIFT_LOOP_POINT_SPACING_M));
  const totalSteps = pointsPerLoop * loopCount;
  const coilPoints: [number, number][] = [anchorPoint];

  for (let step = 1; step <= totalSteps; step += 1) {
    if (step === totalSteps) {
      coilPoints.push(anchorPoint);
      break;
    }
    const deltaDeg =
      (TURN_LIFT_LOOP_DEGREES * loopCount * step) / totalSteps * (turnBlock.loiterDirection > 0 ? 1 : -1);
    const point = geoDestination(
      turnBlock.loiterCenter,
      normaliseDegrees(startBearingDeg + deltaDeg),
      turnBlock.loiterRadiusM,
    );
    coilPoints.push([point[0], point[1]]);
  }

  return coilPoints;
}

function extendTurnaroundForClimbRate(
  turnaroundLine: [number, number][],
  turnBlock: PlannedTurnBlock | undefined,
  previousSweepAltitudeM: number,
  nextSweepAltitudeM: number,
): [number, number][] {
  if (!turnBlock || turnaroundLine.length < 2) return turnaroundLine;

  const altitudeDeltaM = Math.abs(nextSweepAltitudeM - previousSweepAltitudeM);
  const existingTurnLengthM = lineLengthMeters(turnaroundLine);
  const maxLiftM = existingTurnLengthM * TURN_LIFT_ELEVATION_PER_METER;
  if (altitudeDeltaM <= maxLiftM + 1e-6) return turnaroundLine;

  const loopLengthM = 2 * Math.PI * turnBlock.loiterRadiusM;
  if (loopLengthM <= 1e-6) return turnaroundLine;

  const extraDistanceNeededM = altitudeDeltaM / TURN_LIFT_ELEVATION_PER_METER - existingTurnLengthM;
  const requiredLoopCount = Math.max(1, Math.ceil(extraDistanceNeededM / loopLengthM));
  const loiterExitPoint = turnBlock.loiterExitPoint;
  const loiterAnchorIndex =
    loiterExitPoint
      ? getNearestPointIndex(turnaroundLine, loiterExitPoint)
      : getLastLoiterAnchorIndex(turnaroundLine, turnBlock);
  if (loiterAnchorIndex < 0 || loiterAnchorIndex >= turnaroundLine.length - 1) {
    return turnaroundLine;
  }

  const loiterAnchor = loiterExitPoint ?? turnaroundLine[loiterAnchorIndex];
  const coilLoopPoints = generateCoilLoopPoints(turnBlock, loiterAnchor, requiredLoopCount);
  return dedupePolyline2D([
    ...turnaroundLine.slice(0, loiterAnchorIndex),
    ...coilLoopPoints,
    ...turnaroundLine.slice(loiterAnchorIndex + 1),
  ]);
}

export function calculateFlightLineSpacing(
  camera: CameraModel,
  altitudeAGL: number,
  sideOverlapPercent: number,
): number {
  const gsd = (altitudeAGL * camera.sx_m) / camera.f_m;
  const sensorWidthGround = gsd * camera.w_px;
  const overlapFraction = sideOverlapPercent / 100;
  const spacing = sensorWidthGround * (1 - overlapFraction);
  return spacing;
}

export function haversineDistance(coords1: [number, number], coords2: [number, number]): number {
  const R = 6371e3;
  const φ1 = (coords1[1] * Math.PI) / 180;
  const φ2 = (coords2[1] * Math.PI) / 180;
  const Δφ = ((coords2[1] - coords1[1]) * Math.PI) / 180;
  const Δλ = ((coords2[0] - coords1[0]) * Math.PI) / 180;

  const a = Math.sin(Δφ / 2) * Math.sin(Δφ / 2) + Math.cos(φ1) * Math.cos(φ2) * Math.sin(Δλ / 2) * Math.sin(Δλ / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
}

function midpoint2D(line: number[][]): [number, number] {
  const first = line[0] as [number, number];
  const last = line[line.length - 1] as [number, number];
  return [
    (first[0] + last[0]) * 0.5,
    (first[1] + last[1]) * 0.5,
  ];
}

function axialBearingDeltaDeg(a: number, b: number): number {
  const diff = ((a - b + 540) % 360) - 180;
  return Math.abs(Math.abs(diff) > 90 ? 180 - Math.abs(diff) : diff);
}

function projectedOffsetMeters(
  from: [number, number],
  to: [number, number],
  bearingDeg: number,
): { alongTrackM: number; crossTrackM: number } {
  const avgLatRad = ((from[1] + to[1]) * 0.5 * Math.PI) / 180;
  const eastM = (to[0] - from[0]) * 111_320 * Math.cos(avgLatRad);
  const northM = (to[1] - from[1]) * 110_540;
  const bearingRad = (bearingDeg * Math.PI) / 180;
  const alongX = Math.sin(bearingRad);
  const alongY = Math.cos(bearingRad);
  const crossX = Math.sin(bearingRad + Math.PI / 2);
  const crossY = Math.cos(bearingRad + Math.PI / 2);
  return {
    alongTrackM: eastM * alongX + northM * alongY,
    crossTrackM: eastM * crossX + northM * crossY,
  };
}

function isSameSweepLine(
  previousLine: number[][],
  currentLine: number[][],
  lineSpacing: number,
): boolean {
  if (previousLine.length < 2 || currentLine.length < 2) return false;

  const previousBearing = geoBearing(
    previousLine[0] as [number, number],
    previousLine[previousLine.length - 1] as [number, number],
  );
  const currentBearing = geoBearing(
    currentLine[0] as [number, number],
    currentLine[currentLine.length - 1] as [number, number],
  );
  if (axialBearingDeltaDeg(previousBearing, currentBearing) > 10) {
    return false;
  }

  const previousMidpoint = midpoint2D(previousLine);
  const currentMidpoint = midpoint2D(currentLine);
  const { alongTrackM, crossTrackM } = projectedOffsetMeters(previousMidpoint, currentMidpoint, previousBearing);

  return Math.abs(crossTrackM) <= Math.max(6, lineSpacing * 0.25) && Math.abs(alongTrackM) > 2;
}

export function getLineTraversalDirections(
  lines: number[][][],
  lineSpacing: number,
  sweepIndices?: number[],
): boolean[] {
  const traversalDirections = new Array<boolean>(lines.length);
  for (const sweep of groupFlightLinesForTraversal(lines, lineSpacing, sweepIndices)) {
    for (const fragmentIndex of sweep.fragmentIndices) {
      traversalDirections[fragmentIndex] = sweep.directionForward;
    }
  }
  return traversalDirections;
}

type FlightSweepTraversal = {
  sweepIndex: number;
  directionForward: boolean;
  fragments: number[][][];
  fragmentIndices: number[];
};

export function groupFlightLinesForTraversal(
  lines: number[][][],
  lineSpacing: number,
  sweepIndices?: number[],
): FlightSweepTraversal[] {
  const sweeps: FlightSweepTraversal[] = [];
  let directionForward = true;
  let previousRawLine: number[][] | null = null;

  lines.forEach((line, i) => {
    const sameSweepAsPrevious = previousRawLine
      ? (
          sweepIndices &&
          Number.isFinite(sweepIndices[i - 1]) &&
          Number.isFinite(sweepIndices[i])
            ? sweepIndices[i - 1] === sweepIndices[i]
            : isSameSweepLine(previousRawLine, line, lineSpacing)
        )
      : false;
    if (i > 0 && !sameSweepAsPrevious) {
      directionForward = !directionForward;
    }
    if (!sameSweepAsPrevious || sweeps.length === 0) {
      sweeps.push({
        sweepIndex: sweepIndices?.[i] ?? sweeps.length,
        directionForward,
        fragments: [line],
        fragmentIndices: [i],
      });
    } else {
      const currentSweep = sweeps[sweeps.length - 1];
      currentSweep.fragments.push(line);
      currentSweep.fragmentIndices.push(i);
    }
    previousRawLine = line;
  });

  return sweeps;
}

export function getPolygonBounds(ring: number[][]) {
  let minLng = Infinity;
  let minLat = Infinity;
  let maxLng = -Infinity;
  let maxLat = -Infinity;
  for (const [lng, lat] of ring) {
    minLng = Math.min(minLng, lng);
    minLat = Math.min(minLat, lat);
    maxLng = Math.max(maxLng, lng);
    maxLat = Math.max(maxLat, lat);
  }
  return {
    minLng,
    minLat,
    maxLng,
    maxLat,
    centroid: [(minLng + maxLng) / 2, (minLat + maxLat) / 2] as [number, number],
  };
}

export function calculateOptimalLineSpacing(ring: number[][], bearingDeg: number): number {
  const bounds = getPolygonBounds(ring);
  const { centroid } = bounds;

  const perpBearing = (bearingDeg + 90) % 360;

  let minProjection = Infinity;
  let maxProjection = -Infinity;

  for (const point of ring) {
    if (point.length < 2) continue;

    const distanceM = haversineDistance(centroid, [point[0], point[1]]);
    const pointBearing = geoBearing(centroid, [point[0], point[1]]);
    const angleDiff = ((pointBearing - perpBearing + 540) % 360) - 180;
    const projection = distanceM * Math.cos((angleDiff * Math.PI) / 180);

    minProjection = Math.min(minProjection, projection);
    maxProjection = Math.max(maxProjection, projection);
  }

  const perpendicularWidthM = maxProjection - minProjection;

  let spacing: number;
  if (perpendicularWidthM < 200) {
    spacing = 25;
  } else if (perpendicularWidthM < 500) {
    spacing = 50;
  } else if (perpendicularWidthM < 1000) {
    spacing = 100;
  } else if (perpendicularWidthM < 2000) {
    spacing = 150;
  } else {
    spacing = 200;
  }

  const estimatedLines = Math.ceil(perpendicularWidthM / spacing);
  if (estimatedLines < 3) {
    spacing = perpendicularWidthM / 3;
  } else if (estimatedLines > 20) {
    spacing = perpendicularWidthM / 20;
  }

  return spacing;
}

export function extendFlightLineForTurnRunout(
  line: number[][],
  runoutM: number,
): number[][] {
  if (!Array.isArray(line) || line.length < 2 || !(runoutM > 0)) return Array.isArray(line) ? [...line] : [];

  const first = line[0] as [number, number];
  const second = line[1] as [number, number];
  const last = line[line.length - 1] as [number, number];
  const penultimate = line[line.length - 2] as [number, number];

  const startBearing = geoBearing([first[0], first[1]], [second[0], second[1]]);
  const endBearing = geoBearing([penultimate[0], penultimate[1]], [last[0], last[1]]);
  const startExtended = geoDestination([first[0], first[1]], (startBearing + 180) % 360, runoutM);
  const endExtended = geoDestination([last[0], last[1]], endBearing, runoutM);

  return [
    [startExtended[0], startExtended[1]],
    ...line,
    [endExtended[0], endExtended[1]],
  ];
}

function buildFillet(
  P0: [number, number, number],
  P2: [number, number, number],
  dir0: number,
  dir2: number,
  r: number,
): [number, number, number][] {
  const directDistance = haversineDistance([P0[0], P0[1]], [P2[0], P2[1]]);
  if (directDistance < r * 0.1) {
    return [P0, P2];
  }

  const turnAngle = ((dir2 - dir0 + 540) % 360) - 180;
  if (Math.abs(turnAngle) < 5) {
    return [P0, P2];
  }

  const controlDistance = Math.min(r, directDistance * 0.4);
  const P1_control = geoDestination([P0[0], P0[1]], dir0, controlDistance);
  const P2_control = geoDestination([P2[0], P2[1]], (dir2 + 180) % 360, controlDistance);

  const numPoints = Math.max(16, Math.ceil(directDistance / 15));
  const points: [number, number, number][] = [];

  for (let i = 0; i <= numPoints; i++) {
    const t = i / numPoints;
    const t2 = t * t;
    const t3 = t2 * t;
    const mt = 1 - t;
    const mt2 = mt * mt;
    const mt3 = mt2 * mt;

    const lng = mt3 * P0[0] + 3 * mt2 * t * P1_control[0] + 3 * mt * t2 * P2_control[0] + t3 * P2[0];
    const lat = mt3 * P0[1] + 3 * mt2 * t * P1_control[1] + 3 * mt * t2 * P2_control[1] + t3 * P2[1];
    const alt = P0[2] + t * (P2[2] - P0[2]);

    points.push([lng, lat, alt]);
  }
  return points;
}

export function build3DFlightPath(
  lines: number[][][] | PlannedFlightGeometry,
  tiles: TerrainTile[],
  lineSpacing: number,
  opts:
    | { altitudeAGL: number; mode?: 'legacy' | 'min-clearance'; minClearance?: number; turnExtendM?: number; preconnected?: boolean }
    | number = 100,
  sweepIndices?: number[],
): [number, number, number][][] {
  const usingGeometryInput = !Array.isArray(lines);
  const geometryInput = usingGeometryInput ? (lines as PlannedFlightGeometry) : undefined;
  const sourceLines = geometryInput?.connectedLines ?? lines;
  const path: [number, number, number][][] = [];
  const usingLegacySig = typeof opts === 'number';
  const altitudeAGL = usingLegacySig ? (opts as number) : (opts as any).altitudeAGL;
  const mode: 'legacy' | 'min-clearance' = usingLegacySig ? 'legacy' : ((opts as any).mode ?? 'legacy');
  const minClearance = usingLegacySig ? 60 : Math.max(0, (opts as any).minClearance ?? 60);
  const homeAltitude = usingLegacySig ? undefined : ((opts as any).homeAltitude as number | undefined);
  const turnExtendM = usingLegacySig ? 0 : Math.max(0, (opts as any).turnExtendM ?? 0);
  const preconnected = usingGeometryInput || (!usingLegacySig && Boolean((opts as any).preconnected));
  const pointElevationWgs84Cache = new Map<string, number>();
  const polylineElevationExtremaCache = new WeakMap<readonly [number, number][], Map<number, { min: number; max: number }>>();
  const segmentElevationMaxCache = new Map<string, number>();

  const getPointElevationWGS84 = (lng: number, lat: number) => {
    const key = `${lng},${lat}`;
    if (pointElevationWgs84Cache.has(key)) {
      return pointElevationWgs84Cache.get(key)!;
    }
    const elevationEGM96 = queryElevationAtPoint(lng, lat, tiles);
    const elevationWGS84 = Number.isFinite(elevationEGM96) ? convertElevationToWGS84(lat, lng, elevationEGM96) : NaN;
    pointElevationWgs84Cache.set(key, elevationWGS84);
    return elevationWGS84;
  };

  const getPolylineElevationExtrema = (
    line: [number, number][],
    samplesPerSegment: number,
  ): { min: number; max: number } => {
    let samplesCache = polylineElevationExtremaCache.get(line);
    if (!samplesCache) {
      samplesCache = new Map<number, { min: number; max: number }>();
      polylineElevationExtremaCache.set(line, samplesCache);
    }
    const cached = samplesCache.get(samplesPerSegment);
    if (cached) return cached;

    let minElev = +Infinity;
    let maxElev = -Infinity;
    if (Array.isArray(line) && line.length > 0) {
      for (let i = 0; i < line.length - 1; i++) {
        const [startLng, startLat] = line[i];
        const [endLng, endLat] = line[i + 1];
        for (let s = 0; s <= samplesPerSegment; s++) {
          const t = s / samplesPerSegment;
          const lng = startLng + t * (endLng - startLng);
          const lat = startLat + t * (endLat - startLat);
          const elevW = getPointElevationWGS84(lng, lat);
          if (!Number.isFinite(elevW)) continue;
          if (elevW < minElev) minElev = elevW;
          if (elevW > maxElev) maxElev = elevW;
        }
      }

      for (const [lng, lat] of line) {
        const elevW = getPointElevationWGS84(lng, lat);
        if (!Number.isFinite(elevW)) continue;
        if (elevW < minElev) minElev = elevW;
        if (elevW > maxElev) maxElev = elevW;
      }
    }

    const result = { min: minElev, max: maxElev };
    samplesCache.set(samplesPerSegment, result);
    return result;
  };

  const getMaxElevationAlongLineCached = (
    startLng: number,
    startLat: number,
    endLng: number,
    endLat: number,
    samples: number,
  ) => {
    const key = `${startLng},${startLat}|${endLng},${endLat}|${samples}`;
    if (segmentElevationMaxCache.has(key)) {
      return segmentElevationMaxCache.get(key)!;
    }

    let maxElevationWGS84 = -Infinity;
    for (let i = 0; i <= samples; i++) {
      const t = i / samples;
      const lng = startLng + t * (endLng - startLng);
      const lat = startLat + t * (endLat - startLat);
      const elevationWGS84 = getPointElevationWGS84(lng, lat);
      if (Number.isFinite(elevationWGS84)) {
        maxElevationWGS84 = Math.max(maxElevationWGS84, elevationWGS84);
      }
    }

    const result = Number.isFinite(maxElevationWGS84) ? maxElevationWGS84 : -Infinity;
    segmentElevationMaxCache.set(key, result);
    return result;
  };

  const computeLineAltitude = (mergedLine: [number, number][]) => {
    const { min: lineMinElev, max: lineMaxElev } = getPolylineElevationExtrema(mergedLine, 20);
    let extendedMax = lineMaxElev;
    if (!usingLegacySig && (opts as any).turnExtendM && (opts as any).turnExtendM > 0 && mergedLine.length >= 2) {
      const te = Math.max(0, (opts as any).turnExtendM as number);
      const L0 = mergedLine[0] as [number, number];
      const L1 = mergedLine[1] as [number, number];
      const LN = mergedLine[mergedLine.length - 1] as [number, number];
      const sweepBrg = geoBearing([L0[0], L0[1]], [L1[0], L1[1]]);
      const startExt = geoDestination([L0[0], L0[1]], (sweepBrg + 180) % 360, te);
      const endExt = geoDestination([LN[0], LN[1]], sweepBrg, te);
      const startMax = getMaxElevationAlongLineCached(startExt[0], startExt[1], L0[0], L0[1], 20);
      const endMax = getMaxElevationAlongLineCached(LN[0], LN[1], endExt[0], endExt[1], 20);
      if (Number.isFinite(startMax)) extendedMax = Math.max(extendedMax, startMax);
      if (Number.isFinite(endMax)) extendedMax = Math.max(extendedMax, endMax);
    }

    if (mode === 'legacy') {
      const refElev = Number.isFinite(extendedMax) ? extendedMax : 0;
      return refElev + altitudeAGL;
    }

    const a1 = Number.isFinite(lineMinElev) ? (lineMinElev + altitudeAGL) : altitudeAGL;
    const a2 = Number.isFinite(extendedMax) ? (extendedMax + minClearance) : altitudeAGL;
    return Math.max(a1, a2);
  };

  const createEaseInOut = (min: number, max: number, ease = TURN_LIFT_EASING_POWER) => {
    const range = max - min;
    return (value: number) => {
      const normalizedValue = Math.max(0, Math.min(1, value));
      return (
        min +
        range *
          (normalizedValue < 0.5
            ? 0.5 * (2 * normalizedValue) ** ease
            : 1 - 0.5 * (2 * (1 - normalizedValue)) ** ease)
      );
    };
  };

  const setLineAltitudeSmooth = (
    line: [number, number][],
    altitudeA: number,
    altitudeB: number = altitudeA,
  ): [number, number, number][] => {
    if (line.length === 0) return [];
    if (altitudeA === altitudeB) {
      return line.map(([lng, lat]) => [lng, lat, altitudeA] as [number, number, number]);
    }

    const reversed = altitudeA > altitudeB;
    const orientedLine = reversed ? [...line].reverse() : line;
    const ease = createEaseInOut(Math.min(altitudeA, altitudeB), Math.max(altitudeA, altitudeB));

    const stepLengths = orientedLine.map((point, index) =>
      index > 0 ? haversineDistance(orientedLine[index - 1] as [number, number], point as [number, number]) : 0,
    );
    const totalDistance = Math.max(stepLengths.reduce((sum, value) => sum + value, 0), 1e-9);
    let traversedDistance = 0;

    const lifted = orientedLine.map(([lng, lat], index) => {
      if (index > 0) traversedDistance += stepLengths[index];
      return [lng, lat, ease(traversedDistance / totalDistance)] as [number, number, number];
    });

    return reversed ? lifted.reverse() : lifted;
  };

  const getLineMaxElevation = (line: [number, number][]) => getPolylineElevationExtrema(line, 20).max;

  const calculateSweepAltitudeWithAdjacentTurnarounds = (
    sweepLine: [number, number][],
    adjacentLines: [number, number][][],
  ) => {
    const { min: sweepMinElevation, max: sweepMaxElevation } = getPolylineElevationExtrema(sweepLine, 20);
    const adjacentMaxElevation = adjacentLines.reduce((maxElevation, line) => {
      if (!Array.isArray(line) || line.length < 2) return maxElevation;
      return Math.max(maxElevation, getLineMaxElevation(line));
    }, -Infinity);
    const pathMaxElevation = Math.max(sweepMaxElevation, adjacentMaxElevation);

    if (mode === 'legacy') {
      return Number.isFinite(pathMaxElevation) ? pathMaxElevation + altitudeAGL : altitudeAGL;
    }

    const hagAltitude = Number.isFinite(sweepMinElevation) ? sweepMinElevation + altitudeAGL : altitudeAGL;
    const turnaroundSafeAltitude = Number.isFinite(pathMaxElevation)
      ? pathMaxElevation + minClearance + TURN_LIFT_ALTITUDE_MARGIN_M
      : altitudeAGL;
    return Math.max(hagAltitude, turnaroundSafeAltitude);
  };

  const computeConnectorPointAltitude = (
    point: [number, number],
    interpolationRatio: number,
    startAltitude: number,
    endAltitude: number,
  ) => {
    const interpolatedAltitude = startAltitude + (endAltitude - startAltitude) * interpolationRatio;
    const terrainElevationWGS84 = getPointElevationWGS84(point[0], point[1]);
    if (!Number.isFinite(terrainElevationWGS84)) return interpolatedAltitude;
    const clearanceMargin = mode === 'legacy' ? altitudeAGL : minClearance;
    return Math.max(interpolatedAltitude, terrainElevationWGS84 + clearanceMargin);
  };

  const liftConnectedSegments = (
    connectedSegments: [number, number][][],
    sweepAltitudes: number[],
  ) =>
    connectedSegments.map((segment, segmentIndex) => {
      if (segmentIndex % 2 === 0) {
        const sweepAltitude = sweepAltitudes[Math.floor(segmentIndex / 2)] ?? altitudeAGL;
        return segment.map(([lng, lat]) => [lng, lat, sweepAltitude] as [number, number, number]);
      }

      const previousSweepAltitude = sweepAltitudes[Math.floor((segmentIndex - 1) / 2)] ?? altitudeAGL;
      const nextSweepAltitude = sweepAltitudes[Math.floor((segmentIndex + 1) / 2)] ?? previousSweepAltitude;
      return setLineAltitudeSmooth(segment as [number, number][], previousSweepAltitude, nextSweepAltitude);
    });

  if (preconnected && geometryInput) {
    const connectedSegments = geometryInput.connectedLines
      .map((line) => (line as [number, number][]).filter((point) => Array.isArray(point) && point.length >= 2))
      .filter((line) => line.length >= 2);

    if (homeAltitude !== undefined) {
      return connectedSegments.map((segment) =>
        segment.map(([lng, lat]) => [lng, lat, homeAltitude + altitudeAGL] as [number, number, number]),
      );
    }

    const leadInLine =
      geometryInput.leadInPoints.length > 1 ? (geometryInput.leadInPoints.slice(1) as [number, number][]) : undefined;
    const leadOutLine =
      geometryInput.leadOutPoints.length > 1
        ? (geometryInput.leadOutPoints.slice(0, -1) as [number, number][])
        : undefined;

    const calculateSweepAltitudes = (turnaroundSegments: [number, number][][]) => geometryInput.sweepLines.map((sweepLine, sweepIndex) => {
      const adjacentLines: [number, number][][] = [];
      const previousTurnaround = turnaroundSegments[sweepIndex * 2 - 1];
      const nextTurnaround = turnaroundSegments[sweepIndex * 2 + 1];

      if (previousTurnaround) adjacentLines.push(previousTurnaround as [number, number][]);
      if (nextTurnaround) adjacentLines.push(nextTurnaround as [number, number][]);
      if (sweepIndex === 0 && leadInLine) adjacentLines.push(leadInLine);
      if (sweepIndex === geometryInput.sweepLines.length - 1 && leadOutLine) adjacentLines.push(leadOutLine);

      return calculateSweepAltitudeWithAdjacentTurnarounds(sweepLine as [number, number][], adjacentLines);
    });

    const buildTurnaroundSegments = (sweepAltitudes: number[]) =>
      connectedSegments.map((segment, segmentIndex) => {
        if (segmentIndex % 2 === 0) return segment;
        return extendTurnaroundForClimbRate(
          segment as [number, number][],
          geometryInput.turnBlocks[Math.floor(segmentIndex / 2)],
          sweepAltitudes[Math.floor((segmentIndex - 1) / 2)] ?? altitudeAGL,
          sweepAltitudes[Math.floor((segmentIndex + 1) / 2)] ?? altitudeAGL,
        );
      });

    const provisionalSweepAltitudes = calculateSweepAltitudes(connectedSegments);
    const expandedTurnaroundSegments = buildTurnaroundSegments(provisionalSweepAltitudes);
    const turnaroundsChanged = expandedTurnaroundSegments.some(
      (segment, segmentIndex) => segmentIndex % 2 === 1 && segment !== connectedSegments[segmentIndex],
    );
    if (!turnaroundsChanged) {
      return liftConnectedSegments(connectedSegments, provisionalSweepAltitudes);
    }
    const finalSweepAltitudes = calculateSweepAltitudes(expandedTurnaroundSegments);
    const finalConnectedSegments = buildTurnaroundSegments(finalSweepAltitudes);
    const resolvedSweepAltitudes = calculateSweepAltitudes(finalConnectedSegments);

    return liftConnectedSegments(finalConnectedSegments, resolvedSweepAltitudes);
  }

  if (preconnected) {
    const mergedSegments = (sourceLines as number[][][])
      .map((line) => (line as [number, number][]).filter((point) => Array.isArray(point) && point.length >= 2))
      .filter((line) => line.length >= 2);

    const baseAltitudes = mergedSegments.map((segment) => computeLineAltitude(segment));

    mergedSegments.forEach((mergedLine, segmentIndex) => {
      const segmentAltitude = baseAltitudes[segmentIndex];
      if (segmentIndex % 2 === 1) {
        const previousSweepAltitude = baseAltitudes[Math.max(0, segmentIndex - 1)] ?? segmentAltitude;
        const nextSweepAltitude = baseAltitudes[Math.min(baseAltitudes.length - 1, segmentIndex + 1)] ?? segmentAltitude;
        const pointCount = Math.max(mergedLine.length - 1, 1);
        path.push(
          mergedLine.map(([lng, lat], pointIndex) => {
            const interpolationRatio = pointIndex / pointCount;
            const connectorAltitude = computeConnectorPointAltitude(
              [lng, lat],
              interpolationRatio,
              previousSweepAltitude,
              nextSweepAltitude,
            );
            return [lng, lat, connectorAltitude] as [number, number, number];
          }),
        );
        return;
      }
      path.push(mergedLine.map(([lng, lat]) => [lng, lat, segmentAltitude] as [number, number, number]));
    });

    return path;
  }

  const sweeps = groupFlightLinesForTraversal(sourceLines as number[][][], lineSpacing, sweepIndices);

  sweeps.forEach((sweep) => {
    const orderedFragments = sweep.directionForward ? sweep.fragments : [...sweep.fragments].reverse();
    const orientedFragments = orderedFragments
      .map((fragment) => (sweep.directionForward ? fragment : [...fragment].reverse()))
      .filter((fragment) => Array.isArray(fragment) && fragment.length >= 2);
    if (orientedFragments.length === 0) return;

    const mergedLine: [number, number][] = [];
    orientedFragments.forEach((fragment, fragmentIndex) => {
      const typedFragment = fragment as [number, number][];
      if (fragmentIndex === 0) {
        mergedLine.push(...typedFragment);
        return;
      }
      const previous = mergedLine[mergedLine.length - 1];
      const first = typedFragment[0];
      if (!previous || previous[0] !== first[0] || previous[1] !== first[1]) {
        mergedLine.push(first);
      }
      mergedLine.push(...typedFragment.slice(1));
    });
    if (mergedLine.length < 2) return;

    const flightAltitude = computeLineAltitude(mergedLine);
    const coords = mergedLine.map(([lng, lat]) => [lng, lat, flightAltitude] as [number, number, number]);

    if (path.length > 0) {
      const lastSeg = path[path.length - 1];
      const P0 = lastSeg[lastSeg.length - 1];
      const P2 = coords[0];

      const dirPrev =
        lastSeg.length >= 2
          ? geoBearing([lastSeg[lastSeg.length - 2][0], lastSeg[lastSeg.length - 2][1]], [P0[0], P0[1]])
          : geoBearing([P0[0], P0[1]], [P2[0], P2[1]]);
      const dirNext =
        coords.length >= 2
          ? geoBearing([P2[0], P2[1]], [coords[1][0], coords[1][1]])
          : geoBearing([P0[0], P0[1]], [P2[0], P2[1]]);

      const filletRadius = Math.max(30, lineSpacing / 2);
      const startForFillet: [number, number, number] = turnExtendM > 0
        ? (() => { const p = geoDestination([P0[0], P0[1]], dirPrev, turnExtendM); return [p[0], p[1], P0[2]]; })()
        : P0;
      const endForFillet: [number, number, number] = turnExtendM > 0
        ? (() => { const p = geoDestination([P2[0], P2[1]], (dirNext + 180) % 360, turnExtendM); return [p[0], p[1], P2[2]]; })()
        : P2;
      let fillet = buildFillet(startForFillet, endForFillet, dirPrev, dirNext, filletRadius);

      if (fillet.length > 2) {
        if (mode === 'min-clearance') {
          const fillet2D = fillet.map((point) => [point[0], point[1]] as [number, number]);
          const { max: connMaxElev } = queryMinMaxElevationAlongPolylineWGS84(fillet2D, tiles, 12);
          const needed = Number.isFinite(connMaxElev) ? (connMaxElev + minClearance) : undefined;
          if (needed && fillet[0][2] < needed) {
            fillet = fillet.map((point) => [point[0], point[1], needed] as [number, number, number]);
          }
        }
        const connector = turnExtendM > 0
          ? [
              P0,
              startForFillet,
              ...fillet.slice(1, -1),
              endForFillet,
              P2,
            ]
          : fillet;
        path.push(connector);
      } else {
        let connectorAltitude = Math.max(P0[2], P2[2]);
        if (mode === 'min-clearance') {
          const { max: connMaxElev } = queryMinMaxElevationAlongPolylineWGS84([[P0[0], P0[1]], [P2[0], P2[1]]], tiles, 12);
          const needed = Number.isFinite(connMaxElev) ? (connMaxElev + minClearance) : undefined;
          if (needed && connectorAltitude < needed) connectorAltitude = needed;
        }
        if (turnExtendM > 0) {
          const p0ext = geoDestination([P0[0], P0[1]], dirPrev, turnExtendM);
          const p2ext = geoDestination([P2[0], P2[1]], (dirNext + 180) % 360, turnExtendM);
          path.push([
            [P0[0], P0[1], connectorAltitude],
            [p0ext[0], p0ext[1], connectorAltitude],
            [p2ext[0], p2ext[1], connectorAltitude],
            [P2[0], P2[1], connectorAltitude],
          ]);
        } else {
          path.push([[P0[0], P0[1], connectorAltitude], [P2[0], P2[1], connectorAltitude]]);
        }
      }
    }
    path.push(coords);
  });
  return path;
}

export function sampleCameraPositionsOnFlightPath(
  path3D: [number, number, number][][],
  photoSpacingMeters: number,
  opts?: { includeTurns?: boolean },
): [number, number, number, number][] {
  const cameraPositions: [number, number, number, number][] = [];
  const includeTurns = !!opts?.includeTurns;

  for (let segIndex = 0; segIndex < path3D.length; segIndex++) {
    if (!includeTurns && (segIndex % 2 === 1)) continue;
    const segment = path3D[segIndex];
    if (segment.length < 2) continue;

    let totalDistance = 0;
    let lastPhotoDistance = 0;

    if (segment.length >= 2) {
      const initialBearing = geoBearing([segment[0][0], segment[0][1]], [segment[1][0], segment[1][1]]);
      cameraPositions.push([segment[0][0], segment[0][1], segment[0][2], initialBearing]);
    }

    for (let i = 1; i < segment.length; i++) {
      const prevPoint = segment[i - 1];
      const currPoint = segment[i];
      const segmentBearing = geoBearing([prevPoint[0], prevPoint[1]], [currPoint[0], currPoint[1]]);
      const stepDistance = haversineDistance([prevPoint[0], prevPoint[1]], [currPoint[0], currPoint[1]]);
      const stepStartDistance = totalDistance;
      totalDistance += stepDistance;

      while (lastPhotoDistance + photoSpacingMeters <= totalDistance) {
        const targetDistance = lastPhotoDistance + photoSpacingMeters;
        const distanceIntoStep = targetDistance - stepStartDistance;
        const interpolationRatio = stepDistance > 0 ? distanceIntoStep / stepDistance : 0;

        const lng = prevPoint[0] + (currPoint[0] - prevPoint[0]) * interpolationRatio;
        const lat = prevPoint[1] + (currPoint[1] - prevPoint[1]) * interpolationRatio;
        const alt = prevPoint[2] + (currPoint[2] - prevPoint[2]) * interpolationRatio;

        cameraPositions.push([lng, lat, alt, segmentBearing]);
        lastPhotoDistance = targetDistance;
      }
    }

    if (segment.length >= 2) {
      const N = segment.length;
      const endBearing = geoBearing([segment[N - 2][0], segment[N - 2][1]], [segment[N - 1][0], segment[N - 1][1]]);
      const end = segment[N - 1];
      const prevCam = cameraPositions[cameraPositions.length - 1];
      const tailGap = haversineDistance([prevCam[0], prevCam[1]], [end[0], end[1]]);
      if (tailGap > 0.25 * photoSpacingMeters) {
        cameraPositions.push([end[0], end[1], end[2], endBearing]);
      }
    }
  }

  return cameraPositions;
}

export function calculateOptimalTerrainZoom(polygon: { coordinates: number[][] }): number {
  const coords = polygon.coordinates;
  if (coords.length < 3) return 15;

  let area = 0;
  const n = coords.length;

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const [lng1, lat1] = coords[i];
    const [lng2, lat2] = coords[j];

    const φ1 = (lat1 * Math.PI) / 180;
    const φ2 = (lat2 * Math.PI) / 180;
    const Δλ = ((lng2 - lng1) * Math.PI) / 180;

    area += Δλ * (2 + Math.sin(φ1) + Math.sin(φ2));
  }

  area = (Math.abs(area * 6371000 * 6371000)) / 2;

  if (area < 100000) {
    return 15;
  }
  if (area < 1000000) {
    return 14;
  }
  if (area < 10000000) {
    return 13;
  }
  return 12;
}
