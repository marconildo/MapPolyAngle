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
import type { CameraModel } from '@/domain/types';

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
  lines: number[][][],
  tiles: TerrainTile[],
  lineSpacing: number,
  opts: { altitudeAGL: number; mode?: 'legacy' | 'min-clearance'; minClearance?: number; turnExtendM?: number } | number = 100,
  sweepIndices?: number[],
): [number, number, number][][] {
  const path: [number, number, number][][] = [];
  const usingLegacySig = typeof opts === 'number';
  const altitudeAGL = usingLegacySig ? (opts as number) : (opts as any).altitudeAGL;
  const mode: 'legacy' | 'min-clearance' = usingLegacySig ? 'legacy' : ((opts as any).mode ?? 'legacy');
  const minClearance = usingLegacySig ? 60 : Math.max(0, (opts as any).minClearance ?? 60);
  const turnExtendM = usingLegacySig ? 0 : Math.max(0, (opts as any).turnExtendM ?? 0);
  const sweeps = groupFlightLinesForTraversal(lines, lineSpacing, sweepIndices);

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

    const { min: lineMinElev, max: lineMaxElev } = queryMinMaxElevationAlongPolylineWGS84(mergedLine as any, tiles, 20);
    let extendedMax = lineMaxElev;
    if (!usingLegacySig && (opts as any).turnExtendM && (opts as any).turnExtendM > 0 && mergedLine.length >= 2) {
      const te = Math.max(0, (opts as any).turnExtendM as number);
      const L0 = mergedLine[0] as [number, number];
      const L1 = mergedLine[1] as [number, number];
      const LN = mergedLine[mergedLine.length - 1] as [number, number];
      const sweepBrg = geoBearing([L0[0], L0[1]], [L1[0], L1[1]]);
      const startExt = geoDestination([L0[0], L0[1]], (sweepBrg + 180) % 360, te);
      const endExt = geoDestination([LN[0], LN[1]], sweepBrg, te);
      const startMax = queryMaxElevationAlongLineWGS84(startExt[0], startExt[1], L0[0], L0[1], tiles, 20);
      const endMax = queryMaxElevationAlongLineWGS84(LN[0], LN[1], endExt[0], endExt[1], tiles, 20);
      if (Number.isFinite(startMax)) extendedMax = Math.max(extendedMax, startMax);
      if (Number.isFinite(endMax)) extendedMax = Math.max(extendedMax, endMax);
    }

    let flightAltitude: number;
    if (mode === 'legacy') {
      const refElev = Number.isFinite(extendedMax) ? extendedMax : 0;
      flightAltitude = refElev + altitudeAGL;
    } else {
      const a1 = Number.isFinite(lineMinElev) ? (lineMinElev + altitudeAGL) : altitudeAGL;
      const a2 = Number.isFinite(extendedMax) ? (extendedMax + minClearance) : altitudeAGL;
      flightAltitude = Math.max(a1, a2);
    }
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
