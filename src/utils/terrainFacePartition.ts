import type { FlightParams, TerrainTile } from "@/domain/types";
import {
  DJI_ZENMUSE_P1_24MM,
  ILX_LR1_INSPECT_85MM,
  MAP61_17MM,
  RGB61_24MM,
  SONY_RX1R2,
  SONY_RX1R3,
  SONY_A6100_20MM,
  lineSpacingRotated,
} from "@/domain/camera";
import { DEFAULT_LIDAR, LIDAR_REGISTRY, lidarLineSpacing } from "@/domain/lidar";
import {
  dominantContourDirectionPlaneFit,
  destination as geoDestination,
  type Polygon as TerrainPolygon,
  queryElevationAtPoint,
} from "@/utils/terrainAspectHybrid";
import { generatePlannedFlightGeometryForPolygon, summarizePlannedFlightGeometry } from "@/flight/plannedGeometry";

// Turf typings are noisy in this repo. Keep usage narrow and geometry-centric.
// @ts-ignore
import * as turf from "@turf/turf";

type Ring = [number, number][];

const CAMERA_REGISTRY: Record<string, any> = {
  SONY_RX1R2,
  SONY_RX1R3,
  SONY_A6100_20MM,
  DJI_ZENMUSE_P1_24MM,
  ILX_LR1_INSPECT_85MM,
  MAP61_17MM,
  RGB61_24MM,
};

export type TerrainFacePartitionOptions = {
  maxPolygons?: number;
  candidateAngleStepDeg?: number;
  candidateOffsetFractions?: number[];
  minAreaM2?: number;
  maxAspectRatio?: number;
  minConvexity?: number;
  minImprovement?: number;
  splitPenalty?: number;
  mergeSlack?: number;
  forceAtLeastOneSplit?: boolean;
  forceMaxScoreRegression?: number;
  forceMinDirectionDeltaDeg?: number;
  searchSampleStep?: number;
};

export type FlightPatternMetrics = {
  lineCount: number;
  fragmentedLineCount: number;
  meanLineLengthM: number;
  medianLineLengthM: number;
  shortLineCount: number;
};

export type PolygonScoreBreakdown = {
  terrainPenalty: number;
  flightPenalty: number;
  shapePenalty: number;
};

export type EvaluatedPartitionPolygon = {
  ring: Ring;
  areaM2: number;
  contourDirDeg: number;
  lineSpacingM: number;
  score: number;
  fitQuality?: "excellent" | "good" | "fair" | "poor";
  rmse?: number;
  rSquared?: number;
  convexity: number;
  aspectRatio: number;
  crossTrackWidthM: number;
  alongTrackLengthM: number;
  flight: FlightPatternMetrics;
  breakdown: PolygonScoreBreakdown;
};

export type TerrainFacePartitionResult = {
  polygons: Ring[];
  evaluations: EvaluatedPartitionPolygon[];
  iterations: number;
};

export type SplitDebugStats = {
  totalCandidates: number;
  failedSplitGeometry: number;
  failedLeftEval: number;
  failedRightEval: number;
  failedBalance: number;
  failedImprovement: number;
  failedForcedDirection: number;
  failedForcedRegression: number;
  acceptedCandidates: number;
  bestImprovement?: number;
  bestForcedUtility?: number;
};

type TerrainDescriptorPoint = {
  lng: number;
  lat: number;
  x: number;
  y: number;
  contourDirDeg: number;
  slopeMagnitude: number;
  breakStrength: number;
};

type TerrainBreakDescriptor = {
  points: TerrainDescriptorPoint[];
  weightedCentroid?: [number, number];
  principalBearingDeg?: number;
  gridStepM: number;
};

type SplitCandidate = {
  improvement: number;
  splitPenalty: number;
  bearingDeg: number;
  offsetM: number;
  combinedScore: number;
  directionDeltaDeg: number;
  boundaryBonus: number;
  children: [EvaluatedPartitionPolygon, EvaluatedPartitionPolygon];
};

const DEFAULT_OPTIONS: Required<TerrainFacePartitionOptions> = {
  maxPolygons: 4,
  candidateAngleStepDeg: 15,
  candidateOffsetFractions: [-0.35, -0.18, 0, 0.18, 0.35],
  minAreaM2: 4000,
  maxAspectRatio: 8,
  minConvexity: 0.45,
  minImprovement: 0.9,
  splitPenalty: 1.4,
  mergeSlack: 0.35,
  forceAtLeastOneSplit: false,
  forceMaxScoreRegression: 30,
  forceMinDirectionDeltaDeg: 0,
  searchSampleStep: 4,
};

function normalizeRing(ring: Ring): Ring | null {
  const cleaned = ring.filter(
    (coord): coord is [number, number] =>
      Array.isArray(coord) &&
      coord.length >= 2 &&
      Number.isFinite(coord[0]) &&
      Number.isFinite(coord[1]),
  );
  if (cleaned.length < 3) return null;
  const [firstLng, firstLat] = cleaned[0];
  const [lastLng, lastLat] = cleaned[cleaned.length - 1];
  if (firstLng === lastLng && firstLat === lastLat) return cleaned;
  return [...cleaned, [firstLng, firstLat]];
}

function ringFeature(ring: Ring) {
  return turf.polygon([ring]);
}

function featureToSingleRing(feature: any): Ring | null {
  if (!feature?.geometry) return null;
  const cleaned = turf.cleanCoords(feature as any);
  const geom = cleaned?.geometry;
  if (!geom) return null;
  if (geom.type !== "Polygon") return null;
  if (!Array.isArray(geom.coordinates) || geom.coordinates.length !== 1) return null;
  return normalizeRing(geom.coordinates[0] as Ring);
}

function ringAreaM2(ring: Ring): number {
  return turf.area(ringFeature(ring));
}

function degToRad(value: number) {
  return (value * Math.PI) / 180;
}

function lngLatToMercatorMeters(lng: number, lat: number): [number, number] {
  const R = 6378137;
  const lambda = degToRad(lng);
  const phi = Math.max(-85.05112878, Math.min(85.05112878, lat)) * Math.PI / 180;
  return [R * lambda, R * Math.log(Math.tan(Math.PI / 4 + phi / 2))];
}

function mercatorMetersToLngLat(x: number, y: number): [number, number] {
  const R = 6378137;
  const lng = (x / R) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * (180 / Math.PI);
  return [lng, lat];
}

function haversineDistance(a: [number, number], b: [number, number]): number {
  const R = 6371000;
  const phi1 = degToRad(a[1]);
  const phi2 = degToRad(b[1]);
  const dPhi = degToRad(b[1] - a[1]);
  const dLambda = degToRad(b[0] - a[0]);
  const sinPhi = Math.sin(dPhi / 2);
  const sinLambda = Math.sin(dLambda / 2);
  const h =
    sinPhi * sinPhi +
    Math.cos(phi1) * Math.cos(phi2) * sinLambda * sinLambda;
  return 2 * R * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
}

function pointInPolygon(lng: number, lat: number, ring: Ring): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersect =
      (yi > lat) !== (yj > lat) &&
      lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function mean(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function median(values: number[]) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? 0.5 * (sorted[mid - 1] + sorted[mid]) : sorted[mid];
}

function ringMercatorCenter(ring: Ring): [number, number] {
  const coords = ring.slice(0, -1).map(([lng, lat]) => lngLatToMercatorMeters(lng, lat));
  return coords.reduce(
    (acc, [x, y]) => [acc[0] + x / coords.length, acc[1] + y / coords.length] as [number, number],
    [0, 0] as [number, number],
  );
}

function lineSpacingForParams(params: FlightParams): number {
  if ((params.payloadKind ?? "camera") === "lidar") {
    const lidar = params.lidarKey ? LIDAR_REGISTRY[params.lidarKey] || DEFAULT_LIDAR : DEFAULT_LIDAR;
    return lidarLineSpacing(
      params.altitudeAGL,
      params.sideOverlap,
      params.mappingFovDeg ?? lidar.effectiveHorizontalFovDeg,
    );
  }
  const cameraKey = params.cameraKey;
  const camera = cameraKey ? CAMERA_REGISTRY[cameraKey] || SONY_RX1R2 : SONY_RX1R2;
  const yawOffset = params.cameraYawOffsetDeg ?? 0;
  const rotate90 = Math.round((((yawOffset % 180) + 180) % 180)) === 90;
  return lineSpacingRotated(camera, params.altitudeAGL, params.sideOverlap, rotate90);
}

function projectedExtents(ring: Ring, bearingDeg: number) {
  const coords = ring.slice(0, -1).map(([lng, lat]) => lngLatToMercatorMeters(lng, lat));
  const center = ringMercatorCenter(ring);
  const bearingRad = degToRad(bearingDeg);
  const ux = Math.sin(bearingRad);
  const uy = Math.cos(bearingRad);
  const px = Math.sin(bearingRad + Math.PI / 2);
  const py = Math.cos(bearingRad + Math.PI / 2);
  let alongMin = Number.POSITIVE_INFINITY;
  let alongMax = Number.NEGATIVE_INFINITY;
  let crossMin = Number.POSITIVE_INFINITY;
  let crossMax = Number.NEGATIVE_INFINITY;

  for (const [x, y] of coords) {
    const dx = x - center[0];
    const dy = y - center[1];
    const along = dx * ux + dy * uy;
    const cross = dx * px + dy * py;
    alongMin = Math.min(alongMin, along);
    alongMax = Math.max(alongMax, along);
    crossMin = Math.min(crossMin, cross);
    crossMax = Math.max(crossMax, cross);
  }

  return {
    alongTrackLengthM: Math.max(1, alongMax - alongMin),
    crossTrackWidthM: Math.max(1, crossMax - crossMin),
  };
}

function weightedAxialMeanDeg(values: Array<{ angleDeg: number; weight: number }>): number | null {
  let sumSin = 0;
  let sumCos = 0;
  let totalWeight = 0;
  for (const { angleDeg, weight } of values) {
    if (!(weight > 0) || !Number.isFinite(angleDeg)) continue;
    const doubled = degToRad(normalizedAxialBearing(angleDeg) * 2);
    sumSin += Math.sin(doubled) * weight;
    sumCos += Math.cos(doubled) * weight;
    totalWeight += weight;
  }
  if (!(totalWeight > 0)) return null;
  const meanRad = 0.5 * Math.atan2(sumSin, sumCos);
  return normalizedAxialBearing((meanRad * 180) / Math.PI);
}

function buildTerrainBreakDescriptor(ring: Ring, tiles: TerrainTile[]): TerrainBreakDescriptor | null {
  const normalized = normalizeRing(ring);
  if (!normalized || tiles.length === 0) return null;
  const coords = normalized.slice(0, -1).map(([lng, lat]) => lngLatToMercatorMeters(lng, lat));
  const xs = coords.map(([x]) => x);
  const ys = coords.map(([, y]) => y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const areaM2 = ringAreaM2(normalized);
  const gridStepM = clamp(Math.sqrt(Math.max(areaM2, 1)) / 18, 60, 170);
  const diffStepM = clamp(gridStepM * 0.8, 35, 120);
  const points: TerrainDescriptorPoint[] = [];

  for (let y = minY + gridStepM * 0.5; y <= maxY; y += gridStepM) {
    for (let x = minX + gridStepM * 0.5; x <= maxX; x += gridStepM) {
      const [lng, lat] = mercatorMetersToLngLat(x, y);
      if (!pointInPolygon(lng, lat, normalized)) continue;
      const zc = queryElevationAtPoint(lng, lat, tiles as any);
      if (!Number.isFinite(zc)) continue;
      const [eastLng, eastLat] = geoDestination([lng, lat], 90, diffStepM);
      const [westLng, westLat] = geoDestination([lng, lat], 270, diffStepM);
      const [northLng, northLat] = geoDestination([lng, lat], 0, diffStepM);
      const [southLng, southLat] = geoDestination([lng, lat], 180, diffStepM);
      const ze = queryElevationAtPoint(eastLng, eastLat, tiles as any);
      const zw = queryElevationAtPoint(westLng, westLat, tiles as any);
      const zn = queryElevationAtPoint(northLng, northLat, tiles as any);
      const zs = queryElevationAtPoint(southLng, southLat, tiles as any);
      if (![ze, zw, zn, zs].every(Number.isFinite)) continue;

      const gradX = (ze - zw) / (2 * diffStepM);
      const gradY = (zn - zs) / (2 * diffStepM);
      const slopeMagnitude = Math.sqrt(gradX * gradX + gradY * gradY);
      if (!(slopeMagnitude > 1e-4)) continue;

      const aspectRad = (Math.atan2(gradX, gradY) + 2 * Math.PI) % (2 * Math.PI);
      const contourDirDeg = ((aspectRad * 180) / Math.PI + 90) % 360;
      points.push({
        lng,
        lat,
        x,
        y,
        contourDirDeg,
        slopeMagnitude,
        breakStrength: 0,
      });
    }
  }

  if (points.length < 8) return { points, gridStepM };

  const neighborRadiusM = gridStepM * 2.6;
  for (const point of points) {
    let weightedDelta = 0;
    let totalWeight = 0;
    for (const neighbor of points) {
      if (neighbor === point) continue;
      const dx = neighbor.x - point.x;
      const dy = neighbor.y - point.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (!(dist > 0) || dist > neighborRadiusM) continue;
      const weight = 1 - dist / neighborRadiusM;
      weightedDelta += axialAngleDeltaDeg(point.contourDirDeg, neighbor.contourDirDeg) * weight;
      totalWeight += weight;
    }
    const localDisagreement = totalWeight > 0 ? weightedDelta / totalWeight : 0;
    point.breakStrength = localDisagreement * clamp(point.slopeMagnitude / 0.22, 0.35, 1.5);
  }

  const sortedBreaks = points
    .map((point) => point.breakStrength)
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
  const threshold = sortedBreaks.length > 0
    ? sortedBreaks[Math.max(0, Math.floor(sortedBreaks.length * 0.72) - 1)]
    : 0;
  const hotPoints = points.filter((point) => point.breakStrength >= threshold && point.breakStrength > 6);
  if (hotPoints.length < 4) return { points, gridStepM };

  const weightAt = (point: TerrainDescriptorPoint) => Math.max(0.01, point.breakStrength * point.slopeMagnitude);
  let cx = 0;
  let cy = 0;
  let sw = 0;
  for (const point of hotPoints) {
    const weight = weightAt(point);
    cx += point.x * weight;
    cy += point.y * weight;
    sw += weight;
  }
  if (!(sw > 0)) return { points, gridStepM };
  cx /= sw;
  cy /= sw;

  let sxx = 0;
  let syy = 0;
  let sxy = 0;
  for (const point of hotPoints) {
    const weight = weightAt(point);
    const dx = point.x - cx;
    const dy = point.y - cy;
    sxx += weight * dx * dx;
    syy += weight * dy * dy;
    sxy += weight * dx * dy;
  }
  const axisRad = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const principalBearingDeg = normalizedAxialBearing((Math.atan2(Math.sin(axisRad), Math.cos(axisRad)) * 180) / Math.PI + 90);

  return {
    points,
    weightedCentroid: [cx, cy],
    principalBearingDeg,
    gridStepM,
  };
}

function cutBoundaryBonus(
  descriptor: TerrainBreakDescriptor | null,
  ring: Ring,
  cutBearingDeg: number,
  offsetM: number,
): number {
  if (!descriptor || descriptor.points.length < 8) return 0;
  const center = ringMercatorCenter(ring);
  const normalBearingRad = degToRad(cutBearingDeg + 90);
  const nx = Math.sin(normalBearingRad);
  const ny = Math.cos(normalBearingRad);
  const bandWidthM = Math.max(120, descriptor.gridStepM * 2.4);
  const cutCenter = [
    center[0] + nx * offsetM,
    center[1] + ny * offsetM,
  ] as [number, number];

  const left: Array<{ angleDeg: number; weight: number }> = [];
  const right: Array<{ angleDeg: number; weight: number }> = [];
  let breakStrengthSum = 0;
  let breakWeightSum = 0;

  for (const point of descriptor.points) {
    const signedDistance = (point.x - cutCenter[0]) * nx + (point.y - cutCenter[1]) * ny;
    const absDistance = Math.abs(signedDistance);
    if (absDistance > bandWidthM) continue;
    const proximityWeight = 1 - absDistance / bandWidthM;
    const weight = proximityWeight * Math.max(0.05, point.slopeMagnitude) * Math.max(0.1, point.breakStrength / 12);
    if (signedDistance >= 0) {
      left.push({ angleDeg: point.contourDirDeg, weight });
    } else {
      right.push({ angleDeg: point.contourDirDeg, weight });
    }
    breakStrengthSum += point.breakStrength * proximityWeight;
    breakWeightSum += proximityWeight;
  }

  if (left.length < 3 || right.length < 3) return 0;
  const leftMean = weightedAxialMeanDeg(left);
  const rightMean = weightedAxialMeanDeg(right);
  if (!Number.isFinite(leftMean) || !Number.isFinite(rightMean)) return 0;
  const delta = axialAngleDeltaDeg(leftMean!, rightMean!);
  const avgBreakStrength = breakWeightSum > 0 ? breakStrengthSum / breakWeightSum : 0;
  const support = clamp(Math.min(left.length, right.length) / 8, 0, 1);
  return support * clamp(delta / 18, 0, 4) * clamp(avgBreakStrength / 10, 0, 2.5);
}

function dedupeCoords(coords: Ring, toleranceM = 3): Ring {
  const out: Ring = [];
  for (const coord of coords) {
    if (out.length === 0 || haversineDistance(out[out.length - 1], coord) > toleranceM) {
      out.push(coord);
    }
  }
  if (out.length >= 2 && haversineDistance(out[0], out[out.length - 1]) <= toleranceM) {
    out[out.length - 1] = out[0];
  }
  return out;
}

function buildCutEndpoints(ring: Ring, cutBearingDeg: number, offsetM: number): [[number, number], [number, number]] {
  const lons = ring.map((point) => point[0]);
  const lats = ring.map((point) => point[1]);
  const diagonal = haversineDistance(
    [Math.min(...lons), Math.min(...lats)],
    [Math.max(...lons), Math.max(...lats)],
  );
  const centerFeature = turf.centerOfMass(ringFeature(ring));
  const [centerLng, centerLat] = centerFeature.geometry.coordinates as [number, number];
  const cutCenter = geoDestination([centerLng, centerLat], (cutBearingDeg + 90) % 360, offsetM);
  const extendM = Math.max(300, diagonal * 4);
  return [
    geoDestination(cutCenter, (cutBearingDeg + 180) % 360, extendM),
    geoDestination(cutCenter, cutBearingDeg, extendM),
  ];
}

function cutIntersectionsWithRing(ring: Ring, cutBearingDeg: number, offsetM: number): [[number, number], [number, number]] | null {
  try {
    const [p0, p1] = buildCutEndpoints(ring, cutBearingDeg, offsetM);
    const intersections = turf.lineIntersect(
      turf.lineString([p0, p1]),
      turf.lineString(ring),
    )?.features ?? [];
    const unique: Ring = [];
    for (const feature of intersections) {
      const coord = feature?.geometry?.coordinates as [number, number] | undefined;
      if (!coord) continue;
      if (unique.some((existing) => haversineDistance(existing, coord) < 4)) continue;
      unique.push(coord);
    }
    if (unique.length < 2) return null;
    const [sx, sy] = lngLatToMercatorMeters(p0[0], p0[1]);
    const [ex, ey] = lngLatToMercatorMeters(p1[0], p1[1]);
    const ux = ex - sx;
    const uy = ey - sy;
    const ordered = unique
      .map((coord) => {
        const [x, y] = lngLatToMercatorMeters(coord[0], coord[1]);
        return { coord, t: (x - sx) * ux + (y - sy) * uy };
      })
      .sort((a, b) => a.t - b.t);
    return [ordered[0].coord, ordered[ordered.length - 1].coord];
  } catch {
    return null;
  }
}

function pointOnSegmentDistanceM(point: [number, number], a: [number, number], b: [number, number]) {
  const [px, py] = lngLatToMercatorMeters(point[0], point[1]);
  const [ax, ay] = lngLatToMercatorMeters(a[0], a[1]);
  const [bx, by] = lngLatToMercatorMeters(b[0], b[1]);
  const abx = bx - ax;
  const aby = by - ay;
  const ab2 = abx * abx + aby * aby;
  if (!(ab2 > 0)) return Math.sqrt((px - ax) ** 2 + (py - ay) ** 2);
  const t = clamp(((px - ax) * abx + (py - ay) * aby) / ab2, 0, 1);
  const qx = ax + t * abx;
  const qy = ay + t * aby;
  return Math.sqrt((px - qx) ** 2 + (py - qy) ** 2);
}

function insertPointIntoRing(ring: Ring, point: [number, number], toleranceM = 5): Ring | null {
  const open = ring.slice(0, -1);
  for (let i = 0; i < open.length; i++) {
    const a = open[i];
    const b = open[(i + 1) % open.length];
    if (haversineDistance(a, point) <= toleranceM || haversineDistance(b, point) <= toleranceM) continue;
    if (pointOnSegmentDistanceM(point, a, b) <= toleranceM) {
      const nextOpen = [...open.slice(0, i + 1), point, ...open.slice(i + 1)];
      return normalizeRing([...nextOpen, nextOpen[0]]);
    }
  }
  return normalizeRing(ring);
}

function findCoordIndex(ring: Ring, point: [number, number], toleranceM = 5) {
  return ring.findIndex((coord) => haversineDistance(coord, point) <= toleranceM);
}

function chainBetweenIndices(ring: Ring, startIndex: number, endIndex: number): Ring {
  const open = ring.slice(0, -1);
  const out: Ring = [];
  let index = startIndex;
  while (true) {
    out.push(open[index]);
    if (index === endIndex) break;
    index = (index + 1) % open.length;
  }
  return out;
}

function ringsEqualish(a: Ring, b: Ring, toleranceM = 5) {
  if (a.length !== b.length) return false;
  return a.every((coord, index) => haversineDistance(coord, b[index]) <= toleranceM);
}

function tryRidgeFollowSplit(
  parentRing: Ring,
  descriptor: TerrainBreakDescriptor | null,
  cutBearingDeg: number,
  offsetM: number,
): [Ring, Ring] | null {
  if (!descriptor || descriptor.points.length < 8) return null;
  const intersections = cutIntersectionsWithRing(parentRing, cutBearingDeg, offsetM);
  if (!intersections) return null;
  const [endpointA, endpointB] = intersections;
  const [ax, ay] = lngLatToMercatorMeters(endpointA[0], endpointA[1]);
  const [bx, by] = lngLatToMercatorMeters(endpointB[0], endpointB[1]);
  const dx = bx - ax;
  const dy = by - ay;
  const length = Math.sqrt(dx * dx + dy * dy);
  if (!(length > descriptor.gridStepM * 2)) return null;
  const ux = dx / length;
  const uy = dy / length;
  const nx = -uy;
  const ny = ux;
  const bandWidthM = Math.max(110, descriptor.gridStepM * 1.9);
  const bins = Math.max(4, Math.min(10, Math.round(length / Math.max(140, descriptor.gridStepM * 1.2))));
  const bestByBin = new Map<number, { point: TerrainDescriptorPoint; score: number; s: number }>();

  for (const point of descriptor.points) {
    const relX = point.x - ax;
    const relY = point.y - ay;
    const s = relX * ux + relY * uy;
    if (s <= descriptor.gridStepM * 0.4 || s >= length - descriptor.gridStepM * 0.4) continue;
    const cross = relX * nx + relY * ny;
    if (Math.abs(cross) > bandWidthM) continue;
    const bin = clamp(Math.floor((s / length) * bins), 0, bins - 1);
    const score = point.breakStrength - Math.abs(cross) / Math.max(1, descriptor.gridStepM);
    const existing = bestByBin.get(bin);
    if (!existing || score > existing.score) {
      bestByBin.set(bin, { point, score, s });
    }
  }

  const interior = [...bestByBin.values()]
    .sort((a, b) => a.s - b.s)
    .map(({ point }) => [point.lng, point.lat] as [number, number]);
  if (interior.length < 2) return null;

  const ridgePath = dedupeCoords([endpointA, ...interior, endpointB]);
  let ringWithA = insertPointIntoRing(parentRing, endpointA);
  if (!ringWithA) return null;
  let ringWithBoth = insertPointIntoRing(ringWithA, endpointB);
  if (!ringWithBoth) return null;
  const indexA = findCoordIndex(ringWithBoth, endpointA);
  const indexB = findCoordIndex(ringWithBoth, endpointB);
  if (indexA < 0 || indexB < 0 || indexA === indexB) return null;

  const chainAB = chainBetweenIndices(ringWithBoth, indexA, indexB);
  const chainBA = chainBetweenIndices(ringWithBoth, indexB, indexA);
  const ridgeInteriorForward = ridgePath.slice(1, -1);
  const ridgeInteriorReverse = [...ridgeInteriorForward].reverse();
  const child1 = normalizeRing([...chainAB, ...ridgeInteriorReverse, endpointA]);
  const child2 = normalizeRing([...chainBA, ...ridgeInteriorForward, endpointB]);
  if (!child1 || !child2) return null;
  if (ringsEqualish(child1, parentRing) || ringsEqualish(child2, parentRing)) return null;
  return [child1, child2];
}

function convexityRatio(ring: Ring, areaM2: number): number {
  try {
    const points = turf.featureCollection(ring.slice(0, -1).map(([lng, lat]) => turf.point([lng, lat])));
    const hull = turf.convex(points);
    if (!hull) return 1;
    const hullArea = turf.area(hull);
    if (!(hullArea > 0)) return 1;
    return Math.max(0, Math.min(1, areaM2 / hullArea));
  } catch {
    return 1;
  }
}

function estimateFlightPatternMetrics(
  ring: Ring,
  bearingDeg: number,
  lineSpacingM: number,
  shortLineThresholdM: number,
  params: FlightParams,
): FlightPatternMetrics {
  const geometry = generatePlannedFlightGeometryForPolygon(ring, bearingDeg, lineSpacingM, params);
  const summary = summarizePlannedFlightGeometry(geometry);
  const lengths = geometry.sweepLines.map((sweepLine) => {
    let sweepLengthM = 0;
    for (let index = 1; index < sweepLine.length; index += 1) {
      sweepLengthM += haversineDistance(sweepLine[index - 1], sweepLine[index]);
    }
    return sweepLengthM;
  });

  const shortLineCount = lengths.filter((length) => length < shortLineThresholdM).length;

  return {
    lineCount: summary.lineCount,
    fragmentedLineCount: summary.fragmentedLineCount,
    meanLineLengthM: mean(lengths),
    medianLineLengthM: median(lengths),
    shortLineCount,
  };
}

function qualityPenalty(fitQuality?: "excellent" | "good" | "fair" | "poor") {
  switch (fitQuality) {
    case "excellent":
      return 0.05;
    case "good":
      return 0.35;
    case "fair":
      return 1.1;
    case "poor":
    default:
      return 2.4;
  }
}

function evaluatePolygon(
  ring: Ring,
  tiles: TerrainTile[],
  params: FlightParams,
  opts: Required<TerrainFacePartitionOptions>,
  mode: "normal" | "forced" = "normal",
): EvaluatedPartitionPolygon | null {
  const normalized = normalizeRing(ring);
  if (!normalized) return null;
  const areaM2 = ringAreaM2(normalized);
  if (!(areaM2 >= opts.minAreaM2)) return null;

  const terrainResult = dominantContourDirectionPlaneFit(
    { coordinates: normalized } as TerrainPolygon,
    tiles,
    { sampleStep: opts.searchSampleStep },
  );
  if (!Number.isFinite(terrainResult.contourDirDeg)) return null;

  const lineSpacingM = Math.max(1, lineSpacingForParams(params));
  const { alongTrackLengthM, crossTrackWidthM } = projectedExtents(normalized, terrainResult.contourDirDeg);
  const aspectRatio = Math.max(alongTrackLengthM, crossTrackWidthM) / Math.max(1, Math.min(alongTrackLengthM, crossTrackWidthM));
  const convexity = convexityRatio(normalized, areaM2);
  const minWidthM = mode === "forced"
    ? Math.max(45, 2 * lineSpacingM)
    : Math.max(60, 3 * lineSpacingM);
  const shortLineThresholdM = mode === "forced"
    ? Math.max(90, 4 * lineSpacingM)
    : Math.max(120, 6 * lineSpacingM);
  const flight = estimateFlightPatternMetrics(normalized, terrainResult.contourDirDeg, lineSpacingM, shortLineThresholdM, params);

  if (crossTrackWidthM < minWidthM * (mode === "forced" ? 0.55 : 0.75)) return null;
  if (flight.lineCount === 0) return null;
  if (flight.medianLineLengthM < shortLineThresholdM * (mode === "forced" ? 0.3 : 0.45)) return null;
  if (convexity < opts.minConvexity * (mode === "forced" ? 0.5 : 0.7)) return null;

  const terrainPenalty =
    qualityPenalty(terrainResult.fitQuality) +
    Math.max(0, Math.min(4, (terrainResult.rmse ?? 0) / 6)) +
    Math.max(0, Math.min(4, (1 - Math.max(0, terrainResult.rSquared ?? 0)) * 3.5));

  const medianShortfall = Math.max(0, shortLineThresholdM - flight.medianLineLengthM) / shortLineThresholdM;
  const shortFraction = flight.shortLineCount / Math.max(1, flight.lineCount);
  const flightPenalty =
    shortFraction * 4 +
    medianShortfall * 4 +
    flight.fragmentedLineCount * 0.9 +
    Math.max(0, flight.lineCount - 14) * 0.08;

  const widthPenalty = Math.max(0, minWidthM - crossTrackWidthM) / minWidthM;
  const aspectPenalty = Math.max(0, aspectRatio - opts.maxAspectRatio) * 0.45;
  const convexityPenalty = Math.max(0, opts.minConvexity - convexity) * 5;
  const shapePenalty = widthPenalty * 6 + aspectPenalty + convexityPenalty;

  return {
    ring: normalized,
    areaM2,
    contourDirDeg: terrainResult.contourDirDeg,
    lineSpacingM,
    score: terrainPenalty * 1.7 + flightPenalty + shapePenalty,
    fitQuality: terrainResult.fitQuality,
    rmse: terrainResult.rmse,
    rSquared: terrainResult.rSquared,
    convexity,
    aspectRatio,
    crossTrackWidthM,
    alongTrackLengthM,
    flight,
    breakdown: {
      terrainPenalty,
      flightPenalty,
      shapePenalty,
    },
  };
}

function normalizedAxialBearing(value: number) {
  const wrapped = ((value % 180) + 180) % 180;
  return wrapped;
}

function uniqueBearings(values: number[]): number[] {
  const out: number[] = [];
  for (const value of values) {
    const normalized = normalizedAxialBearing(value);
    if (out.some((existing) => Math.abs(existing - normalized) < 1e-6)) continue;
    out.push(normalized);
  }
  return out;
}

function axialAngleDeltaDeg(a: number, b: number): number {
  const aa = normalizedAxialBearing(a);
  const bb = normalizedAxialBearing(b);
  const delta = Math.abs(aa - bb);
  return Math.min(delta, 180 - delta);
}

function halfPlanePolygon(
  center: [number, number],
  cutBearingDeg: number,
  sideBearingDeg: number,
  distanceM: number,
) {
  const a = geoDestination(center, cutBearingDeg, distanceM);
  const b = geoDestination(center, (cutBearingDeg + 180) % 360, distanceM);
  const c = geoDestination(b, sideBearingDeg, distanceM * 3);
  const d = geoDestination(a, sideBearingDeg, distanceM * 3);
  return turf.polygon([[a, b, c, d, a]]);
}

function splitPolygonByCut(ring: Ring, cutBearingDeg: number, offsetM: number): [Ring, Ring] | null {
  const polygon = ringFeature(ring);
  const centerFeature = turf.centerOfMass(polygon);
  const [centerLng, centerLat] = centerFeature.geometry.coordinates as [number, number];
  const cutCenter = geoDestination([centerLng, centerLat], (cutBearingDeg + 90) % 360, offsetM);

  const lons = ring.map((point) => point[0]);
  const lats = ring.map((point) => point[1]);
  const diagonal = haversineDistance(
    [Math.min(...lons), Math.min(...lats)],
    [Math.max(...lons), Math.max(...lats)],
  );
  const extendM = Math.max(300, diagonal * 4);

  try {
    const positive = turf.intersect(polygon, halfPlanePolygon(cutCenter, cutBearingDeg, (cutBearingDeg + 90) % 360, extendM));
    const negative = turf.intersect(polygon, halfPlanePolygon(cutCenter, cutBearingDeg, (cutBearingDeg + 270) % 360, extendM));
    if (!positive || !negative) return null;
    const ringA = featureToSingleRing(positive);
    const ringB = featureToSingleRing(negative);
    if (!ringA || !ringB) return null;
    const areaSum = ringAreaM2(ringA) + ringAreaM2(ringB);
    const parentArea = ringAreaM2(ring);
    if (parentArea > 0 && areaSum / parentArea < 0.95) return null;
    return [ringA, ringB];
  } catch {
    return null;
  }
}

function candidateCutAngles(
  parent: EvaluatedPartitionPolygon,
  descriptor: TerrainBreakDescriptor | null,
  opts: Required<TerrainFacePartitionOptions>,
): number[] {
  const sampled: number[] = [];
  for (let angle = 0; angle < 180; angle += opts.candidateAngleStepDeg) sampled.push(angle);
  sampled.push(parent.contourDirDeg);
  sampled.push(parent.contourDirDeg + 90);
  if (Number.isFinite(descriptor?.principalBearingDeg)) {
    sampled.push(descriptor!.principalBearingDeg!);
    sampled.push(descriptor!.principalBearingDeg! + opts.candidateAngleStepDeg);
    sampled.push(descriptor!.principalBearingDeg! - opts.candidateAngleStepDeg);
  }
  return uniqueBearings(sampled);
}

function projectedNormalSpan(ring: Ring, cutBearingDeg: number) {
  const coords = ring.slice(0, -1).map(([lng, lat]) => lngLatToMercatorMeters(lng, lat));
  const center = coords.reduce(
    (acc, [x, y]) => [acc[0] + x / coords.length, acc[1] + y / coords.length] as [number, number],
    [0, 0] as [number, number],
  );
  const normalBearingRad = degToRad(cutBearingDeg + 90);
  const nx = Math.sin(normalBearingRad);
  const ny = Math.cos(normalBearingRad);
  let minProjection = Number.POSITIVE_INFINITY;
  let maxProjection = Number.NEGATIVE_INFINITY;
  for (const [x, y] of coords) {
    const projection = (x - center[0]) * nx + (y - center[1]) * ny;
    minProjection = Math.min(minProjection, projection);
    maxProjection = Math.max(maxProjection, projection);
  }
  return { minProjection, maxProjection, spanM: maxProjection - minProjection };
}

function findBestSplit(
  polygon: EvaluatedPartitionPolygon,
  tiles: TerrainTile[],
  params: FlightParams,
  opts: Required<TerrainFacePartitionOptions>,
  mode: "normal" | "forced" = "normal",
  debug?: SplitDebugStats,
): SplitCandidate | null {
  const descriptor = buildTerrainBreakDescriptor(polygon.ring, tiles);
  const candidates = candidateCutAngles(polygon, descriptor, opts);
  let best: SplitCandidate | null = null;
  const center = ringMercatorCenter(polygon.ring);

  for (const cutBearingDeg of candidates) {
    const { minProjection, maxProjection, spanM } = projectedNormalSpan(polygon.ring, cutBearingDeg);
    if (!(spanM > 0)) continue;
    const normalBearingRad = degToRad(cutBearingDeg + 90);
    const nx = Math.sin(normalBearingRad);
    const ny = Math.cos(normalBearingRad);
    const candidateOffsets = new Set<number>();
    for (const fraction of opts.candidateOffsetFractions) {
      candidateOffsets.add(fraction * spanM * 0.5);
    }
    if (descriptor?.weightedCentroid) {
      const [cx, cy] = descriptor.weightedCentroid;
      const hotspotOffset = (cx - center[0]) * nx + (cy - center[1]) * ny;
      if (hotspotOffset >= minProjection && hotspotOffset <= maxProjection) {
        candidateOffsets.add(hotspotOffset);
        candidateOffsets.add(hotspotOffset - descriptor.gridStepM);
        candidateOffsets.add(hotspotOffset + descriptor.gridStepM);
      }
    }

    for (const offsetM of [...candidateOffsets]) {
      if (debug) debug.totalCandidates += 1;
      if (offsetM < minProjection || offsetM > maxProjection) continue;
      const straightSplit = splitPolygonByCut(polygon.ring, cutBearingDeg, offsetM);
      const split = tryRidgeFollowSplit(polygon.ring, descriptor, cutBearingDeg, offsetM) ?? straightSplit;
      if (!split) {
        if (debug) debug.failedSplitGeometry += 1;
        continue;
      }
      const left = evaluatePolygon(split[0], tiles, params, opts, mode);
      const right = evaluatePolygon(split[1], tiles, params, opts, mode);
      if (!left) {
        if (debug) debug.failedLeftEval += 1;
        continue;
      }
      if (!right) {
        if (debug) debug.failedRightEval += 1;
        continue;
      }

      const minArea = Math.min(left.areaM2, right.areaM2);
      const maxArea = Math.max(left.areaM2, right.areaM2);
      const balance = maxArea > 0 ? minArea / maxArea : 0;
      if (balance < 0.12) {
        if (debug) debug.failedBalance += 1;
        continue;
      }

      const boundaryBonus = cutBoundaryBonus(descriptor, polygon.ring, cutBearingDeg, offsetM);
      const combinedScore = left.score + right.score + opts.splitPenalty - boundaryBonus * 1.4;
      const improvement = polygon.score - combinedScore;
      const directionDeltaDeg = axialAngleDeltaDeg(left.contourDirDeg, right.contourDirDeg);

      if (mode === "normal") {
        if (!(improvement > opts.minImprovement)) {
          if (debug) debug.failedImprovement += 1;
          continue;
        }
      } else {
        if (directionDeltaDeg < opts.forceMinDirectionDeltaDeg) {
          if (debug) debug.failedForcedDirection += 1;
          continue;
        }
        if (combinedScore > polygon.score + opts.forceMaxScoreRegression) {
          if (debug) debug.failedForcedRegression += 1;
          continue;
        }
      }
      if (debug) debug.acceptedCandidates += 1;

      const candidate: SplitCandidate = {
        improvement,
        splitPenalty: opts.splitPenalty,
        bearingDeg: cutBearingDeg,
        offsetM,
        combinedScore,
        directionDeltaDeg,
        boundaryBonus,
        children: [left, right],
      };
      if (mode === "normal") {
        if (
          !best ||
          candidate.improvement > best.improvement + 1e-6 ||
          (
            Math.abs(candidate.improvement - best.improvement) <= 1e-6 &&
            candidate.boundaryBonus > best.boundaryBonus + 1e-6
          )
        ) {
          best = candidate;
          if (debug) debug.bestImprovement = candidate.improvement;
        }
      } else {
        const candidateUtility =
          candidate.directionDeltaDeg * 2.2 +
          candidate.boundaryBonus * 2 -
          Math.max(0, candidate.combinedScore - polygon.score);
        const bestUtility = best
          ? best.directionDeltaDeg * 2.2 + best.boundaryBonus * 2 - Math.max(0, best.combinedScore - polygon.score)
          : Number.NEGATIVE_INFINITY;
        if (
          !best ||
          candidateUtility > bestUtility + 1e-6 ||
          (
            Math.abs(candidateUtility - bestUtility) <= 1e-6 &&
            candidate.combinedScore < best.combinedScore - 1e-6
          ) ||
          (
            Math.abs(candidateUtility - bestUtility) <= 1e-6 &&
            Math.abs(candidate.combinedScore - best.combinedScore) <= 1e-6 &&
            candidate.directionDeltaDeg > best.directionDeltaDeg
          )
        ) {
          best = candidate;
          if (debug) debug.bestForcedUtility = candidateUtility;
        }
      }
    }
  }

  return best;
}

export function debugBestSplitForPolygon(
  ring: Ring,
  tiles: TerrainTile[],
  params: FlightParams,
  options: TerrainFacePartitionOptions = {},
): {
  root: EvaluatedPartitionPolygon | null;
  normal: SplitCandidate | null;
  forced: SplitCandidate | null;
  normalStats: SplitDebugStats;
  forcedStats: SplitDebugStats;
} {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const normalized = normalizeRing(ring);
  if (!normalized) {
    const emptyStats: SplitDebugStats = {
      totalCandidates: 0,
      failedSplitGeometry: 0,
      failedLeftEval: 0,
      failedRightEval: 0,
      failedBalance: 0,
      failedImprovement: 0,
      failedForcedDirection: 0,
      failedForcedRegression: 0,
      acceptedCandidates: 0,
    };
    return { root: null, normal: null, forced: null, normalStats: emptyStats, forcedStats: { ...emptyStats } };
  }

  const root = evaluatePolygon(normalized, tiles, params, opts);
  const normalStats: SplitDebugStats = {
    totalCandidates: 0,
    failedSplitGeometry: 0,
    failedLeftEval: 0,
    failedRightEval: 0,
    failedBalance: 0,
    failedImprovement: 0,
    failedForcedDirection: 0,
    failedForcedRegression: 0,
    acceptedCandidates: 0,
  };
  const forcedStats: SplitDebugStats = { ...normalStats };

  if (!root) {
    return { root: null, normal: null, forced: null, normalStats, forcedStats };
  }

  const normal = findBestSplit(root, tiles, params, opts, "normal", normalStats);
  const forced = findBestSplit(root, tiles, params, opts, "forced", forcedStats);
  return { root, normal, forced, normalStats, forcedStats };
}

function unionToSingleRing(a: Ring, b: Ring): Ring | null {
  try {
    const unionFn = (turf as any).union;
    if (typeof unionFn !== "function") return null;
    const merged =
      unionFn.length >= 2
        ? unionFn(ringFeature(a), ringFeature(b))
        : unionFn(turf.featureCollection([ringFeature(a), ringFeature(b)]));
    return featureToSingleRing(merged);
  } catch {
    return null;
  }
}

function mergeAdjacentLeaves(
  leaves: EvaluatedPartitionPolygon[],
  tiles: TerrainTile[],
  params: FlightParams,
  opts: Required<TerrainFacePartitionOptions>,
): EvaluatedPartitionPolygon[] {
  const mergedLeaves = [...leaves];

  while (mergedLeaves.length > 1) {
    let bestPair: { i: number; j: number; merged: EvaluatedPartitionPolygon; benefit: number } | null = null;

    for (let i = 0; i < mergedLeaves.length; i++) {
      for (let j = i + 1; j < mergedLeaves.length; j++) {
        const mergedRing = unionToSingleRing(mergedLeaves[i].ring, mergedLeaves[j].ring);
        if (!mergedRing) continue;
        const mergedEval = evaluatePolygon(mergedRing, tiles, params, opts);
        if (!mergedEval) continue;

        const childrenScore = mergedLeaves[i].score + mergedLeaves[j].score;
        const benefit = childrenScore + opts.splitPenalty - mergedEval.score;
        if (benefit < -opts.mergeSlack) continue;

        if (!bestPair || benefit > bestPair.benefit) {
          bestPair = { i, j, merged: mergedEval, benefit };
        }
      }
    }

    if (!bestPair) break;

    mergedLeaves.splice(bestPair.j, 1);
    mergedLeaves.splice(bestPair.i, 1, bestPair.merged);
  }

  return mergedLeaves;
}

export function partitionPolygonByTerrainFaces(
  ring: Ring,
  tiles: TerrainTile[],
  params: FlightParams,
  options: TerrainFacePartitionOptions = {},
): TerrainFacePartitionResult {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const normalized = normalizeRing(ring);
  if (!normalized) {
    return { polygons: [], evaluations: [], iterations: 0 };
  }

  const root = evaluatePolygon(normalized, tiles, params, opts);
  if (!root) {
    return { polygons: [normalized], evaluations: [], iterations: 0 };
  }

  const leaves: EvaluatedPartitionPolygon[] = [root];
  let iterations = 0;
  let forcedInitialSplitApplied = false;

  while (leaves.length < opts.maxPolygons) {
    let bestLeafIndex = -1;
    let bestProposal: SplitCandidate | null = null;

    for (let i = 0; i < leaves.length; i++) {
      const proposal = findBestSplit(leaves[i], tiles, params, opts);
      if (!proposal) continue;
      if (!bestProposal || proposal.improvement > bestProposal.improvement) {
        bestProposal = proposal;
        bestLeafIndex = i;
      }
    }

    if (bestLeafIndex < 0 || !bestProposal) break;
    leaves.splice(bestLeafIndex, 1, bestProposal.children[0], bestProposal.children[1]);
    iterations += 1;
  }

  if (opts.forceAtLeastOneSplit && leaves.length === 1) {
    const forced = findBestSplit(leaves[0], tiles, params, opts, "forced");
    if (forced) {
      leaves.splice(0, 1, forced.children[0], forced.children[1]);
      iterations += 1;
      forcedInitialSplitApplied = true;
    }
  }

  const simplifiedLeaves = forcedInitialSplitApplied
    ? leaves
    : mergeAdjacentLeaves(leaves, tiles, params, opts);

  return {
    polygons: simplifiedLeaves.map((leaf) => leaf.ring),
    evaluations: simplifiedLeaves,
    iterations,
  };
}
