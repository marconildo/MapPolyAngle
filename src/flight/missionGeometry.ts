import type {
  FlightParams,
  InterAreaConnectionGeometry,
  LngLat,
  MissionFlightGeometry,
  MissionTravelSummary,
  PlannedFlightGeometry,
  PlannedLeadManeuver,
} from '@/domain/types';
import { lngLatToMeters } from '@/overlap/mercator';
import type { TerrainTile } from '@/utils/terrainAspectHybrid';
import { haversineDistance, queryMinMaxElevationAlongPolylineWGS84 } from './geometry';
import { getCruiseSpeedMps } from './speeds';

const METERS_PER_DEG_LAT = 111_320;
const MIN_ARC_SEGMENTS = 18;
const ARC_POINT_SPACING_M = 8;
const COIL_POINT_SPACING_M = 8;
export const CONNECTOR_ELEVATION_PER_METER = 0.25;

type LocalPoint = { x: number; y: number };

export interface InterAreaConnectionInput {
  key: string;
  fromPolygonId: string;
  toPolygonId: string;
  sourceManeuver: PlannedLeadManeuver;
  targetManeuver: PlannedLeadManeuver;
  sourceAnchorAltitudeM: number;
  targetAnchorAltitudeM: number;
  cruiseSpeedMps: number;
  terrainZoom: number;
  terrainTileCount: number;
  terrainTiles?: TerrainTile[];
  altitudeMode?: 'legacy' | 'min-clearance';
  minClearanceM?: number;
}

export interface MissionAreaSummaryInput {
  polygonId: string;
  params: FlightParams;
  geometry: Pick<PlannedFlightGeometry, 'connectedLines'>;
}

function normaliseAngleRad(angleRad: number) {
  let normalized = angleRad % (Math.PI * 2);
  if (normalized <= -Math.PI) normalized += Math.PI * 2;
  if (normalized > Math.PI) normalized -= Math.PI * 2;
  return normalized;
}

function toLocal(point: LngLat, reference: LngLat): LocalPoint {
  const latScale = Math.cos((reference[1] * Math.PI) / 180);
  return {
    x: (point[0] - reference[0]) * METERS_PER_DEG_LAT * latScale,
    y: (point[1] - reference[1]) * METERS_PER_DEG_LAT,
  };
}

function fromLocal(point: LocalPoint, reference: LngLat): LngLat {
  const latScale = Math.cos((reference[1] * Math.PI) / 180);
  const safeScale = Math.abs(latScale) > 1e-6 ? latScale : 1e-6;
  return [
    reference[0] + point.x / (METERS_PER_DEG_LAT * safeScale),
    reference[1] + point.y / METERS_PER_DEG_LAT,
  ];
}

function subtract(left: LocalPoint, right: LocalPoint): LocalPoint {
  return { x: left.x - right.x, y: left.y - right.y };
}

function add(left: LocalPoint, right: LocalPoint): LocalPoint {
  return { x: left.x + right.x, y: left.y + right.y };
}

function scale(point: LocalPoint, factor: number): LocalPoint {
  return { x: point.x * factor, y: point.y * factor };
}

function dot(left: LocalPoint, right: LocalPoint): number {
  return left.x * right.x + left.y * right.y;
}

function magnitude(point: LocalPoint): number {
  return Math.sqrt(dot(point, point));
}

function perp(point: LocalPoint): LocalPoint {
  return { x: -point.y, y: point.x };
}

function clockwisePerp(point: LocalPoint): LocalPoint {
  return { x: point.y, y: -point.x };
}

function localAngle(center: LocalPoint, point: LocalPoint): number {
  return Math.atan2(point.y - center.y, point.x - center.x);
}

function dedupe2DPath(path: LngLat[]): LngLat[] {
  return path.filter((point, index) => index === 0 || point[0] !== path[index - 1][0] || point[1] !== path[index - 1][1]);
}

function polylineLengthMeters(path: readonly LngLat[]): number {
  let total = 0;
  for (let index = 1; index < path.length; index += 1) {
    total += haversineDistance(path[index - 1], path[index]);
  }
  return total;
}

function connectedPathLengthMetersLocal(geometry: Pick<PlannedFlightGeometry, 'connectedLines'>): number {
  return geometry.connectedLines.reduce((sum, line) => sum + polylineLengthMeters(line), 0);
}

function path3DLengthMeters(path3D: readonly [number, number, number][][]): number {
  let total = 0;
  for (const segment of path3D) {
    for (let index = 1; index < segment.length; index += 1) {
      const start = segment[index - 1];
      const end = segment[index];
      const horizontalDistanceM = haversineDistance([start[0], start[1]], [end[0], end[1]]);
      const dz = end[2] - start[2];
      total += Math.sqrt(horizontalDistanceM ** 2 + dz ** 2);
    }
  }
  return total;
}

function inferArcDirection(
  center: LngLat,
  start: LngLat,
  witness: LngLat,
): 1 | -1 {
  const reference = center;
  const startLocal = toLocal(start, reference);
  const witnessLocal = toLocal(witness, reference);
  const cross = startLocal.x * witnessLocal.y - startLocal.y * witnessLocal.x;
  return cross < 0 ? 1 : -1;
}

function normalizeLocal(point: LocalPoint): LocalPoint | null {
  const length = magnitude(point);
  if (length <= 1e-9) return null;
  return scale(point, 1 / length);
}

function rotateClockwise(point: LocalPoint): LocalPoint {
  return { x: point.y, y: -point.x };
}

function rotateCounterClockwise(point: LocalPoint): LocalPoint {
  return { x: -point.y, y: point.x };
}

function firstDistinctPointAfterAnchor(connectedLines: LngLat[][], anchorPoint: LngLat): LngLat | undefined {
  for (const segment of connectedLines) {
    for (const point of segment) {
      if (point[0] !== anchorPoint[0] || point[1] !== anchorPoint[1]) return point;
    }
  }
  return undefined;
}

function lastDistinctPointBeforeAnchor(connectedLines: LngLat[][], anchorPoint: LngLat): LngLat | undefined {
  for (let segmentIndex = connectedLines.length - 1; segmentIndex >= 0; segmentIndex -= 1) {
    const segment = connectedLines[segmentIndex]!;
    for (let pointIndex = segment.length - 1; pointIndex >= 0; pointIndex -= 1) {
      const point = segment[pointIndex]!;
      if (point[0] !== anchorPoint[0] || point[1] !== anchorPoint[1]) return point;
    }
  }
  return undefined;
}

function buildLeadManeuverFromHeading(
  anchorPoint: LngLat,
  tangentHeading: LocalPoint,
  preferredCenterPoint: LngLat | undefined,
  loiterRadiusM: number,
  loiterDirection: 1 | -1,
): PlannedLeadManeuver {
  const safeRadiusM = Math.max(1, loiterRadiusM);
  const headingUnit = normalizeLocal(tangentHeading);
  const buildPathJoinPoint = (centerPoint: LngLat) => {
    if (!headingUnit) return undefined;
    const centerLocal = toLocal(centerPoint, anchorPoint);
    const radialUnit = loiterDirection > 0
      ? rotateCounterClockwise(headingUnit)
      : rotateClockwise(headingUnit);
    const joinPointLocal = add(centerLocal, scale(radialUnit, safeRadiusM));
    const tangentLineOffsetM = Math.abs(dot(scale(joinPointLocal, -1), radialUnit));
    if (tangentLineOffsetM > 2) return undefined;
    return fromLocal(joinPointLocal, anchorPoint);
  };

  if (preferredCenterPoint) {
    return {
      anchorPoint,
      loiterCenter: preferredCenterPoint,
      loiterRadiusM: safeRadiusM,
      loiterDirection,
      pathJoinPoint: buildPathJoinPoint(preferredCenterPoint),
    };
  }
  if (!headingUnit) {
    return {
      anchorPoint,
      loiterCenter: anchorPoint,
      loiterRadiusM: safeRadiusM,
      loiterDirection,
    };
  }
  const radialUnit = loiterDirection > 0
    ? rotateCounterClockwise(headingUnit)
    : rotateClockwise(headingUnit);
  const centerLocal = scale(radialUnit, -safeRadiusM);
  return {
    anchorPoint,
    loiterCenter: fromLocal(centerLocal, anchorPoint),
    loiterRadiusM: safeRadiusM,
    loiterDirection,
    pathJoinPoint: fromLocal(add(centerLocal, scale(radialUnit, safeRadiusM)), anchorPoint),
  };
}

export function buildLeadManeuversForGeometry(
  connectedLines: LngLat[][],
  leadInPoints: LngLat[],
  leadOutPoints: LngLat[],
  loiterRadiusM: number,
): { leadIn?: PlannedLeadManeuver; leadOut?: PlannedLeadManeuver } {
  const firstAnchor = connectedLines[0]?.[0];
  const lastSegment = connectedLines.at(-1);
  const lastAnchor = lastSegment?.[lastSegment.length - 1];

  const leadIn =
    firstAnchor && leadInPoints.length >= 4
      ? (() => {
          const nextPoint = firstDistinctPointAfterAnchor(connectedLines, firstAnchor);
          if (!nextPoint) return undefined;
          const previewCenter = leadInPoints[0];
          const loiterDirection = inferArcDirection(leadInPoints[0], leadInPoints[1], leadInPoints[2]);
          return buildLeadManeuverFromHeading(
            firstAnchor,
            subtract(toLocal(nextPoint, firstAnchor), { x: 0, y: 0 }),
            previewCenter,
            loiterRadiusM,
            loiterDirection,
          );
        })()
      : undefined;
  const leadOut =
    lastAnchor && leadOutPoints.length >= 4
      ? (() => {
          const previousPoint = lastDistinctPointBeforeAnchor(connectedLines, lastAnchor);
          if (!previousPoint) return undefined;
          const previewCenter = leadOutPoints[3];
          const loiterDirection = inferArcDirection(leadOutPoints[3], leadOutPoints[0], leadOutPoints[1]);
          return buildLeadManeuverFromHeading(
            lastAnchor,
            subtract(toLocal(lastAnchor, previousPoint), { x: 0, y: 0 }),
            previewCenter,
            loiterRadiusM,
            loiterDirection,
          );
        })()
      : undefined;

  return { leadIn, leadOut };
}

function calculateTangentPointOnCircle(
  anchorPoint: LngLat,
  maneuver: PlannedLeadManeuver,
  loiterDirection: 1 | -1,
  reference: LngLat,
): LocalPoint {
  const anchorLocal = toLocal(anchorPoint, reference);
  const centerLocal = toLocal(maneuver.loiterCenter, reference);
  const delta = subtract(centerLocal, anchorLocal);
  const distanceToCenterM = magnitude(delta);

  if (distanceToCenterM <= maneuver.loiterRadiusM + 1e-6) {
    const radial = normalizeLocal(subtract(anchorLocal, centerLocal));
    if (!radial) return add(centerLocal, { x: maneuver.loiterRadiusM, y: 0 });
    return add(centerLocal, scale(radial, maneuver.loiterRadiusM));
  }

  const tangentAngleRad = Math.asin(maneuver.loiterRadiusM / distanceToCenterM);
  const centerBearingRad = Math.atan2(delta.y, delta.x);
  const centerToTangentAngleRad = Math.PI / 2 - tangentAngleRad;
  const sign = loiterDirection > 0 ? -1 : 1;
  const centerToTangentBearingRad = centerBearingRad + Math.PI + sign * centerToTangentAngleRad;
  return add(centerLocal, {
    x: Math.cos(centerToTangentBearingRad) * maneuver.loiterRadiusM,
    y: Math.sin(centerToTangentBearingRad) * maneuver.loiterRadiusM,
  });
}

function circleTravelTangentAtPoint(
  center: LocalPoint,
  point: LocalPoint,
  loiterDirection: 1 | -1,
): LocalPoint | null {
  const radial = normalizeLocal(subtract(point, center));
  if (!radial) return null;
  return loiterDirection > 0
    ? clockwisePerp(radial)
    : perp(radial);
}

function sampleArcPoints(
  center: LocalPoint,
  radiusM: number,
  startPoint: LocalPoint,
  endPoint: LocalPoint,
  loiterDirection: 1 | -1,
): LocalPoint[] {
  const startAngle = localAngle(center, startPoint);
  const endAngle = localAngle(center, endPoint);
  let delta = normaliseAngleRad(endAngle - startAngle);
  if (loiterDirection > 0 && delta > 0) delta -= Math.PI * 2;
  if (loiterDirection < 0 && delta < 0) delta += Math.PI * 2;

  const arcLengthM = Math.abs(delta) * radiusM;
  const segmentCount = Math.max(MIN_ARC_SEGMENTS, Math.ceil(arcLengthM / ARC_POINT_SPACING_M));
  const points: LocalPoint[] = [];
  for (let index = 0; index <= segmentCount; index += 1) {
    const t = index / segmentCount;
    const angle = startAngle + delta * t;
    points.push({
      x: center.x + Math.cos(angle) * radiusM,
      y: center.y + Math.sin(angle) * radiusM,
    });
  }
  return points;
}

function buildCoilLoop(
  center: LocalPoint,
  radiusM: number,
  direction: 1 | -1,
  anchorPoint: LocalPoint,
  loopCount: number,
): LocalPoint[] {
  if (loopCount <= 0) return [anchorPoint];
  const startAngle = localAngle(center, anchorPoint);
  const circumferenceM = 2 * Math.PI * radiusM;
  const pointsPerLoop = Math.max(36, Math.ceil(circumferenceM / COIL_POINT_SPACING_M));
  const totalSteps = pointsPerLoop * loopCount;
  const points: LocalPoint[] = [anchorPoint];
  for (let step = 1; step <= totalSteps; step += 1) {
    if (step === totalSteps) {
      points.push(anchorPoint);
      break;
    }
    const delta = ((Math.PI * 2 * loopCount * step) / totalSteps) * (direction > 0 ? -1 : 1);
    points.push({
      x: center.x + Math.cos(startAngle + delta) * radiusM,
      y: center.y + Math.sin(startAngle + delta) * radiusM,
    });
  }
  return points;
}

type TangentCandidate = {
  start: LocalPoint;
  end: LocalPoint;
};

function buildTangentCandidate(
  sourceCenter: LocalPoint,
  sourceRadiusM: number,
  targetCenter: LocalPoint,
  targetRadiusM: number,
  kind: 'direct' | 'transverse',
  sign: 1 | -1,
): TangentCandidate | null {
  const delta = subtract(targetCenter, sourceCenter);
  const distanceSquared = dot(delta, delta);
  if (distanceSquared <= 1e-9) return null;
  const radiusMix = kind === 'direct' ? sourceRadiusM - targetRadiusM : sourceRadiusM + targetRadiusM;
  const heightSquared = distanceSquared - radiusMix * radiusMix;
  if (heightSquared < 0) return null;
  const direction = scale(
    add(scale(delta, radiusMix), scale(perp(delta), sign * Math.sqrt(Math.max(0, heightSquared)))),
    1 / distanceSquared,
  );
  const start = add(sourceCenter, scale(direction, sourceRadiusM));
  const end = kind === 'direct'
    ? add(targetCenter, scale(direction, targetRadiusM))
    : add(targetCenter, scale(direction, -targetRadiusM));
  return { start, end };
}

function buildFallbackConnection2D(
  reference: LngLat,
  source: PlannedLeadManeuver,
  target: PlannedLeadManeuver,
): { leadOut: LngLat[]; transfer: LngLat[]; leadIn: LngLat[] } {
  const sourceCirclePoint = source.pathJoinPoint ?? fromLocal(
    calculateTangentPointOnCircle(source.anchorPoint, source, -source.loiterDirection as 1 | -1, reference),
    reference,
  );
  const targetCirclePoint = target.pathJoinPoint ?? fromLocal(
    calculateTangentPointOnCircle(target.anchorPoint, target, target.loiterDirection, reference),
    reference,
  );
  return {
    leadOut: dedupe2DPath([source.anchorPoint, sourceCirclePoint]),
    transfer: dedupe2DPath([sourceCirclePoint, targetCirclePoint]),
    leadIn: dedupe2DPath([targetCirclePoint, target.anchorPoint]),
  };
}

function buildConnection2D(
  source: PlannedLeadManeuver,
  target: PlannedLeadManeuver,
): { leadOut: LngLat[]; transfer: LngLat[]; leadIn: LngLat[] } {
  const reference: LngLat = [
    (source.loiterCenter[0] + target.loiterCenter[0]) / 2,
    (source.loiterCenter[1] + target.loiterCenter[1]) / 2,
  ];
  const sourceCenterLocal = toLocal(source.loiterCenter, reference);
  const targetCenterLocal = toLocal(target.loiterCenter, reference);
  const sourceCircleAnchor = source.pathJoinPoint
    ? toLocal(source.pathJoinPoint, reference)
    : calculateTangentPointOnCircle(
        source.anchorPoint,
        source,
        -source.loiterDirection as 1 | -1,
        reference,
      );
  const targetCircleAnchor = target.pathJoinPoint
    ? toLocal(target.pathJoinPoint, reference)
    : calculateTangentPointOnCircle(
        target.anchorPoint,
        target,
        target.loiterDirection,
        reference,
      );
  const tangentKind = source.loiterDirection === target.loiterDirection ? 'direct' : 'transverse';

  let bestCandidate: {
    leadOut: LocalPoint[];
    transfer: LocalPoint[];
    leadIn: LocalPoint[];
    score: number;
    sourceContinuity: number;
    targetContinuity: number;
  } | null = null;

  for (const sign of [-1, 1] as const) {
    const tangent = buildTangentCandidate(
      sourceCenterLocal,
      source.loiterRadiusM,
      targetCenterLocal,
      target.loiterRadiusM,
      tangentKind,
      sign,
    );
    if (!tangent) continue;
    const transferDirection = normalizeLocal(subtract(tangent.end, tangent.start));
    const sourceTravelTangent = circleTravelTangentAtPoint(sourceCenterLocal, tangent.start, source.loiterDirection);
    const targetTravelTangent = circleTravelTangentAtPoint(targetCenterLocal, tangent.end, target.loiterDirection);
    if (!transferDirection || !sourceTravelTangent || !targetTravelTangent) continue;
    const sourceContinuity = dot(sourceTravelTangent, transferDirection);
    const targetContinuity = dot(targetTravelTangent, transferDirection);
    if (sourceContinuity <= 0.95 || targetContinuity <= 0.95) {
      continue;
    }

    const leadOut = dedupe2DPath([
      source.anchorPoint,
      fromLocal(sourceCircleAnchor, reference),
      ...sampleArcPoints(
        sourceCenterLocal,
        source.loiterRadiusM,
        sourceCircleAnchor,
        tangent.start,
        source.loiterDirection,
      ).slice(1).map((point) => fromLocal(point, reference)),
    ]);
    const transfer = dedupe2DPath([
      fromLocal(tangent.start, reference),
      fromLocal(tangent.end, reference),
    ]);
    const leadIn = dedupe2DPath([
      fromLocal(tangent.end, reference),
      ...sampleArcPoints(
        targetCenterLocal,
        target.loiterRadiusM,
        tangent.end,
        targetCircleAnchor,
        target.loiterDirection,
      ).slice(1).map((point) => fromLocal(point, reference)),
      target.anchorPoint,
    ]);
    const score = polylineLengthMeters(leadOut) + polylineLengthMeters(transfer) + polylineLengthMeters(leadIn);
    if (!bestCandidate || score < bestCandidate.score - 1e-6) {
      bestCandidate = {
        leadOut: leadOut.map((point) => toLocal(point, reference)),
        transfer: transfer.map((point) => toLocal(point, reference)),
        leadIn: leadIn.map((point) => toLocal(point, reference)),
        score,
        sourceContinuity,
        targetContinuity,
      };
    }
  }

  if (!bestCandidate) {
    return buildFallbackConnection2D(reference, source, target);
  }

  return {
    leadOut: dedupe2DPath(bestCandidate.leadOut.map((point) => fromLocal(point, reference))),
    transfer: dedupe2DPath(bestCandidate.transfer.map((point) => fromLocal(point, reference))),
    leadIn: dedupe2DPath(bestCandidate.leadIn.map((point) => fromLocal(point, reference))),
  };
}

function requiredLoopCount(altitudeDeltaM: number, baseDistanceM: number, radiusM: number): number {
  if (altitudeDeltaM <= 1e-6) return 0;
  const maxLiftWithoutLoops = baseDistanceM * CONNECTOR_ELEVATION_PER_METER;
  if (altitudeDeltaM <= maxLiftWithoutLoops + 1e-6) return 0;
  const circumferenceM = 2 * Math.PI * radiusM;
  if (circumferenceM <= 1e-6) return 0;
  const extraDistanceNeededM = altitudeDeltaM / CONNECTOR_ELEVATION_PER_METER - baseDistanceM;
  return Math.max(1, Math.ceil(extraDistanceNeededM / circumferenceM));
}

function liftPolyline2D(
  path: LngLat[],
  startAltitudeM: number,
  endAltitudeM: number,
): [number, number, number][] {
  if (path.length === 0) return [];
  if (path.length === 1 || Math.abs(endAltitudeM - startAltitudeM) <= 1e-6) {
    return path.map(([lng, lat]) => [lng, lat, startAltitudeM] as [number, number, number]);
  }

  const segmentLengths = path.map((point, index) =>
    index > 0 ? haversineDistance(path[index - 1], point) : 0,
  );
  const totalDistance = Math.max(segmentLengths.reduce((sum, value) => sum + value, 0), 1e-9);
  let traversedDistance = 0;
  return path.map(([lng, lat], index) => {
    if (index > 0) traversedDistance += segmentLengths[index];
    const t = traversedDistance / totalDistance;
    return [lng, lat, startAltitudeM + (endAltitudeM - startAltitudeM) * t] as [number, number, number];
  });
}

function requiredSafeAltitudeForPolyline(
  path: LngLat[],
  tiles: TerrainTile[],
  minClearanceM: number,
): number {
  if (path.length === 0 || tiles.length === 0) return -Infinity;
  const { max } = queryMinMaxElevationAlongPolylineWGS84(path, tiles, 12);
  return Number.isFinite(max) ? max + minClearanceM : -Infinity;
}

function requiredSafeAltitudeAtPoint(
  point: LngLat,
  tiles: TerrainTile[],
  minClearanceM: number,
): number {
  if (tiles.length === 0) return -Infinity;
  const { max } = queryMinMaxElevationAlongPolylineWGS84([point], tiles, 1);
  return Number.isFinite(max) ? max + minClearanceM : -Infinity;
}

function liftPolyline2DTerrainAware(
  path: LngLat[],
  startAltitudeM: number,
  endAltitudeM: number,
  tiles: TerrainTile[],
  altitudeMode: 'legacy' | 'min-clearance',
  minClearanceM: number,
): [number, number, number][] {
  if (altitudeMode !== 'min-clearance' || tiles.length === 0) {
    return liftPolyline2D(path, startAltitudeM, endAltitudeM);
  }
  if (path.length === 0) return [];
  if (path.length === 1) {
    const [lng, lat] = path[0]!;
    const safeAltitudeM = requiredSafeAltitudeAtPoint(path[0]!, tiles, minClearanceM);
    return [[lng, lat, Math.max(startAltitudeM, endAltitudeM, safeAltitudeM)]];
  }

  const segmentLengths = path.map((point, index) =>
    index > 0 ? haversineDistance(path[index - 1]!, point) : 0,
  );
  const totalDistance = Math.max(segmentLengths.reduce((sum, value) => sum + value, 0), 1e-9);
  let traversedDistance = 0;
  return path.map((point, index) => {
    if (index > 0) traversedDistance += segmentLengths[index]!;
    const t = traversedDistance / totalDistance;
    const interpolatedAltitudeM = startAltitudeM + (endAltitudeM - startAltitudeM) * t;
    const safeAltitudeM = requiredSafeAltitudeAtPoint(point, tiles, minClearanceM);
    return [
      point[0],
      point[1],
      Number.isFinite(safeAltitudeM) ? Math.max(interpolatedAltitudeM, safeAltitudeM) : interpolatedAltitudeM,
    ] as [number, number, number];
  });
}

function prependCoilLoop(
  path: LngLat[],
  maneuver: PlannedLeadManeuver,
  loopCount: number,
): LngLat[] {
  if (path.length === 0 || loopCount <= 0) return path;
  const reference = maneuver.loiterCenter;
  const coil = buildCoilLoop(
    toLocal(maneuver.loiterCenter, reference),
    maneuver.loiterRadiusM,
    maneuver.loiterDirection,
    toLocal(path[0], reference),
    loopCount,
  );
  return dedupe2DPath([
    ...coil.map((point) => fromLocal(point, reference)),
    ...path.slice(1),
  ]);
}

function appendCoilLoop(
  path: LngLat[],
  maneuver: PlannedLeadManeuver,
  loopCount: number,
): LngLat[] {
  if (path.length === 0 || loopCount <= 0) return path;
  const reference = maneuver.loiterCenter;
  const anchorPoint = path[path.length - 1];
  const coil = buildCoilLoop(
    toLocal(maneuver.loiterCenter, reference),
    maneuver.loiterRadiusM,
    maneuver.loiterDirection,
    toLocal(anchorPoint, reference),
    loopCount,
  );
  return dedupe2DPath([
    ...path,
    ...coil.slice(1).map((point) => fromLocal(point, reference)),
  ]);
}

export function buildInterAreaConnectionGeometry(
  input: InterAreaConnectionInput,
): InterAreaConnectionGeometry {
  const { sourceManeuver, targetManeuver } = input;
  const terrainTiles = input.terrainTiles ?? [];
  const altitudeMode = input.altitudeMode ?? 'legacy';
  const minClearanceM = Math.max(0, input.minClearanceM ?? 0);
  const base2D = buildConnection2D(sourceManeuver, targetManeuver);
  const transfer = base2D.transfer;
  let transferAltitudeM = Math.max(input.sourceAnchorAltitudeM, input.targetAnchorAltitudeM);
  let leadOut = base2D.leadOut;
  let leadIn = base2D.leadIn;

  for (let iteration = 0; iteration < 4; iteration += 1) {
    const sourceIsLower = input.sourceAnchorAltitudeM < transferAltitudeM - 1e-6;
    const targetIsLower = input.targetAnchorAltitudeM < transferAltitudeM - 1e-6;
    const sourceLoopCount = sourceIsLower
      ? requiredLoopCount(
          transferAltitudeM - input.sourceAnchorAltitudeM,
          polylineLengthMeters(base2D.leadOut),
          sourceManeuver.loiterRadiusM,
        )
      : 0;
    const targetLoopCount = targetIsLower
      ? requiredLoopCount(
          transferAltitudeM - input.targetAnchorAltitudeM,
          polylineLengthMeters(base2D.leadIn),
          targetManeuver.loiterRadiusM,
        )
      : 0;

    leadOut = appendCoilLoop(base2D.leadOut, sourceManeuver, sourceLoopCount);
    leadIn = prependCoilLoop(base2D.leadIn, targetManeuver, targetLoopCount);

    if (altitudeMode !== 'min-clearance' || terrainTiles.length === 0) {
      break;
    }

    const requiredTransferAltitudeM = Math.max(
      transferAltitudeM,
      requiredSafeAltitudeForPolyline(leadOut, terrainTiles, minClearanceM),
      requiredSafeAltitudeForPolyline(transfer, terrainTiles, minClearanceM),
      requiredSafeAltitudeForPolyline(leadIn, terrainTiles, minClearanceM),
    );
    if (requiredTransferAltitudeM <= transferAltitudeM + 1e-6) {
      break;
    }
    transferAltitudeM = requiredTransferAltitudeM;
  }

  const path3D: [number, number, number][][] = [];
  if (leadOut.length >= 2) {
    path3D.push(
      liftPolyline2DTerrainAware(
        leadOut,
        input.sourceAnchorAltitudeM,
        transferAltitudeM,
        terrainTiles,
        altitudeMode,
        minClearanceM,
      ),
    );
  }
  if (transfer.length >= 2) {
    path3D.push(transfer.map(([lng, lat]) => [lng, lat, transferAltitudeM] as [number, number, number]));
  }
  if (leadIn.length >= 2) {
    path3D.push(
      liftPolyline2DTerrainAware(
        leadIn,
        transferAltitudeM,
        input.targetAnchorAltitudeM,
        terrainTiles,
        altitudeMode,
        minClearanceM,
      ),
    );
  }

  const distanceM = path3DLengthMeters(path3D);

  return {
    key: input.key,
    fromPolygonId: input.fromPolygonId,
    toPolygonId: input.toPolygonId,
    leadOut,
    transfer,
    leadIn,
    path3D,
    terrainZoom: input.terrainZoom,
    terrainTileCount: input.terrainTileCount,
    sourceAltitudeM: input.sourceAnchorAltitudeM,
    targetAltitudeM: input.targetAnchorAltitudeM,
    transferAltitudeM,
    distanceM,
    timeSec: input.cruiseSpeedMps > 0 ? distanceM / input.cruiseSpeedMps : 0,
  };
}

export function buildInterAreaConnectionCorridorRing(
  connection: Pick<InterAreaConnectionGeometry, 'leadOut' | 'transfer' | 'leadIn'>,
  paddingM = 90,
): LngLat[] | null {
  const points = [...connection.leadOut, ...connection.transfer, ...connection.leadIn];
  if (points.length === 0) return null;
  const reference = points[0];
  const locals = points.map((point) => toLocal(point, reference));
  const xs = locals.map((point) => point.x);
  const ys = locals.map((point) => point.y);
  const minX = Math.min(...xs) - paddingM;
  const maxX = Math.max(...xs) + paddingM;
  const minY = Math.min(...ys) - paddingM;
  const maxY = Math.max(...ys) + paddingM;
  return [
    fromLocal({ x: minX, y: minY }, reference),
    fromLocal({ x: maxX, y: minY }, reference),
    fromLocal({ x: maxX, y: maxY }, reference),
    fromLocal({ x: minX, y: maxY }, reference),
    fromLocal({ x: minX, y: minY }, reference),
  ];
}

export function buildMissionTravelSummary(
  areas: MissionAreaSummaryInput[],
  connections: InterAreaConnectionGeometry[],
): MissionTravelSummary {
  const areaDistanceM = areas.reduce((sum, area) => sum + connectedPathLengthMetersLocal(area.geometry), 0);
  const areaTimeSec = areas.reduce((sum, area) => {
    const distanceM = connectedPathLengthMetersLocal(area.geometry);
    const cruiseSpeedMps = getCruiseSpeedMps(area.params);
    return sum + (cruiseSpeedMps > 0 ? distanceM / cruiseSpeedMps : 0);
  }, 0);
  const connectorDistanceM = connections.reduce((sum, connection) => sum + connection.distanceM, 0);
  const connectorTimeSec = connections.reduce((sum, connection) => sum + connection.timeSec, 0);
  return {
    areaDistanceM,
    areaTimeSec,
    connectorDistanceM,
    connectorTimeSec,
    totalDistanceM: areaDistanceM + connectorDistanceM,
    totalTimeSec: areaTimeSec + connectorTimeSec,
  };
}

export function buildMissionFlightGeometry(
  orderedPolygonIds: string[],
  connections: InterAreaConnectionGeometry[],
  areas: MissionAreaSummaryInput[],
): MissionFlightGeometry {
  return {
    orderedPolygonIds: [...orderedPolygonIds],
    connections,
    summary: buildMissionTravelSummary(areas, connections),
  };
}
