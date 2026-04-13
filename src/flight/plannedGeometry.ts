import type { FlightParams, LngLat, PlannedFlightGeometry, PlannedTurnBlock } from "@/domain/types";
import { generateFlightLinesForPolygon } from "./flightLines";
import { groupFlightLinesForTraversal, haversineDistance } from "./geometry";

type LocalPoint = {
  north: number;
  east: number;
};

type LocalSegment = {
  start: LocalPoint;
  end: LocalPoint;
};

type TurnPreviewModelParams = {
  trimSpeedMps: number;
  turnSpeedMps: number;
  l1PeriodS: number;
  l1Damping: number;
  yawRateEfficiency: number;
  rollNaturalFrequencyRadS: number;
  rollDampingRatio: number;
  maxRollDeg: number;
  maxRollRateDegS: number;
  slowdownDistanceM: number;
  speedTimeConstantS: number;
  dtS: number;
  maxPreviewDurationS: number;
};

type L1State = {
  rollSpRad: number;
  previousLatAccMps2: number;
  l1Ratio: number;
  l1DistanceM: number;
  gain: number;
};

type TurnBlockGeometry = {
  startSweep: LocalPoint;
  endSweep: LocalPoint;
  turnOff: LocalPoint;
  loiterCenter: LocalPoint;
  nextSweepStart: LocalPoint;
  nextSweepEnd: LocalPoint;
  loiterRadiusM: number;
  loiterDirection: 1 | -1;
  turnOffAcceptanceRadiusM: number;
};

type TurnFramePoint = {
  along: number;
  cross: number;
};

type CachedTurnPreview = {
  turnaroundPath: TurnFramePoint[];
  loiterEntryPoint?: TurnFramePoint;
  loiterExitPoint?: TurnFramePoint;
};

type FlightTurnModel = {
  defaultTurnaroundRadiusM: number;
  turnaroundRadiusFunctionSlope: number;
  obliqueTiltFlightPlanDeg: number;
  trimSpeedMps?: number;
};

const GEODESY_EARTH_RADIUS_M = 6378137.0;
const PHYSICS_GRAVITY_M_S2 = 9.80665;

const MISSION_DEFAULT_TURNAROUND_DISTANCE_M = 80.0;
const MISSION_DEFAULT_TURNAROUND_SIDE_OFFSET_M = 70.0;
const MISSION_TRIGGER_TO_POLYGON_DISTANCE_M = 15.0;
const TURN_PREVIEW_V5_DEFAULT_TURNAROUND_RADIUS_THRESHOLD_M = 40.0;

const PX4_MIS_YAW_ERR_DEG = 15.0;
const PX4_L1_WAYPOINT_REACQUIRE_BEARING_LIMIT_DEG = 100.0;
const PX4_LOITER_EXIT_ROLL_SOFT_OFFSET_START_DEG = 20.0;
const PX4_LOITER_EXIT_ROLL_SOFT_OFFSET_END_DEG = 45.0;
const PX4_LOITER_EXIT_COG_OFFSET_MAX_DEG = 30.0;
const PX4_LOITER_EXIT_THRESHOLD_MULTIPLIER_MAX = 4.0;
const PX4_V4_FW_L1_PERIOD_S = 6.9;
const PX4_V5_FW_L1_PERIOD_S = 10.0;
const PX4_FW_L1_DAMPING = 0.75;
const PX4_FW_L1_R_SLEW_MAX_DEG_S = 150.0;
const PX4_FW_R_LIM_DEG = 55.0;
const PX4_V4_FW_AIRSPD_TRIM_MPS = 16.0;
const PX4_V5_FW_AIRSPD_TRIM_MPS = 19.0;
const PX4_FW_AIRSPD_T_TRIM_MPS = 17.0;

const FINETUNE_YAW_RATE_EFFICIENCY = 0.93;
const FINETUNE_ROLL_RESPONSE_NATURAL_FREQUENCY_RAD_S = 3.0;
const FINETUNE_ROLL_RESPONSE_DAMPING_RATIO = 0.5;
const FINETUNE_SPEED_RESPONSE_TIME_CONSTANT_S = 2.0;
const FINETUNE_SIMULATION_DT_S = 0.02;
const FINETUNE_MIN_PREVIEW_DURATION_S = 18.0;
const FINETUNE_PREVIEW_DURATION_BUFFER_S = 8.0;
const FINETUNE_PREVIEW_LOITER_DISTANCE_MULTIPLIER = 2.5;
const FINETUNE_MIN_PREVIEW_SPEED_MPS = 8.0;
const FINETUNE_MIN_CUSTOM_ACCEPTANCE_RADIUS_TO_HONOR_M = 15.0;
const FINETUNE_LOITER_LOOP_DELTA_THRESHOLD_RAD = 1.0;
const FINETUNE_MIN_YAW_RATE_SPEED_MPS = 1.0;
const FINETUNE_NEXT_SWEEP_REACHED_DISTANCE_M = 5.0;
const FINETUNE_NEXT_SWEEP_ALIGNMENT_DISTANCE_M = 5.0;
const FINETUNE_NEXT_SWEEP_APPROACH_MIN_DISTANCE_M = 1.5;
const FINETUNE_NEXT_SWEEP_APPROACH_MIN_ADVANCE_M = 2.0;
const FINETUNE_NEXT_SWEEP_APPROACH_MAX_ADVANCE_M = 8.0;
const FINETUNE_PREVIEW_POINT_SPACING_M = 2.0;
const FINETUNE_SMOOTH_JOIN_POINT_SPACING_M = 5.0;
const TURN_PREVIEW_CACHE_PRECISION_M = 0.1;
const TURN_PREVIEW_CACHE_LIMIT = 2048;
const PLANNED_GEOMETRY_CACHE_LIMIT = 128;

const turnPreviewCache = new Map<string, CachedTurnPreview>();
const plannedGeometryCache = new Map<
  string,
  PlannedFlightGeometry & { bounds: { minLng: number; minLat: number; maxLng: number; maxLat: number; centroid: [number, number] } }
>();

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const wrapPi = (angleRad: number) => {
  let wrappedAngle = angleRad;
  while (wrappedAngle > Math.PI) wrappedAngle -= Math.PI * 2;
  while (wrappedAngle < -Math.PI) wrappedAngle += Math.PI * 2;
  return wrappedAngle;
};

const normaliseDegrees = (angleDeg: number) => {
  const normalisedAngle = angleDeg % 360;
  return normalisedAngle >= 0 ? normalisedAngle : normalisedAngle + 360;
};

const addLocal = (a: LocalPoint, b: LocalPoint): LocalPoint => ({ north: a.north + b.north, east: a.east + b.east });
const subtractLocal = (a: LocalPoint, b: LocalPoint): LocalPoint => ({ north: a.north - b.north, east: a.east - b.east });
const multiplyLocal = (value: LocalPoint, scalar: number): LocalPoint => ({ north: value.north * scalar, east: value.east * scalar });
const dotLocal = (a: LocalPoint, b: LocalPoint) => a.north * b.north + a.east * b.east;
const crossLocal = (a: LocalPoint, b: LocalPoint) => a.north * b.east - a.east * b.north;
const normLocal = (value: LocalPoint) => Math.hypot(value.north, value.east);
const unitLocal = (value: LocalPoint): LocalPoint => {
  const norm = normLocal(value);
  return norm > 1e-9 ? { north: value.north / norm, east: value.east / norm } : { north: 0, east: 0 };
};
const negateLocal = (value: LocalPoint): LocalPoint => ({ north: -value.north, east: -value.east });
const addOffsetToLocalPoint = (point: LocalPoint, direction: LocalPoint, distanceM: number): LocalPoint =>
  addLocal(point, multiplyLocal(unitLocal(direction), distanceM));

const perpendicularRightLocal = (direction: LocalPoint): LocalPoint => ({
  north: -direction.east,
  east: direction.north,
});

const toLocalPoint = (coordinate: LngLat, referenceCoordinate: LngLat): LocalPoint => {
  const dLat = ((coordinate[1] - referenceCoordinate[1]) * Math.PI) / 180;
  const dLon = ((coordinate[0] - referenceCoordinate[0]) * Math.PI) / 180;
  const cosLat = Math.cos((referenceCoordinate[1] * Math.PI) / 180);
  return {
    north: dLat * GEODESY_EARTH_RADIUS_M,
    east: dLon * GEODESY_EARTH_RADIUS_M * cosLat,
  };
};

const fromLocalPoint = (point: LocalPoint, referenceCoordinate: LngLat): LngLat => {
  const lat = referenceCoordinate[1] + (point.north / GEODESY_EARTH_RADIUS_M) * (180 / Math.PI);
  const lon =
    referenceCoordinate[0] +
    (point.east / (GEODESY_EARTH_RADIUS_M * Math.cos((referenceCoordinate[1] * Math.PI) / 180))) * (180 / Math.PI);
  return [lon, lat];
};

const bearingRad = (start: LocalPoint, end: LocalPoint) => {
  const delta = subtractLocal(end, start);
  return Math.atan2(delta.east, delta.north);
};

const advanceLocalPoint = (start: LocalPoint, distanceM: number, bearingAngleRad: number): LocalPoint => ({
  north: start.north + distanceM * Math.cos(bearingAngleRad),
  east: start.east + distanceM * Math.sin(bearingAngleRad),
});

const simplifyPreviewPathLocal = (path: LocalPoint[], minSpacingM: number) => {
  if (path.length <= 2 || !(minSpacingM > 0)) return path;

  const simplified: LocalPoint[] = [path[0]];
  let lastKeptPoint = path[0];

  for (let index = 1; index < path.length - 1; index += 1) {
    const candidatePoint = path[index];
    if (normLocal(subtractLocal(candidatePoint, lastKeptPoint)) < minSpacingM) continue;
    simplified.push(candidatePoint);
    lastKeptPoint = candidatePoint;
  }

  const lastPoint = path[path.length - 1];
  if (lastKeptPoint !== lastPoint) {
    simplified.push(lastPoint);
  }

  return simplified;
};

const dedupeCoordinates = (coordinates: LngLat[]) =>
  coordinates.filter((coordinate, index) => {
    if (index === 0) return true;
    const previousCoordinate = coordinates[index - 1];
    return coordinate[0] !== previousCoordinate[0] || coordinate[1] !== previousCoordinate[1];
  });

const getPlannedGeometryCacheKey = (
  ring: LngLat[],
  bearingDeg: number,
  lineSpacingM: number,
  params: FlightParams,
  includeTurnPreview: boolean,
) => [
  includeTurnPreview ? "preview" : "raw",
  normaliseDegrees(bearingDeg).toFixed(6),
  lineSpacingM.toFixed(6),
  params.payloadKind,
  params.cameraKey ?? "",
  params.lidarKey ?? "",
  params.altitudeAGL.toFixed(3),
  (params.speedMps ?? 0).toFixed(3),
  ring.map(([lng, lat]) => `${lng.toFixed(10)},${lat.toFixed(10)}`).join(";"),
].join("|");

const rememberPlannedGeometry = (
  cacheKey: string,
  geometry: PlannedFlightGeometry & { bounds: { minLng: number; minLat: number; maxLng: number; maxLat: number; centroid: [number, number] } },
) => {
  if (plannedGeometryCache.has(cacheKey)) {
    plannedGeometryCache.delete(cacheKey);
  }
  plannedGeometryCache.set(cacheKey, geometry);
  if (plannedGeometryCache.size <= PLANNED_GEOMETRY_CACHE_LIMIT) return;
  const oldestKey = plannedGeometryCache.keys().next().value;
  if (oldestKey) plannedGeometryCache.delete(oldestKey);
};

const toTurnFramePoint = (
  point: LocalPoint,
  origin: LocalPoint,
  forwardDirection: LocalPoint,
): TurnFramePoint => {
  const rightDirection = perpendicularRightLocal(forwardDirection);
  const delta = subtractLocal(point, origin);
  return {
    along: dotLocal(delta, forwardDirection),
    cross: dotLocal(delta, rightDirection),
  };
};

const fromTurnFramePoint = (
  point: TurnFramePoint,
  origin: LocalPoint,
  forwardDirection: LocalPoint,
): LocalPoint => {
  const rightDirection = perpendicularRightLocal(forwardDirection);
  return addLocal(
    origin,
    addLocal(
      multiplyLocal(forwardDirection, point.along),
      multiplyLocal(rightDirection, point.cross),
    ),
  );
};

const roundTurnPreviewCacheValue = (value: number) =>
  Math.round(value / TURN_PREVIEW_CACHE_PRECISION_M) * TURN_PREVIEW_CACHE_PRECISION_M;

const serializeTurnFramePoint = (point: TurnFramePoint) =>
  `${roundTurnPreviewCacheValue(point.along).toFixed(1)},${roundTurnPreviewCacheValue(point.cross).toFixed(1)}`;

const getTurnPreviewCacheKey = (
  geometry: TurnBlockGeometry,
  previewStart: LocalPoint,
  previewParams: TurnPreviewModelParams,
  forwardDirection: LocalPoint,
) => {
  if (normLocal(forwardDirection) <= 1e-9) return undefined;

  return [
    serializeTurnFramePoint(toTurnFramePoint(previewStart, geometry.endSweep, forwardDirection)),
    serializeTurnFramePoint(toTurnFramePoint(geometry.turnOff, geometry.endSweep, forwardDirection)),
    serializeTurnFramePoint(toTurnFramePoint(geometry.loiterCenter, geometry.endSweep, forwardDirection)),
    serializeTurnFramePoint(toTurnFramePoint(geometry.nextSweepStart, geometry.endSweep, forwardDirection)),
    serializeTurnFramePoint(toTurnFramePoint(geometry.nextSweepEnd, geometry.endSweep, forwardDirection)),
    roundTurnPreviewCacheValue(geometry.loiterRadiusM).toFixed(1),
    roundTurnPreviewCacheValue(geometry.turnOffAcceptanceRadiusM).toFixed(1),
    geometry.loiterDirection.toString(),
    roundTurnPreviewCacheValue(previewParams.trimSpeedMps).toFixed(1),
    roundTurnPreviewCacheValue(previewParams.turnSpeedMps).toFixed(1),
    roundTurnPreviewCacheValue(previewParams.slowdownDistanceM).toFixed(1),
  ].join("|");
};

const cacheTurnPreview = (
  cacheKey: string,
  geometry: TurnBlockGeometry,
  forwardDirection: LocalPoint,
  turnaroundPathLocal: LocalPoint[],
  loiterEntryPoint?: LocalPoint,
  loiterExitPoint?: LocalPoint,
) => {
  turnPreviewCache.set(cacheKey, {
    turnaroundPath: turnaroundPathLocal.map((point) => toTurnFramePoint(point, geometry.endSweep, forwardDirection)),
    loiterEntryPoint: loiterEntryPoint
      ? toTurnFramePoint(loiterEntryPoint, geometry.endSweep, forwardDirection)
      : undefined,
    loiterExitPoint: loiterExitPoint
      ? toTurnFramePoint(loiterExitPoint, geometry.endSweep, forwardDirection)
      : undefined,
  });
  if (turnPreviewCache.size <= TURN_PREVIEW_CACHE_LIMIT) return;
  const oldestKey = turnPreviewCache.keys().next().value;
  if (oldestKey) turnPreviewCache.delete(oldestKey);
};

const restoreTurnPreviewFromCache = (
  cachedPreview: CachedTurnPreview,
  geometry: TurnBlockGeometry,
  forwardDirection: LocalPoint,
) => ({
  turnaroundPathLocal: cachedPreview.turnaroundPath.map((point) =>
    fromTurnFramePoint(point, geometry.endSweep, forwardDirection),
  ),
  loiterEntryPoint: cachedPreview.loiterEntryPoint
    ? fromTurnFramePoint(cachedPreview.loiterEntryPoint, geometry.endSweep, forwardDirection)
    : undefined,
  loiterExitPoint: cachedPreview.loiterExitPoint
    ? fromTurnFramePoint(cachedPreview.loiterExitPoint, geometry.endSweep, forwardDirection)
    : undefined,
});

const projectPointOntoPolyline = (point: LngLat, line: LngLat[]) => {
  if (line.length < 2) return undefined;

  const referenceCoordinate = line[0];
  const pointLocal = toLocalPoint(point, referenceCoordinate);
  let bestProjection:
    | {
        coordinate: LngLat;
        segmentIndex: number;
        distanceM: number;
      }
    | undefined;

  for (let segmentIndex = 0; segmentIndex < line.length - 1; segmentIndex += 1) {
    const segmentStart = toLocalPoint(line[segmentIndex], referenceCoordinate);
    const segmentEnd = toLocalPoint(line[segmentIndex + 1], referenceCoordinate);
    const segmentVector = subtractLocal(segmentEnd, segmentStart);
    const segmentLengthSquared = dotLocal(segmentVector, segmentVector);
    const projectionScalar =
      segmentLengthSquared > 1e-9
        ? clamp(dotLocal(subtractLocal(pointLocal, segmentStart), segmentVector) / segmentLengthSquared, 0, 1)
        : 0;
    const projectedPoint = addLocal(segmentStart, multiplyLocal(segmentVector, projectionScalar));
    const distanceM = normLocal(subtractLocal(pointLocal, projectedPoint));

    if (!bestProjection || distanceM < bestProjection.distanceM) {
      bestProjection = {
        coordinate: fromLocalPoint(projectedPoint, referenceCoordinate),
        segmentIndex,
        distanceM,
      };
    }
  }

  return bestProjection;
};

const getMissionTurnaroundRadius = (gridSpacingM: number, defaultTurnaroundRadiusM: number) => {
  let radiusM = MISSION_DEFAULT_TURNAROUND_DISTANCE_M / 2;
  const maxTurnaroundSideOffset = Math.max(MISSION_DEFAULT_TURNAROUND_SIDE_OFFSET_M - gridSpacingM, 0);
  radiusM = Math.min((maxTurnaroundSideOffset / 2 + gridSpacingM) / 2, radiusM);
  return Math.max(radiusM, defaultTurnaroundRadiusM);
};

const getTurnOffAcceptanceRadius = (obliqueTiltFlightPlanDeg: number, heightAboveGroundM: number) => {
  const obliqueLookAheadM = Math.tan((obliqueTiltFlightPlanDeg * Math.PI) / 180) * heightAboveGroundM;
  return MISSION_DEFAULT_TURNAROUND_DISTANCE_M - MISSION_TRIGGER_TO_POLYGON_DISTANCE_M + obliqueLookAheadM;
};

export const getFlightTurnModel = (params: FlightParams): FlightTurnModel => {
  const key = `${params.cameraKey ?? ""} ${params.lidarKey ?? ""}`.toLowerCase();
  const explicitV4 = key.includes("v4") || key.includes("rx1r2");
  const explicitV5 =
    key.includes("v5") ||
    key.includes("survey61") ||
    key.includes("survey24") ||
    key.includes("a6100") ||
    key.includes("rx1r3") ||
    key.includes("zenmuse") ||
    key.includes("p1") ||
    key.includes("lidar");
  const isV5Variant = explicitV5 || !explicitV4;
  const isObliquePayload = key.includes("map61") || key.includes("mapstaroblique");

  return {
    defaultTurnaroundRadiusM: isV5Variant ? 40 : 35,
    turnaroundRadiusFunctionSlope: isV5Variant ? 0.3 : 0.2,
    obliqueTiltFlightPlanDeg: isObliquePayload ? 45 : 0,
    trimSpeedMps: params.payloadKind === "lidar" ? params.speedMps : undefined,
  };
};

const createTurnPreviewModelParams = (
  defaultTurnaroundRadius: number,
  trimSpeedOverrideMps?: number,
): TurnPreviewModelParams => {
  const isV5Variant = defaultTurnaroundRadius >= TURN_PREVIEW_V5_DEFAULT_TURNAROUND_RADIUS_THRESHOLD_M;
  const trimSpeedMps = trimSpeedOverrideMps ?? (isV5Variant ? PX4_V5_FW_AIRSPD_TRIM_MPS : PX4_V4_FW_AIRSPD_TRIM_MPS);
  return {
    trimSpeedMps,
    turnSpeedMps: Math.min(PX4_FW_AIRSPD_T_TRIM_MPS, trimSpeedMps),
    l1PeriodS: isV5Variant ? PX4_V5_FW_L1_PERIOD_S : PX4_V4_FW_L1_PERIOD_S,
    l1Damping: PX4_FW_L1_DAMPING,
    yawRateEfficiency: FINETUNE_YAW_RATE_EFFICIENCY,
    rollNaturalFrequencyRadS: FINETUNE_ROLL_RESPONSE_NATURAL_FREQUENCY_RAD_S,
    rollDampingRatio: FINETUNE_ROLL_RESPONSE_DAMPING_RATIO,
    maxRollDeg: PX4_FW_R_LIM_DEG,
    maxRollRateDegS: PX4_FW_L1_R_SLEW_MAX_DEG_S,
    slowdownDistanceM: MISSION_DEFAULT_TURNAROUND_DISTANCE_M,
    speedTimeConstantS: FINETUNE_SPEED_RESPONSE_TIME_CONSTANT_S,
    dtS: FINETUNE_SIMULATION_DT_S,
    maxPreviewDurationS: FINETUNE_MIN_PREVIEW_DURATION_S,
  };
};

const createInitialL1State = ({ l1Damping, l1PeriodS }: TurnPreviewModelParams): L1State => ({
  rollSpRad: 0,
  previousLatAccMps2: 0,
  l1Ratio: (l1Damping * l1PeriodS) / Math.PI,
  l1DistanceM: 0,
  gain: 4 * l1Damping * l1Damping,
});

const updateRollSetpoint = (latAccMps2: number, state: L1State, previewParams: TurnPreviewModelParams) => {
  const maxRollRad = (previewParams.maxRollDeg * Math.PI) / 180;
  const maxRollRateRadS = (previewParams.maxRollRateDegS * Math.PI) / 180;
  const unclampedRollSpRad = Math.atan2(latAccMps2, PHYSICS_GRAVITY_M_S2);
  const clampedRollSpRad = clamp(unclampedRollSpRad, -maxRollRad, maxRollRad);
  const rollDelta = maxRollRateRadS * previewParams.dtS;
  const limitedRollSpRad = clamp(clampedRollSpRad, state.rollSpRad - rollDelta, state.rollSpRad + rollDelta);
  const latAccSlewRate = PHYSICS_GRAVITY_M_S2 * Math.tan(maxRollRateRadS * previewParams.dtS);
  const limitedLatAccMps2 = clamp(
    latAccMps2,
    state.previousLatAccMps2 - latAccSlewRate,
    state.previousLatAccMps2 + latAccSlewRate,
  );

  state.rollSpRad = limitedRollSpRad;
  state.previousLatAccMps2 = limitedLatAccMps2;
  return { latAccMps2: limitedLatAccMps2, rollSpRad: limitedRollSpRad };
};

const navigateWaypoints = (
  waypointA: LocalPoint,
  waypointB: LocalPoint,
  currentPosition: LocalPoint,
  groundSpeed: LocalPoint,
  l1State: L1State,
  previewParams: TurnPreviewModelParams,
) => {
  const currentGroundSpeed = Math.max(normLocal(groundSpeed), 0.1);
  l1State.l1Ratio = (previewParams.l1Damping * previewParams.l1PeriodS) / Math.PI;
  l1State.l1DistanceM = l1State.l1Ratio * currentGroundSpeed;

  let trackDirection = subtractLocal(waypointB, waypointA);
  if (normLocal(trackDirection) < 1e-6) trackDirection = subtractLocal(waypointB, currentPosition);
  const trackDirectionUnit = unitLocal(trackDirection);
  const waypointAToCurrent = subtractLocal(currentPosition, waypointA);
  const distanceToWaypointA = normLocal(waypointAToCurrent);
  const alongTrackDistance = dotLocal(waypointAToCurrent, trackDirectionUnit);

  const waypointBToCurrentUnit = unitLocal(subtractLocal(currentPosition, waypointB));
  const waypointBToCurrentBearing = Math.atan2(
    crossLocal(waypointBToCurrentUnit, trackDirectionUnit),
    dotLocal(waypointBToCurrentUnit, trackDirectionUnit),
  );

  let etaRad = 0;
  if (
    distanceToWaypointA > l1State.l1DistanceM &&
    alongTrackDistance / Math.max(distanceToWaypointA, 1.0) < -Math.SQRT1_2
  ) {
    const reverseDirection = multiplyLocal(unitLocal(waypointAToCurrent), -1);
    etaRad = Math.atan2(crossLocal(groundSpeed, reverseDirection), dotLocal(groundSpeed, reverseDirection));
  } else if (Math.abs(waypointBToCurrentBearing) < (PX4_L1_WAYPOINT_REACQUIRE_BEARING_LIMIT_DEG * Math.PI) / 180) {
    const reverseDirection = multiplyLocal(waypointBToCurrentUnit, -1);
    etaRad = Math.atan2(crossLocal(groundSpeed, reverseDirection), dotLocal(groundSpeed, reverseDirection));
  } else {
    const eta2Rad = Math.atan2(crossLocal(groundSpeed, trackDirectionUnit), dotLocal(groundSpeed, trackDirectionUnit));
    const crossTrackErrorM = crossLocal(waypointAToCurrent, trackDirectionUnit);
    const eta1Rad = Math.asin(clamp(crossTrackErrorM / l1State.l1DistanceM, -Math.SQRT1_2, Math.SQRT1_2));
    etaRad = eta1Rad + eta2Rad;
  }

  const limitedEtaRad = clamp(etaRad, -Math.PI / 2, Math.PI / 2);
  const latAccMps2 =
    (l1State.gain * currentGroundSpeed * currentGroundSpeed * Math.sin(limitedEtaRad)) / Math.max(l1State.l1DistanceM, 0.1);
  return updateRollSetpoint(latAccMps2, l1State, previewParams);
};

const navigateLoiter = (
  loiterCenter: LocalPoint,
  currentPosition: LocalPoint,
  loiterRadiusM: number,
  loiterDirection: 1 | -1,
  groundSpeed: LocalPoint,
  l1State: L1State,
  previewParams: TurnPreviewModelParams,
) => {
  const centerToCurrent = subtractLocal(currentPosition, loiterCenter);
  const centerToCurrentUnit = unitLocal(centerToCurrent);
  const targetBearingRad = Math.atan2(-centerToCurrentUnit.east, -centerToCurrentUnit.north);
  const currentGroundSpeed = normLocal(groundSpeed);

  l1State.l1DistanceM = l1State.l1Ratio * currentGroundSpeed;
  if (loiterRadiusM > 1e-6 && currentGroundSpeed > 0 && l1State.l1DistanceM / loiterRadiusM > 1) {
    l1State.l1Ratio = loiterRadiusM / currentGroundSpeed;
    l1State.l1DistanceM = l1State.l1Ratio * currentGroundSpeed;
  }

  const distanceToCircleM = normLocal(centerToCurrent) - loiterRadiusM;
  l1State.l1DistanceM = Math.max(
    Math.abs(distanceToCircleM),
    Math.min(2 * loiterRadiusM + distanceToCircleM, l1State.l1DistanceM),
  );

  let etaRad = 0;
  if (currentGroundSpeed > 1e-6) {
    let navBearingRad = targetBearingRad;
    const radialDistanceM = distanceToCircleM + loiterRadiusM;

    if (radialDistanceM * radialDistanceM - loiterRadiusM * loiterRadiusM > l1State.l1DistanceM * l1State.l1DistanceM) {
      navBearingRad = Math.asin(clamp((-loiterDirection * loiterRadiusM) / radialDistanceM, -1, 1)) + targetBearingRad;
    } else {
      const cosGamma =
        (l1State.l1DistanceM * l1State.l1DistanceM + radialDistanceM * radialDistanceM - loiterRadiusM * loiterRadiusM) /
        (2 * Math.max(l1State.l1DistanceM, 0.1) * Math.max(radialDistanceM, 0.1));
      const gammaRad = Math.acos(clamp(cosGamma, -1, 1));
      navBearingRad = Math.atan2(-centerToCurrent.east, -centerToCurrent.north) - loiterDirection * gammaRad;
    }

    const groundTrackBearingRad = Math.atan2(groundSpeed.east, groundSpeed.north);
    etaRad = wrapPi(navBearingRad - groundTrackBearingRad);
  }

  const limitedEtaRad = clamp(etaRad, -Math.PI / 2, Math.PI / 2);
  const latAccMps2 = (l1State.gain * currentGroundSpeed * Math.sin(limitedEtaRad)) / Math.max(l1State.l1Ratio, 0.1);
  return updateRollSetpoint(latAccMps2, l1State, previewParams);
};

const getAcceptanceRadius = (customRadiusM: number, controllerAcceptanceRadiusM: number) => {
  if (customRadiusM < FINETUNE_MIN_CUSTOM_ACCEPTANCE_RADIUS_TO_HONOR_M && controllerAcceptanceRadiusM > customRadiusM) {
    return controllerAcceptanceRadiusM;
  }
  return customRadiusM;
};

const getBearingLoiterExitRad = (
  loiterCenter: LocalPoint,
  nextWaypoint: LocalPoint,
  loiterRadiusM: number,
  loiterDirection: 1 | -1,
) => {
  const centerToNextBearingRad = bearingRad(loiterCenter, nextWaypoint);
  const centerToNextDistanceM = normLocal(subtractLocal(nextWaypoint, loiterCenter));
  if (centerToNextDistanceM < loiterRadiusM) return centerToNextBearingRad;
  const innerAngleRad = Math.asin(clamp(loiterRadiusM / centerToNextDistanceM, -1, 1));
  return wrapPi(centerToNextBearingRad + (loiterDirection > 0 ? innerAngleRad : -innerAngleRad));
};

const getCogErrorOffsetRad = (rollSetpointRad: number) => {
  const absoluteRollSetpointRad = Math.abs(rollSetpointRad);
  const softOffsetStartRad = (PX4_LOITER_EXIT_ROLL_SOFT_OFFSET_START_DEG * Math.PI) / 180;
  const softOffsetEndRad = (PX4_LOITER_EXIT_ROLL_SOFT_OFFSET_END_DEG * Math.PI) / 180;
  const maxOffsetRad = (PX4_LOITER_EXIT_COG_OFFSET_MAX_DEG * Math.PI) / 180;
  if (absoluteRollSetpointRad <= softOffsetStartRad) return 0;
  if (absoluteRollSetpointRad >= softOffsetEndRad) return maxOffsetRad;
  return ((absoluteRollSetpointRad - softOffsetStartRad) / (softOffsetEndRad - softOffsetStartRad)) * maxOffsetRad;
};

const estimatePreviewDurationS = (geometry: TurnBlockGeometry, previewParams: TurnPreviewModelParams) => {
  const turnOffDistanceM = normLocal(subtractLocal(geometry.turnOff, geometry.endSweep));
  const reconnectDistanceM = normLocal(subtractLocal(geometry.nextSweepStart, geometry.turnOff));
  const loiterBudgetDistanceM = FINETUNE_PREVIEW_LOITER_DISTANCE_MULTIPLIER * Math.PI * geometry.loiterRadiusM;
  const totalDistanceBudgetM = turnOffDistanceM + reconnectDistanceM + loiterBudgetDistanceM;
  const conservativeSpeedMps = Math.max(
    Math.min(previewParams.turnSpeedMps, previewParams.trimSpeedMps),
    FINETUNE_MIN_PREVIEW_SPEED_MPS,
  );
  return Math.max(
    previewParams.maxPreviewDurationS,
    totalDistanceBudgetM / conservativeSpeedMps + FINETUNE_PREVIEW_DURATION_BUFFER_S,
  );
};

const mergeSweepLines = (flightLines: LngLat[][], lineSpacing: number, sweepIndices: number[]) =>
  groupFlightLinesForTraversal(flightLines as number[][][], lineSpacing, sweepIndices).map((sweep) => {
    const orderedFragments = sweep.directionForward ? sweep.fragments : [...sweep.fragments].reverse();
    const orientedFragments = orderedFragments.map((fragment) => (sweep.directionForward ? fragment : [...fragment].reverse()));
    const mergedLine: LngLat[] = [];
    orientedFragments.forEach((fragment, fragmentIndex) => {
      const typedFragment = fragment as LngLat[];
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
    return mergedLine;
  }).filter((line) => line.length >= 2);

const normalizeSweepLineDirectionsForGrid = (sweepLines: LngLat[][]) => {
  if (sweepLines.length === 0) return [];

  const referenceCoordinate = sweepLines[0][0];
  const referenceDirection = unitLocal(
    subtractLocal(
      toLocalPoint(sweepLines[0][sweepLines[0].length - 1], referenceCoordinate),
      toLocalPoint(sweepLines[0][0], referenceCoordinate),
    ),
  );

  return sweepLines.map((line) => {
    if (line.length < 2) return line;
    const lineDirection = unitLocal(
      subtractLocal(
        toLocalPoint(line[line.length - 1], referenceCoordinate),
        toLocalPoint(line[0], referenceCoordinate),
      ),
    );
    return dotLocal(referenceDirection, lineDirection) < 0 ? [...line].reverse() : line;
  });
};

const getOffsetForGridLine = (line: LocalSegment) => unitLocal(subtractLocal(line.end, line.start));

const getTurnaroundLateralOffsetForGridLine = (line: LocalSegment, nextLinePointEnd: LocalPoint, isLastLine: boolean) => {
  let lateralOffset: LocalPoint = {
    north: line.end.east - line.start.east,
    east: line.start.north - line.end.north,
  };
  const space = subtractLocal(nextLinePointEnd, line.end);
  const offset = dotLocal(space, lateralOffset);
  if ((offset > 0) !== isLastLine) lateralOffset = negateLocal(lateralOffset);
  if (isLastLine) lateralOffset = negateLocal(lateralOffset);
  return unitLocal(lateralOffset);
};

const buildSurveyGridPoints = (
  sweepLines: LngLat[][],
  lineSpacing: number,
  defaultTurnaroundRadiusM: number,
) => {
  if (sweepLines.length === 0) {
    return { gridPoints: [] as LngLat[], leadInPoints: [] as LngLat[], leadOutPoints: [] as LngLat[] };
  }

  const referenceCoordinate = sweepLines[0][0];
  const sweepSegments = sweepLines.map((line) => ({
    start: toLocalPoint(line[0], referenceCoordinate),
    end: toLocalPoint(line[line.length - 1], referenceCoordinate),
  }));

  const gridPointsLocal: LocalPoint[] = [];
  const leadInPointsLocal: LocalPoint[] = [];
  const leadOutPointsLocal: LocalPoint[] = [];
  let leadInCompRoot = sweepSegments[0].start;

  for (let index = 0; index < sweepSegments.length; index += 1) {
    const line = sweepSegments[index];
    const nextLine = index < sweepSegments.length - 1 ? sweepSegments[index + 1] : sweepSegments[index];
    const turnaroundSideOffset = Math.max(MISSION_DEFAULT_TURNAROUND_SIDE_OFFSET_M - lineSpacing, 0);
    const turnaroundOffset = getOffsetForGridLine(line);
    const turnaroundLateralOffset = getTurnaroundLateralOffsetForGridLine(line, nextLine.end, index === sweepSegments.length - 1);
    const compensationP1Out = dotLocal(nextLine.start, turnaroundOffset) - dotLocal(line.start, turnaroundOffset);
    const compensationP2Out = dotLocal(nextLine.end, turnaroundOffset) - dotLocal(line.end, turnaroundOffset);
    const compensationP1In = Math.max(dotLocal(turnaroundOffset, subtractLocal(line.start, leadInCompRoot)), 0);
    const compensationP2In = Math.max(dotLocal(turnaroundOffset, subtractLocal(leadInCompRoot, line.end)), 0);

    if (index % 2 === 1) {
      gridPointsLocal.push(addOffsetToLocalPoint(line.end, turnaroundOffset, MISSION_DEFAULT_TURNAROUND_DISTANCE_M + compensationP2In));
      gridPointsLocal.push(addOffsetToLocalPoint(line.end, turnaroundOffset, compensationP2In));
      gridPointsLocal.push(addOffsetToLocalPoint(line.start, turnaroundOffset, Math.min(compensationP1Out, 0)));
      if (index !== sweepSegments.length - 1) {
        gridPointsLocal.push(
          addOffsetToLocalPoint(
            addOffsetToLocalPoint(
              line.start,
              negateLocal(turnaroundOffset),
              MISSION_DEFAULT_TURNAROUND_DISTANCE_M - Math.min(compensationP1Out, 0),
            ),
            turnaroundLateralOffset,
            turnaroundSideOffset,
          ),
        );
      } else {
        gridPointsLocal.push(
          addOffsetToLocalPoint(
            line.start,
            negateLocal(turnaroundOffset),
            MISSION_DEFAULT_TURNAROUND_DISTANCE_M - Math.min(compensationP1Out, 0),
          ),
        );
      }
      leadInCompRoot = line.start;
    } else {
      gridPointsLocal.push(addOffsetToLocalPoint(line.start, negateLocal(turnaroundOffset), MISSION_DEFAULT_TURNAROUND_DISTANCE_M + compensationP1In));
      gridPointsLocal.push(addOffsetToLocalPoint(line.start, negateLocal(turnaroundOffset), compensationP1In));
      gridPointsLocal.push(addOffsetToLocalPoint(line.end, turnaroundOffset, Math.max(compensationP2Out, 0)));
      if (index !== sweepSegments.length - 1) {
        gridPointsLocal.push(
          addOffsetToLocalPoint(
            addOffsetToLocalPoint(
              line.end,
              turnaroundOffset,
              MISSION_DEFAULT_TURNAROUND_DISTANCE_M + Math.max(compensationP2Out, 0),
            ),
            turnaroundLateralOffset,
            turnaroundSideOffset,
          ),
        );
      } else {
        gridPointsLocal.push(addOffsetToLocalPoint(line.end, turnaroundOffset, MISSION_DEFAULT_TURNAROUND_DISTANCE_M + Math.max(compensationP2Out, 0)));
      }
      leadInCompRoot = line.end;
    }

    if (index === 0) {
      const firstGridPoint = gridPointsLocal[0];
      const leadInPoint1 = addOffsetToLocalPoint(firstGridPoint, negateLocal(turnaroundLateralOffset), defaultTurnaroundRadiusM * 2);
      leadInPointsLocal.unshift(leadInPoint1);
      const leadInPoint2 = addOffsetToLocalPoint(leadInPoint1, turnaroundOffset, defaultTurnaroundRadiusM * 2);
      leadInPointsLocal.unshift(leadInPoint2);
      const leadInPoint3 = addOffsetToLocalPoint(leadInPoint2, turnaroundLateralOffset, defaultTurnaroundRadiusM * 2);
      leadInPointsLocal.unshift(leadInPoint3);
      const leadInCenterPoint = addOffsetToLocalPoint(
        addOffsetToLocalPoint(firstGridPoint, negateLocal(turnaroundLateralOffset), defaultTurnaroundRadiusM),
        turnaroundOffset,
        defaultTurnaroundRadiusM,
      );
      leadInPointsLocal.unshift(leadInCenterPoint);
    }
  }

  const lastIndex = sweepSegments.length - 1;
  const lastLine = sweepSegments[lastIndex];
  const previousLine = sweepSegments[Math.max(0, lastIndex - 1)];
  let leadoutOffset = getOffsetForGridLine(lastLine);
  if (sweepSegments.length % 2 === 1) leadoutOffset = negateLocal(leadoutOffset);
  let leadoutLateralOffset: LocalPoint = {
    north: lastLine.end.east - lastLine.start.east,
    east: lastLine.start.north - lastLine.end.north,
  };
  const space = subtractLocal(previousLine.end, lastLine.end);
  if (dotLocal(space, leadoutLateralOffset) < 0) leadoutLateralOffset = negateLocal(leadoutLateralOffset);
  leadoutLateralOffset = unitLocal(leadoutLateralOffset);

  const lastGridPoint = gridPointsLocal.at(-1)!;
  const leadOutPoint1 = addOffsetToLocalPoint(lastGridPoint, leadoutLateralOffset, defaultTurnaroundRadiusM * 2);
  leadOutPointsLocal.push(leadOutPoint1);
  const leadOutPoint2 = addOffsetToLocalPoint(leadOutPoint1, leadoutOffset, defaultTurnaroundRadiusM * 2);
  leadOutPointsLocal.push(leadOutPoint2);
  const leadOutPoint3 = addOffsetToLocalPoint(leadOutPoint2, negateLocal(leadoutLateralOffset), defaultTurnaroundRadiusM * 2);
  leadOutPointsLocal.push(leadOutPoint3);
  const leadOutCenterPoint = addOffsetToLocalPoint(
    addOffsetToLocalPoint(lastGridPoint, leadoutLateralOffset, defaultTurnaroundRadiusM),
    leadoutOffset,
    defaultTurnaroundRadiusM,
  );
  leadOutPointsLocal.push(leadOutCenterPoint);

  return {
    gridPoints: gridPointsLocal.map((point) => fromLocalPoint(point, referenceCoordinate)),
    leadInPoints: leadInPointsLocal.map((point) => fromLocalPoint(point, referenceCoordinate)),
    leadOutPoints: leadOutPointsLocal.map((point) => fromLocalPoint(point, referenceCoordinate)),
  };
};

const buildTurnBlockGeometry = (
  gridPoints: LngLat[],
  sweepIndex: number,
  currentSweepLine: LngLat[],
  nextSweepLine: LngLat[],
  referenceCoordinate: LngLat,
  gridSpacing: number,
  defaultTurnaroundRadius: number,
  heightAboveGroundM: number,
  obliqueTiltFlightPlanDeg: number,
): TurnBlockGeometry | undefined => {
  const startSweepIndex = sweepIndex * 4 + 1;
  const endSweepIndex = startSweepIndex + 1;
  const turnOffIndex = startSweepIndex + 2;
  const turnCoordIndex = startSweepIndex + 3;
  const nextSweepStartIndex = startSweepIndex + 4;
  const nextSweepEndIndex = nextSweepStartIndex + 1;
  if (nextSweepEndIndex >= gridPoints.length) return undefined;

  const startSweepCoordinate = currentSweepLine[0] ?? gridPoints[startSweepIndex];
  const endSweepCoordinate = currentSweepLine.at(-1) ?? gridPoints[endSweepIndex];
  const nextSweepStartCoordinate = nextSweepLine[0] ?? gridPoints[nextSweepStartIndex];
  const nextSweepEndCoordinate = nextSweepLine[1] ?? nextSweepLine[0] ?? gridPoints[nextSweepEndIndex];

  const startSweep = toLocalPoint(startSweepCoordinate, referenceCoordinate);
  const endSweep = toLocalPoint(endSweepCoordinate, referenceCoordinate);
  const turnOff = toLocalPoint(gridPoints[turnOffIndex], referenceCoordinate);
  const turnCoord = toLocalPoint(gridPoints[turnCoordIndex], referenceCoordinate);
  const nextSweepStart = toLocalPoint(nextSweepStartCoordinate, referenceCoordinate);
  const nextSweepEnd = toLocalPoint(nextSweepEndCoordinate, referenceCoordinate);
  const loiterRadiusM = getMissionTurnaroundRadius(gridSpacing, defaultTurnaroundRadius);
  let loiterCenter = advanceLocalPoint(turnCoord, loiterRadiusM, bearingRad(turnCoord, turnOff));
  loiterCenter = advanceLocalPoint(loiterCenter, loiterRadiusM, bearingRad(turnCoord, nextSweepStart));
  const bearingTurnToNextDeg = normaliseDegrees((Math.atan2(nextSweepStart.east - turnCoord.east, nextSweepStart.north - turnCoord.north) * 180) / Math.PI);
  const bearingOffToTurnDeg = normaliseDegrees((Math.atan2(turnCoord.east - turnOff.east, turnCoord.north - turnOff.north) * 180) / Math.PI);
  const directionClockwise = (bearingTurnToNextDeg - bearingOffToTurnDeg + 540) % 360 > 180;

  return {
    startSweep,
    endSweep,
    turnOff,
    loiterCenter,
    nextSweepStart,
    nextSweepEnd,
    loiterRadiusM,
    loiterDirection: directionClockwise ? 1 : -1,
    turnOffAcceptanceRadiusM: getTurnOffAcceptanceRadius(obliqueTiltFlightPlanDeg, heightAboveGroundM),
  };
};

const sampleLoiterArcBearings = (
  startBearingRad: number,
  endBearingRad: number,
  loiterDirection: 1 | -1,
  segmentCount: number,
) => {
  const normalizedStartRad = wrapPi(startBearingRad);
  let deltaRad = wrapPi(endBearingRad - normalizedStartRad);
  if (loiterDirection > 0 && deltaRad < 0) deltaRad += Math.PI * 2;
  if (loiterDirection < 0 && deltaRad > 0) deltaRad -= Math.PI * 2;

  const bearingsRad: number[] = [];
  for (let index = 0; index <= segmentCount; index += 1) {
    const t = index / segmentCount;
    bearingsRad.push(normalizedStartRad + deltaRad * t);
  }
  return bearingsRad;
};

const buildMissionTurnFallbackLine = (
  geometry: TurnBlockGeometry,
  referenceCoordinate: LngLat,
) => {
  const entryBearingRad = bearingRad(geometry.loiterCenter, geometry.turnOff);
  const entryPoint = advanceLocalPoint(geometry.loiterCenter, geometry.loiterRadiusM, entryBearingRad);
  const idealCogRad = getBearingLoiterExitRad(
    geometry.loiterCenter,
    geometry.nextSweepStart,
    geometry.loiterRadiusM,
    geometry.loiterDirection,
  );
  const exitBearingRad = idealCogRad + (geometry.loiterDirection > 0 ? -Math.PI / 2 : Math.PI / 2);
  const exitPoint = advanceLocalPoint(geometry.loiterCenter, geometry.loiterRadiusM, exitBearingRad);
  const approximateArcLengthM =
    Math.abs(wrapPi(exitBearingRad - entryBearingRad)) * geometry.loiterRadiusM;
  const segmentCount = Math.max(16, Math.ceil(approximateArcLengthM / 4));
  const arcCoordinates = sampleLoiterArcBearings(entryBearingRad, exitBearingRad, geometry.loiterDirection, segmentCount)
    .map((bearingAngleRad) =>
      fromLocalPoint(advanceLocalPoint(geometry.loiterCenter, geometry.loiterRadiusM, bearingAngleRad), referenceCoordinate),
    );

  const turnaroundCoordinates = dedupeCoordinates([
    fromLocalPoint(geometry.endSweep, referenceCoordinate),
    fromLocalPoint(geometry.turnOff, referenceCoordinate),
    fromLocalPoint(entryPoint, referenceCoordinate),
    ...arcCoordinates.slice(1, -1),
    fromLocalPoint(exitPoint, referenceCoordinate),
    fromLocalPoint(geometry.nextSweepStart, referenceCoordinate),
  ]);

  return {
    turnaroundLine: turnaroundCoordinates.length >= 2 ? turnaroundCoordinates : undefined,
    loiterEntryPoint: entryPoint,
    loiterExitPoint: exitPoint,
  };
};

const toPlannedTurnBlock = (
  turnBlock: TurnBlockGeometry,
  referenceCoordinate: LngLat,
  previewMetadata?: { loiterEntryPoint?: LocalPoint; loiterExitPoint?: LocalPoint },
): PlannedTurnBlock => ({
  startSweep: fromLocalPoint(turnBlock.startSweep, referenceCoordinate),
  endSweep: fromLocalPoint(turnBlock.endSweep, referenceCoordinate),
  turnOff: fromLocalPoint(turnBlock.turnOff, referenceCoordinate),
  loiterCenter: fromLocalPoint(turnBlock.loiterCenter, referenceCoordinate),
  nextSweepStart: fromLocalPoint(turnBlock.nextSweepStart, referenceCoordinate),
  loiterRadiusM: turnBlock.loiterRadiusM,
  loiterDirection: turnBlock.loiterDirection,
  turnOffAcceptanceRadiusM: turnBlock.turnOffAcceptanceRadiusM,
  loiterEntryPoint: previewMetadata?.loiterEntryPoint
    ? fromLocalPoint(previewMetadata.loiterEntryPoint, referenceCoordinate)
    : undefined,
  loiterExitPoint: previewMetadata?.loiterExitPoint
    ? fromLocalPoint(previewMetadata.loiterExitPoint, referenceCoordinate)
    : undefined,
});

const runTurnPreviewBlock = (
  geometry: TurnBlockGeometry,
  referenceCoordinate: LngLat,
  previewParams: TurnPreviewModelParams,
) => {
  const sweepDirection = unitLocal(subtractLocal(geometry.endSweep, geometry.startSweep));
  if (normLocal(sweepDirection) <= 1e-9) return undefined;
  const sweepDistanceM = normLocal(subtractLocal(geometry.endSweep, geometry.startSweep));
  const previewRunInDistanceM = Math.min(
    sweepDistanceM,
    Math.max(geometry.loiterRadiusM * 4, previewParams.slowdownDistanceM * 2),
  );
  const nextSweepDirection = unitLocal(subtractLocal(geometry.nextSweepEnd, geometry.nextSweepStart));
  const hasNextSweepDirection = normLocal(nextSweepDirection) > 1e-9;
  const nextSweepApproachDistanceM = Math.min(12, geometry.loiterRadiusM * 0.3);
  const nextSweepApproachStart = hasNextSweepDirection
    ? addOffsetToLocalPoint(
        geometry.nextSweepStart,
        negateLocal(nextSweepDirection),
        nextSweepApproachDistanceM,
      )
    : geometry.nextSweepStart;
  const previewStart = addOffsetToLocalPoint(geometry.endSweep, negateLocal(sweepDirection), previewRunInDistanceM);
  const cacheKey = getTurnPreviewCacheKey(geometry, previewStart, previewParams, sweepDirection);
  if (cacheKey) {
    const cachedPreview = turnPreviewCache.get(cacheKey);
    if (cachedPreview) {
      const restored = restoreTurnPreviewFromCache(cachedPreview, geometry, sweepDirection);
      const turnaroundCoordinates = dedupeCoordinates(
        restored.turnaroundPathLocal.map((point) => fromLocalPoint(point, referenceCoordinate)),
      );
      if (turnaroundCoordinates.length >= 2) {
        return {
          turnaroundLine: turnaroundCoordinates,
          loiterEntryPoint: restored.loiterEntryPoint,
          loiterExitPoint: restored.loiterExitPoint,
        };
      }
    }
  }
  let currentPosition = previewStart;
  let headingRad = Math.atan2(sweepDirection.east, sweepDirection.north);
  let speedMps = previewParams.trimSpeedMps;
  let rollRad = 0;
  let rollRateRadS = 0;
  let lastLoiterAngleRad: number | undefined;
  let loiterLoops = 0;
  let stage: "end" | "turnoff" | "loiter" | "next" = "end";
  let turnStartIndex = -1;
  let reachedNextSweep = false;
  let loiterEntryPoint: LocalPoint | undefined;
  let loiterExitPoint: LocalPoint | undefined;
  const l1State = createInitialL1State(previewParams);
  const trajectory: LocalPoint[] = [currentPosition];
  const maxSteps = Math.ceil(
    (estimatePreviewDurationS(geometry, previewParams) +
      previewRunInDistanceM / Math.max(previewParams.trimSpeedMps, FINETUNE_MIN_PREVIEW_SPEED_MPS)) /
      previewParams.dtS,
  );

  for (let step = 0; step < maxSteps; step += 1) {
    const groundSpeed = { north: speedMps * Math.cos(headingRad), east: speedMps * Math.sin(headingRad) };
    let rollSetpointRad = 0;

    if (stage === "end") {
      ({ rollSpRad: rollSetpointRad } = navigateWaypoints(previewStart, geometry.endSweep, currentPosition, groundSpeed, l1State, previewParams));
      if (normLocal(subtractLocal(currentPosition, geometry.endSweep)) <= getAcceptanceRadius(0, l1State.l1DistanceM)) {
        currentPosition = geometry.endSweep;
        trajectory[trajectory.length - 1] = geometry.endSweep;
        stage = "turnoff";
        turnStartIndex = trajectory.length - 1;
      }
    } else if (stage === "turnoff") {
      ({ rollSpRad: rollSetpointRad } = navigateWaypoints(geometry.endSweep, geometry.turnOff, currentPosition, groundSpeed, l1State, previewParams));
      if (normLocal(subtractLocal(currentPosition, geometry.turnOff)) <= getAcceptanceRadius(geometry.turnOffAcceptanceRadiusM, l1State.l1DistanceM)) {
        stage = "loiter";
        loiterEntryPoint = currentPosition;
      }
    } else if (stage === "loiter") {
      ({ rollSpRad: rollSetpointRad } = navigateLoiter(geometry.loiterCenter, currentPosition, geometry.loiterRadiusM, geometry.loiterDirection, groundSpeed, l1State, previewParams));
      const idealCogRad = getBearingLoiterExitRad(geometry.loiterCenter, geometry.nextSweepStart, geometry.loiterRadiusM, geometry.loiterDirection);
      const cogErrorRad = wrapPi(idealCogRad - headingRad);
      const cogErrorOffsetRad = getCogErrorOffsetRad(rollSetpointRad);
      const thresholdMultiplier = clamp(1 + loiterLoops, 1, PX4_LOITER_EXIT_THRESHOLD_MULTIPLIER_MAX);
      const shouldEvaluateExit =
        Math.abs(cogErrorRad) > cogErrorOffsetRad &&
        Math.abs(cogErrorRad) < (PX4_MIS_YAW_ERR_DEG * thresholdMultiplier * Math.PI) / 180 + cogErrorOffsetRad &&
        (cogErrorRad < 0) !== (geometry.loiterDirection > 0) &&
        (rollSetpointRad < 0) !== (geometry.loiterDirection > 0);

      if (shouldEvaluateExit) {
        const exitBearingRad = idealCogRad + (geometry.loiterDirection > 0 ? -Math.PI / 2 : Math.PI / 2);
        const exitPoint = advanceLocalPoint(geometry.loiterCenter, geometry.loiterRadiusM, exitBearingRad);
        const distanceExitToNextM = normLocal(subtractLocal(geometry.nextSweepStart, exitPoint));
        const distanceCurrentToNextM = normLocal(subtractLocal(geometry.nextSweepStart, currentPosition));
        const distanceCurrentToLoiterM = normLocal(subtractLocal(currentPosition, geometry.loiterCenter));
        const distanceCurrentToExitM = normLocal(subtractLocal(currentPosition, exitPoint));
        if (distanceCurrentToNextM > distanceExitToNextM && distanceCurrentToExitM < distanceCurrentToLoiterM) {
          stage = "next";
          loiterExitPoint = exitPoint;
        }
      }

      if (stage === "loiter") {
        const currentLoiterAngleRad = Math.atan2(currentPosition.east - geometry.loiterCenter.east, currentPosition.north - geometry.loiterCenter.north);
        if (lastLoiterAngleRad !== undefined) {
          const loiterDelta = wrapPi(currentLoiterAngleRad - lastLoiterAngleRad);
          if (geometry.loiterDirection > 0 && loiterDelta < -FINETUNE_LOITER_LOOP_DELTA_THRESHOLD_RAD) loiterLoops += 1;
          if (geometry.loiterDirection < 0 && loiterDelta > FINETUNE_LOITER_LOOP_DELTA_THRESHOLD_RAD) loiterLoops += 1;
        }
        lastLoiterAngleRad = currentLoiterAngleRad;
      }
    } else {
      ({ rollSpRad: rollSetpointRad } = navigateWaypoints(
        hasNextSweepDirection ? nextSweepApproachStart : geometry.nextSweepStart,
        hasNextSweepDirection ? geometry.nextSweepEnd : geometry.nextSweepStart,
        currentPosition,
        groundSpeed,
        l1State,
        previewParams,
      ));
    }

    let targetSpeedMps = previewParams.trimSpeedMps;
    if (stage === "turnoff" || stage === "loiter") {
      const slowdownTarget = stage === "turnoff" ? geometry.turnOff : geometry.loiterCenter;
      if (normLocal(subtractLocal(currentPosition, slowdownTarget)) < previewParams.slowdownDistanceM) {
        targetSpeedMps = previewParams.turnSpeedMps;
      }
    }

    speedMps += (targetSpeedMps - speedMps) * Math.min(1, previewParams.dtS / previewParams.speedTimeConstantS);
    const rollAccelerationRadS2 =
      previewParams.rollNaturalFrequencyRadS * previewParams.rollNaturalFrequencyRadS * (rollSetpointRad - rollRad) -
      2 * previewParams.rollDampingRatio * previewParams.rollNaturalFrequencyRadS * rollRateRadS;
    rollRateRadS += rollAccelerationRadS2 * previewParams.dtS;
    const maxRollRateRadS = (previewParams.maxRollRateDegS * Math.PI) / 180;
    rollRateRadS = clamp(rollRateRadS, -maxRollRateRadS, maxRollRateRadS);
    rollRad += rollRateRadS * previewParams.dtS;

      const yawRateRadS =
      (previewParams.yawRateEfficiency * PHYSICS_GRAVITY_M_S2 * Math.tan(rollRad)) /
      Math.max(speedMps, FINETUNE_MIN_YAW_RATE_SPEED_MPS);
    headingRad = wrapPi(headingRad + yawRateRadS * previewParams.dtS);
    currentPosition = addLocal(currentPosition, {
      north: speedMps * Math.cos(headingRad) * previewParams.dtS,
      east: speedMps * Math.sin(headingRad) * previewParams.dtS,
    });
    trajectory.push(currentPosition);

    if (stage === "next") {
      const distanceToNextSweepStartM = normLocal(subtractLocal(currentPosition, geometry.nextSweepStart));
      const alignedWithNextSweep =
        hasNextSweepDirection &&
        dotLocal(subtractLocal(currentPosition, geometry.nextSweepStart), nextSweepDirection) >= -FINETUNE_NEXT_SWEEP_REACHED_DISTANCE_M &&
        Math.abs(crossLocal(subtractLocal(currentPosition, geometry.nextSweepStart), nextSweepDirection)) <=
          FINETUNE_NEXT_SWEEP_ALIGNMENT_DISTANCE_M;
      if (distanceToNextSweepStartM < FINETUNE_NEXT_SWEEP_REACHED_DISTANCE_M || alignedWithNextSweep) {
        trajectory[trajectory.length - 1] = geometry.nextSweepStart;
        reachedNextSweep = true;
        break;
      }
    }
  }

  if (turnStartIndex < 0 || !reachedNextSweep) return undefined;
  let turnaroundPathLocal = simplifyPreviewPathLocal(
    trajectory.slice(Math.max(0, turnStartIndex)),
    FINETUNE_PREVIEW_POINT_SPACING_M,
  );
  if (turnaroundPathLocal.length >= 2 && normLocal(nextSweepDirection) > 0) {
    turnaroundPathLocal = smoothTurnaroundTailIntoSweep(
      turnaroundPathLocal,
      geometry.nextSweepStart,
      nextSweepDirection,
      Math.min(12, geometry.loiterRadiusM * 0.3),
    );
  }
  const turnaroundCoordinates = dedupeCoordinates(
    turnaroundPathLocal.map((point) => fromLocalPoint(point, referenceCoordinate)),
  );
  if (turnaroundCoordinates.length < 2) return undefined;
  if (cacheKey) {
    cacheTurnPreview(
      cacheKey,
      geometry,
      sweepDirection,
      turnaroundPathLocal,
      loiterEntryPoint,
      loiterExitPoint,
    );
  }
  return {
    turnaroundLine: turnaroundCoordinates,
    loiterEntryPoint,
    loiterExitPoint,
  };
};

const connectSweepLinesFallback = (sweepLines: LngLat[][]) => {
  const connectedLines: LngLat[][] = [];
  sweepLines.forEach((sweepLine, index) => {
    if (index > 0) {
      connectedLines.push([connectedLines.at(-1)!.at(-1)!, sweepLine[0]]);
    }
    connectedLines.push(sweepLine);
  });
  return connectedLines;
};

const stitchLineStart = (line: LngLat[], targetStart: LngLat) => {
  if (line.length === 0) return line;
  if (line[0][0] === targetStart[0] && line[0][1] === targetStart[1]) return line;

  const projectedStart = projectPointOntoPolyline(targetStart, line);
  if (!projectedStart || projectedStart.distanceM > FINETUNE_NEXT_SWEEP_ALIGNMENT_DISTANCE_M) {
    return line;
  }

  const projectedStartShiftM = haversineDistance(line[0], projectedStart.coordinate);
  if (projectedStartShiftM > FINETUNE_NEXT_SWEEP_ALIGNMENT_DISTANCE_M) {
    return line;
  }

  const stitchedLine = [projectedStart.coordinate, ...line.slice(projectedStart.segmentIndex + 1)];
  if (stitchedLine.length < 2) {
    return [projectedStart.coordinate, line.at(-1)!];
  }
  return dedupeCoordinates(stitchedLine);
};

const buildSmoothJoinToLineStart = (
  previousLine: LngLat[],
  nextLine: LngLat[],
) => {
  if (previousLine.length < 2 || nextLine.length < 2) return [];

  const startPoint = previousLine.at(-1)!;
  const previousPoint = previousLine.at(-2)!;
  const endPoint = nextLine[0];
  const nextPoint = nextLine[1];
  if (startPoint[0] === endPoint[0] && startPoint[1] === endPoint[1]) return [];

  const referenceCoordinate = startPoint;
  const startLocal = toLocalPoint(startPoint, referenceCoordinate);
  const previousLocal = toLocalPoint(previousPoint, referenceCoordinate);
  const endLocal = toLocalPoint(endPoint, referenceCoordinate);
  const nextLocal = toLocalPoint(nextPoint, referenceCoordinate);
  const joinDistanceM = normLocal(subtractLocal(endLocal, startLocal));
  if (joinDistanceM < 1e-6) return [];

  const tangentScaleM = joinDistanceM * 0.5;
  const startTangent = multiplyLocal(unitLocal(subtractLocal(startLocal, previousLocal)), tangentScaleM);
  const endTangent = multiplyLocal(unitLocal(subtractLocal(nextLocal, endLocal)), tangentScaleM);
  const controlPointCount = Math.max(12, Math.ceil(joinDistanceM / FINETUNE_SMOOTH_JOIN_POINT_SPACING_M));
  const joinPoints: LngLat[] = [];

  for (let pointIndex = 1; pointIndex <= controlPointCount; pointIndex += 1) {
    const t = pointIndex / controlPointCount;
    const t2 = t * t;
    const t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1;
    const h10 = t3 - 2 * t2 + t;
    const h01 = -2 * t3 + 3 * t2;
    const h11 = t3 - t2;
    const pointLocal = addLocal(
      addLocal(multiplyLocal(startLocal, h00), multiplyLocal(startTangent, h10)),
      addLocal(multiplyLocal(endLocal, h01), multiplyLocal(endTangent, h11)),
    );
    joinPoints.push(fromLocalPoint(pointLocal, referenceCoordinate));
  }

  return dedupeCoordinates(joinPoints);
};

const blendTurnaroundEndIntoSweep = (
  turnaroundLine: LngLat[],
  nextSweepLine: LngLat[],
) => {
  if (turnaroundLine.length < 3 || nextSweepLine.length < 2) return turnaroundLine;

  const referenceCoordinate = nextSweepLine[0];
  const nextSweepStartLocal = toLocalPoint(nextSweepLine[0], referenceCoordinate);
  const nextSweepDirection = unitLocal(
    subtractLocal(toLocalPoint(nextSweepLine[1], referenceCoordinate), nextSweepStartLocal),
  );
  if (normLocal(nextSweepDirection) <= 1e-9) return turnaroundLine;

  const penultimatePoint = turnaroundLine.at(-2);
  if (!penultimatePoint) return turnaroundLine;
  const penultimateFrame = toTurnFramePoint(
    toLocalPoint(penultimatePoint, referenceCoordinate),
    nextSweepStartLocal,
    nextSweepDirection,
  );
  const desiredBlendDistanceM = clamp(
    Math.max(8, Math.abs(penultimateFrame.cross) * 4),
    8,
    18,
  );

  let anchorIndex = turnaroundLine.length - 2;
  for (let pointIndex = turnaroundLine.length - 2; pointIndex >= 1; pointIndex -= 1) {
    anchorIndex = pointIndex;
    const framePoint = toTurnFramePoint(
      toLocalPoint(turnaroundLine[pointIndex], referenceCoordinate),
      nextSweepStartLocal,
      nextSweepDirection,
    );
    if (framePoint.along <= -desiredBlendDistanceM) {
      break;
    }
  }

  const turnaroundPrefix = turnaroundLine.slice(0, anchorIndex + 1);
  const smoothTail = buildSmoothJoinToLineStart(turnaroundPrefix, nextSweepLine);
  if (smoothTail.length === 0) return turnaroundLine;
  return dedupeCoordinates([...turnaroundPrefix, ...smoothTail]);
};

const smoothTurnaroundTailIntoSweep = (
  turnaroundPathLocal: LocalPoint[],
  nextSweepStart: LocalPoint,
  nextSweepDirection: LocalPoint,
  preferredApproachDistanceM: number,
) => {
  if (turnaroundPathLocal.length < 2) return turnaroundPathLocal;

  const sweepDirection = unitLocal(nextSweepDirection);
  if (normLocal(sweepDirection) <= 1e-9) return turnaroundPathLocal;

  const finalPoint = turnaroundPathLocal.at(-1)!;
  if (normLocal(subtractLocal(finalPoint, nextSweepStart)) > 1e-6) return turnaroundPathLocal;

  const previousPoint = turnaroundPathLocal.at(-2)!;
  const previousFramePoint = toTurnFramePoint(previousPoint, nextSweepStart, sweepDirection);
  if (previousFramePoint.along >= -FINETUNE_NEXT_SWEEP_APPROACH_MIN_DISTANCE_M) {
    return turnaroundPathLocal;
  }

  const minAdvanceM = clamp(
    Math.abs(previousFramePoint.along) * 0.35,
    FINETUNE_NEXT_SWEEP_APPROACH_MIN_ADVANCE_M,
    FINETUNE_NEXT_SWEEP_APPROACH_MAX_ADVANCE_M,
  );
  const minApproachAlong = previousFramePoint.along + minAdvanceM;
  const maxApproachAlong = -FINETUNE_NEXT_SWEEP_APPROACH_MIN_DISTANCE_M;
  if (minApproachAlong >= maxApproachAlong) {
    return turnaroundPathLocal;
  }

  const desiredApproachAlong = -Math.max(
    preferredApproachDistanceM,
    FINETUNE_NEXT_SWEEP_APPROACH_MIN_DISTANCE_M,
  );
  const approachAlong = clamp(
    desiredApproachAlong,
    minApproachAlong,
    maxApproachAlong,
  );
  if (approachAlong <= previousFramePoint.along + 1e-6) {
    return turnaroundPathLocal;
  }

  const alongRatio = clamp(approachAlong / previousFramePoint.along, 0, 1);
  const approachPoint = fromTurnFramePoint(
    {
      along: approachAlong,
      cross: previousFramePoint.cross * alongRatio * alongRatio * alongRatio,
    },
    nextSweepStart,
    sweepDirection,
  );
  if (normLocal(subtractLocal(approachPoint, previousPoint)) <= 0.5) {
    return turnaroundPathLocal;
  }

  return [...turnaroundPathLocal.slice(0, -1), approachPoint, nextSweepStart];
};

const connectSweepLinesUsingDynamicFlightTurn = (
  sweepLines: LngLat[][],
  gridPoints: LngLat[],
  lineSpacing: number,
  flightTurnModel: FlightTurnModel,
  heightAboveGroundM: number,
) => {
  if (sweepLines.length <= 1) {
    return {
      connectedLines: sweepLines,
      turnaroundRadiusM: getMissionTurnaroundRadius(lineSpacing, flightTurnModel.defaultTurnaroundRadiusM),
      turnBlocks: [] as PlannedTurnBlock[],
    };
  }

  const referenceCoordinate = sweepLines[0][0];
  if (!referenceCoordinate || gridPoints.length < 6) {
    return {
      connectedLines: connectSweepLinesFallback(sweepLines),
      turnaroundRadiusM: getMissionTurnaroundRadius(lineSpacing, flightTurnModel.defaultTurnaroundRadiusM),
      turnBlocks: [] as PlannedTurnBlock[],
    };
  }

  const previewParams = createTurnPreviewModelParams(
    flightTurnModel.defaultTurnaroundRadiusM,
    flightTurnModel.trimSpeedMps,
  );
  const connectedLines: LngLat[][] = [];
  const missionTurnaroundRadiusM = getMissionTurnaroundRadius(lineSpacing, flightTurnModel.defaultTurnaroundRadiusM);
  const turnBlocks: PlannedTurnBlock[] = [];

  for (let sweepIndex = 0; sweepIndex < sweepLines.length - 1; sweepIndex += 1) {
    const turnBlockGeometry = buildTurnBlockGeometry(
      gridPoints,
      sweepIndex,
      sweepLines[sweepIndex]!,
      sweepLines[sweepIndex + 1]!,
      referenceCoordinate,
      lineSpacing,
      flightTurnModel.defaultTurnaroundRadiusM,
      heightAboveGroundM,
      flightTurnModel.obliqueTiltFlightPlanDeg,
    );
    if (!turnBlockGeometry) {
      const currentSweepLine =
        connectedLines.length > 0 ? stitchLineStart(sweepLines[sweepIndex], connectedLines.at(-1)!.at(-1)!) : sweepLines[sweepIndex];
      const nextSweepStart = sweepLines[sweepIndex + 1]?.[0];
      connectedLines.push(currentSweepLine);
      if (nextSweepStart) {
        connectedLines.push([currentSweepLine.at(-1)!, nextSweepStart]);
      }
      continue;
    }

    const previewLines = runTurnPreviewBlock(turnBlockGeometry, referenceCoordinate, previewParams);
    const sweepLineForTurn = sweepLines[sweepIndex];
    const stitchedSweepLine =
      connectedLines.length > 0 ? stitchLineStart(sweepLineForTurn, connectedLines.at(-1)!.at(-1)!) : sweepLineForTurn;
    if (connectedLines.length > 0) {
      const previousTurnaroundLine = connectedLines.at(-1)!;
      const smoothJoin = buildSmoothJoinToLineStart(previousTurnaroundLine, stitchedSweepLine);
      if (smoothJoin.length > 0) previousTurnaroundLine.push(...smoothJoin);
    }

    let turnaroundLine = previewLines?.turnaroundLine;
    let loiterEntryPoint = previewLines?.loiterEntryPoint;
    let loiterExitPoint = previewLines?.loiterExitPoint;

    if (!turnaroundLine) {
      const fallbackTurn = buildMissionTurnFallbackLine(turnBlockGeometry, referenceCoordinate);
      turnaroundLine = fallbackTurn.turnaroundLine;
      loiterEntryPoint = fallbackTurn.loiterEntryPoint;
      loiterExitPoint = fallbackTurn.loiterExitPoint;
    }

    if (turnaroundLine && turnaroundLine.length > 0) {
      turnaroundLine = blendTurnaroundEndIntoSweep(
        turnaroundLine,
        sweepLines[sweepIndex + 1]!,
      );
      const sweepToTurnJoin = buildSmoothJoinToLineStart(stitchedSweepLine, turnaroundLine);
      if (sweepToTurnJoin.length > 0) {
        turnaroundLine = dedupeCoordinates([
          stitchedSweepLine.at(-1)!,
          ...sweepToTurnJoin,
          ...turnaroundLine.slice(1),
        ]);
      } else if (
        stitchedSweepLine.at(-1) &&
        (turnaroundLine[0][0] !== stitchedSweepLine.at(-1)![0] || turnaroundLine[0][1] !== stitchedSweepLine.at(-1)![1])
      ) {
        turnaroundLine = dedupeCoordinates([stitchedSweepLine.at(-1)!, ...turnaroundLine]);
      }
    }

    connectedLines.push(stitchedSweepLine, turnaroundLine ?? [stitchedSweepLine.at(-1)!, sweepLines[sweepIndex + 1][0]]);
    turnBlocks.push(
      toPlannedTurnBlock(turnBlockGeometry, referenceCoordinate, {
        loiterEntryPoint,
        loiterExitPoint,
      }),
    );
  }

  const lastSweepLine = sweepLines.at(-1)!;
  const stitchedLastSweepLine =
    connectedLines.length > 0 ? stitchLineStart(lastSweepLine, connectedLines.at(-1)!.at(-1)!) : lastSweepLine;
  if (connectedLines.length > 0) {
    const previousTurnaroundLine = connectedLines.at(-1)!;
    const smoothJoin = buildSmoothJoinToLineStart(previousTurnaroundLine, stitchedLastSweepLine);
    if (smoothJoin.length > 0) previousTurnaroundLine.push(...smoothJoin);
  }
  connectedLines.push(stitchedLastSweepLine);
  return { connectedLines, turnaroundRadiusM: missionTurnaroundRadiusM, turnBlocks };
};

export const generatePlannedFlightGeometryForPolygon = (
  ring: LngLat[],
  bearingDeg: number,
  lineSpacingM: number,
  params: FlightParams,
  options?: {
    includeTurnPreview?: boolean;
  },
): PlannedFlightGeometry & { bounds: { minLng: number; minLat: number; maxLng: number; maxLat: number; centroid: [number, number] } } => {
  const includeTurnPreview = options?.includeTurnPreview ?? true;
  const cacheKey = getPlannedGeometryCacheKey(ring, bearingDeg, lineSpacingM, params, includeTurnPreview);
  const cachedGeometry = plannedGeometryCache.get(cacheKey);
  if (cachedGeometry) {
    plannedGeometryCache.delete(cacheKey);
    plannedGeometryCache.set(cacheKey, cachedGeometry);
    return cachedGeometry;
  }

  const raw = generateFlightLinesForPolygon(ring, bearingDeg, lineSpacingM);
  const sweepLines = mergeSweepLines(raw.flightLines as LngLat[][], raw.lineSpacing, raw.sweepIndices);
  if (!includeTurnPreview) {
    const geometry = {
      ...raw,
      flightLines: raw.flightLines as LngLat[][],
      sweepLines,
      gridPoints: [],
      leadInPoints: [],
      leadOutPoints: [],
      connectedLines: sweepLines,
      turnaroundRadiusM: 0,
      turnBlocks: [],
    };
    rememberPlannedGeometry(cacheKey, geometry);
    return geometry;
  }
  const canonicalSweepLines = normalizeSweepLineDirectionsForGrid(sweepLines);
  const flightTurnModel = getFlightTurnModel(params);
  const { gridPoints, leadInPoints, leadOutPoints } = buildSurveyGridPoints(
    canonicalSweepLines,
    raw.lineSpacing,
    flightTurnModel.defaultTurnaroundRadiusM,
  );
  const { connectedLines, turnaroundRadiusM, turnBlocks } = connectSweepLinesUsingDynamicFlightTurn(
    sweepLines,
    gridPoints,
    raw.lineSpacing,
    flightTurnModel,
    params.altitudeAGL,
  );

  const geometry = {
    ...raw,
    flightLines: raw.flightLines as LngLat[][],
    sweepLines,
    gridPoints,
    leadInPoints,
    leadOutPoints,
    connectedLines,
    turnaroundRadiusM,
    turnBlocks,
  };
  rememberPlannedGeometry(cacheKey, geometry);
  return geometry;
};

export const connectedPathLengthMeters = (geometry: Pick<PlannedFlightGeometry, "connectedLines">) => {
  let total = 0;
  for (const line of geometry.connectedLines) {
    for (let index = 1; index < line.length; index += 1) {
      total += haversineDistance(line[index - 1] as [number, number], line[index] as [number, number]);
    }
  }
  return total;
};

export const summarizePlannedFlightGeometry = (
  geometry: Pick<PlannedFlightGeometry, "flightLines" | "sweepIndices" | "lineSpacing" | "sweepLines" | "connectedLines">,
) => {
  const sweeps = groupFlightLinesForTraversal(
    geometry.flightLines as number[][][],
    geometry.lineSpacing,
    geometry.sweepIndices,
  );
  let interSegmentGapLengthM = 0;
  let fragmentedLineCount = 0;

  sweeps.forEach((sweep) => {
    if (sweep.fragments.length <= 1) return;
    fragmentedLineCount += sweep.fragments.length - 1;
    for (let fragmentIndex = 1; fragmentIndex < sweep.fragments.length; fragmentIndex += 1) {
      const previousFragment = sweep.fragments[fragmentIndex - 1];
      const currentFragment = sweep.fragments[fragmentIndex];
      if (!previousFragment?.length || !currentFragment?.length) continue;
      interSegmentGapLengthM += haversineDistance(
        previousFragment[previousFragment.length - 1] as [number, number],
        currentFragment[0] as [number, number],
      );
    }
  });

  const totalFlightLineLengthM = geometry.sweepLines.reduce((sum, sweepLine) => {
    let lengthM = 0;
    for (let index = 1; index < sweepLine.length; index += 1) {
      lengthM += haversineDistance(sweepLine[index - 1] as [number, number], sweepLine[index] as [number, number]);
    }
    return sum + lengthM;
  }, 0);

  const totalConnectedPathLengthM = connectedPathLengthMeters(geometry);
  const connectorLengthM = Math.max(0, totalConnectedPathLengthM - totalFlightLineLengthM);
  const turnCount = Math.max(0, geometry.sweepLines.length - 1);

  return {
    lineCount: geometry.sweepLines.length,
    fragmentedLineCount,
    interSegmentGapLengthM,
    totalFlightLineLengthM,
    totalConnectedPathLengthM,
    connectorLengthM,
    overflightTransitFraction:
      totalConnectedPathLengthM > 0 ? connectorLengthM / totalConnectedPathLengthM : 0,
    fragmentedLineFraction:
      geometry.sweepLines.length > 0 ? fragmentedLineCount / geometry.sweepLines.length : 0,
    turnCount,
  };
};
