import { DJI_ZENMUSE_P1_24MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM, SONY_RX1R2, SONY_RX1R3, SONY_A6100_20MM, forwardSpacingRotated, lineSpacingRotated } from "@/domain/camera";
import { DEFAULT_LIDAR, DEFAULT_LIDAR_MAX_RANGE_M, LIDAR_REGISTRY, getLidarMappingFovDeg, getLidarModel, lidarDeliverableDensity, lidarLineSpacing, lidarSinglePassDensity, lidarSwathWidth } from "@/domain/lidar";
import type { FlightParams, TerrainTile } from "@/domain/types";
import { build3DFlightPath, calculateOptimalTerrainZoom, queryMinMaxElevationAlongPolylineWGS84, sampleCameraPositionsOnFlightPath } from "@/flight/geometry";
import { generatePlannedFlightGeometryForPolygon } from "@/flight/plannedGeometry";
import type {
  ExactCameraTileInput,
  ExactCameraTileOutput,
  ExactLidarTileInput,
  ExactLidarTileOutput,
  ExactScoreBreakdown,
} from "@/overlap/exact-core";
import { scoreExactCameraStats, scoreExactLidarStats } from "@/overlap/exact-core";
import { aggregateMetricStats } from "@/overlap/metricAggregation";
import { lngLatToMeters, tileMetersBounds } from "@/overlap/mercator";
import type { GSDStats, LidarStripMeters, PaddedDemTileRGBA, PoseMeters, TileRGBA } from "@/overlap/types";
import { tilesCoveringPolygon } from "@/overlap/tileCoverage";
import type { TerrainPartitionSolutionPreview } from "@/terrain-partition/types";

const CAMERA_REGISTRY: Record<string, typeof SONY_RX1R2> = {
  SONY_RX1R2,
  SONY_RX1R3,
  SONY_A6100_20MM,
  DJI_ZENMUSE_P1_24MM,
  ILX_LR1_INSPECT_85MM,
  MAP61_17MM,
  RGB61_24MM,
};

const DEFAULT_CAMERA = SONY_RX1R2;
const DEFAULT_EXACT_OPTIMIZE_ZOOM = 14;
const DEFAULT_TIME_WEIGHT = 0.1;
const DEFAULT_CAMERA_SPEED_MPS = 12;
const DEFAULT_MIN_OVERLAP_FOR_GSD = 3;

export type ExactMetricKind = "gsd" | "density";
export type ExactSearchMode = "local" | "global";

export type ExactTileRef = { z: number; x: number; y: number };

export type ExactTileWithHalo = {
  tile: TileRGBA;
  demTile: PaddedDemTileRGBA;
};

export interface ExactTerrainProvider {
  getTerrainTiles(tileRefs: ExactTileRef[]): Promise<Map<string, TileRGBA>>;
  getTerrainTilesWithHalo(tileRefs: ExactTileRef[], padTiles: number): Promise<Map<string, ExactTileWithHalo>>;
}

export interface ExactTileEvaluator {
  evaluateCameraTile(input: ExactCameraTileInput): Promise<ExactCameraTileOutput>;
  evaluateLidarTile(input: ExactLidarTileInput): Promise<ExactLidarTileOutput>;
  dispose?(): void | Promise<void>;
}

export interface ExactRegionRuntime {
  terrainProvider: ExactTerrainProvider;
  tileEvaluator: ExactTileEvaluator;
  yieldToEventLoop?: () => Promise<void>;
}

export interface ExactRegionCommonArgs {
  scopeId: string;
  ring: [number, number][];
  params: FlightParams;
  altitudeMode: "legacy" | "min-clearance";
  minClearanceM: number;
  turnExtendM: number;
  exactOptimizeZoom?: number;
  timeWeight?: number;
  clipInnerBufferM?: number;
  minOverlapForGsd?: number;
  geometryTiles?: TerrainTile[];
}

export interface ExactBearingCandidate {
  bearingDeg: number;
  exactCost: number;
  qualityCost: number;
  missionTimeSec: number;
  normalizedTimeCost: number;
  metricKind: ExactMetricKind;
  stats: GSDStats;
  diagnostics: Record<string, number>;
  qualityBreakdown: ExactScoreBreakdown;
  costBreakdown: ExactScoreBreakdown;
  missionBreakdown: ExactMissionBreakdown;
}

export interface ExactBearingSearchResult {
  best: ExactBearingCandidate | null;
  evaluated: ExactBearingCandidate[];
  seedBearingDeg: number;
  lineSpacingM: number;
  safeParams: FlightParams;
}

export interface ExactMissionBreakdown {
  totalLengthM: number;
  speedMps: number;
  lineCount: number;
  sampleCount?: number;
  segmentCount?: number;
  sampleLabel?: string;
}

export interface ExactRegionSummary {
  exactScore?: number;
  qualityCost?: number;
  missionTimeSec?: number;
  normalizedTimeCost?: number;
  metricKind?: ExactMetricKind;
  seedBearingDeg?: number;
}

export interface ExactPartitionPreview {
  solution: TerrainPartitionSolutionPreview;
  metricKind: ExactMetricKind;
  stats: GSDStats;
  regionStats: GSDStats[];
  regionCount: number;
  sampleCount: number;
  sampleLabel: string;
}

export interface ExactPartitionRerankResult {
  bestIndex: number;
  solutions: TerrainPartitionSolutionPreview[];
  previewsBySignature: Record<string, ExactPartitionPreview>;
  debugBySignature?: Record<string, ExactSolutionDebugTrace>;
}

export interface ExactPartitionSolutionEvaluation {
  solution: TerrainPartitionSolutionPreview;
  preview: ExactPartitionPreview;
  score: number;
  debugTrace?: ExactSolutionDebugTrace;
}

export interface ExactRegionSearchTrace {
  regionIndex: number;
  originalBearingDeg: number;
  seedBearingDeg: number;
  chosenBearingDeg: number | null;
  chosenExactCost: number | null;
  searchMode: ExactSearchMode;
  halfWindowDeg: number;
  lineSpacingM: number;
  elapsedMs: number;
  evaluatedBearings: ExactBearingCandidate[];
}

export interface ExactSolutionDebugTrace {
  signature: string;
  polygonId: string;
  rankingSource: "backend-exact" | "frontend-exact";
  exactOptimizeZoom: number;
  timeWeight: number;
  qualityWeight: number;
  fastestMissionTimeSec: number;
  partitionScoreBreakdown: ExactScoreBreakdown;
  preview: ExactPartitionPreview;
  timings: {
    totalElapsedMs: number;
    previewElapsedMs: number;
    regionSearchElapsedMs: number[];
  };
  regions: ExactRegionSearchTrace[];
}

function tileKey(tileRef: ExactTileRef) {
  return `${tileRef.z}/${tileRef.x}/${tileRef.y}`;
}

function normalizeTileRef(tileRef: ExactTileRef): ExactTileRef {
  const tilesPerAxis = 1 << tileRef.z;
  return {
    z: tileRef.z,
    x: ((tileRef.x % tilesPerAxis) + tilesPerAxis) % tilesPerAxis,
    y: Math.max(0, Math.min(tilesPerAxis - 1, tileRef.y)),
  };
}

function normalizeAxialBearingDeg(value: number) {
  const normalized = ((value % 180) + 180) % 180;
  return Number.isFinite(normalized) ? normalized : 0;
}

function pointInRing(lng: number, lat: number, ring: [number, number][]) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersect = ((yi > lat) !== (yj > lat)) &&
      (lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function path3dLengthMeters(path3d: [number, number, number][][]) {
  let total = 0;
  for (const line of path3d) {
    if (!Array.isArray(line) || line.length < 2) continue;
    for (let index = 1; index < line.length; index++) {
      const start = line[index - 1];
      const end = line[index];
      const [x1, y1] = lngLatToMeters(start[0], start[1]);
      const [x2, y2] = lngLatToMeters(end[0], end[1]);
      const dz = end[2] - start[2];
      total += Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + dz ** 2);
    }
  }
  return total;
}

function buildScoreBreakdown(
  modelVersion: string,
  signals: Record<string, number>,
  weights: Record<string, number>,
) {
  const contributions = Object.fromEntries(
    Object.entries(weights).map(([key, weight]) => [key, (signals[key] ?? 0) * weight]),
  );
  return {
    modelVersion,
    total: Object.values(contributions).reduce((sum, value) => sum + value, 0),
    signals,
    weights,
    contributions,
  } satisfies ExactScoreBreakdown;
}

function buildExactCostBreakdown(
  modelVersion: string,
  qualityWeight: number,
  qualityCost: number,
  timeWeight: number,
  normalizedTimeCost: number,
) {
  return buildScoreBreakdown(
    modelVersion,
    {
      qualityCost,
      normalizedTimeCost,
    },
    {
      qualityCost: qualityWeight,
      normalizedTimeCost: timeWeight,
    },
  );
}

function isLidarParams(params: FlightParams): boolean {
  return (params.payloadKind ?? "camera") === "lidar";
}

function getCameraForParams(params: FlightParams) {
  const cameraKey = params.cameraKey;
  return cameraKey ? CAMERA_REGISTRY[cameraKey] || DEFAULT_CAMERA : DEFAULT_CAMERA;
}

function getLineSpacingForParams(params: FlightParams): number {
  if (isLidarParams(params)) {
    const lidar = params.lidarKey ? LIDAR_REGISTRY[params.lidarKey] || DEFAULT_LIDAR : DEFAULT_LIDAR;
    const mappingFovDeg = params.mappingFovDeg ?? lidar.effectiveHorizontalFovDeg;
    return lidarLineSpacing(params.altitudeAGL, params.sideOverlap, mappingFovDeg);
  }
  const camera = getCameraForParams(params);
  const yawOffset = params.cameraYawOffsetDeg ?? 0;
  const rotate90 = Math.round((((yawOffset % 180) + 180) % 180)) === 90;
  return lineSpacingRotated(camera, params.altitudeAGL, params.sideOverlap, rotate90);
}

function getForwardSpacingForParams(params: FlightParams): number | null {
  if (isLidarParams(params)) return null;
  const camera = getCameraForParams(params);
  const yawOffset = params.cameraYawOffsetDeg ?? 0;
  const rotate90 = Math.round((((yawOffset % 180) + 180) % 180)) === 90;
  return forwardSpacingRotated(camera, params.altitudeAGL, params.frontOverlap, rotate90);
}

function lidarStripMayAffectTile(strip: LidarStripMeters, tileRef: ExactTileRef) {
  const bounds = tileMetersBounds(tileRef.z, tileRef.x, tileRef.y);
  const reachPadM = Math.max(
    strip.halfWidthM ?? 0,
    typeof strip.maxRangeM === "number" && Number.isFinite(strip.maxRangeM) ? strip.maxRangeM : 0,
  );
  const minXs = Math.min(strip.x1, strip.x2) - reachPadM;
  const maxXs = Math.max(strip.x1, strip.x2) + reachPadM;
  const minYs = Math.min(strip.y1, strip.y2) - reachPadM;
  const maxYs = Math.max(strip.y1, strip.y2) + reachPadM;
  return !(
    maxXs < bounds.minX ||
    minXs > bounds.maxX ||
    maxYs < bounds.minY ||
    minYs > bounds.maxY
  );
}

function toTerrainTiles(tiles: Map<string, TileRGBA>): TerrainTile[] {
  return [...tiles.values()].map((tile) => ({
    z: tile.z,
    x: tile.x,
    y: tile.y,
    width: tile.size,
    height: tile.size,
    data: tile.data,
  }));
}

async function resolveGeometryTiles(
  runtime: ExactRegionRuntime,
  ring: [number, number][],
  geometryTiles?: TerrainTile[],
) {
  if (Array.isArray(geometryTiles) && geometryTiles.length > 0) return geometryTiles;
  const geometryZoom = calculateOptimalTerrainZoom({ coordinates: ring });
  const refs = tilesCoveringPolygon({ ring }, geometryZoom).map((tile) => normalizeTileRef({ z: geometryZoom, x: tile.x, y: tile.y }));
  const tiles = await runtime.terrainProvider.getTerrainTiles(refs);
  return toTerrainTiles(tiles);
}

function estimateMissionBreakdown(
  ring: [number, number][],
  bearingDeg: number,
  lineSpacing: number,
  terrainTiles: TerrainTile[],
  params: FlightParams,
  altitudeMode: "legacy" | "min-clearance",
  minClearanceM: number,
) {
  const geometry = generatePlannedFlightGeometryForPolygon(ring, bearingDeg, lineSpacing, params);
  if (!geometry.flightLines.length) {
    return {
      missionTimeSec: 0,
      totalLengthM: 0,
      speedMps: isLidarParams(params)
        ? (params.speedMps ?? getLidarModel(params.lidarKey).defaultSpeedMps)
        : DEFAULT_CAMERA_SPEED_MPS,
      lineCount: 0,
    };
  }
  const path3d = build3DFlightPath(
    geometry,
    terrainTiles,
    lineSpacing,
    { altitudeAGL: params.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, preconnected: true },
  );
  const totalLengthM = path3dLengthMeters(path3d);
  const speedMps = isLidarParams(params)
    ? (params.speedMps ?? getLidarModel(params.lidarKey).defaultSpeedMps)
    : DEFAULT_CAMERA_SPEED_MPS;
  return {
    missionTimeSec: totalLengthM / Math.max(0.1, speedMps),
    totalLengthM,
    speedMps,
    lineCount: geometry.sweepLines.length,
  };
}

async function buildCameraPosesForBearing(
  scopeId: string,
  ring: [number, number][],
  params: FlightParams,
  bearingDeg: number,
  geometryTiles: TerrainTile[],
  altitudeMode: "legacy" | "min-clearance",
  minClearanceM: number,
) {
  const lineSpacing = getLineSpacingForParams(params);
  const photoSpacing = getForwardSpacingForParams(params);
  if (!photoSpacing || !(photoSpacing > 0)) {
    return { poses: [] as PoseMeters[], lineSpacingM: lineSpacing };
  }
  const yawOffset = params.cameraYawOffsetDeg ?? 0;
  const normalizeDeg = (value: number) => ((value % 360) + 360) % 360;
  const geometry = generatePlannedFlightGeometryForPolygon(ring, bearingDeg, lineSpacing, params);
  const path3d = build3DFlightPath(
    geometry,
    geometryTiles,
    lineSpacing,
    { altitudeAGL: params.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, preconnected: true },
  );
  const cameraPositions = sampleCameraPositionsOnFlightPath(path3d, photoSpacing, { includeTurns: false });
  const filtered = ring.length >= 3
    ? cameraPositions.filter(([lng, lat]) => pointInRing(lng, lat, ring))
    : cameraPositions;
  const poses: PoseMeters[] = filtered.map(([lng, lat, altMSL, yawDeg], index) => {
    const [x, y] = lngLatToMeters(lng, lat);
    return {
      id: `exact_pose_${scopeId}_${index}`,
      x,
      y,
      z: altMSL,
      omega_deg: 0,
      phi_deg: 0,
      kappa_deg: normalizeDeg(-yawDeg + yawOffset),
      polygonId: scopeId,
    };
  });
  return { poses, lineSpacingM: lineSpacing };
}

async function buildLidarStripsForBearing(
  scopeId: string,
  ring: [number, number][],
  params: FlightParams,
  bearingDeg: number,
  geometryTiles: TerrainTile[],
  altitudeMode: "legacy" | "min-clearance",
  minClearanceM: number,
) {
  const lineSpacing = getLineSpacingForParams(params);
  const model = getLidarModel(params.lidarKey);
  const altitudeAGL = params.altitudeAGL;
  const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
  const speedMps = params.speedMps ?? model.defaultSpeedMps;
  const returnMode = params.lidarReturnMode ?? "single";
  const maxLidarRangeM = params.maxLidarRangeM ?? model.defaultMaxRangeM ?? DEFAULT_LIDAR_MAX_RANGE_M;
  const frameRateHz = params.lidarFrameRateHz ?? model.defaultFrameRateHz;
  const azimuthSectorCenterDeg = params.lidarAzimuthSectorCenterDeg ?? model.defaultAzimuthSectorCenterDeg ?? 0;
  const boresightYawDeg = params.lidarBoresightYawDeg ?? model.boresightYawDeg ?? 0;
  const boresightPitchDeg = params.lidarBoresightPitchDeg ?? model.boresightPitchDeg ?? 0;
  const boresightRollDeg = params.lidarBoresightRollDeg ?? model.boresightRollDeg ?? 0;
  const comparisonMode = params.lidarComparisonMode ?? "first-return";
  const densityPerPass = lidarSinglePassDensity(model, altitudeAGL, speedMps, returnMode, mappingFovDeg);
  const halfFovTan = Math.tan((mappingFovDeg * Math.PI) / 360);
  const strips: LidarStripMeters[] = [];
  const geometry = generatePlannedFlightGeometryForPolygon(ring, bearingDeg, lineSpacing, params);
  const sweeps = geometry.sweepLines;
  let passIndex = 0;
  for (const activeSweepLine of sweeps) {
    if (!Array.isArray(activeSweepLine) || activeSweepLine.length < 2) continue;
    const sweepPath3d = build3DFlightPath(
      [activeSweepLine],
      geometryTiles,
      lineSpacing,
      { altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, turnExtendM: 0 },
    )[0];
    if (!Array.isArray(sweepPath3d) || sweepPath3d.length < 2) continue;
    const localPassIndex = passIndex++;
    for (let i = 1; i < sweepPath3d.length; i++) {
      const start = sweepPath3d[i - 1];
      const end = sweepPath3d[i];
      if (!Array.isArray(start) || !Array.isArray(end) || start.length < 3 || end.length < 3) continue;
      const [x1, y1] = lngLatToMeters(start[0], start[1]);
      const [x2, y2] = lngLatToMeters(end[0], end[1]);
      const terrainMin = queryMinMaxElevationAlongPolylineWGS84([[start[0], start[1]], [end[0], end[1]]], geometryTiles, 12).min;
      const maxSensorAltitude = Math.max(start[2], end[2]);
      const maxHalfWidth = Number.isFinite(terrainMin)
        ? Math.max(lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2, Math.max(1, (maxSensorAltitude - terrainMin) * halfFovTan))
        : lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2;
      strips.push({
        id: `${scopeId}-sweep-${localPassIndex}-seg-${i - 1}`,
        polygonId: scopeId,
        x1,
        y1,
        z1: start[2],
        x2,
        y2,
        z2: end[2],
        plannedAltitudeAGL: altitudeAGL,
        halfWidthM: maxHalfWidth,
        densityPerPass,
        speedMps,
        effectivePointRate: model.effectivePointRates[returnMode],
        halfFovTan,
        maxRangeM: maxLidarRangeM,
        passIndex: localPassIndex,
        frameRateHz,
        nativeHorizontalFovDeg: model.nativeHorizontalFovDeg,
        mappingFovDeg,
        verticalAnglesDeg: model.verticalAnglesDeg,
        returnMode,
        comparisonMode,
        azimuthSectorCenterDeg,
        boresightYawDeg,
        boresightPitchDeg,
        boresightRollDeg,
      });
    }
  }
  return { strips, lineSpacingM: lineSpacing };
}

export async function evaluateRegionBearingExact(
  runtime: ExactRegionRuntime,
  args: ExactRegionCommonArgs & { bearingDeg: number },
): Promise<ExactBearingCandidate | null> {
  const safeParams = { ...args.params, useCustomBearing: false, customBearingDeg: undefined };
  const lineSpacingM = getLineSpacingForParams(safeParams);
  const geometryTiles = await resolveGeometryTiles(runtime, args.ring, args.geometryTiles);
  const exactZoom = args.exactOptimizeZoom ?? DEFAULT_EXACT_OPTIMIZE_ZOOM;
  const timeWeight = args.timeWeight ?? DEFAULT_TIME_WEIGHT;
  const exactQualityWeight = 1 - timeWeight;
  const normalizedBearingDeg = normalizeAxialBearingDeg(args.bearingDeg);
  const tileRefs = (() => {
    const seen = new Set<string>();
    const refs: ExactTileRef[] = [];
    for (const tile of tilesCoveringPolygon({ ring: args.ring }, exactZoom)) {
      const ref = normalizeTileRef({ z: exactZoom, x: tile.x, y: tile.y });
      const key = tileKey(ref);
      if (seen.has(key)) continue;
      seen.add(key);
      refs.push(ref);
    }
    return refs;
  })();
  if (runtime.yieldToEventLoop) {
    await runtime.yieldToEventLoop();
  }

  if (isLidarParams(safeParams)) {
      const { strips } = await buildLidarStripsForBearing(
        args.scopeId,
        args.ring,
        safeParams,
        normalizedBearingDeg,
        geometryTiles,
        args.altitudeMode,
        args.minClearanceM,
      );
    if (!strips.length) return null;
    const tileMapWithHalo = await runtime.terrainProvider.getTerrainTilesWithHalo(tileRefs, 1);
    const perTileStats: GSDStats[] = [];
    for (const tileRef of tileRefs) {
      const tileBundle = tileMapWithHalo.get(tileKey(tileRef));
      if (!tileBundle) continue;
      const tileStrips = strips.filter((strip) => lidarStripMayAffectTile(strip, tileRef));
      if (!tileStrips.length) continue;
      const response = await runtime.tileEvaluator.evaluateLidarTile({
        tile: tileBundle.tile,
        demTile: tileBundle.demTile,
        polygons: [{ id: args.scopeId, ring: args.ring }],
        strips: tileStrips,
        options: { clipInnerBufferM: args.clipInnerBufferM ?? 0 },
      });
      const densityStats = response.perPolygon?.find((entry) => entry.polygonId === args.scopeId)?.densityStats;
      if (densityStats) perTileStats.push(densityStats);
    }
    if (!perTileStats.length) return null;
    const stats = aggregateMetricStats(perTileStats);
    const scored = scoreExactLidarStats(stats, safeParams);
    const mission = estimateMissionBreakdown(
      args.ring,
      normalizedBearingDeg,
      lineSpacingM,
      geometryTiles,
      safeParams,
      args.altitudeMode,
      args.minClearanceM,
    );
    const normalizedTimeCost = mission.missionTimeSec / 180;
    const costBreakdown = buildExactCostBreakdown(
      scored.breakdown.modelVersion,
      exactQualityWeight,
      scored.qualityCost,
      timeWeight,
      normalizedTimeCost,
    );
    const missionBreakdown: ExactMissionBreakdown = {
      totalLengthM: mission.totalLengthM,
      speedMps: mission.speedMps,
      lineCount: mission.lineCount,
      sampleCount: new Set(strips.map((strip) => strip.passIndex ?? -1)).size,
      segmentCount: strips.length,
      sampleLabel: "Flight lines",
    };
    return {
      bearingDeg: normalizedBearingDeg,
      exactCost: costBreakdown.total,
      qualityCost: scored.qualityCost,
      missionTimeSec: mission.missionTimeSec,
      normalizedTimeCost,
      metricKind: "density",
      stats,
      diagnostics: {
        qualityCost: scored.qualityCost,
        missionTimeSec: mission.missionTimeSec,
        normalizedTimeCost,
        targetDensityPtsM2: scored.targetDensityPtsM2,
        holeFraction: scored.holeFraction,
        lowFraction: scored.lowFraction,
        q10: scored.q10,
        q25: scored.q25,
      },
      qualityBreakdown: scored.breakdown,
      costBreakdown,
      missionBreakdown,
    };
  }

  const { poses } = await buildCameraPosesForBearing(
    args.scopeId,
    args.ring,
    safeParams,
    normalizedBearingDeg,
    geometryTiles,
    args.altitudeMode,
    args.minClearanceM,
  );
  if (!poses.length) return null;
  const camera = getCameraForParams(safeParams);
  const tileMap = await runtime.terrainProvider.getTerrainTiles(tileRefs);
  const perTileStats: GSDStats[] = [];
  for (const tileRef of tileRefs) {
    const tile = tileMap.get(tileKey(tileRef));
    if (!tile) continue;
    const response = await runtime.tileEvaluator.evaluateCameraTile({
      tile,
      polygons: [{ id: args.scopeId, ring: args.ring }],
      poses,
      cameras: [camera],
      poseCameraIndices: new Uint16Array(poses.length),
      camera: undefined,
      options: {
        clipInnerBufferM: args.clipInnerBufferM ?? 0,
        minOverlapForGsd: args.minOverlapForGsd ?? DEFAULT_MIN_OVERLAP_FOR_GSD,
      },
    });
    const gsdStats = response.perPolygon?.find((entry) => entry.polygonId === args.scopeId)?.gsdStats;
    if (gsdStats) perTileStats.push(gsdStats);
  }
  if (!perTileStats.length) return null;
  const stats = aggregateMetricStats(perTileStats);
  const scored = scoreExactCameraStats(stats, safeParams);
  const mission = estimateMissionBreakdown(
    args.ring,
    normalizedBearingDeg,
    lineSpacingM,
    geometryTiles,
    safeParams,
    args.altitudeMode,
    args.minClearanceM,
  );
  const normalizedTimeCost = mission.missionTimeSec / 180;
  const costBreakdown = buildExactCostBreakdown(
    scored.breakdown.modelVersion,
    exactQualityWeight,
    scored.qualityCost,
    timeWeight,
    normalizedTimeCost,
  );
  const missionBreakdown: ExactMissionBreakdown = {
    totalLengthM: mission.totalLengthM,
    speedMps: mission.speedMps,
    lineCount: mission.lineCount,
    sampleCount: poses.length,
    sampleLabel: "Images",
  };
  return {
    bearingDeg: normalizedBearingDeg,
    exactCost: costBreakdown.total,
    qualityCost: scored.qualityCost,
    missionTimeSec: mission.missionTimeSec,
    normalizedTimeCost,
    metricKind: "gsd",
    stats,
    diagnostics: {
      qualityCost: scored.qualityCost,
      missionTimeSec: mission.missionTimeSec,
      normalizedTimeCost,
      targetGsdM: scored.targetGsdM,
      overTargetAreaFraction: scored.overTargetAreaFraction,
      q75: scored.q75,
      q90: scored.q90,
    },
    qualityBreakdown: scored.breakdown,
    costBreakdown,
    missionBreakdown,
  };
}

export async function optimizeBearingExact(
  runtime: ExactRegionRuntime,
  args: ExactRegionCommonArgs & {
    seedBearingDeg: number;
    mode?: ExactSearchMode;
    halfWindowDeg?: number;
  },
): Promise<ExactBearingSearchResult> {
  const normalizedSeedBearingDeg = Number.isFinite(args.seedBearingDeg)
    ? normalizeAxialBearingDeg(args.seedBearingDeg)
    : 0;
  const safeParams = { ...args.params, useCustomBearing: false, customBearingDeg: undefined };
  const lineSpacingM = getLineSpacingForParams(safeParams);
  const geometryTiles = await resolveGeometryTiles(runtime, args.ring, args.geometryTiles);
  const coarseOffsets = [-30, -20, -10, 0, 10, 20, 30];
  const halfWindowDeg = Math.max(1, args.halfWindowDeg ?? 30);
  const refineStepsDeg = [8, 4, 2, 1].filter((step) => step <= halfWindowDeg);
  const globalCoarseBearings = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165];
  const minImprovement = 1e-4;
  const mode = args.mode ?? "local";
  const evaluationCache = new Map<number, Promise<ExactBearingCandidate | null>>();

  const evaluateBearing = async (bearingDeg: number) => {
    const normalized = normalizeAxialBearingDeg(bearingDeg);
    const cacheKey = Math.round(normalized * 1000);
    if (!evaluationCache.has(cacheKey)) {
      evaluationCache.set(cacheKey, evaluateRegionBearingExact(runtime, {
        ...args,
        params: safeParams,
        bearingDeg: normalized,
        geometryTiles,
      }));
    }
    return evaluationCache.get(cacheKey)!;
  };

  const evaluateOffset = async (offsetDeg: number) => {
    if (Math.abs(offsetDeg) > halfWindowDeg + 1e-6) return null;
    return evaluateBearing(normalizedSeedBearingDeg + offsetDeg);
  };

  let best: ExactBearingCandidate | null = null;
  let bestOffset = 0;
  if (mode === "global") {
    const coarseCandidates = Array.from(new Set([...globalCoarseBearings, Math.round(normalizedSeedBearingDeg * 10) / 10]));
    for (const bearingDeg of coarseCandidates) {
      const candidate = await evaluateBearing(bearingDeg);
      if (candidate && (!best || candidate.exactCost < best.exactCost)) best = candidate;
    }
  } else {
    best = await evaluateOffset(0);
    for (const offsetDeg of coarseOffsets.filter((offset) => Math.abs(offset) <= halfWindowDeg + 1e-6)) {
      const candidate = await evaluateOffset(offsetDeg);
      if (candidate && (!best || candidate.exactCost < best.exactCost)) {
        best = candidate;
        bestOffset = offsetDeg;
      }
    }
  }

  if (best) {
    for (const stepDeg of refineStepsDeg) {
      let improved = true;
      while (improved) {
        improved = false;
        const currentBest: ExactBearingCandidate = best;
        const left: ExactBearingCandidate | null = mode === "global"
          ? await evaluateBearing(currentBest.bearingDeg - stepDeg)
          : await evaluateOffset(bestOffset - stepDeg);
        const right: ExactBearingCandidate | null = mode === "global"
          ? await evaluateBearing(currentBest.bearingDeg + stepDeg)
          : await evaluateOffset(bestOffset + stepDeg);
        const nextBest: { offsetDeg: number; candidate: ExactBearingCandidate } | null =
          [
            { offsetDeg: mode === "global" ? 0 : (bestOffset - stepDeg), candidate: left },
            { offsetDeg: mode === "global" ? 0 : (bestOffset + stepDeg), candidate: right },
          ]
            .filter((value): value is { offsetDeg: number; candidate: ExactBearingCandidate } => value.candidate !== null)
            .sort((a, b) => a.candidate.exactCost - b.candidate.exactCost)[0] ?? null;
        if (nextBest && nextBest.candidate.exactCost + minImprovement < currentBest.exactCost) {
          best = nextBest.candidate;
          if (mode !== "global") bestOffset = nextBest.offsetDeg;
          improved = true;
        }
      }
    }
  }

  const evaluated = (await Promise.all([...evaluationCache.values()]))
    .filter((value): value is ExactBearingCandidate => value !== null)
    .sort((left, right) => left.exactCost - right.exactCost);

  return {
    best,
    evaluated,
    seedBearingDeg: normalizedSeedBearingDeg,
    lineSpacingM,
    safeParams,
  };
}

export async function evaluatePartitionSolutionExact(
  runtime: ExactRegionRuntime,
  args: ExactRegionCommonArgs & {
    polygonId: string;
    solution: TerrainPartitionSolutionPreview;
  },
): Promise<ExactPartitionPreview> {
  const params = args.params;
  const geometryTiles = await resolveGeometryTiles(runtime, args.ring, args.geometryTiles);
  const virtualPolygons = args.solution.regions.map((region, index) => ({
    id: `${args.polygonId}::${index}`,
    ring: region.ring as [number, number][],
    bearingDeg: region.bearingDeg,
  }));
  const exactZoom = args.exactOptimizeZoom ?? DEFAULT_EXACT_OPTIMIZE_ZOOM;
  const tileRefs = (() => {
    const seen = new Set<string>();
    const refs: ExactTileRef[] = [];
    for (const polygon of virtualPolygons) {
      for (const tile of tilesCoveringPolygon({ ring: polygon.ring }, exactZoom)) {
        const ref = normalizeTileRef({ z: exactZoom, x: tile.x, y: tile.y });
        const key = tileKey(ref);
        if (seen.has(key)) continue;
        seen.add(key);
        refs.push(ref);
      }
    }
    return refs;
  })();

  if (isLidarParams(params)) {
    const model = getLidarModel(params.lidarKey);
    const altitudeAGL = params.altitudeAGL;
    const lineSpacing = getLineSpacingForParams(params);
    const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
    const speedMps = params.speedMps ?? model.defaultSpeedMps;
    const returnMode = params.lidarReturnMode ?? "single";
    const maxLidarRangeM = params.maxLidarRangeM ?? model.defaultMaxRangeM ?? DEFAULT_LIDAR_MAX_RANGE_M;
    const frameRateHz = params.lidarFrameRateHz ?? model.defaultFrameRateHz;
    const azimuthSectorCenterDeg = params.lidarAzimuthSectorCenterDeg ?? model.defaultAzimuthSectorCenterDeg ?? 0;
    const boresightYawDeg = params.lidarBoresightYawDeg ?? model.boresightYawDeg ?? 0;
    const boresightPitchDeg = params.lidarBoresightPitchDeg ?? model.boresightPitchDeg ?? 0;
    const boresightRollDeg = params.lidarBoresightRollDeg ?? model.boresightRollDeg ?? 0;
    const comparisonMode = params.lidarComparisonMode ?? "first-return";
    const densityPerPass = lidarSinglePassDensity(model, altitudeAGL, speedMps, returnMode, mappingFovDeg);
    const halfFovTan = Math.tan((mappingFovDeg * Math.PI) / 360);
    const strips: LidarStripMeters[] = [];
    let passIndex = 0;

    for (const region of virtualPolygons) {
      const geometry = generatePlannedFlightGeometryForPolygon(region.ring, region.bearingDeg, lineSpacing, params);
      for (let lineIndex = 0; lineIndex < geometry.sweepLines.length; lineIndex++) {
        const activeSweepLine = geometry.sweepLines[lineIndex];
        if (!Array.isArray(activeSweepLine) || activeSweepLine.length < 2) continue;
        const sweepPath3d = build3DFlightPath(
          [activeSweepLine],
          geometryTiles,
          lineSpacing,
          { altitudeAGL, mode: args.altitudeMode, minClearance: args.minClearanceM, turnExtendM: 0 },
        )[0];
        if (!Array.isArray(sweepPath3d) || sweepPath3d.length < 2) continue;
        const localPassIndex = passIndex++;
        for (let i = 1; i < sweepPath3d.length; i++) {
          const start = sweepPath3d[i - 1];
          const end = sweepPath3d[i];
          const [x1, y1] = lngLatToMeters(start[0], start[1]);
          const [x2, y2] = lngLatToMeters(end[0], end[1]);
          const terrainMin = queryMinMaxElevationAlongPolylineWGS84([[start[0], start[1]], [end[0], end[1]]], geometryTiles, 12).min;
          const maxSensorAltitude = Math.max(start[2], end[2]);
          const maxHalfWidth = Number.isFinite(terrainMin)
            ? Math.max(lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2, Math.max(1, (maxSensorAltitude - terrainMin) * halfFovTan))
            : lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2;
          strips.push({
            id: `${region.id}-line-${lineIndex}-seg-${i - 1}`,
            polygonId: region.id,
            x1,
            y1,
            z1: start[2],
            x2,
            y2,
            z2: end[2],
            plannedAltitudeAGL: altitudeAGL,
            halfWidthM: maxHalfWidth,
            densityPerPass,
            speedMps,
            effectivePointRate: model.effectivePointRates[returnMode],
            halfFovTan,
            maxRangeM: maxLidarRangeM,
            passIndex: localPassIndex,
            frameRateHz,
            nativeHorizontalFovDeg: model.nativeHorizontalFovDeg,
            mappingFovDeg,
            verticalAnglesDeg: model.verticalAnglesDeg,
            returnMode,
            comparisonMode,
            azimuthSectorCenterDeg,
            boresightYawDeg,
            boresightPitchDeg,
            boresightRollDeg,
          });
        }
      }
    }

    const tileMapWithHalo = await runtime.terrainProvider.getTerrainTilesWithHalo(tileRefs, 1);
    const perRegionStats = new Map<string, GSDStats[]>();
    for (const tileRef of tileRefs) {
      const tileBundle = tileMapWithHalo.get(tileKey(tileRef));
      if (!tileBundle) continue;
      const tileStrips = strips.filter((strip) => lidarStripMayAffectTile(strip, tileRef));
      if (!tileStrips.length) continue;
      const result = await runtime.tileEvaluator.evaluateLidarTile({
        tile: tileBundle.tile,
        demTile: tileBundle.demTile,
        polygons: virtualPolygons.map(({ id, ring }) => ({ id, ring })),
        strips: tileStrips,
        options: { clipInnerBufferM: args.clipInnerBufferM ?? 0 },
      });
      (result.perPolygon ?? []).forEach((polyStats) => {
        if (!polyStats.densityStats) return;
        const list = perRegionStats.get(polyStats.polygonId) ?? [];
        list.push(polyStats.densityStats);
        perRegionStats.set(polyStats.polygonId, list);
      });
    }
    const regionSummaries = Array.from(perRegionStats.values()).map((statsList) => aggregateMetricStats(statsList)).filter((stats) => stats.count > 0);
    if (!regionSummaries.length) {
      throw new Error("No lidar density preview could be computed for this partition.");
    }
    return {
      solution: args.solution,
      metricKind: "density",
      stats: aggregateMetricStats(regionSummaries),
      regionStats: regionSummaries,
      regionCount: args.solution.regionCount,
      sampleCount: new Set(strips.map((strip) => strip.passIndex ?? -1)).size,
      sampleLabel: "Flight lines",
    };
  }

  const camera = getCameraForParams(params);
  const altitudeAGL = params.altitudeAGL;
  const photoSpacing = getForwardSpacingForParams(params);
  const lineSpacing = getLineSpacingForParams(params);
  const yawOffset = params.cameraYawOffsetDeg ?? 0;
  const normalizeDeg = (value: number) => ((value % 360) + 360) % 360;
  const poses: PoseMeters[] = [];
  let poseId = 0;
  for (const region of virtualPolygons) {
    const geometry = generatePlannedFlightGeometryForPolygon(region.ring, region.bearingDeg, lineSpacing, params);
    const path3d = build3DFlightPath(
      geometry,
      geometryTiles,
      lineSpacing,
      { altitudeAGL, mode: args.altitudeMode, minClearance: args.minClearanceM, preconnected: true },
    );
    const cameraPositions = sampleCameraPositionsOnFlightPath(path3d, photoSpacing ?? 0, { includeTurns: false });
    const filtered = region.ring.length >= 3
      ? cameraPositions.filter(([lng, lat]) => pointInRing(lng, lat, region.ring))
      : cameraPositions;
    filtered.forEach(([lng, lat, altMSL, yawDeg]) => {
      const [x, y] = lngLatToMeters(lng, lat);
      poses.push({
        id: `partition_pose_${poseId++}`,
        x,
        y,
        z: altMSL,
        omega_deg: 0,
        phi_deg: 0,
        kappa_deg: normalizeDeg(-yawDeg + yawOffset),
        polygonId: region.id,
      });
    });
  }
  if (!poses.length) {
    throw new Error("No camera poses could be generated for this partition.");
  }

  const tileMap = await runtime.terrainProvider.getTerrainTiles(tileRefs);
  const perRegionStats = new Map<string, GSDStats[]>();
  for (const tileRef of tileRefs) {
    const tile = tileMap.get(tileKey(tileRef));
    if (!tile) continue;
    const result = await runtime.tileEvaluator.evaluateCameraTile({
      tile,
      polygons: virtualPolygons.map(({ id, ring }) => ({ id, ring })),
      poses,
      cameras: [camera],
      poseCameraIndices: new Uint16Array(poses.length),
      camera: undefined,
      options: {
        clipInnerBufferM: args.clipInnerBufferM ?? 0,
        minOverlapForGsd: args.minOverlapForGsd ?? DEFAULT_MIN_OVERLAP_FOR_GSD,
      },
    });
    (result.perPolygon ?? []).forEach((polyStats) => {
      if (!polyStats.gsdStats) return;
      const list = perRegionStats.get(polyStats.polygonId) ?? [];
      list.push(polyStats.gsdStats);
      perRegionStats.set(polyStats.polygonId, list);
    });
  }
  const regionSummaries = Array.from(perRegionStats.values()).map((statsList) => aggregateMetricStats(statsList)).filter((stats) => stats.count > 0);
  if (!regionSummaries.length) {
    throw new Error("No camera GSD preview could be computed for this partition.");
  }
  return {
    solution: args.solution,
    metricKind: "gsd",
    stats: aggregateMetricStats(regionSummaries),
    regionStats: regionSummaries,
    regionCount: args.solution.regionCount,
    sampleCount: poses.length,
    sampleLabel: "Images",
  };
}

function scoreLidarPartitionPreview(
  params: FlightParams,
  solution: TerrainPartitionSolutionPreview,
  preview: ExactPartitionPreview,
  fastestMissionTimeSec: number,
) {
  const model = getLidarModel(params.lidarKey);
  const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
  const speedMps = params.speedMps ?? model.defaultSpeedMps;
  const returnMode = params.lidarReturnMode ?? "single";
  const targetDensityPtsM2 = params.pointDensityPtsM2
    ?? lidarDeliverableDensity(model, params.altitudeAGL, params.sideOverlap, speedMps, returnMode, mappingFovDeg);
  const totalAreaM2 = Math.max(1, preview.stats.totalAreaM2 || 0);
  const holeThreshold = Math.max(5, targetDensityPtsM2 * 0.2);
  const weakThreshold = Math.max(holeThreshold + 1e-6, targetDensityPtsM2 * 0.7);
  const overall = scoreExactLidarStats(preview.stats, params);
  const regionStats = preview.regionStats.length > 0 ? preview.regionStats : [preview.stats];
  const regionSignals = regionStats.map((stats) => {
    const regionScore = scoreExactLidarStats(stats, params);
    return {
      holeFraction: regionScore.holeFraction,
      lowFraction: regionScore.lowFraction,
      q10Deficit: Math.max(0, 1 - regionScore.q10 / Math.max(1e-6, targetDensityPtsM2)),
      meanDeficit: Math.max(0, 1 - stats.mean / Math.max(1e-6, targetDensityPtsM2)),
    };
  });
  const worstRegionHoleFraction = Math.max(...regionSignals.map((signal) => signal.holeFraction));
  const worstRegionLowFraction = Math.max(...regionSignals.map((signal) => signal.lowFraction));
  const worstRegionQ10Deficit = Math.max(...regionSignals.map((signal) => signal.q10Deficit));
  const worstRegionMeanDeficit = Math.max(...regionSignals.map((signal) => signal.meanDeficit));
  const relativeTimePenalty = fastestMissionTimeSec > 0
    ? Math.max(0, solution.totalMissionTimeSec / fastestMissionTimeSec - 1)
    : 0;
  const breakdown = buildScoreBreakdown(
    "lidar-partition-v1",
    {
      worstRegionHoleFraction,
      worstRegionLowFraction,
      worstRegionQ10Deficit,
      worstRegionMeanDeficit,
      overallHoleFraction: overall.holeFraction,
      overallLowFraction: overall.lowFraction,
      overallQ10Deficit: Math.max(0, 1 - overall.q10 / Math.max(1e-6, targetDensityPtsM2)),
      overallQ25Deficit: Math.max(0, 1 - overall.q25 / Math.max(1e-6, targetDensityPtsM2)),
      overallMeanDeficit: Math.max(0, 1 - preview.stats.mean / Math.max(1e-6, targetDensityPtsM2)),
      relativeTimePenalty,
      regionPenalty: solution.regionCount > 1 ? solution.regionCount - 1 : 0,
    },
    {
      worstRegionHoleFraction: 4.8,
      worstRegionLowFraction: 2.9,
      worstRegionQ10Deficit: 2.1,
      worstRegionMeanDeficit: 0.9,
      overallHoleFraction: 4.2,
      overallLowFraction: 2.4,
      overallQ10Deficit: 1.9,
      overallQ25Deficit: 1.2,
      overallMeanDeficit: 0.8,
      relativeTimePenalty: 0.18,
      regionPenalty: 0.035,
    },
  );
  return { score: breakdown.total, totalAreaM2, holeThreshold, weakThreshold, breakdown };
}

function scoreCameraPartitionPreview(
  params: FlightParams,
  solution: TerrainPartitionSolutionPreview,
  preview: ExactPartitionPreview,
  fastestMissionTimeSec: number,
) {
  const overall = scoreExactCameraStats(preview.stats, params);
  const overallMeanCm = preview.stats.mean * 100;
  const overallQ75Cm = overall.q75 * 100;
  const overallQ90Cm = overall.q90 * 100;
  const overallMaxCm = preview.stats.max * 100;
  const regionStats = preview.regionStats.length > 0 ? preview.regionStats : [preview.stats];
  const worstRegionMeanCm = Math.max(...regionStats.map((stats) => stats.mean * 100));
  const worstRegionQ90Cm = Math.max(...regionStats.map((stats) => scoreExactCameraStats(stats, params).q90 * 100));
  const worstRegionMaxCm = Math.max(...regionStats.map((stats) => stats.max * 100));
  const relativeTimePenalty = fastestMissionTimeSec > 0
    ? Math.max(0, solution.totalMissionTimeSec / fastestMissionTimeSec - 1)
    : 0;
  const breakdown = buildScoreBreakdown(
    "camera-partition-v1",
    {
      worstRegionQ90Cm,
      worstRegionMeanCm,
      worstRegionMaxCm,
      overallQ90Cm,
      overallQ75Cm,
      overallMeanCm,
      overallMaxCm,
      relativeTimePenalty,
      regionPenalty: solution.regionCount > 1 ? solution.regionCount - 1 : 0,
    },
    {
      worstRegionQ90Cm: 2.6,
      worstRegionMeanCm: 2.0,
      worstRegionMaxCm: 1.2,
      overallQ90Cm: 1.0,
      overallQ75Cm: 0.7,
      overallMeanCm: 0.4,
      overallMaxCm: 0.25,
      relativeTimePenalty: 0.35,
      regionPenalty: 0.12,
    },
  );
  return {
    score: breakdown.total,
    breakdown,
  };
}

function withExactSummary(
  params: FlightParams,
  solution: TerrainPartitionSolutionPreview,
  preview: ExactPartitionPreview,
  rankingSource: "backend-exact" | "frontend-exact",
  exactScore: number,
) {
  const exactQualityCost = preview.metricKind === "density"
    ? scoreExactLidarStats(preview.stats, params).qualityCost
    : scoreExactCameraStats(preview.stats, params).qualityCost;
  return {
    ...solution,
    rankingSource,
    exactScore,
    exactQualityCost,
    exactMissionTimeSec: solution.totalMissionTimeSec,
    exactMetricKind: preview.metricKind,
  } as TerrainPartitionSolutionPreview;
}

export async function evaluatePartitionSolutionCandidateExact(
  runtime: ExactRegionRuntime,
  args: ExactRegionCommonArgs & {
    polygonId: string;
    solution: TerrainPartitionSolutionPreview;
    fastestMissionTimeSec: number;
    rankingSource?: "backend-exact" | "frontend-exact";
    debugTrace?: boolean;
  },
): Promise<ExactPartitionSolutionEvaluation> {
  const evaluationStartedAt = performance.now();
  const rankingSource = args.rankingSource ?? "frontend-exact";
  const timeWeight = args.timeWeight ?? DEFAULT_TIME_WEIGHT;
  const qualityWeight = 1 - timeWeight;
  const exactOptimizeZoom = args.exactOptimizeZoom ?? DEFAULT_EXACT_OPTIMIZE_ZOOM;
  const solution = args.solution;
  const refinedRegions = [];
  const regionTraces: ExactRegionSearchTrace[] = [];
  const regionSearchElapsedMs: number[] = [];
  for (let regionIndex = 0; regionIndex < solution.regions.length; regionIndex++) {
    const region = solution.regions[regionIndex];
    const regionStartedAt = performance.now();
    const local = await optimizeBearingExact(runtime, {
      ...args,
      scopeId: `${args.polygonId}::${regionIndex}`,
      ring: region.ring as [number, number][],
      params: args.params,
      seedBearingDeg: region.bearingDeg,
      mode: "local",
      halfWindowDeg: 30,
    });
    const elapsedMs = performance.now() - regionStartedAt;
    regionSearchElapsedMs.push(elapsedMs);
    const best = local.best;
    refinedRegions.push({
      ...region,
      bearingDeg: best?.bearingDeg ?? region.bearingDeg,
      exactScore: best?.exactCost ?? null,
      exactSeedBearingDeg: local.seedBearingDeg,
    });
    if (args.debugTrace) {
      regionTraces.push({
        regionIndex,
        originalBearingDeg: region.bearingDeg,
        seedBearingDeg: local.seedBearingDeg,
        chosenBearingDeg: best?.bearingDeg ?? region.bearingDeg,
        chosenExactCost: best?.exactCost ?? null,
        searchMode: "local",
        halfWindowDeg: 30,
        lineSpacingM: local.lineSpacingM,
        elapsedMs,
        evaluatedBearings: local.evaluated,
      });
    }
  }
  const refinedSolution: TerrainPartitionSolutionPreview = {
    ...solution,
    regions: refinedRegions,
  };
  const previewStartedAt = performance.now();
  const preview = await evaluatePartitionSolutionExact(runtime, {
    ...args,
    solution: refinedSolution,
  });
  const previewElapsedMs = performance.now() - previewStartedAt;
  const partitionScore = isLidarParams(args.params)
    ? scoreLidarPartitionPreview(args.params, refinedSolution, preview, args.fastestMissionTimeSec)
    : scoreCameraPartitionPreview(args.params, refinedSolution, preview, args.fastestMissionTimeSec);
  const score = partitionScore.score;
  return {
    solution: withExactSummary(args.params, refinedSolution, preview, rankingSource, score),
    preview,
    score,
    debugTrace: args.debugTrace
      ? {
        signature: solution.signature,
        polygonId: args.polygonId,
        rankingSource,
        exactOptimizeZoom,
        timeWeight,
        qualityWeight,
        fastestMissionTimeSec: args.fastestMissionTimeSec,
        partitionScoreBreakdown: partitionScore.breakdown,
        preview,
        timings: {
          totalElapsedMs: performance.now() - evaluationStartedAt,
          previewElapsedMs,
          regionSearchElapsedMs,
        },
        regions: regionTraces,
      }
      : undefined,
  };
}

export async function rerankPartitionSolutionsExact(
  runtime: ExactRegionRuntime,
  args: ExactRegionCommonArgs & {
    polygonId: string;
    solutions: TerrainPartitionSolutionPreview[];
    rankingSource?: "backend-exact" | "frontend-exact";
    debugTrace?: boolean;
  },
): Promise<ExactPartitionRerankResult> {
  if (args.solutions.length === 0) return { bestIndex: 0, solutions: [], previewsBySignature: {} };
  const fastestMissionTimeSec = args.solutions.reduce(
    (best, solution) => Math.min(best, solution.totalMissionTimeSec),
    Number.POSITIVE_INFINITY,
  );
  const preparedSolutions = [...args.solutions];
  const previewsBySignature: Record<string, ExactPartitionPreview> = {};
  const debugBySignature: Record<string, ExactSolutionDebugTrace> = {};
  let bestIndex = 0;
  let bestScore = Number.POSITIVE_INFINITY;

  for (let index = 0; index < args.solutions.length; index++) {
    const evaluated = await evaluatePartitionSolutionCandidateExact(runtime, {
      ...args,
      solution: args.solutions[index],
      fastestMissionTimeSec,
    });
    previewsBySignature[args.solutions[index].signature] = evaluated.preview;
    preparedSolutions[index] = evaluated.solution;
    if (evaluated.debugTrace) {
      debugBySignature[args.solutions[index].signature] = evaluated.debugTrace;
    }
    if (evaluated.score < bestScore - 1e-9) {
      bestScore = evaluated.score;
      bestIndex = index;
    }
  }

  return {
    bestIndex,
    solutions: preparedSolutions,
    previewsBySignature,
    debugBySignature: args.debugTrace ? debugBySignature : undefined,
  };
}
