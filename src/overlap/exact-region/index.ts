import { DJI_ZENMUSE_P1_24MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM, SONY_RX1R2, SONY_RX1R3, SONY_A6100_20MM, forwardSpacingRotated, lineSpacingRotated } from "@/domain/camera";
import { DEFAULT_LIDAR, DEFAULT_LIDAR_MAX_RANGE_M, LIDAR_REGISTRY, getLidarMappingFovDeg, getLidarModel, lidarDeliverableDensity, lidarLineSpacing, lidarSinglePassDensity, lidarSwathWidth } from "@/domain/lidar";
import type { CameraModel, FlightParams, PlannedFlightGeometry, TerrainTile } from "@/domain/types";
import { build3DFlightPath, calculateOptimalTerrainZoom, isPointInRing, queryMinMaxElevationAlongPolylineWGS84, sampleCameraPositionsOnPlannedFlightGeometry } from "@/flight/geometry";
import { generatePlannedFlightGeometryForPolygon, summarizePlannedFlightGeometry } from "@/flight/plannedGeometry";
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
import { evaluatePreparedRegionOrientation, prepareRegionOrientationContext } from "@/utils/terrainPartitionObjective";

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
const HYBRID_SURROGATE_POSE_THRESHOLD = 300;
const HYBRID_SURROGATE_STEP_DEG = 10;
const HYBRID_SURROGATE_MAX_HOTSPOTS = 3;
const HYBRID_SURROGATE_MIN_SEPARATION_DEG = 30;
const HYBRID_SURROGATE_ADDITIONAL_HOTSPOT_RELATIVE_COST_WINDOW = 0.03;

function isExactRuntimeProfilingEnabled() {
  return typeof process !== "undefined" && process.env?.EXACT_RUNTIME_PROFILE === "1";
}

function logExactRuntimeProfile(message: string) {
  if (!isExactRuntimeProfilingEnabled()) return;
  console.error(`[exact-runtime-profile] ${message}`);
}
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
  candidateConcurrency?: number;
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

type ExactMissionEstimate = {
  missionTimeSec: number;
  totalLengthM: number;
  speedMps: number;
  lineCount: number;
};

type ExactBearingArtifacts = {
  normalizedBearingDeg: number;
  geometry: PlannedFlightGeometry;
  path3d: [number, number, number][][];
  mission: ExactMissionEstimate;
  camera?: {
    camera: CameraModel;
    poses: PoseMeters[];
    poseCameraIndices: Uint16Array;
  };
  lidar?: {
    strips: LidarStripMeters[];
    stripsByTile: Map<string, LidarStripMeters[]>;
  };
};

type ExactCandidateCacheEntry = {
  requestOrder: number;
  promise?: Promise<ExactBearingCandidate | null>;
};

type ExactBearingEvaluationContext = {
  runtime: ExactRegionRuntime;
  scopeId: string;
  ring: [number, number][];
  safeParams: FlightParams;
  altitudeMode: "legacy" | "min-clearance";
  minClearanceM: number;
  exactZoom: number;
  timeWeight: number;
  clipInnerBufferM: number;
  minOverlapForGsd: number;
  lineSpacingM: number;
  geometryTiles: TerrainTile[];
  tileRefs: ExactTileRef[];
  terrainTilesPromise?: Promise<Map<string, TileRGBA>>;
  terrainTilesWithHaloPromise?: Promise<Map<string, ExactTileWithHalo>>;
  artifactCache: Map<number, Promise<ExactBearingArtifacts>>;
  candidateCache: Map<number, ExactCandidateCacheEntry>;
  nextRequestOrder: number;
};

type SurrogateBasinCandidate = {
  bearingDeg: number;
  totalCost: number;
};

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

function getSpeedForParams(params: FlightParams) {
  return isLidarParams(params)
    ? (params.speedMps ?? getLidarModel(params.lidarKey).defaultSpeedMps)
    : DEFAULT_CAMERA_SPEED_MPS;
}

function buildLidarStripBuckets(
  tileRefs: ExactTileRef[],
  strips: LidarStripMeters[],
) {
  const buckets = new Map<string, LidarStripMeters[]>();
  for (const tileRef of tileRefs) {
    const affectingStrips: LidarStripMeters[] = [];
    for (const strip of strips) {
      if (lidarStripMayAffectTile(strip, tileRef)) {
        affectingStrips.push(strip);
      }
    }
    if (affectingStrips.length > 0) {
      buckets.set(tileKey(tileRef), affectingStrips);
    }
  }
  return buckets;
}

function getCandidateConcurrency(runtime: ExactRegionRuntime) {
  return Math.max(1, Math.floor(runtime.candidateConcurrency ?? 1));
}

function getTileConcurrency(runtime: ExactRegionRuntime) {
  return Math.max(1, Math.floor(runtime.candidateConcurrency ?? 1));
}

function estimatePoseCountFromMissionLength(totalFlightLineLengthM: number, photoSpacingM: number | null) {
  if (!(photoSpacingM && photoSpacingM > 0)) return 0;
  return totalFlightLineLengthM / photoSpacingM;
}

function axialDistanceDeg(left: number, right: number) {
  const diff = Math.abs(normalizeAxialBearingDeg(left - right));
  return Math.min(diff, 180 - diff);
}

function collectBestSurrogateBasins(
  candidates: SurrogateBasinCandidate[],
  maxHotspots: number,
  minSeparationDeg: number,
  additionalHotspotRelativeCostWindow: number,
) {
  if (candidates.length === 0) return [] as SurrogateBasinCandidate[];
  const sorted = [...candidates].sort((left, right) => left.totalCost - right.totalCost || left.bearingDeg - right.bearingDeg);
  const selected: SurrogateBasinCandidate[] = [];
  const bestCost = sorted[0].totalCost;
  for (const candidate of sorted) {
    if (
      selected.length > 0
      && candidate.totalCost > bestCost * (1 + additionalHotspotRelativeCostWindow)
    ) {
      break;
    }
    if (selected.every((selectedCandidate) => axialDistanceDeg(selectedCandidate.bearingDeg, candidate.bearingDeg) >= minSeparationDeg)) {
      selected.push(candidate);
    }
    if (selected.length >= maxHotspots) break;
  }
  return selected;
}

async function mapWithConcurrencyLimit<T, R>(
  values: readonly T[],
  maxConcurrency: number,
  mapper: (value: T, index: number) => Promise<R>,
) {
  const results = new Array<R>(values.length);
  let nextIndex = 0;
  const workerCount = Math.min(Math.max(1, maxConcurrency), values.length);

  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      while (nextIndex < values.length) {
        const currentIndex = nextIndex;
        nextIndex += 1;
        results[currentIndex] = await mapper(values[currentIndex], currentIndex);
      }
    }),
  );

  return results;
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

async function createExactBearingEvaluationContext(
  runtime: ExactRegionRuntime,
  args: ExactRegionCommonArgs,
  safeParams: FlightParams,
): Promise<ExactBearingEvaluationContext> {
  const lineSpacingM = getLineSpacingForParams(safeParams);
  const geometryTiles = await resolveGeometryTiles(runtime, args.ring, args.geometryTiles);
  const exactZoom = args.exactOptimizeZoom ?? DEFAULT_EXACT_OPTIMIZE_ZOOM;
  const seen = new Set<string>();
  const tileRefs: ExactTileRef[] = [];
  for (const tile of tilesCoveringPolygon({ ring: args.ring }, exactZoom)) {
    const ref = normalizeTileRef({ z: exactZoom, x: tile.x, y: tile.y });
    const key = tileKey(ref);
    if (seen.has(key)) continue;
    seen.add(key);
    tileRefs.push(ref);
  }
  return {
    runtime,
    scopeId: args.scopeId,
    ring: args.ring,
    safeParams,
    altitudeMode: args.altitudeMode,
    minClearanceM: args.minClearanceM,
    exactZoom,
    timeWeight: args.timeWeight ?? DEFAULT_TIME_WEIGHT,
    clipInnerBufferM: args.clipInnerBufferM ?? 0,
    minOverlapForGsd: args.minOverlapForGsd ?? DEFAULT_MIN_OVERLAP_FOR_GSD,
    lineSpacingM,
    geometryTiles,
    tileRefs,
    artifactCache: new Map(),
    candidateCache: new Map(),
    nextRequestOrder: 0,
  };
}

function getTerrainTilesForContext(context: ExactBearingEvaluationContext) {
  if (!context.terrainTilesPromise) {
    context.terrainTilesPromise = context.runtime.terrainProvider.getTerrainTiles(context.tileRefs);
  }
  return context.terrainTilesPromise;
}

function getTerrainTilesWithHaloForContext(context: ExactBearingEvaluationContext) {
  if (!context.terrainTilesWithHaloPromise) {
    context.terrainTilesWithHaloPromise = context.runtime.terrainProvider.getTerrainTilesWithHalo(context.tileRefs, 1);
  }
  return context.terrainTilesWithHaloPromise;
}

async function getExactBearingArtifacts(
  context: ExactBearingEvaluationContext,
  bearingDeg: number,
) {
  const normalizedBearingDeg = normalizeAxialBearingDeg(bearingDeg);
  const cacheKey = Math.round(normalizedBearingDeg * 1000);
  if (!context.artifactCache.has(cacheKey)) {
    context.artifactCache.set(cacheKey, (async (): Promise<ExactBearingArtifacts> => {
      const startedAt = performance.now();
      const geometry = generatePlannedFlightGeometryForPolygon(
        context.ring,
        normalizedBearingDeg,
        context.lineSpacingM,
        context.safeParams,
      );
      const path3d = geometry.flightLines.length > 0
        ? build3DFlightPath(
          geometry,
          context.geometryTiles,
          context.lineSpacingM,
          {
            altitudeAGL: context.safeParams.altitudeAGL,
            mode: context.altitudeMode,
            minClearance: context.minClearanceM,
            preconnected: true,
          },
        )
        : [];
      const speedMps = getSpeedForParams(context.safeParams);
      const totalLengthM = path3dLengthMeters(path3d);
      const mission: ExactMissionEstimate = {
        missionTimeSec: totalLengthM / Math.max(0.1, speedMps),
        totalLengthM,
        speedMps,
        lineCount: geometry.sweepLines.length,
      };

      if (isLidarParams(context.safeParams)) {
        const model = getLidarModel(context.safeParams.lidarKey);
        const altitudeAGL = context.safeParams.altitudeAGL;
        const mappingFovDeg = getLidarMappingFovDeg(model, context.safeParams.mappingFovDeg);
        const speedMpsLidar = context.safeParams.speedMps ?? model.defaultSpeedMps;
        const returnMode = context.safeParams.lidarReturnMode ?? "single";
        const maxLidarRangeM = context.safeParams.maxLidarRangeM ?? model.defaultMaxRangeM ?? DEFAULT_LIDAR_MAX_RANGE_M;
        const frameRateHz = context.safeParams.lidarFrameRateHz ?? model.defaultFrameRateHz;
        const azimuthSectorCenterDeg = context.safeParams.lidarAzimuthSectorCenterDeg ?? model.defaultAzimuthSectorCenterDeg ?? 0;
        const boresightYawDeg = context.safeParams.lidarBoresightYawDeg ?? model.boresightYawDeg ?? 0;
        const boresightPitchDeg = context.safeParams.lidarBoresightPitchDeg ?? model.boresightPitchDeg ?? 0;
        const boresightRollDeg = context.safeParams.lidarBoresightRollDeg ?? model.boresightRollDeg ?? 0;
        const comparisonMode = context.safeParams.lidarComparisonMode ?? "first-return";
        const densityPerPass = lidarSinglePassDensity(model, altitudeAGL, speedMpsLidar, returnMode, mappingFovDeg);
        const halfFovTan = Math.tan((mappingFovDeg * Math.PI) / 360);
        const strips: LidarStripMeters[] = [];
        let passIndex = 0;
        for (const activeSweepLine of geometry.sweepLines) {
          if (!Array.isArray(activeSweepLine) || activeSweepLine.length < 2) continue;
          const sweepPath3d = build3DFlightPath(
            [activeSweepLine],
            context.geometryTiles,
            context.lineSpacingM,
            { altitudeAGL, mode: context.altitudeMode, minClearance: context.minClearanceM, turnExtendM: 0 },
          )[0];
          if (!Array.isArray(sweepPath3d) || sweepPath3d.length < 2) continue;
          const localPassIndex = passIndex++;
          for (let index = 1; index < sweepPath3d.length; index += 1) {
            const start = sweepPath3d[index - 1];
            const end = sweepPath3d[index];
            const [x1, y1] = lngLatToMeters(start[0], start[1]);
            const [x2, y2] = lngLatToMeters(end[0], end[1]);
            const terrainMin = queryMinMaxElevationAlongPolylineWGS84(
              [[start[0], start[1]], [end[0], end[1]]],
              context.geometryTiles,
              12,
            ).min;
            const maxSensorAltitude = Math.max(start[2], end[2]);
            const maxHalfWidth = Number.isFinite(terrainMin)
              ? Math.max(lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2, Math.max(1, (maxSensorAltitude - terrainMin) * halfFovTan))
              : lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2;
            strips.push({
              id: `${context.scopeId}-sweep-${localPassIndex}-seg-${index - 1}`,
              polygonId: context.scopeId,
              x1,
              y1,
              z1: start[2],
              x2,
              y2,
              z2: end[2],
              plannedAltitudeAGL: altitudeAGL,
              halfWidthM: maxHalfWidth,
              densityPerPass,
              speedMps: speedMpsLidar,
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
        const result = {
          normalizedBearingDeg,
          geometry,
          path3d,
          mission,
          lidar: {
            strips,
            stripsByTile: buildLidarStripBuckets(context.tileRefs, strips),
          },
        };
        logExactRuntimeProfile(
          `bearing=${normalizedBearingDeg.toFixed(2)} phase=artifacts payload=lidar elapsedMs=${(performance.now() - startedAt).toFixed(1)} sweeps=${geometry.sweepLines.length} pathSegments=${path3d.length} strips=${strips.length}`,
        );
        return result;
      }

      const camera = getCameraForParams(context.safeParams);
      const photoSpacing = getForwardSpacingForParams(context.safeParams);
      const yawOffset = context.safeParams.cameraYawOffsetDeg ?? 0;
      const normalizeDeg = (value: number) => ((value % 360) + 360) % 360;
      const cameraPositions = photoSpacing && photoSpacing > 0
        ? sampleCameraPositionsOnPlannedFlightGeometry(geometry, path3d, photoSpacing)
        : [];
      const filteredPositions = context.ring.length >= 3
        ? cameraPositions.filter(([lng, lat]) => isPointInRing(lng, lat, context.ring))
        : cameraPositions;
      const poses = filteredPositions.map(([lng, lat, altMSL, yawDeg], index) => {
        const [x, y] = lngLatToMeters(lng, lat);
        return {
          id: `exact_pose_${context.scopeId}_${index}`,
          x,
          y,
          z: altMSL,
          omega_deg: 0,
          phi_deg: 0,
          kappa_deg: normalizeDeg(-yawDeg + yawOffset),
          polygonId: context.scopeId,
        };
      });
      const result = {
        normalizedBearingDeg,
        geometry,
        path3d,
        mission,
        camera: {
          camera,
          poses,
          poseCameraIndices: new Uint16Array(poses.length),
        },
      };
      logExactRuntimeProfile(
        `bearing=${normalizedBearingDeg.toFixed(2)} phase=artifacts payload=camera elapsedMs=${(performance.now() - startedAt).toFixed(1)} sweeps=${geometry.sweepLines.length} pathSegments=${path3d.length} poses=${poses.length}`,
      );
      return result;
    })());
  }
  return context.artifactCache.get(cacheKey)!;
}

async function evaluateRegionBearingExactWithContext(
  context: ExactBearingEvaluationContext,
  bearingDeg: number,
): Promise<ExactBearingCandidate | null> {
  const candidateStartedAt = performance.now();
  if (context.runtime.yieldToEventLoop) {
    await context.runtime.yieldToEventLoop();
  }
  const exactQualityWeight = 1 - context.timeWeight;
  const artifacts = await getExactBearingArtifacts(context, bearingDeg);

  if (artifacts.lidar) {
    if (!artifacts.lidar.strips.length) return null;
    const terrainStartedAt = performance.now();
    const tileMapWithHalo = await getTerrainTilesWithHaloForContext(context);
    const terrainElapsedMs = performance.now() - terrainStartedAt;
    const tileEvalStartedAt = performance.now();
    const tileResults = await mapWithConcurrencyLimit(
      context.tileRefs,
      getTileConcurrency(context.runtime),
      async (tileRef) => {
        const key = tileKey(tileRef);
        const tileBundle = tileMapWithHalo.get(key);
        const tileStrips = artifacts.lidar!.stripsByTile.get(key);
        if (!tileBundle || !tileStrips || tileStrips.length === 0) return null;
        const response = await context.runtime.tileEvaluator.evaluateLidarTile({
          tile: tileBundle.tile,
          demTile: tileBundle.demTile,
          polygons: [{ id: context.scopeId, ring: context.ring }],
          strips: tileStrips,
          options: { clipInnerBufferM: context.clipInnerBufferM },
        });
        const densityStats = response.perPolygon?.find((entry) => entry.polygonId === context.scopeId)?.densityStats;
        return densityStats
          ? { touched: true, stats: densityStats }
          : { touched: true, stats: null };
      },
    );
    const perTileStats: GSDStats[] = [];
    let touchedTileCount = 0;
    for (const tileResult of tileResults) {
      if (!tileResult) continue;
      touchedTileCount += tileResult.touched ? 1 : 0;
      if (tileResult.stats) perTileStats.push(tileResult.stats);
    }
    if (!perTileStats.length) return null;
    const tileEvalElapsedMs = performance.now() - tileEvalStartedAt;
    const stats = aggregateMetricStats(perTileStats);
    const scored = scoreExactLidarStats(stats, context.safeParams);
    const normalizedTimeCost = artifacts.mission.missionTimeSec / 180;
    const costBreakdown = buildExactCostBreakdown(
      scored.breakdown.modelVersion,
      exactQualityWeight,
      scored.qualityCost,
      context.timeWeight,
      normalizedTimeCost,
    );
    const missionBreakdown: ExactMissionBreakdown = {
      totalLengthM: artifacts.mission.totalLengthM,
      speedMps: artifacts.mission.speedMps,
      lineCount: artifacts.mission.lineCount,
      sampleCount: new Set(artifacts.lidar.strips.map((strip) => strip.passIndex ?? -1)).size,
      segmentCount: artifacts.lidar.strips.length,
      sampleLabel: "Flight lines",
    };
    const result: ExactBearingCandidate = {
      bearingDeg: artifacts.normalizedBearingDeg,
      exactCost: costBreakdown.total,
      qualityCost: scored.qualityCost,
      missionTimeSec: artifacts.mission.missionTimeSec,
      normalizedTimeCost,
      metricKind: "density",
      stats,
      diagnostics: {
        qualityCost: scored.qualityCost,
        missionTimeSec: artifacts.mission.missionTimeSec,
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
    logExactRuntimeProfile(
      `bearing=${artifacts.normalizedBearingDeg.toFixed(2)} payload=lidar totalMs=${(performance.now() - candidateStartedAt).toFixed(1)} terrainMs=${terrainElapsedMs.toFixed(1)} tileEvalMs=${tileEvalElapsedMs.toFixed(1)} touchedTiles=${touchedTileCount} strips=${artifacts.lidar.strips.length} exactCost=${result.exactCost.toFixed(4)}`,
    );
    return result;
  }

  const cameraArtifacts = artifacts.camera;
  if (!cameraArtifacts || cameraArtifacts.poses.length === 0) return null;
  const terrainStartedAt = performance.now();
  const tileMap = await getTerrainTilesForContext(context);
  const terrainElapsedMs = performance.now() - terrainStartedAt;
  const tileEvalStartedAt = performance.now();
  const tileResults = await mapWithConcurrencyLimit(
    context.tileRefs,
    getTileConcurrency(context.runtime),
    async (tileRef) => {
      const key = tileKey(tileRef);
      const tile = tileMap.get(key);
      if (!tile) return null;
      const response = await context.runtime.tileEvaluator.evaluateCameraTile({
        tile,
        polygons: [{ id: context.scopeId, ring: context.ring }],
        poses: cameraArtifacts.poses,
        cameras: [cameraArtifacts.camera],
        poseCameraIndices: cameraArtifacts.poseCameraIndices,
        camera: undefined,
        options: {
          clipInnerBufferM: context.clipInnerBufferM,
          minOverlapForGsd: context.minOverlapForGsd,
        },
      });
      const gsdStats = response.perPolygon?.find((entry) => entry.polygonId === context.scopeId)?.gsdStats;
      return gsdStats
        ? { touched: true, stats: gsdStats }
        : { touched: true, stats: null };
    },
  );
  const perTileStats: GSDStats[] = [];
  let touchedTileCount = 0;
  for (const tileResult of tileResults) {
    if (!tileResult) continue;
    touchedTileCount += tileResult.touched ? 1 : 0;
    if (tileResult.stats) perTileStats.push(tileResult.stats);
  }
  if (!perTileStats.length) return null;
  const tileEvalElapsedMs = performance.now() - tileEvalStartedAt;
  const stats = aggregateMetricStats(perTileStats);
  const scored = scoreExactCameraStats(stats, context.safeParams);
  const normalizedTimeCost = artifacts.mission.missionTimeSec / 180;
  const costBreakdown = buildExactCostBreakdown(
    scored.breakdown.modelVersion,
    exactQualityWeight,
    scored.qualityCost,
    context.timeWeight,
    normalizedTimeCost,
  );
  const missionBreakdown: ExactMissionBreakdown = {
    totalLengthM: artifacts.mission.totalLengthM,
    speedMps: artifacts.mission.speedMps,
    lineCount: artifacts.mission.lineCount,
    sampleCount: cameraArtifacts.poses.length,
    sampleLabel: "Images",
  };
    const result: ExactBearingCandidate = {
    bearingDeg: artifacts.normalizedBearingDeg,
    exactCost: costBreakdown.total,
    qualityCost: scored.qualityCost,
    missionTimeSec: artifacts.mission.missionTimeSec,
    normalizedTimeCost,
    metricKind: "gsd",
    stats,
    diagnostics: {
      qualityCost: scored.qualityCost,
      missionTimeSec: artifacts.mission.missionTimeSec,
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
  logExactRuntimeProfile(
    `bearing=${artifacts.normalizedBearingDeg.toFixed(2)} payload=camera totalMs=${(performance.now() - candidateStartedAt).toFixed(1)} terrainMs=${terrainElapsedMs.toFixed(1)} tileEvalMs=${tileEvalElapsedMs.toFixed(1)} touchedTiles=${touchedTileCount} poses=${cameraArtifacts.poses.length} exactCost=${result.exactCost.toFixed(4)}`,
  );
  return result;
}

export async function evaluateRegionBearingExact(
  runtime: ExactRegionRuntime,
  args: ExactRegionCommonArgs & { bearingDeg: number },
): Promise<ExactBearingCandidate | null> {
  const safeParams = { ...args.params, useCustomBearing: false, customBearingDeg: undefined };
  const context = await createExactBearingEvaluationContext(runtime, args, safeParams);
  return evaluateRegionBearingExactWithContext(context, args.bearingDeg);
}

async function collectEvaluatedCandidates(context: ExactBearingEvaluationContext) {
  return (await Promise.all(
    [...context.candidateCache.values()].map(async (entry) => ({
      requestOrder: entry.requestOrder,
      candidate: entry.promise ? await entry.promise : null,
    })),
  ))
    .filter((value): value is { requestOrder: number; candidate: ExactBearingCandidate } => value.candidate !== null)
    .sort((left, right) => {
      if (left.candidate.exactCost !== right.candidate.exactCost) {
        return left.candidate.exactCost - right.candidate.exactCost;
      }
      return left.requestOrder - right.requestOrder;
    })
    .map((value) => value.candidate);
}

async function runExactBearingSearchWithContext(
  context: ExactBearingEvaluationContext,
  search: {
    seedBearingDeg: number;
    mode: ExactSearchMode;
    halfWindowDeg: number;
  },
) {
  const searchStartedAt = performance.now();
  const normalizedSeedBearingDeg = Number.isFinite(search.seedBearingDeg)
    ? normalizeAxialBearingDeg(search.seedBearingDeg)
    : 0;
  const coarseOffsets = [-10, 0, 10];
  const refineStepsDeg = [8, 4, 2, 1].filter((step) => step <= search.halfWindowDeg);
  const globalCoarseBearings = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165];
  const minImprovement = 1e-4;
  const candidateConcurrency = getCandidateConcurrency(context.runtime);

  const getCandidateEntry = (bearingDeg: number) => {
    const normalized = normalizeAxialBearingDeg(bearingDeg);
    const cacheKey = Math.round(normalized * 1000);
    let entry = context.candidateCache.get(cacheKey);
    if (!entry) {
      entry = { requestOrder: context.nextRequestOrder++ };
      context.candidateCache.set(cacheKey, entry);
    }
    return { normalized, entry };
  };

  const evaluateBearing = (bearingDeg: number) => {
    const { normalized, entry } = getCandidateEntry(bearingDeg);
    if (!entry.promise) {
      entry.promise = evaluateRegionBearingExactWithContext(context, normalized);
    }
    return entry.promise;
  };

  const evaluateOffset = (offsetDeg: number) => {
    if (Math.abs(offsetDeg) > search.halfWindowDeg + 1e-6) return Promise.resolve<ExactBearingCandidate | null>(null);
    return evaluateBearing(normalizedSeedBearingDeg + offsetDeg);
  };

  let best: ExactBearingCandidate | null = null;
  let bestOffset = 0;
  if (search.mode === "global") {
    const coarseCandidates = Array.from(new Set([...globalCoarseBearings, Math.round(normalizedSeedBearingDeg * 10) / 10]));
    const coarseStartedAt = performance.now();
    const coarseResults = await mapWithConcurrencyLimit(
      coarseCandidates,
      candidateConcurrency,
      (bearingDeg) => evaluateBearing(bearingDeg),
    );
    for (const candidate of coarseResults) {
      if (candidate && (!best || candidate.exactCost < best.exactCost)) best = candidate;
    }
    logExactRuntimeProfile(
      `search scopeId=${context.scopeId} mode=global coarseCandidates=${coarseCandidates.length} coarseMs=${(performance.now() - coarseStartedAt).toFixed(1)} bestBearing=${best?.bearingDeg?.toFixed(2) ?? "none"} bestCost=${best?.exactCost?.toFixed(4) ?? "none"}`,
    );
  } else {
    const coarseCandidateOffsets = coarseOffsets.filter((offset) => Math.abs(offset) <= search.halfWindowDeg + 1e-6);
    const coarseStartedAt = performance.now();
    const coarseResults = await mapWithConcurrencyLimit(
      coarseCandidateOffsets,
      candidateConcurrency,
      (offsetDeg) => evaluateOffset(offsetDeg),
    );
    for (let index = 0; index < coarseCandidateOffsets.length; index += 1) {
      const candidate = coarseResults[index];
      if (candidate && (!best || candidate.exactCost < best.exactCost)) {
        best = candidate;
        bestOffset = coarseCandidateOffsets[index];
      }
    }
    logExactRuntimeProfile(
      `search scopeId=${context.scopeId} mode=local seed=${normalizedSeedBearingDeg.toFixed(2)} coarseCandidates=${coarseCandidateOffsets.length} coarseMs=${(performance.now() - coarseStartedAt).toFixed(1)} bestBearing=${best?.bearingDeg?.toFixed(2) ?? "none"} bestCost=${best?.exactCost?.toFixed(4) ?? "none"}`,
    );
  }

  if (best) {
    for (const stepDeg of refineStepsDeg) {
      const refineStepStartedAt = performance.now();
      let refineIterations = 0;
      let improved = true;
      while (improved) {
        improved = false;
        refineIterations += 1;
        const currentBest: ExactBearingCandidate = best;
        const neighborSpecs: Array<{ offsetDeg: number; promise: Promise<ExactBearingCandidate | null> }> = search.mode === "global"
          ? [
            { offsetDeg: 0, promise: evaluateBearing(currentBest.bearingDeg - stepDeg) },
            { offsetDeg: 0, promise: evaluateBearing(currentBest.bearingDeg + stepDeg) },
          ]
          : [
            { offsetDeg: bestOffset - stepDeg, promise: evaluateOffset(bestOffset - stepDeg) },
            { offsetDeg: bestOffset + stepDeg, promise: evaluateOffset(bestOffset + stepDeg) },
          ];
        const neighborResults = await mapWithConcurrencyLimit(
          neighborSpecs,
          Math.min(candidateConcurrency, neighborSpecs.length),
          (spec) => spec.promise,
        );
        let nextBest: { offsetDeg: number; candidate: ExactBearingCandidate } | null = null;
        for (let index = 0; index < neighborSpecs.length; index += 1) {
          const candidate = neighborResults[index];
          if (!candidate) continue;
          if (!nextBest || candidate.exactCost < nextBest.candidate.exactCost) {
            nextBest = { offsetDeg: neighborSpecs[index].offsetDeg, candidate };
          }
        }
        if (nextBest && nextBest.candidate.exactCost + minImprovement < currentBest.exactCost) {
          best = nextBest.candidate;
          if (search.mode !== "global") bestOffset = nextBest.offsetDeg;
          improved = true;
        }
      }
      logExactRuntimeProfile(
        `search scopeId=${context.scopeId} mode=${search.mode} stepDeg=${stepDeg} refineIterations=${refineIterations} refineMs=${(performance.now() - refineStepStartedAt).toFixed(1)} bestBearing=${best.bearingDeg.toFixed(2)} bestCost=${best.exactCost.toFixed(4)}`,
      );
    }
  }

  const evaluated = await collectEvaluatedCandidates(context);
  logExactRuntimeProfile(
    `search scopeId=${context.scopeId} mode=${search.mode} seed=${normalizedSeedBearingDeg.toFixed(2)} totalMs=${(performance.now() - searchStartedAt).toFixed(1)} evaluated=${evaluated.length} winner=${best?.bearingDeg?.toFixed(2) ?? "none"} winnerCost=${best?.exactCost?.toFixed(4) ?? "none"}`,
  );
  return {
    best,
    evaluated,
    seedBearingDeg: normalizedSeedBearingDeg,
  };
}

function getHybridHotspotBearings(
  context: ExactBearingEvaluationContext,
  normalizedSeedBearingDeg: number,
) {
  if (isLidarParams(context.safeParams)) {
    logExactRuntimeProfile(`hybrid-search scopeId=${context.scopeId} skipped=lidar`);
    return null;
  }
  const photoSpacingM = getForwardSpacingForParams(context.safeParams);
  if (!(photoSpacingM && photoSpacingM > 0)) {
    logExactRuntimeProfile(`hybrid-search scopeId=${context.scopeId} skipped=no-photo-spacing`);
    return null;
  }

  const coarseBearings = Array.from({ length: Math.ceil(180 / HYBRID_SURROGATE_STEP_DEG) }, (_, index) => index * HYBRID_SURROGATE_STEP_DEG);
  const prepared = prepareRegionOrientationContext(
    context.ring,
    context.geometryTiles,
    context.safeParams,
    { tradeoff: 1 - context.timeWeight },
  );

  let estimatedPoseCount = 0;
  let surrogateCandidates: SurrogateBasinCandidate[] = [];
  if (prepared) {
    const seedObjective = evaluatePreparedRegionOrientation(prepared, normalizedSeedBearingDeg);
    estimatedPoseCount = estimatePoseCountFromMissionLength(seedObjective?.flightTime.totalFlightLineLengthM ?? 0, photoSpacingM);
    surrogateCandidates = coarseBearings
      .map((bearingDeg) => {
        const objective = evaluatePreparedRegionOrientation(prepared, bearingDeg);
        if (!objective) return null;
        return { bearingDeg: objective.bearingDeg, totalCost: objective.totalCost } satisfies SurrogateBasinCandidate;
      })
      .filter((candidate): candidate is SurrogateBasinCandidate => candidate !== null);
  } else {
    logExactRuntimeProfile(`hybrid-search scopeId=${context.scopeId} fallback=geometry-only`);
    surrogateCandidates = coarseBearings.map((bearingDeg) => {
      const geometry = generatePlannedFlightGeometryForPolygon(
        context.ring,
        bearingDeg,
        context.lineSpacingM,
        context.safeParams,
      );
      const summary = summarizePlannedFlightGeometry(geometry);
      if (axialDistanceDeg(bearingDeg, normalizedSeedBearingDeg) < HYBRID_SURROGATE_STEP_DEG / 2) {
        estimatedPoseCount = estimatePoseCountFromMissionLength(summary.totalFlightLineLengthM, photoSpacingM);
      }
      return {
        bearingDeg,
        totalCost: summary.totalConnectedPathLengthM,
      } satisfies SurrogateBasinCandidate;
    });
  }

  if (estimatedPoseCount <= HYBRID_SURROGATE_POSE_THRESHOLD) {
    logExactRuntimeProfile(
      `hybrid-search scopeId=${context.scopeId} skipped=below-threshold estimatedPoseCount=${estimatedPoseCount.toFixed(1)}`,
    );
    return null;
  }
  if (surrogateCandidates.length === 0) {
    logExactRuntimeProfile(`hybrid-search scopeId=${context.scopeId} skipped=no-surrogate-candidates`);
    return null;
  }

  const localMinima = surrogateCandidates.filter((candidate, index) => {
    const left = surrogateCandidates[(index + surrogateCandidates.length - 1) % surrogateCandidates.length];
    const right = surrogateCandidates[(index + 1) % surrogateCandidates.length];
    return candidate.totalCost <= left.totalCost && candidate.totalCost <= right.totalCost;
  });
  const hotspotCandidates = collectBestSurrogateBasins(
    localMinima.length > 0 ? localMinima : surrogateCandidates,
    HYBRID_SURROGATE_MAX_HOTSPOTS,
    HYBRID_SURROGATE_MIN_SEPARATION_DEG,
    HYBRID_SURROGATE_ADDITIONAL_HOTSPOT_RELATIVE_COST_WINDOW,
  );
  const hotspotBearings = hotspotCandidates.map((candidate) => candidate.bearingDeg);
  if (hotspotBearings.length === 0) {
    logExactRuntimeProfile(`hybrid-search scopeId=${context.scopeId} skipped=no-hotspots estimatedPoseCount=${estimatedPoseCount.toFixed(1)}`);
    return null;
  }

  logExactRuntimeProfile(
    `hybrid-search scopeId=${context.scopeId} estimatedPoseCount=${estimatedPoseCount.toFixed(1)} hotspots=${hotspotCandidates.map((candidate) => `${candidate.bearingDeg.toFixed(1)}@${candidate.totalCost.toFixed(4)}`).join(",")} costWindow=${(HYBRID_SURROGATE_ADDITIONAL_HOTSPOT_RELATIVE_COST_WINDOW * 100).toFixed(1)}%`,
  );
  return hotspotBearings;
}

export async function optimizeBearingExact(
  runtime: ExactRegionRuntime,
  args: ExactRegionCommonArgs & {
    seedBearingDeg: number;
    mode?: ExactSearchMode;
    halfWindowDeg?: number;
  },
): Promise<ExactBearingSearchResult> {
  const optimizeStartedAt = performance.now();
  const normalizedSeedBearingDeg = Number.isFinite(args.seedBearingDeg)
    ? normalizeAxialBearingDeg(args.seedBearingDeg)
    : 0;
  const safeParams = { ...args.params, useCustomBearing: false, customBearingDeg: undefined };
  const contextStartedAt = performance.now();
  const context = await createExactBearingEvaluationContext(runtime, args, safeParams);
  logExactRuntimeProfile(
    `optimize scopeId=${context.scopeId} contextMs=${(performance.now() - contextStartedAt).toFixed(1)} lineSpacingM=${context.lineSpacingM.toFixed(2)} timeWeight=${context.timeWeight.toFixed(3)}`,
  );
  const mode = args.mode ?? "local";
  const halfWindowDeg = Math.max(1, args.halfWindowDeg ?? 30);

  let best: ExactBearingCandidate | null = null;
  if (mode === "global") {
    const hotspotStartedAt = performance.now();
    const hotspotBearings = getHybridHotspotBearings(context, normalizedSeedBearingDeg);
    logExactRuntimeProfile(
      `optimize scopeId=${context.scopeId} hotspotSelectionMs=${(performance.now() - hotspotStartedAt).toFixed(1)} hotspotCount=${hotspotBearings?.length ?? 0}`,
    );
    if (hotspotBearings && hotspotBearings.length > 0) {
      const hybridStartedAt = performance.now();
      for (const hotspotBearingDeg of hotspotBearings) {
        const localSearchStartedAt = performance.now();
        const search = await runExactBearingSearchWithContext(context, {
          seedBearingDeg: hotspotBearingDeg,
          mode: "local",
          halfWindowDeg,
        });
        logExactRuntimeProfile(
          `optimize scopeId=${context.scopeId} hotspot=${hotspotBearingDeg.toFixed(2)} localSearchMs=${(performance.now() - localSearchStartedAt).toFixed(1)} bestBearing=${search.best?.bearingDeg?.toFixed(2) ?? "none"} bestCost=${search.best?.exactCost?.toFixed(4) ?? "none"}`,
        );
        if (search.best && (!best || search.best.exactCost < best.exactCost)) {
          best = search.best;
        }
      }
      logExactRuntimeProfile(
        `optimize scopeId=${context.scopeId} hybridTotalMs=${(performance.now() - hybridStartedAt).toFixed(1)} winner=${best?.bearingDeg?.toFixed(2) ?? "none"} winnerCost=${best?.exactCost?.toFixed(4) ?? "none"}`,
      );
    } else {
      const globalSearchStartedAt = performance.now();
      const search = await runExactBearingSearchWithContext(context, {
        seedBearingDeg: normalizedSeedBearingDeg,
        mode: "global",
        halfWindowDeg,
      });
      best = search.best;
      logExactRuntimeProfile(
        `optimize scopeId=${context.scopeId} globalSearchMs=${(performance.now() - globalSearchStartedAt).toFixed(1)} winner=${best?.bearingDeg?.toFixed(2) ?? "none"} winnerCost=${best?.exactCost?.toFixed(4) ?? "none"}`,
      );
    }
  } else {
    const localSearchStartedAt = performance.now();
    const search = await runExactBearingSearchWithContext(context, {
      seedBearingDeg: normalizedSeedBearingDeg,
      mode,
      halfWindowDeg,
    });
    best = search.best;
    logExactRuntimeProfile(
      `optimize scopeId=${context.scopeId} localSearchMs=${(performance.now() - localSearchStartedAt).toFixed(1)} winner=${best?.bearingDeg?.toFixed(2) ?? "none"} winnerCost=${best?.exactCost?.toFixed(4) ?? "none"}`,
    );
  }

  const evaluated = await collectEvaluatedCandidates(context);
  logExactRuntimeProfile(
    `optimize scopeId=${context.scopeId} mode=${mode} totalMs=${(performance.now() - optimizeStartedAt).toFixed(1)} evaluated=${evaluated.length} winner=${best?.bearingDeg?.toFixed(2) ?? "none"} winnerCost=${best?.exactCost?.toFixed(4) ?? "none"}`,
  );
  return {
    best,
    evaluated,
    seedBearingDeg: normalizedSeedBearingDeg,
    lineSpacingM: context.lineSpacingM,
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
    const tileResults = await mapWithConcurrencyLimit(
      tileRefs,
      getTileConcurrency(runtime),
      async (tileRef) => {
        const tileBundle = tileMapWithHalo.get(tileKey(tileRef));
        if (!tileBundle) return null;
        const tileStrips = strips.filter((strip) => lidarStripMayAffectTile(strip, tileRef));
        if (!tileStrips.length) return null;
        return runtime.tileEvaluator.evaluateLidarTile({
          tile: tileBundle.tile,
          demTile: tileBundle.demTile,
          polygons: virtualPolygons.map(({ id, ring }) => ({ id, ring })),
          strips: tileStrips,
          options: { clipInnerBufferM: args.clipInnerBufferM ?? 0 },
        });
      },
    );
    tileResults.forEach((result) => {
      if (!result) return;
      (result.perPolygon ?? []).forEach((polyStats) => {
        if (!polyStats.densityStats) return;
        const list = perRegionStats.get(polyStats.polygonId) ?? [];
        list.push(polyStats.densityStats);
        perRegionStats.set(polyStats.polygonId, list);
      });
    });
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
    const cameraPositions = sampleCameraPositionsOnPlannedFlightGeometry(geometry, path3d, photoSpacing ?? 0);
    const filtered = region.ring.length >= 3
      ? cameraPositions.filter(([lng, lat]) => isPointInRing(lng, lat, region.ring))
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
  const tileResults = await mapWithConcurrencyLimit(
    tileRefs,
    getTileConcurrency(runtime),
    async (tileRef) => {
      const tile = tileMap.get(tileKey(tileRef));
      if (!tile) return null;
      return runtime.tileEvaluator.evaluateCameraTile({
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
    },
  );
  tileResults.forEach((result) => {
    if (!result) return;
    (result.perPolygon ?? []).forEach((polyStats) => {
      if (!polyStats.gsdStats) return;
      const list = perRegionStats.get(polyStats.polygonId) ?? [];
      list.push(polyStats.gsdStats);
      perRegionStats.set(polyStats.polygonId, list);
    });
  });
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
