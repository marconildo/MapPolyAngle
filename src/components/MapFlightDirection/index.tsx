// src/components/MapFlightDirection/index.tsx
/***********************************************************************
 * MapFlightDirection.tsx
 ***********************************************************************/
import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Map as MapboxMap, LngLatLike } from 'mapbox-gl';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import { MapboxOverlay } from '@deck.gl/mapbox';

import 'mapbox-gl/dist/mapbox-gl.css';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';

import { forceReloadTerrainDemSourceOnMap, reassertTerrainDemSourceOnMap, setTerrainDemSourceOnMap, useMapInitialization, waitForTerrainDemSourceOnMap } from './hooks/useMapInitialization';
import { usePolygonAnalysis } from './hooks/usePolygonAnalysis';
import {
  addFlightLinesForPolygon,
  animateProcessingPerimeter,
  removeFlightLinesForPolygon,
  clearAllFlightLines,
  removeTriggerPointsForPolygon,
  clearAllTriggerPoints,
  setProcessingPerimeterPolygons,
  setSelectedPolygonHighlight,
  setNonSelectedPolygonDimMask,
  setFlightLineSelectionEmphasis,
  renderFlightLinesForPolygon,
} from './utils/mapbox-layers';
import { update3DPathLayer, remove3DPathLayer, update3DCameraPointsLayer, remove3DCameraPointsLayer, update3DTriggerPointsLayer, remove3DTriggerPointsLayer } from './utils/deckgl-layers';
import { build3DFlightPath, calculateOptimalTerrainZoom, isPointInRing, sampleCameraPositionsOnPlannedFlightGeometry, extendFlightLineForTurnRunout, groupFlightLinesForTraversal, queryMinMaxElevationAlongPolylineWGS84 } from './utils/geometry';
import { PolygonAnalysisResult, PolygonParams } from './types';
import { parseKmlPolygons, calculateKmlBounds, extractKmlFromKmz } from '@/utils/kml';
import { SONY_RX1R2, SONY_RX1R3, SONY_A6100_20MM, DJI_ZENMUSE_P1_24MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM, calculateGSD, forwardSpacingRotated, lineSpacingRotated } from '@/domain/camera';
import { DEFAULT_LIDAR, DEFAULT_LIDAR_MAX_RANGE_M, LIDAR_REGISTRY, getLidarMappingFovDeg, getLidarModel, lidarDeliverableDensity, lidarLineSpacing, lidarSinglePassDensity, lidarSwathWidth } from '@/domain/lidar';
import type { PlannedFlightGeometry } from '@/domain/types';
import type {
  BearingOverride,
  ImportedFlightplanArea,
  MapFlightDirectionAPI,
  PolygonHistoryState,
  PolygonMergeState,
  PolygonOperationTransaction,
  PolygonSnapshot,
  TerrainPartitionSolutionPreview,
  WingtraFreshExportConfig,
} from './api';
import { fetchTilesForPolygon } from './utils/terrain';
import { isTerrainPartitionBackendEnabled, solveTerrainPartitionWithBackend } from '@/services/terrainPartitionBackend';
import { isAreaSequenceBackendEnabled, optimizeAreaSequenceWithBackend } from '@/services/areaSequenceBackend';
import { isExactTerrainBackendEnabled, optimizeBearingWithBackend } from '@/services/exactTerrainBackend';
import type { TerrainSourceSelection } from '@/terrain/types';
import type { GSDStats, LidarStripMeters, PoseMeters } from '@/overlap/types';
import { lngLatToMeters, tileMetersBounds } from '@/overlap/mercator';
import type { ExactBearingCandidate as SharedExactBearingCandidate, ExactBearingSearchResult as SharedExactBearingSearchResult } from '@/overlap/exact-region';
// @ts-ignore Turf typings are inconsistent in this repo.
import * as turf from '@turf/turf';

import { shouldApplyAsyncPolygonUpdate, shouldRunAsyncGeneration } from '@/state/asyncUpdateGuard';
import { shouldConsumeClearAllEpoch } from '@/state/clearAllState';
import {
  applyPolygonSnapshotsToMetadata,
  clearPolygonOperationHistory,
  clearPolygonOperationRedo as clearPolygonOperationRedoState,
  clonePolygonSnapshot,
  collectAffectedPolygonIds,
  createEmptyPolygonOperationHistory,
  createIdlePolygonMergeState,
  createPolygonFeatureSnapshot,
  createPolygonHistoryState,
  derivePolygonMergeState,
  popRedoPolygonOperation,
  popUndoPolygonOperation,
  pushPolygonOperationTransaction,
  toMergeablePolygonFeature,
} from '@/state/polygonOperations';
import { PlannedGeometryWorkerController } from '@/flight/plannedGeometryController';

type ExactRegionModule = typeof import('@/overlap/exact-region');
type ExactBrowserRuntimeModule = typeof import('@/overlap/exactBrowserRuntime');
type TerrainFacePartitionModule = typeof import('@/utils/terrainFacePartition');
type TerrainPartitionGraphModule = typeof import('@/utils/terrainPartitionGraph');
type WingtraConvertModule = typeof import('@/interop/wingtra/convert');

let exactRegionModulePromise: Promise<ExactRegionModule> | null = null;
let exactBrowserRuntimeModulePromise: Promise<ExactBrowserRuntimeModule> | null = null;
let terrainFacePartitionModulePromise: Promise<TerrainFacePartitionModule> | null = null;
let terrainPartitionGraphModulePromise: Promise<TerrainPartitionGraphModule> | null = null;
let wingtraConvertModulePromise: Promise<WingtraConvertModule> | null = null;

const CAMERA_REGISTRY: Record<string, any> = {
  SONY_RX1R2,
  SONY_RX1R3,
  SONY_A6100_20MM,
  DJI_ZENMUSE_P1_24MM,
  ILX_LR1_INSPECT_85MM,
  MAP61_17MM,
  RGB61_24MM,
};
const DEFAULT_CAMERA = SONY_RX1R2;
const DEFAULT_PAYLOAD_KIND = 'camera';
const DEFAULT_ALTITUDE_AGL = 100;
const DEFAULT_FRONT_OVERLAP = 70;
const DEFAULT_SIDE_OVERLAP = 70;
const DEFAULT_PARTITION_TRADEOFF_SAMPLES = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85];
const DEFAULT_PARTITION_TARGET_TRADEOFF = 0.8;
const DEFAULT_EXPORT_SEQUENCE_MAX_HEIGHT_ABOVE_GROUND_M = 200;
const DEFAULT_WINGTRA_EXPORT_CRUISE_SPEED_MPS = 15.375008;
const EXACT_OPTIMIZE_ZOOM = 14;
const EXACT_MIN_OVERLAP_FOR_GSD = 3;
const EXACT_OPTIMIZE_TIME_WEIGHT = 0.1;
const TERRAIN_SPLIT_DEBUG = true;
const GEOMETRY_RING_EPSILON_DEG = 1e-7;

function loadExactRegionModule(): Promise<ExactRegionModule> {
  if (!exactRegionModulePromise) {
    exactRegionModulePromise = import('@/overlap/exact-region');
  }
  return exactRegionModulePromise;
}

function loadExactBrowserRuntimeModule(): Promise<ExactBrowserRuntimeModule> {
  if (!exactBrowserRuntimeModulePromise) {
    exactBrowserRuntimeModulePromise = import('@/overlap/exactBrowserRuntime');
  }
  return exactBrowserRuntimeModulePromise;
}

function loadTerrainFacePartitionModule(): Promise<TerrainFacePartitionModule> {
  if (!terrainFacePartitionModulePromise) {
    terrainFacePartitionModulePromise = import('@/utils/terrainFacePartition');
  }
  return terrainFacePartitionModulePromise;
}

function loadTerrainPartitionGraphModule(): Promise<TerrainPartitionGraphModule> {
  if (!terrainPartitionGraphModulePromise) {
    terrainPartitionGraphModulePromise = import('@/utils/terrainPartitionGraph');
  }
  return terrainPartitionGraphModulePromise;
}

function loadWingtraConvertModule(): Promise<WingtraConvertModule> {
  if (!wingtraConvertModulePromise) {
    wingtraConvertModulePromise = import('@/interop/wingtra/convert');
  }
  return wingtraConvertModulePromise;
}

function getPlaneHardwareVersionFromWingtraPayloadUniqueStringLocal(payloadUniqueString: string | undefined): '4' | '5' | undefined {
  if (!payloadUniqueString) return undefined;
  if (payloadUniqueString.endsWith('_v4')) return '4';
  if (payloadUniqueString.endsWith('_v5')) return '5';
  return undefined;
}

function isWingtraFlightPlanTemplateExportReadyLocal(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const maybeFlightPlan = (value as { flightPlan?: unknown }).flightPlan;
  if (!maybeFlightPlan || typeof maybeFlightPlan !== 'object') return false;
  const items = (maybeFlightPlan as { items?: unknown }).items;
  const payloadUniqueString = (maybeFlightPlan as { payloadUniqueString?: unknown }).payloadUniqueString;
  const planeHardwareVersion = (maybeFlightPlan as { planeHardware?: { hwVersion?: unknown } }).planeHardware?.hwVersion;
  if (!Array.isArray(items)) return false;
  if (typeof payloadUniqueString !== 'string' || payloadUniqueString.length === 0) return false;
  if (planeHardwareVersion !== '4' && planeHardwareVersion !== '5') return false;
  if (getPlaneHardwareVersionFromWingtraPayloadUniqueStringLocal(payloadUniqueString) !== planeHardwareVersion) return false;
  if (!('geofence' in (value as object)) || !('safety' in (value as object))) return false;
  return typeof (maybeFlightPlan as { creationTime?: unknown }).creationTime === 'number';
}

function resolveWingtraTemplateMaxGroundClearanceM(value: unknown): number | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const raw = (value as { safety?: { maxGroundClearance?: unknown } }).safety?.maxGroundClearance;
  const numeric = typeof raw === 'number' ? raw : Number(raw);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : undefined;
}

function resolveWingtraExportMaxHeightAboveGroundM(value: unknown): number {
  return resolveWingtraTemplateMaxGroundClearanceM(value) ?? DEFAULT_EXPORT_SEQUENCE_MAX_HEIGHT_ABOVE_GROUND_M;
}

function resolveWingtraTemplateCruiseSpeedMps(value: unknown): number | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const raw = (value as { flightPlan?: { cruiseSpeed?: unknown } }).flightPlan?.cruiseSpeed;
  const numeric = typeof raw === 'number' ? raw : Number(raw);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : undefined;
}

function resolveWingtraExportCruiseSpeedMps(value: unknown): number {
  return resolveWingtraTemplateCruiseSpeedMps(value) ?? DEFAULT_WINGTRA_EXPORT_CRUISE_SPEED_MPS;
}

function splitPerfNow() {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function splitPerfLog(scope: string, message: string, data?: unknown) {
  if (!TERRAIN_SPLIT_DEBUG) return;
  if (data === undefined) {
    console.debug(`[terrain-split][${scope}] ${message}`);
    return;
  }
  console.debug(`[terrain-split][${scope}] ${message}`, data);
}

function clampNumber(value: number | undefined, min: number, max: number, fallback: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value as number));
}

function normalizeBearing(value?: number): number | undefined {
  if (!Number.isFinite(value)) return undefined;
  return (((value as number) % 360) + 360) % 360;
}

const DECK_FLIGHT_LAYER_ID_PREFIXES = [
  'drone-path-',
  'drone-centerline-',
  'trigger-points-',
  'camera-points-',
] as const;

function getDeckFlightLayerPolygonId(layerId: string): string {
  const prefix = DECK_FLIGHT_LAYER_ID_PREFIXES.find((candidate) => layerId.startsWith(candidate));
  return prefix ? layerId.slice(prefix.length) : '';
}

function aggregateMetricStats(tileStats: GSDStats[]): GSDStats {
  const valid = tileStats.filter((s) => s && s.count > 0 && isFinite(s.min) && isFinite(s.max) && s.max > 0);
  if (valid.length === 0) return { min: 0, max: 0, mean: 0, count: 0, totalAreaM2: 0, histogram: [] };

  let totalCount = 0;
  let totalArea = 0;
  let weightedSum = 0;
  let globalMin = Number.POSITIVE_INFINITY;
  let globalMax = Number.NEGATIVE_INFINITY;
  for (const s of valid) {
    totalCount += s.count;
    const areaWeight = (s.totalAreaM2 && s.totalAreaM2 > 0) ? s.totalAreaM2 : s.count;
    totalArea += areaWeight;
    weightedSum += s.mean * areaWeight;
    globalMin = Math.min(globalMin, s.min);
    globalMax = Math.max(globalMax, s.max);
  }
  const accurateMean = totalArea > 0 ? weightedSum / totalArea : 0;
  const span = globalMax - globalMin;
  if (!(span > 0)) {
    return {
      min: globalMin,
      max: globalMax,
      mean: accurateMean,
      count: totalCount,
      totalAreaM2: totalArea,
      histogram: [{ bin: globalMin, count: totalCount, areaM2: totalArea }],
    };
  }
  const targetBins = 20;
  const binSize = span / targetBins;
  const bins = new Array<{ bin: number; count: number; areaM2: number }>(targetBins);
  for (let i = 0; i < targetBins; i++) {
    bins[i] = { bin: globalMin + (i + 0.5) * binSize, count: 0, areaM2: 0 };
  }
  for (const s of valid) {
    for (const hb of s.histogram || []) {
      if (!hb || hb.count === 0) continue;
      let index = Math.floor((hb.bin - globalMin) / binSize);
      if (index < 0) index = 0;
      if (index >= targetBins) index = targetBins - 1;
      bins[index].count += hb.count;
      bins[index].areaM2 += hb.areaM2 || 0;
    }
  }
  return {
    min: globalMin,
    max: globalMax,
    mean: accurateMean,
    count: totalCount,
    totalAreaM2: totalArea,
    histogram: bins.filter((bin) => bin.count > 0 || bin.areaM2 > 0),
  };
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

function lidarStripMayAffectTile(
  strip: LidarStripMeters,
  tileRef: { z: number; x: number; y: number },
) {
  const bounds = tileMetersBounds(tileRef.z, tileRef.x, tileRef.y);
  const reachPadM = Math.max(
    strip.halfWidthM ?? 0,
    typeof strip.maxRangeM === 'number' && Number.isFinite(strip.maxRangeM) ? strip.maxRangeM : 0,
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

function normalizeAxialBearingDeg(value: number) {
  const normalized = ((value % 180) + 180) % 180;
  return Number.isFinite(normalized) ? normalized : 0;
}

function path3dLengthMeters(path3d: number[][][]) {
  let total = 0;
  for (const line of path3d) {
    if (!Array.isArray(line) || line.length < 2) continue;
    for (let index = 1; index < line.length; index++) {
      const start = line[index - 1];
      const end = line[index];
      if (!Array.isArray(start) || !Array.isArray(end) || start.length < 3 || end.length < 3) continue;
      const [x1, y1] = lngLatToMeters(start[0], start[1]);
      const [x2, y2] = lngLatToMeters(end[0], end[1]);
      const dz = end[2] - start[2];
      total += Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + dz ** 2);
    }
  }
  return total;
}

function areCoordsNear(a: [number, number], b: [number, number], epsilon = GEOMETRY_RING_EPSILON_DEG): boolean {
  return Math.abs(a[0] - b[0]) <= epsilon && Math.abs(a[1] - b[1]) <= epsilon;
}

function roundCoordPair(coord: [number, number]): [number, number] {
  return [Number(coord[0].toFixed(12)), Number(coord[1].toFixed(12))];
}

function isLidarParams(params?: PolygonParams | null): boolean {
  return (params?.payloadKind ?? DEFAULT_PAYLOAD_KIND) === 'lidar';
}

function sanitizePolygonParams(params: PolygonParams): PolygonParams {
  const payloadKind = params.payloadKind ?? DEFAULT_PAYLOAD_KIND;
  const isLidar = payloadKind === 'lidar';
  const preservedSpeedMps = Number.isFinite(params.speedMps) && (params.speedMps as number) > 0
    ? Math.max(0.1, params.speedMps as number)
    : undefined;
  return {
    ...params,
    payloadKind,
    altitudeAGL: Math.max(1, Number.isFinite(params.altitudeAGL) ? params.altitudeAGL : DEFAULT_ALTITUDE_AGL),
    frontOverlap: isLidar ? 0 : clampNumber(params.frontOverlap, 0, 95, DEFAULT_FRONT_OVERLAP),
    sideOverlap: clampNumber(params.sideOverlap, 0, 95, DEFAULT_SIDE_OVERLAP),
    speedMps: isLidar ? (preservedSpeedMps ?? DEFAULT_LIDAR.defaultSpeedMps) : preservedSpeedMps,
    mappingFovDeg: isLidar ? clampNumber(params.mappingFovDeg, 1, 180, DEFAULT_LIDAR.effectiveHorizontalFovDeg) : undefined,
    maxLidarRangeM: isLidar ? Math.max(1, Number.isFinite(params.maxLidarRangeM) ? params.maxLidarRangeM! : DEFAULT_LIDAR_MAX_RANGE_M) : undefined,
    lidarReturnMode: isLidar ? (params.lidarReturnMode ?? 'single') : undefined,
    useCustomBearing: !!params.useCustomBearing,
    customBearingDeg: params.useCustomBearing ? normalizeBearing(params.customBearingDeg) : undefined,
  };
}

function normalizeRingForGeometryOps(ring: [number, number][]): [number, number][] | null {
  const coords = Array.isArray(ring)
    ? ring.filter((coord): coord is [number, number] => (
        Array.isArray(coord) &&
        coord.length >= 2 &&
        Number.isFinite(coord[0]) &&
        Number.isFinite(coord[1])
      )).map((coord) => roundCoordPair(coord))
    : [];
  if (coords.length < 3) return null;

  const deduped: [number, number][] = [];
  for (const coord of coords) {
    if (deduped.length === 0 || !areCoordsNear(deduped[deduped.length - 1], coord)) {
      deduped.push(coord);
    }
  }
  if (deduped.length < 3) return null;

  const first = deduped[0];
  const last = deduped[deduped.length - 1];
  if (areCoordsNear(first, last)) {
    deduped[deduped.length - 1] = first;
  } else {
    deduped.push(first);
  }

  if (deduped.length < 4) return null;
  return deduped;
}

function normalizePolygonFeature(feature: any) {
  if (!feature?.geometry) return feature;
  let next = feature;
  try {
    next = turf.cleanCoords(next as any);
  } catch {}
  try {
    const truncateFn = (turf as any).truncate;
    if (typeof truncateFn === 'function') {
      next = truncateFn(next, { precision: 10, coordinates: 2, mutate: false });
    }
  } catch {}
  try {
    const bufferFn = (turf as any).buffer;
    if (typeof bufferFn === 'function' && (next.geometry?.type === 'Polygon' || next.geometry?.type === 'MultiPolygon')) {
      const repaired = bufferFn(next, 0, { units: 'meters' });
      if (repaired?.geometry) next = repaired;
    }
  } catch {}
  try {
    next = turf.cleanCoords(next as any);
  } catch {}
  return next;
}

function unionTurfFeatures(a: any, b: any) {
  const unionFn = (turf as any).union;
  if (typeof unionFn !== 'function') return null;
  const attempt = (left: any, right: any) => (
    unionFn.length >= 2
      ? unionFn(left, right)
      : unionFn(turf.featureCollection([left, right]))
  );
  try {
    return attempt(a, b);
  } catch {
    try {
      return attempt(normalizePolygonFeature(a), normalizePolygonFeature(b));
    } catch {
      return null;
    }
  }
}

function intersectTurfFeatures(a: any, b: any) {
  const intersectFn = (turf as any).intersect;
  if (typeof intersectFn !== 'function') return null;
  const attempt = (left: any, right: any) => (
    intersectFn.length >= 2
      ? intersectFn(left, right)
      : intersectFn(turf.featureCollection([left, right]))
  );
  try {
    return attempt(a, b);
  } catch {
    try {
      return attempt(normalizePolygonFeature(a), normalizePolygonFeature(b));
    } catch {
      return null;
    }
  }
}

function computePartitionCoverageMetrics(
  parentRing: [number, number][],
  childRings: [number, number][][],
) {
  const normalizedParent = normalizeRingForGeometryOps(parentRing);
  const normalizedChildren = childRings
    .map((ring) => normalizeRingForGeometryOps(ring))
    .filter((ring): ring is [number, number][] => ring !== null);
  if (!normalizedParent || normalizedChildren.length === 0) {
    return { coverageRatio: 0, overlapRatio: 1, normalizedChildren: [] as [number, number][][] };
  }

  const parentFeature = normalizePolygonFeature(turf.polygon([normalizedParent]));
  const parentArea = Math.max(1e-6, turf.area(parentFeature));
  let unionFeature: any = null;
  let summedArea = 0;

  for (const childRing of normalizedChildren) {
    const feature = normalizePolygonFeature(turf.polygon([childRing]));
    summedArea += turf.area(feature);
    unionFeature = unionFeature ? unionTurfFeatures(unionFeature, feature) : feature;
    if (!unionFeature) {
      return { coverageRatio: 0, overlapRatio: 1, normalizedChildren };
    }
  }

  const coveredFeature = intersectTurfFeatures(parentFeature, unionFeature);
  const coveredArea = coveredFeature ? turf.area(coveredFeature) : 0;
  const unionArea = turf.area(unionFeature);
  return {
    coverageRatio: coveredArea / parentArea,
    overlapRatio: Math.max(0, summedArea - unionArea) / parentArea,
    normalizedChildren,
  };
}

function getLineSpacingForParams(params: PolygonParams): number {
  if (isLidarParams(params)) {
    const lidar = params.lidarKey ? LIDAR_REGISTRY[params.lidarKey] || DEFAULT_LIDAR : DEFAULT_LIDAR;
    const mappingFovDeg = params.mappingFovDeg ?? lidar.effectiveHorizontalFovDeg;
    return lidarLineSpacing(params.altitudeAGL, params.sideOverlap, mappingFovDeg);
  }

  const cameraKey = params.cameraKey;
  const camera = cameraKey ? CAMERA_REGISTRY[cameraKey] || DEFAULT_CAMERA : DEFAULT_CAMERA;
  const yawOffset = params.cameraYawOffsetDeg ?? 0;
  const rotate90 = Math.round((((yawOffset % 180) + 180) % 180)) === 90;
  return lineSpacingRotated(camera, params.altitudeAGL, params.sideOverlap, rotate90);
}

function getForwardSpacingForParams(params: PolygonParams): number | null {
  if (isLidarParams(params)) return null;
  const cameraKey = params.cameraKey;
  const camera = cameraKey ? CAMERA_REGISTRY[cameraKey] || DEFAULT_CAMERA : DEFAULT_CAMERA;
  const yawOffset = params.cameraYawOffsetDeg ?? 0;
  const rotate90 = Math.round((((yawOffset % 180) + 180) % 180)) === 90;
  return forwardSpacingRotated(camera, params.altitudeAGL, params.frontOverlap, rotate90);
}

interface Props {
  mapboxToken: string;
  clearAllEpoch?: number;
  center?: LngLatLike;
  zoom?: number;
  sampleStep?: number;
  terrainDemUrlTemplate?: string | null;
  terrainSource?: TerrainSourceSelection;
  onTerrainSourceReady?: (terrainSource: TerrainSourceSelection) => void;

  onRequestParams?: (polygonId: string, ring: [number, number][]) => void;
  onAnalysisComplete?: (results: PolygonAnalysisResult[]) => void;
  onAnalysisStart?: (polygonId: string) => void;
  onError?: (error: string, polygonId?: string) => void;
  onFlightLinesUpdated?: (changed: string | '__all__') => void;
  onClearGSD?: () => void;
  onPolygonSelected?: (polygonId: string | null) => void;
  onMergeStateChange?: (state: PolygonMergeState) => void;
  onHistoryStateChange?: (state: PolygonHistoryState) => void;
  selectedPolygonId?: string | null;
}

const MapFlightDirectionComponent = React.forwardRef<MapFlightDirectionAPI, Props>(
  (
    {
      mapboxToken,
      clearAllEpoch = 0,
      center = [8.54, 47.37],
      zoom = 13,
      sampleStep = 2,
      terrainDemUrlTemplate = null,
      terrainSource = { mode: 'mapbox', datasetId: null },
      onTerrainSourceReady,
      onRequestParams,
      onAnalysisComplete,
      onAnalysisStart,
      onError,
      onFlightLinesUpdated,
      onClearGSD,
      onPolygonSelected,
      onMergeStateChange,
      onHistoryStateChange,
      selectedPolygonId = null,
    },
    ref
  ) => {
    const mapContainer = useRef<HTMLDivElement>(null);
    const mapRef = useRef<MapboxMap>();
    const drawRef = useRef<MapboxDraw>();
    const deckOverlayRef = useRef<MapboxOverlay>();
    const terrainDemSourceTemplateRef = useRef<string | null>(terrainDemUrlTemplate);
    const terrainSourceRef = useRef<TerrainSourceSelection>(terrainSource);
    const onTerrainSourceReadyRef = useRef(onTerrainSourceReady);
    const onMergeStateChangeRef = useRef(onMergeStateChange);
    const onHistoryStateChangeRef = useRef(onHistoryStateChange);
    const terrainSourceApplySeqRef = useRef(0);
    const flightLinesVisibleRef = useRef(true);
    const selectedPolygonIdRef = useRef<string | null>(selectedPolygonId);
    const vertexClickCandidateRef = useRef<{
      parentId: string;
      coordPath: string;
      point: { x: number; y: number };
      timestampMs: number;
    } | null>(null);

    // File inputs
    const kmlInputRef = useRef<HTMLInputElement>(null);
    const flightplanInputRef = useRef<HTMLInputElement>(null);
    const [isDraggingKml, setIsDraggingKml] = useState(false);

    // Suspend auto-analysis during programmatic imports
    const suspendAutoAnalysisRef = useRef(false);

    const [polygonResults, setPolygonResults] = useState<Map<string, PolygonAnalysisResult>>(new Map());
    const [polygonTiles, setPolygonTiles] = useState<Map<string, any[]>>(new Map());
    const backendPartitionSolutionsRef = useRef<Map<string, TerrainPartitionSolutionPreview[]>>(new Map());

    // Flight lines + spacing + altitude actually used to build 3D path.
    const [polygonFlightLines, setPolygonFlightLines] = useState<
      Map<string, PlannedFlightGeometry & { altitudeAGL: number }>
    >(new Map());

    // Per‑polygon parameters provided by user (or by importer).
    const [polygonParams, setPolygonParams] = useState<Map<string, PolygonParams>>(new Map());
    // NEW: Queue of polygons awaiting parameter input (so multiple imports show dialogs sequentially)
    const [pendingParamPolygons, setPendingParamPolygons] = useState<string[]>([]);

    // Overrides: force a heading/spacing (e.g., from Wingtra file)
    const [bearingOverrides, setBearingOverrides] = useState<
      Map<string, BearingOverride>
    >(new Map());

    // Original file meta for revert
    const [importedOriginals, setImportedOriginals] = useState<
      Map<string, { bearingDeg: number; lineSpacingM: number }>
    >(new Map());

    const [, setDeckLayers] = useState<any[]>([]);
    const [lastImportedFlightplan, setLastImportedFlightplan] = useState<any | null>(null);
    const lastImportedFlightplanNameRef = useRef<string | undefined>(undefined);
    // Bulk apply guard to suppress sequential dialog popping
    const bulkApplyRef = useRef(false);
    // NEW: If user clicks "Apply All" while some polygons still analyzing, preset params
    const bulkPresetParamsRef = useRef<PolygonParams | null>(null);

    // Keep live copies so async callbacks always see current values (avoid stale closures)
    const polygonParamsRef = React.useRef(polygonParams);
    const bearingOverridesRef = React.useRef(bearingOverrides);
    const polygonTilesRef = React.useRef(polygonTiles);
    const polygonFlightLinesRef = React.useRef(polygonFlightLines);
    const polygonResultsRef = React.useRef(polygonResults);
    const importedOriginalsRef = React.useRef(importedOriginals);
    const pendingGeometryRefreshRef = React.useRef<Set<string>>(new Set());
    const processingPolygonIdsRef = React.useRef<Set<string>>(new Set());
    const processingAnimationFrameRef = React.useRef<number | null>(null);
    // NEW: Suppress per‑polygon flight line update events during batched imports
    const suppressFlightLineEventsRef = React.useRef(false);
    const pendingOptimizeRef = React.useRef<Set<string>>(new Set());
    const plannedGeometryWorkerRef = React.useRef<PlannedGeometryWorkerController | null>(null);
    const resetGenerationRef = React.useRef(0);
    const lastHandledClearAllEpochRef = React.useRef(clearAllEpoch);
    const guardedTimeoutsRef = React.useRef<Set<number>>(new Set());
    const pendingProgrammaticDeletesRef = React.useRef<Set<string>>(new Set());
    const suppressHistoryInvalidationRef = React.useRef(false);
    const isApplyingPolygonOperationRef = React.useRef(false);
    const polygonOperationAffectedIdsRef = React.useRef<Set<string>>(new Set());
    const suppressSelectionDialogUntilRef = React.useRef(0);
    const suppressNextEmptySelectionRef = React.useRef(0);
    const polygonFeatureIdSeqRef = React.useRef(0);
    // NEW: Altitude mode + minimum clearance configuration (global)
    const [altitudeMode, setAltitudeMode] = useState<'legacy' | 'min-clearance'>('legacy');
    const [minClearanceM, setMinClearanceM] = useState<number>(60);
    const [turnExtendM, setTurnExtendM] = useState<number>(96);
    const [mergeState, setMergeState] = useState<PolygonMergeState>(() => createIdlePolygonMergeState());
    const mergeStateRef = React.useRef<PolygonMergeState>(createIdlePolygonMergeState());
    const polygonOperationHistoryRef = React.useRef(createEmptyPolygonOperationHistory());

    React.useEffect(() => { polygonParamsRef.current = polygonParams; }, [polygonParams]);
    React.useEffect(() => { bearingOverridesRef.current = bearingOverrides; }, [bearingOverrides]);
    React.useEffect(() => { polygonTilesRef.current = polygonTiles; }, [polygonTiles]);
    React.useEffect(() => { polygonFlightLinesRef.current = polygonFlightLines; }, [polygonFlightLines]);
    React.useEffect(() => { polygonResultsRef.current = polygonResults; }, [polygonResults]);
    React.useEffect(() => { importedOriginalsRef.current = importedOriginals; }, [importedOriginals]);
    React.useEffect(() => { terrainSourceRef.current = terrainSource; }, [terrainSource]);
    React.useEffect(() => { onTerrainSourceReadyRef.current = onTerrainSourceReady; }, [onTerrainSourceReady]);
    React.useEffect(() => { onMergeStateChangeRef.current = onMergeStateChange; }, [onMergeStateChange]);
    React.useEffect(() => { onHistoryStateChangeRef.current = onHistoryStateChange; }, [onHistoryStateChange]);

    const publishMergeState = useCallback((next: PolygonMergeState) => {
      mergeStateRef.current = next;
      setMergeState(next);
      onMergeStateChangeRef.current?.(next);
    }, []);

    const publishHistoryState = useCallback(() => {
      onHistoryStateChangeRef.current?.(
        createPolygonHistoryState(
          polygonOperationHistoryRef.current,
          isApplyingPolygonOperationRef.current,
        ),
      );
    }, []);

    const setPolygonOperationApplying = useCallback((next: boolean) => {
      if (isApplyingPolygonOperationRef.current === next) return;
      isApplyingPolygonOperationRef.current = next;
      publishHistoryState();
    }, [publishHistoryState]);

    const replacePolygonOperationHistory = useCallback((nextHistory: ReturnType<typeof createEmptyPolygonOperationHistory>) => {
      polygonOperationHistoryRef.current = nextHistory;
      publishHistoryState();
    }, [publishHistoryState]);

    const invalidatePolygonOperationHistory = useCallback(() => {
      const currentHistory = polygonOperationHistoryRef.current;
      if (currentHistory.undoStack.length === 0 && currentHistory.redoStack.length === 0) return;
      replacePolygonOperationHistory(clearPolygonOperationHistory());
    }, [replacePolygonOperationHistory]);

    const clearPolygonOperationRedoStack = useCallback(() => {
      const currentHistory = polygonOperationHistoryRef.current;
      if (currentHistory.redoStack.length === 0) return;
      replacePolygonOperationHistory(clearPolygonOperationRedoState(currentHistory));
    }, [replacePolygonOperationHistory]);

    const createPolygonFeatureId = useCallback(() => {
      polygonFeatureIdSeqRef.current += 1;
      return `polyop-${Date.now().toString(36)}-${polygonFeatureIdSeqRef.current.toString(36)}`;
    }, []);

    const getAllDrawPolygonFeatures = useCallback(() => {
      const draw = drawRef.current as any;
      const features = Array.isArray(draw?.getAll?.()?.features) ? draw.getAll().features : [];
      return features
        .map((feature: any) => toMergeablePolygonFeature(feature))
        .filter(Boolean) as Array<ReturnType<typeof toMergeablePolygonFeature> extends infer T ? Exclude<T, null> : never>;
    }, []);

    const syncPolygonMergeState = useCallback((
      primaryPolygonId = mergeStateRef.current.primaryPolygonId,
      selectedPolygonIds = mergeStateRef.current.selectedPolygonIds,
    ) => {
      if (!primaryPolygonId) {
        publishMergeState(createIdlePolygonMergeState());
        return createIdlePolygonMergeState();
      }
      const next = derivePolygonMergeState({
        features: getAllDrawPolygonFeatures(),
        primaryPolygonId,
        selectedPolygonIds,
      });
      publishMergeState(next);
      return next;
    }, [getAllDrawPolygonFeatures, publishMergeState]);

    const cancelPolygonMerge = useCallback(() => {
      if (mergeStateRef.current.mode === 'idle') return;
      publishMergeState(createIdlePolygonMergeState());
    }, [publishMergeState]);

    const snapshotPolygonFeature = useCallback((polygonId: string): PolygonSnapshot | null => {
      const draw = drawRef.current as any;
      const feature = draw?.get?.(polygonId);
      if (feature?.geometry?.type !== 'Polygon') return null;
      const snapshotFeature = createPolygonFeatureSnapshot({
        id: polygonId,
        ring: feature.geometry.coordinates?.[0] as [number, number][],
        properties: { ...(feature.properties ?? {}) },
      });
      if (!snapshotFeature) return null;

      const params = polygonParamsRef.current.get(polygonId);
      const override = bearingOverridesRef.current.get(polygonId);
      const importedOriginal = importedOriginalsRef.current.get(polygonId);
      return {
        feature: snapshotFeature,
        params: params ? { ...params } : undefined,
        override: override ? { ...override } : undefined,
        importedOriginal: importedOriginal ? { ...importedOriginal } : undefined,
      };
    }, []);

    const addPolygonSnapshotToDraw = useCallback((snapshot: PolygonSnapshot) => {
      const draw = drawRef.current as any;
      if (!draw) return;
      draw.add({
        type: 'Feature',
        id: snapshot.feature.id,
        properties: { ...(snapshot.feature.properties ?? {}) },
        geometry: {
          type: 'Polygon',
          coordinates: [[...snapshot.feature.geometry.coordinates[0].map((coord) => [coord[0], coord[1]])]],
        },
      });
    }, []);

    useEffect(() => {
      onMergeStateChangeRef.current?.(mergeStateRef.current);
      onHistoryStateChangeRef.current?.(
        createPolygonHistoryState(
          polygonOperationHistoryRef.current,
          isApplyingPolygonOperationRef.current,
        ),
      );
    }, []);

    const getPlannedGeometryWorker = useCallback(() => {
      if (!plannedGeometryWorkerRef.current) {
        plannedGeometryWorkerRef.current = new PlannedGeometryWorkerController();
      }
      return plannedGeometryWorkerRef.current;
    }, []);

    const applyTerrainSourceToMap = useCallback((map: MapboxMap, tileUrlTemplate: string | null, nextTerrainSource: TerrainSourceSelection) => {
      const applySeq = terrainSourceApplySeqRef.current + 1;
      terrainSourceApplySeqRef.current = applySeq;
      console.log('[terrain-source] applying terrain source to map', {
        applySeq,
        terrainMode: nextTerrainSource.mode,
        datasetId: nextTerrainSource.datasetId ?? null,
        tileUrlTemplate,
      });
      setTerrainDemSourceOnMap(map, tileUrlTemplate);
      if (typeof window !== 'undefined') {
        window.requestAnimationFrame(() => {
          if (terrainSourceApplySeqRef.current !== applySeq) return;
          console.debug('[terrain-source] reasserting terrain source after animation frame', {
            applySeq,
            terrainMode: nextTerrainSource.mode,
            datasetId: nextTerrainSource.datasetId ?? null,
          });
          reassertTerrainDemSourceOnMap(map, tileUrlTemplate);
        });
      }
      void (async () => {
        let waitResult = await waitForTerrainDemSourceOnMap(map, tileUrlTemplate, tileUrlTemplate ? 4000 : 10000);
        if (terrainSourceApplySeqRef.current !== applySeq) return;

        if (tileUrlTemplate && (waitResult.timedOut || !waitResult.sawRenderableContent)) {
          console.warn('[terrain-source] backend terrain source did not render on first apply; retrying full source bind', {
            applySeq,
            terrainMode: nextTerrainSource.mode,
            datasetId: nextTerrainSource.datasetId ?? null,
            tileUrlTemplate,
            timedOut: waitResult.timedOut,
            sawRenderableContent: waitResult.sawRenderableContent,
          });
          forceReloadTerrainDemSourceOnMap(map, tileUrlTemplate);
          waitResult = await waitForTerrainDemSourceOnMap(map, tileUrlTemplate, 6000);
          if (terrainSourceApplySeqRef.current !== applySeq) return;
        }

        reassertTerrainDemSourceOnMap(map, tileUrlTemplate);
        console.log('[terrain-source] terrain source apply completed', {
          applySeq,
          terrainMode: nextTerrainSource.mode,
          datasetId: nextTerrainSource.datasetId ?? null,
          tileUrlTemplate,
          timedOut: waitResult.timedOut,
          sawRenderableContent: waitResult.sawRenderableContent,
        });
        onTerrainSourceReadyRef.current?.(nextTerrainSource);
      })();
    }, []);

    const syncFlightLinesVisibility = useCallback((visible: boolean) => {
      flightLinesVisibleRef.current = visible;
      const emphasizedPolygonId = (() => {
        const selectedPolygonId = selectedPolygonIdRef.current;
        if (!selectedPolygonId) return null;
        return polygonFlightLinesRef.current.has(selectedPolygonId) ? selectedPolygonId : null;
      })();

      const map = mapRef.current;
      if (map) {
        setFlightLineSelectionEmphasis(map, emphasizedPolygonId, visible);
      }

      const overlay = deckOverlayRef.current;
      if (!overlay) return;
      setDeckLayers((currentLayers) => {
        const nextLayers = currentLayers.map((layer: any) => {
          const id = String(layer?.id ?? '');
          const isPathLayer = id.startsWith('drone-path-') || id.startsWith('drone-centerline-');
          const isTriggerLayer = id.startsWith('trigger-points-');
          const isCameraLayer = id.startsWith('camera-points-');
          const isFlightLayer =
            isPathLayer ||
            isTriggerLayer ||
            isCameraLayer;
          if (!isFlightLayer) return layer;
          const polygonId = getDeckFlightLayerPolygonId(id);
          const isSelected = !!emphasizedPolygonId && polygonId === emphasizedPolygonId;
          const isDimmed = !!emphasizedPolygonId && !isSelected;
          if (typeof layer?.clone !== 'function') return layer;

          if (isPathLayer) {
            return layer.clone({
              visible,
              getColor: isDimmed ? [100, 200, 255, 70] : isSelected ? [100, 200, 255, 255] : [100, 200, 255, 230],
              getWidth: isSelected ? 3.5 : 2,
            });
          }

          if (isTriggerLayer) {
            return layer.clone({
              visible,
              getRadius: isSelected ? 5 : 4,
              getFillColor: isDimmed ? [12, 36, 97, 55] : [12, 36, 97, 230],
              getLineColor: isDimmed ? [230, 240, 255, 40] : [230, 240, 255, 220],
            });
          }

          return layer.clone({
            visible,
            getFillColor: isDimmed ? [255, 71, 87, 80] : [255, 71, 87, 255],
            getLineColor: isDimmed ? [255, 255, 255, 70] : [255, 255, 255, 255],
          });
        });
        overlay.setProps({ layers: nextLayers });
        return nextLayers;
      });
    }, []);

    const syncProcessingPerimeterOverlay = useCallback(() => {
      const map = mapRef.current;
      const draw = drawRef.current as any;
      if (!map || !draw) return;
      const polygons = Array.from(processingPolygonIdsRef.current)
        .map((polygonId) => {
          const feature = draw.get?.(polygonId);
          const ring = feature?.geometry?.type === 'Polygon'
            ? feature.geometry.coordinates?.[0] as [number, number][] | undefined
            : undefined;
          if (!ring || ring.length < 3) return null;
          return { polygonId, ring };
        })
        .filter((polygon): polygon is NonNullable<typeof polygon> => polygon !== null);
      setProcessingPerimeterPolygons(map, polygons);
    }, []);

    const stopProcessingPerimeterAnimation = useCallback(() => {
      if (processingAnimationFrameRef.current != null && typeof window !== 'undefined') {
        window.cancelAnimationFrame(processingAnimationFrameRef.current);
        processingAnimationFrameRef.current = null;
      }
    }, []);

    const startProcessingPerimeterAnimation = useCallback(() => {
      if (typeof window === 'undefined' || processingAnimationFrameRef.current != null) return;
      const tick = (timestamp: number) => {
        const map = mapRef.current;
        if (!map || processingPolygonIdsRef.current.size === 0) {
          processingAnimationFrameRef.current = null;
          return;
        }
        animateProcessingPerimeter(map, timestamp);
        processingAnimationFrameRef.current = window.requestAnimationFrame(tick);
      };
      processingAnimationFrameRef.current = window.requestAnimationFrame(tick);
    }, []);

    const setProcessingPolygonIds = useCallback((polygonIds: string[]) => {
      processingPolygonIdsRef.current = new Set(polygonIds);
      syncProcessingPerimeterOverlay();
      if (processingPolygonIdsRef.current.size === 0) {
        stopProcessingPerimeterAnimation();
        return;
      }
      startProcessingPerimeterAnimation();
    }, [startProcessingPerimeterAnimation, stopProcessingPerimeterAnimation, syncProcessingPerimeterOverlay]);

    const cancelAllGuardedTimeouts = useCallback(() => {
      for (const timeoutId of guardedTimeoutsRef.current) {
        window.clearTimeout(timeoutId);
      }
      guardedTimeoutsRef.current.clear();
    }, []);

    const scheduleGuardedTimeout = useCallback((task: () => void, delayMs = 0, generation = resetGenerationRef.current) => {
      const timeoutId = window.setTimeout(() => {
        guardedTimeoutsRef.current.delete(timeoutId);
        if (generation !== resetGenerationRef.current) return;
        task();
      }, delayMs);
      guardedTimeoutsRef.current.add(timeoutId);
      return timeoutId;
    }, []);

    // Debounced callback to prevent React render conflicts when multiple analyses complete
    const debouncedAnalysisComplete = useCallback(() => {
      const generation = resetGenerationRef.current;
      const timeoutId = scheduleGuardedTimeout(() => {
        onAnalysisComplete?.(Array.from(polygonResultsRef.current.values()));
      }, 0, generation);
      return timeoutId;
    }, [onAnalysisComplete, scheduleGuardedTimeout]);

    const applyPolygonParams = useCallback(async (polygonId: string, params: PolygonParams, opts?: { skipEvent?: boolean; skipQueue?: boolean }) => {
      const safeParams = sanitizePolygonParams(params);
      const nextParams = new Map(polygonParamsRef.current);
      nextParams.set(polygonId, safeParams);
      polygonParamsRef.current = nextParams;
      setPolygonParams(nextParams);

      const res = polygonResultsRef.current.get(polygonId);
      const tiles = polygonTilesRef.current.get(polygonId) || [];
      if (!res || !mapRef.current) return;

      const defaultSpacing = getLineSpacingForParams(safeParams);

      const customBearing = safeParams.useCustomBearing && Number.isFinite(safeParams.customBearingDeg)
        ? (((safeParams.customBearingDeg ?? 0) % 360) + 360) % 360
        : undefined;

      let override = bearingOverridesRef.current.get(polygonId);

      if (customBearing !== undefined) {
        const entry = { bearingDeg: customBearing, lineSpacingM: defaultSpacing, source: 'user' as const };
        setBearingOverrides((prev) => {
          const next = new Map(prev);
          next.set(polygonId, entry);
          return next;
        });
        const nextOverrides = new Map(bearingOverridesRef.current);
        nextOverrides.set(polygonId, entry);
        bearingOverridesRef.current = nextOverrides;
        override = entry;
      } else {
        if (override?.source === 'user') {
          setBearingOverrides((prev) => {
            const next = new Map(prev);
            next.delete(polygonId);
            return next;
          });
          const nextOverrides = new Map(bearingOverridesRef.current);
          nextOverrides.delete(polygonId);
          bearingOverridesRef.current = nextOverrides;
          override = bearingOverridesRef.current.get(polygonId);
        } else if (override) {
          const updated = { ...override, lineSpacingM: defaultSpacing };
          setBearingOverrides((prev) => {
            const next = new Map(prev);
            next.set(polygonId, updated);
            return next;
          });
          const nextOverrides = new Map(bearingOverridesRef.current);
          nextOverrides.set(polygonId, updated);
          bearingOverridesRef.current = nextOverrides;
          override = updated;
        }
      }

      const bearingDeg = override ? override.bearingDeg : res.result.contourDirDeg;
      const spacing = override?.lineSpacingM ?? defaultSpacing;
      const generation = resetGenerationRef.current;
      const fl = await buildFlightLinesForPolygonAsync({
        polygonId,
        ring: res.polygon.coordinates as [number, number][],
        bearingDeg,
        lineSpacingM: spacing,
        params: safeParams,
        quality: res.result.fitQuality,
        startedGeneration: generation,
      });
      if (!fl) return;

      const nextFlightLines = new Map(polygonFlightLinesRef.current);
      nextFlightLines.set(polygonId, { ...fl, altitudeAGL: safeParams.altitudeAGL });
      polygonFlightLinesRef.current = nextFlightLines;
      setPolygonFlightLines(nextFlightLines);
      syncFlightLinesVisibility(flightLinesVisibleRef.current);

      if (!opts?.skipEvent && !suppressFlightLineEventsRef.current) {
        onFlightLinesUpdated?.(polygonId);
      }

      if (deckOverlayRef.current && fl.flightLines.length > 0) {
        const path3d = build3DFlightPath(
          fl,
          tiles,
          fl.lineSpacing,
          { altitudeAGL: safeParams.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, preconnected: true },
        );
        update3DPathLayer(deckOverlayRef.current, polygonId, path3d, setDeckLayers);
        const spacingForward = getForwardSpacingForParams(safeParams);
        if (spacingForward && spacingForward > 0) {
          const samples = sampleCameraPositionsOnPlannedFlightGeometry(fl, path3d, spacingForward);
          const zOffset = 1;
          const ring = res.polygon.coordinates as [number, number][];
          const positions: [number, number, number][] = samples
            .filter(([lng, lat]) => isPointInRing(lng, lat, ring))
            .map(([lng,lat,alt]) => [lng,lat,alt + zOffset]);
          update3DTriggerPointsLayer(deckOverlayRef.current, polygonId, positions, setDeckLayers);
        } else {
          remove3DTriggerPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
        }
        syncFlightLinesVisibility(flightLinesVisibleRef.current);
      }

      if (opts?.skipQueue) return;

      setPendingParamPolygons(prev => {
        const rest = prev.filter(id => id !== polygonId);
        if (rest.length === 0) {
          bulkPresetParamsRef.current = null;
        }
        return rest;
      });
    }, [polygonResults, polygonTiles, onFlightLinesUpdated, altitudeMode, minClearanceM, syncFlightLinesVisibility, buildFlightLinesForPolygonAsync]);

    const applyPolygonParamsBatch = useCallback((updates: Array<{ polygonId: string; params: PolygonParams }>) => {
      const latestByPolygon = new Map<string, PolygonParams>();
      for (const update of updates) {
        if (!update?.polygonId || !update?.params) continue;
        latestByPolygon.set(update.polygonId, update.params);
      }
      if (latestByPolygon.size === 0) return;

      const prevSuppress = suppressFlightLineEventsRef.current;
      suppressFlightLineEventsRef.current = true;
      latestByPolygon.forEach((params, polygonId) => {
        void applyPolygonParams(polygonId, params, { skipEvent: true, skipQueue: true });
      });
      suppressFlightLineEventsRef.current = prevSuppress;

      if (!prevSuppress) {
        onFlightLinesUpdated?.(latestByPolygon.size === 1 ? Array.from(latestByPolygon.keys())[0] : '__all__');
      }
    }, [applyPolygonParams, onFlightLinesUpdated]);

    const withProcessingPolygon = useCallback(async <T,>(polygonId: string, task: () => Promise<T>) => {
      const startIds = new Set(processingPolygonIdsRef.current);
      startIds.add(polygonId);
      setProcessingPolygonIds(Array.from(startIds));
      try {
        return await task();
      } finally {
        const endIds = new Set(processingPolygonIdsRef.current);
        endIds.delete(polygonId);
        setProcessingPolygonIds(Array.from(endIds));
      }
    }, [setProcessingPolygonIds]);

    const isAsyncPolygonUpdateStillValid = useCallback((polygonId: string, generation: number) => {
      const draw = drawRef.current as any;
      return shouldApplyAsyncPolygonUpdate({
        startedGeneration: generation,
        currentGeneration: resetGenerationRef.current,
        polygonStillExists: !!draw?.get?.(polygonId),
      });
    }, []);

    const isAsyncGenerationStillValid = useCallback((generation: number) => {
      return shouldRunAsyncGeneration(generation, resetGenerationRef.current);
    }, []);

    type ExactBearingCandidate = SharedExactBearingCandidate;
    type ExactBearingSearchResult = SharedExactBearingSearchResult;

    const searchExactBearingNearSeed = useCallback(async ({
      scopeId,
      ring,
      params,
      tiles,
      seedBearingDeg,
      mode = 'local',
    }: {
      scopeId: string;
      ring: [number, number][];
      params: PolygonParams;
      tiles: any[];
      seedBearingDeg: number;
      mode?: 'local' | 'global';
    }): Promise<ExactBearingSearchResult> => {
      const safeParams = sanitizePolygonParams({ ...params, useCustomBearing: false });
      if ('customBearingDeg' in safeParams) delete (safeParams as any).customBearingDeg;
      const [{ optimizeBearingExact }, { withBrowserExactRegionRuntime }] = await Promise.all([
        loadExactRegionModule(),
        loadExactBrowserRuntimeModule(),
      ]);
      return withBrowserExactRegionRuntime(mapboxToken, async (runtime) => {
        const result = await optimizeBearingExact(runtime, {
          scopeId,
          ring,
          params: safeParams,
          altitudeMode,
          minClearanceM,
          turnExtendM,
          exactOptimizeZoom: EXACT_OPTIMIZE_ZOOM,
          timeWeight: EXACT_OPTIMIZE_TIME_WEIGHT,
          minOverlapForGsd: EXACT_MIN_OVERLAP_FOR_GSD,
          geometryTiles: tiles as any[],
          seedBearingDeg,
          mode,
          halfWindowDeg: mode === 'global' ? 90 : 30,
        });
        return {
          ...result,
          safeParams: result.safeParams as PolygonParams,
        };
      });
    }, [altitudeMode, mapboxToken, minClearanceM, turnExtendM]);

    const runOptimizedBearingSearch = useCallback(async (
      polygonId: string,
      params: PolygonParams,
      result: PolygonAnalysisResult,
      tiles: any[],
      generation: number,
    ) => {
      const ring = result.polygon.coordinates as [number, number][];
      const safeParams = sanitizePolygonParams({ ...params, useCustomBearing: false });
      if ('customBearingDeg' in safeParams) delete (safeParams as any).customBearingDeg;
      const seedBearingDeg = result.result.contourDirDeg;
      let best: ExactBearingCandidate | null = null;
      let evaluated: ExactBearingCandidate[] = [];
      let lineSpacingM = getLineSpacingForParams(safeParams);

      if (isExactTerrainBackendEnabled()) {
        try {
          const backendResult = await optimizeBearingWithBackend({
            polygonId,
            ring,
            payloadKind: isLidarParams(safeParams) ? 'lidar' : 'camera',
            params: safeParams,
            terrainSource,
            altitudeMode,
            minClearanceM,
            turnExtendM,
            seedBearingDeg,
            mode: 'global',
            halfWindowDeg: 90,
          });
          lineSpacingM = backendResult.lineSpacingM ?? lineSpacingM;
          if (Number.isFinite(backendResult.bearingDeg)) {
            best = {
              bearingDeg: backendResult.bearingDeg!,
              exactCost: backendResult.exactScore ?? Number.POSITIVE_INFINITY,
              qualityCost: backendResult.qualityCost ?? Number.POSITIVE_INFINITY,
              missionTimeSec: backendResult.missionTimeSec ?? 0,
              normalizedTimeCost: backendResult.normalizedTimeCost ?? 0,
              metricKind: backendResult.metricKind ?? (isLidarParams(safeParams) ? 'density' : 'gsd'),
              stats: { min: 0, max: 0, mean: 0, count: 0, totalAreaM2: 0, histogram: [] },
              diagnostics: backendResult.diagnostics ?? {},
              qualityBreakdown: {
                modelVersion: isLidarParams(safeParams) ? 'lidar-region-v1' : 'camera-region-v1',
                total: backendResult.qualityCost ?? Number.POSITIVE_INFINITY,
                signals: {},
                weights: {},
                contributions: {},
              },
              costBreakdown: {
                modelVersion: isLidarParams(safeParams) ? 'lidar-region-v1' : 'camera-region-v1',
                total: backendResult.exactScore ?? Number.POSITIVE_INFINITY,
                signals: {},
                weights: {},
                contributions: {},
              },
              missionBreakdown: {
                totalLengthM: 0,
                speedMps: 0,
                lineCount: 0,
              },
            };
            evaluated = best ? [best] : [];
          }
        } catch (error) {
          splitPerfLog(polygonId, 'backend exact optimize failed; falling back to frontend exact search', {
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }

      if (!best) {
        const localResult = await searchExactBearingNearSeed({
          scopeId: polygonId,
          ring,
          params,
          tiles,
          seedBearingDeg,
          mode: 'global',
        });
        best = localResult.best;
        evaluated = localResult.evaluated;
        lineSpacingM = localResult.lineSpacingM;
      }

      if (!isAsyncPolygonUpdateStillValid(polygonId, generation)) {
        splitPerfLog(polygonId, 'dropping stale optimized bearing result after reset', {
          generation,
          currentGeneration: resetGenerationRef.current,
        });
        return;
      }

      if (!best || !Number.isFinite(best.bearingDeg)) {
        splitPerfLog(polygonId, 'optimizePolygonDirection fallback to plane-fit bearing', {
          seedBearingDeg,
        });
        setBearingOverrides((prev) => {
          if (!prev.has(polygonId)) return prev;
          const next = new Map(prev);
          next.delete(polygonId);
          return next;
        });
        const nextOverrides = new Map(bearingOverridesRef.current);
        nextOverrides.delete(polygonId);
        bearingOverridesRef.current = nextOverrides;
        void applyPolygonParams(polygonId, safeParams, { skipQueue: true });
        return;
      }

      const optimizedOverride = {
        bearingDeg: best.bearingDeg,
        lineSpacingM,
        source: 'optimized' as const,
      };
      splitPerfLog(polygonId, 'optimizePolygonDirection selected exact optimized bearing', {
        seedBearingDeg,
        bestBearingDeg: best.bearingDeg,
        evaluatedBearingDegs: evaluated.map((candidate) => Math.round(candidate.bearingDeg * 10) / 10),
        bestExactCost: best.exactCost,
        bestQualityCost: best.qualityCost,
        bestMissionTimeSec: best.missionTimeSec,
        bestNormalizedTimeCost: best.normalizedTimeCost,
        metricKind: best.metricKind,
        timeWeight: EXACT_OPTIMIZE_TIME_WEIGHT,
        diagnostics: best.diagnostics,
      });
      setBearingOverrides((prev) => {
        const next = new Map(prev);
        next.set(polygonId, optimizedOverride);
        return next;
      });
      const nextOverrides = new Map(bearingOverridesRef.current);
      nextOverrides.set(polygonId, optimizedOverride);
      bearingOverridesRef.current = nextOverrides;
      void applyPolygonParams(polygonId, safeParams, { skipQueue: true });
    }, [
      altitudeMode,
      applyPolygonParams,
      isAsyncPolygonUpdateStillValid,
      minClearanceM,
      searchExactBearingNearSeed,
      terrainSource,
      turnExtendM,
    ]);

    // Rebuild 3D paths when altitude mode or minimum clearance changes
    useEffect(() => {
      if (!deckOverlayRef.current) return;
      const overlay = deckOverlayRef.current;
      // For each polygon with flight lines and tiles, rebuild path3D and update layer
      polygonFlightLines.forEach((fl, pid) => {
        const tiles = polygonTiles.get(pid) || [];
        if (!tiles || fl.flightLines.length === 0) return;
        const path3d = build3DFlightPath(
          fl,
          tiles,
          fl.lineSpacing,
          { altitudeAGL: fl.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, preconnected: true },
        );
        update3DPathLayer(overlay, pid, path3d, setDeckLayers);
      });
      syncFlightLinesVisibility(flightLinesVisibleRef.current);
    }, [altitudeMode, minClearanceM, syncFlightLinesVisibility]);
    // ---------- helpers ----------
    const fitMapToRings = useCallback((rings: [number, number][][]) => {
      if (!mapRef.current || rings.length === 0) return;
      let minLng = +Infinity, minLat = +Infinity, maxLng = -Infinity, maxLat = -Infinity;
      for (const ring of rings) {
        for (const [lng, lat] of ring) {
          minLng = Math.min(minLng, lng);
          maxLng = Math.max(maxLng, lng);
          minLat = Math.min(minLat, lat);
          maxLat = Math.max(maxLat, lat);
        }
      }
      if (Number.isFinite(minLng) && Number.isFinite(minLat) && Number.isFinite(maxLng) && Number.isFinite(maxLat)) {
        mapRef.current.fitBounds([[minLng, minLat], [maxLng, maxLat]], {
          padding: 50,
          duration: 1000,
          maxZoom: 18
        });
      }
    }, []);

    async function buildFlightLinesForPolygonAsync({
      polygonId,
      ring,
      bearingDeg,
      lineSpacingM,
      params,
      quality,
      startedGeneration,
    }: {
      polygonId: string;
      ring: [number, number][];
      bearingDeg: number;
      lineSpacingM: number;
      params?: PolygonParams;
      quality?: string;
      startedGeneration?: number;
    }): Promise<PlannedFlightGeometry | undefined> {
      const isStillValid = () => {
        if (startedGeneration === undefined) return true;
        const draw = drawRef.current as any;
        return shouldApplyAsyncPolygonUpdate({
          startedGeneration,
          currentGeneration: resetGenerationRef.current,
          polygonStillExists: !!draw?.get?.(polygonId),
        });
      };

      if (!params || !mapRef.current) {
        if (!mapRef.current || !isStillValid()) return undefined;
        return addFlightLinesForPolygon(mapRef.current, polygonId, ring, bearingDeg, lineSpacingM, params, quality);
      }

      try {
        const geometry = await getPlannedGeometryWorker().run({
          ring,
          bearingDeg,
          lineSpacingM,
          params,
        });

        if (!mapRef.current || !isStillValid()) return undefined;
        return renderFlightLinesForPolygon(mapRef.current, polygonId, geometry, quality, bearingDeg);
      } catch (error) {
        console.warn('[flight-lines] planned geometry worker failed, falling back to main thread', {
          polygonId,
          error: error instanceof Error ? error.message : String(error),
        });
        if (!mapRef.current || !isStillValid()) return undefined;
        return addFlightLinesForPolygon(mapRef.current, polygonId, ring, bearingDeg, lineSpacingM, params, quality);
      }
    }

    // ---------- analysis callbacks ----------
    const handleAnalysisResult = useCallback(
      async (result: PolygonAnalysisResult, tiles: any[]) => {
        if (pendingProgrammaticDeletesRef.current.has(result.polygonId)) {
          splitPerfLog(result.polygonId, 'ignoring late analysis result for polygon pending deletion', {
            polygonId: result.polygonId,
          });
          return;
        }
        // If polygon no longer exists (deleted), ignore late results
        const draw = drawRef.current as any;
        const stillExists = !!draw?.get?.(result.polygonId);
        if (!stillExists) {
          splitPerfLog(result.polygonId, 'ignoring late analysis result for polygon no longer in draw', {
            polygonId: result.polygonId,
          });
          return;
        }
        // 1) Commit result and defer parent notification to avoid React 18 render conflicts
        setPolygonResults((prev) => {
          const next = new Map(prev);
          next.set(result.polygonId, result);
          // Keep ref in sync immediately so batch callbacks (import) can read it before effect runs
            polygonResultsRef.current = next;
          debouncedAnalysisComplete();
          return next;
        });

        // 2) Store tiles
        const nextTiles = new Map(polygonTilesRef.current);
        nextTiles.set(result.polygonId, tiles);
        polygonTilesRef.current = nextTiles;
        setPolygonTiles(nextTiles);

        // Decide which heading/spacing to use when drawing
      let params = polygonParamsRef.current.get(result.polygonId);
      const override = bearingOverridesRef.current.get(result.polygonId);

        // If bulk preset exists (user clicked Apply All early) adopt params automatically
        if (!params && !override && bulkPresetParamsRef.current) {
          const preset = bulkPresetParamsRef.current;
          setPolygonParams(prev => {
            if (prev.has(result.polygonId)) return prev;
            const next = new Map(prev);
          next.set(result.polygonId, sanitizePolygonParams(preset));
          return next;
        });
          params = sanitizePolygonParams(preset);
      }

        if (!params) {
          if (override) {
            console.log(`Skipping params dialog for imported polygon ${result.polygonId}`);
            return;
          }
          setPendingParamPolygons(prev => {
            if (bulkApplyRef.current || prev.includes(result.polygonId) || bulkPresetParamsRef.current) return prev;
            return [...prev, result.polygonId];
          });
          return;
        }

        if (!mapRef.current) return;

        // Use override if present (e.g., file direction), otherwise terrain-optimal
        const safeParams = sanitizePolygonParams(params);
        const bearingDeg = override ? override.bearingDeg : result.result.contourDirDeg;

        const generation = resetGenerationRef.current;

        // Spacing: keep override spacing if present, otherwise recompute from params
        const spacing =
          override?.lineSpacingM ??
          getLineSpacingForParams(safeParams);

        const lines = await buildFlightLinesForPolygonAsync({
          polygonId: result.polygonId,
          ring: result.polygon.coordinates as [number, number][],
          bearingDeg,
          lineSpacingM: spacing,
          params: safeParams,
          quality: result.result.fitQuality,
          startedGeneration: generation,
        });
        if (!lines) return;

        const nextFlightLines = new Map(polygonFlightLinesRef.current);
        nextFlightLines.set(result.polygonId, { ...lines, altitudeAGL: safeParams.altitudeAGL });
        polygonFlightLinesRef.current = nextFlightLines;
        setPolygonFlightLines(nextFlightLines);
        syncFlightLinesVisibility(flightLinesVisibleRef.current);

        if (!suppressFlightLineEventsRef.current) {
          if (pendingGeometryRefreshRef.current.delete(result.polygonId)) {
            onFlightLinesUpdated?.('__all__');
          } else {
            onFlightLinesUpdated?.(result.polygonId);
          }
        }

        if (deckOverlayRef.current && lines.flightLines.length > 0) {
          const path3d = build3DFlightPath(
            lines,
            tiles,
            lines.lineSpacing,
            { altitudeAGL: safeParams.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, preconnected: true },
          );
          update3DPathLayer(deckOverlayRef.current, result.polygonId, path3d, setDeckLayers);
          const spacingForward = getForwardSpacingForParams(safeParams);
          if (spacingForward && spacingForward > 0) {
            // 3D trigger points sampled along the 3D path
            const samples = sampleCameraPositionsOnPlannedFlightGeometry(lines, path3d, spacingForward);
            const zOffset = 1; // lift triggers slightly above the path for visibility
            const ring = result.polygon.coordinates as [number, number][];
            const positions: [number, number, number][] = samples
              .filter(([lng, lat]) => isPointInRing(lng, lat, ring))
              .map(([lng,lat,alt]) => [lng,lat,alt + zOffset]);
            update3DTriggerPointsLayer(deckOverlayRef.current, result.polygonId, positions, setDeckLayers);
          } else {
            remove3DTriggerPointsLayer(deckOverlayRef.current, result.polygonId, setDeckLayers);
          }
          syncFlightLinesVisibility(flightLinesVisibleRef.current);
        }

        if (pendingOptimizeRef.current.has(result.polygonId)) {
          pendingOptimizeRef.current.delete(result.polygonId);
          const currentParams = polygonParamsRef.current.get(result.polygonId) ?? { altitudeAGL: 100, frontOverlap: 80, sideOverlap: 70 };
          const paramsToApply: PolygonParams = { ...currentParams, useCustomBearing: false };
          if ('customBearingDeg' in paramsToApply) delete (paramsToApply as any).customBearingDeg;
          const generation = resetGenerationRef.current;
          void withProcessingPolygon(result.polygonId, () =>
            runOptimizedBearingSearch(result.polygonId, paramsToApply, result, tiles, generation),
          );
        }
      },
      [debouncedAnalysisComplete, onFlightLinesUpdated, altitudeMode, minClearanceM, runOptimizedBearingSearch, scheduleGuardedTimeout, withProcessingPolygon, buildFlightLinesForPolygonAsync]
    );

    const memoizedOnAnalysisStart = useCallback((polygonId: string) => {
      onAnalysisStart?.(polygonId);
    }, [onAnalysisStart]);

    const memoizedOnError = useCallback((message: string, polygonId?: string) => {
      onError?.(message, polygonId);
    }, [onError]);

    const { analyzePolygon, cancelAnalysis, cancelAllAnalyses } = usePolygonAnalysis({
      mapboxToken,
      sampleStep,
      onAnalysisStart: memoizedOnAnalysisStart,
      onAnalysisComplete: handleAnalysisResult,
      onError: memoizedOnError,
    });

    const cleanupPolygonState = useCallback((polygonId: string) => {
      splitPerfLog(polygonId, 'cleanupPolygonState start', {
        hadResult: polygonResultsRef.current.has(polygonId),
        hadTiles: polygonTilesRef.current.has(polygonId),
        hadLines: polygonFlightLinesRef.current.has(polygonId),
        hadParams: polygonParamsRef.current.has(polygonId),
        hadOverride: bearingOverridesRef.current.has(polygonId),
        hadImportedOriginal: importedOriginalsRef.current.has(polygonId),
      });
      cancelAnalysis(polygonId);
      pendingOptimizeRef.current.delete(polygonId);

      if (mapRef.current) {
        removeFlightLinesForPolygon(mapRef.current, polygonId);
        removeTriggerPointsForPolygon(mapRef.current, polygonId);
      }
      if (deckOverlayRef.current) {
        remove3DPathLayer(deckOverlayRef.current, polygonId, setDeckLayers);
        remove3DTriggerPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
        remove3DCameraPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
      }

      setPolygonResults((prev) => {
        if (!prev.has(polygonId)) return prev;
        const next = new Map(prev);
        next.delete(polygonId);
        polygonResultsRef.current = next;
        debouncedAnalysisComplete();
        return next;
      });
      setPolygonFlightLines((prev) => {
        if (!prev.has(polygonId)) return prev;
        const next = new Map(prev);
        next.delete(polygonId);
        polygonFlightLinesRef.current = next;
        return next;
      });
      setPolygonTiles((prev) => {
        if (!prev.has(polygonId)) return prev;
        const next = new Map(prev);
        next.delete(polygonId);
        polygonTilesRef.current = next;
        return next;
      });
      setPolygonParams((prev) => {
        if (!prev.has(polygonId)) return prev;
        const next = new Map(prev);
        next.delete(polygonId);
        polygonParamsRef.current = next;
        return next;
      });
      setBearingOverrides((prev) => {
        if (!prev.has(polygonId)) return prev;
        const next = new Map(prev);
        next.delete(polygonId);
        bearingOverridesRef.current = next;
        return next;
      });
      setImportedOriginals((prev) => {
        if (!prev.has(polygonId)) return prev;
        const next = new Map(prev);
        next.delete(polygonId);
        importedOriginalsRef.current = next;
        return next;
      });
      setPendingParamPolygons((prev) => prev.filter((id) => id !== polygonId));
      if (processingPolygonIdsRef.current.delete(polygonId)) {
        syncProcessingPerimeterOverlay();
        if (processingPolygonIdsRef.current.size === 0) {
          stopProcessingPerimeterAnimation();
        }
      }

      scheduleGuardedTimeout(() => {
        const isTransactionCleanup =
          isApplyingPolygonOperationRef.current &&
          polygonOperationAffectedIdsRef.current.has(polygonId);
        const draw = drawRef.current as any;
        const remainingPolygonIds = Array.isArray(draw?.getAll?.()?.features)
          ? draw.getAll().features
              .filter((feature: any) => feature?.geometry?.type === 'Polygon')
              .map((feature: any) => String(feature.id))
          : [];
        splitPerfLog(polygonId, 'cleanupPolygonState settled', {
          remainingPolygonIds,
          resultIds: Array.from(polygonResultsRef.current.keys()),
          importedOriginalIds: Array.from(importedOriginalsRef.current.keys()),
        });
        if (!isTransactionCleanup) {
          onPolygonSelected?.(null);
        }
        if (!isTransactionCleanup && !suppressFlightLineEventsRef.current) {
          onFlightLinesUpdated?.('__all__');
        }
        try {
          const coll = (drawRef.current as any)?.getAll?.();
          const hasAny = Array.isArray(coll?.features) && coll.features.some((f: any) => f?.geometry?.type === 'Polygon');
          if (!isTransactionCleanup && !hasAny) onClearGSD?.();
        } catch {}
      }, 0);
    }, [cancelAnalysis, debouncedAnalysisComplete, onClearGSD, onFlightLinesUpdated, onPolygonSelected, scheduleGuardedTimeout]);

    const deletePolygonFeature = useCallback((polygonId: string) => {
      const draw = drawRef.current as any;
      if (!draw) {
        if (pendingProgrammaticDeletesRef.current.delete(polygonId)) {
          cleanupPolygonState(polygonId);
        }
        return;
      }

      const attemptDelete = () => {
        const beforeExists = !!draw.get?.(polygonId);
        try {
          draw.changeMode?.('simple_select');
        } catch {}
        try {
          draw.delete?.(polygonId);
        } catch {}
        if (!draw.get?.(polygonId)) {
          splitPerfLog(polygonId, 'deletePolygonFeature removed polygon via draw.delete(id)', {
            beforeExists,
          });
          return true;
        }
        try {
          draw.delete?.([polygonId]);
        } catch {}
        if (!draw.get?.(polygonId)) {
          splitPerfLog(polygonId, 'deletePolygonFeature removed polygon via draw.delete([id])', {
            beforeExists,
          });
          return true;
        }
        try {
          const collection = draw.getAll?.();
          const features = Array.isArray(collection?.features)
            ? collection.features.filter((feature: any) => String(feature?.id ?? '') !== polygonId)
            : null;
          if (features && typeof draw.set === 'function') {
            draw.set({
              type: 'FeatureCollection',
              features,
            });
          }
        } catch {}
        const removed = !draw.get?.(polygonId);
        splitPerfLog(polygonId, 'deletePolygonFeature attempt finished', {
          beforeExists,
          removed,
          remainingPolygonIds: Array.isArray(draw.getAll?.()?.features)
            ? draw.getAll().features
                .filter((feature: any) => feature?.geometry?.type === 'Polygon')
                .map((feature: any) => String(feature.id))
            : [],
        });
        return removed;
      };

      pendingProgrammaticDeletesRef.current.add(polygonId);
      splitPerfLog(polygonId, 'deletePolygonFeature scheduled', {
        remainingPolygonIds: Array.isArray(draw.getAll?.()?.features)
          ? draw.getAll().features
              .filter((feature: any) => feature?.geometry?.type === 'Polygon')
              .map((feature: any) => String(feature.id))
          : [],
      });
      attemptDelete();
      scheduleGuardedTimeout(() => {
        if (draw.get?.(polygonId)) {
          splitPerfLog(polygonId, 'deletePolygonFeature retrying after initial failure');
          attemptDelete();
        }
        if (pendingProgrammaticDeletesRef.current.delete(polygonId)) {
          cleanupPolygonState(polygonId);
        }
      }, 0);
    }, [cleanupPolygonState, scheduleGuardedTimeout]);

    const restorePolygonOperationSnapshots = useCallback(async (
      snapshots: PolygonSnapshot[],
      affectedPolygonIds: string[],
      selectionAfter: string | null,
    ) => {
      const draw = drawRef.current as any;
      if (!draw) {
        onError?.('Map is not ready for polygon operations.');
        return false;
      }

      const affectedIds = Array.from(new Set(affectedPolygonIds));
      const nextMetadata = applyPolygonSnapshotsToMetadata({
        params: polygonParamsRef.current,
        overrides: bearingOverridesRef.current,
        importedOriginals: importedOriginalsRef.current,
      }, snapshots, affectedIds);

      cancelPolygonMerge();
      setPolygonOperationApplying(true);
      polygonOperationAffectedIdsRef.current = new Set(affectedIds);
      suppressHistoryInvalidationRef.current = true;
      suspendAutoAnalysisRef.current = true;
      const prevSuppressEvents = suppressFlightLineEventsRef.current;
      suppressFlightLineEventsRef.current = true;

      try {
        for (const polygonId of affectedIds) {
          cancelAnalysis(polygonId);
          pendingOptimizeRef.current.delete(polygonId);
          backendPartitionSolutionsRef.current.delete(polygonId);
          pendingGeometryRefreshRef.current.delete(polygonId);
          if (processingPolygonIdsRef.current.delete(polygonId)) {
            syncProcessingPerimeterOverlay();
          }
          if (mapRef.current) {
            removeFlightLinesForPolygon(mapRef.current, polygonId);
            removeTriggerPointsForPolygon(mapRef.current, polygonId);
          }
          if (deckOverlayRef.current) {
            remove3DPathLayer(deckOverlayRef.current, polygonId, setDeckLayers);
            remove3DTriggerPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
            remove3DCameraPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
          }
        }

        setPolygonResults((prev) => {
          const next = new Map(prev);
          for (const polygonId of affectedIds) next.delete(polygonId);
          polygonResultsRef.current = next;
          debouncedAnalysisComplete();
          return next;
        });
        setPolygonTiles((prev) => {
          const next = new Map(prev);
          for (const polygonId of affectedIds) next.delete(polygonId);
          polygonTilesRef.current = next;
          return next;
        });
        setPolygonFlightLines((prev) => {
          const next = new Map(prev);
          for (const polygonId of affectedIds) next.delete(polygonId);
          polygonFlightLinesRef.current = next;
          return next;
        });

        polygonParamsRef.current = nextMetadata.params;
        setPolygonParams(new Map(nextMetadata.params));
        bearingOverridesRef.current = nextMetadata.overrides;
        setBearingOverrides(new Map(nextMetadata.overrides));
        importedOriginalsRef.current = nextMetadata.importedOriginals;
        setImportedOriginals(new Map(nextMetadata.importedOriginals));
        setPendingParamPolygons((prev) => prev.filter((polygonId) => !affectedIds.includes(polygonId)));

        try {
          draw.changeMode?.('simple_select', { featureIds: [] });
        } catch {
          try {
            draw.changeMode?.('simple_select');
          } catch {}
        }

        for (const polygonId of affectedIds) {
          pendingProgrammaticDeletesRef.current.add(polygonId);
        }
        try {
          draw.delete?.(affectedIds);
        } catch {}

        const hasRemainingAffected = affectedIds.some((polygonId) => !!draw.get?.(polygonId));
        if (hasRemainingAffected) {
          const remainingFeatures = Array.isArray(draw?.getAll?.()?.features)
            ? draw.getAll().features.filter((feature: any) => !affectedIds.includes(String(feature?.id ?? '')))
            : [];
          try {
            if (typeof draw.set === 'function') {
              draw.set({
                type: 'FeatureCollection',
                features: remainingFeatures,
              });
            }
          } catch {}
        }
        for (const polygonId of affectedIds) {
          pendingProgrammaticDeletesRef.current.delete(polygonId);
        }

        for (const snapshot of snapshots) {
          addPolygonSnapshotToDraw(snapshot);
        }

        if (snapshots.length > 0) {
          fitMapToRings(snapshots.map((snapshot) => snapshot.feature.geometry.coordinates[0] as [number, number][]));
        }

        const optimizeAfterIds = snapshots
          .filter((snapshot) => snapshot.feature.properties?.source === 'terrain-face-merge' && !snapshot.override)
          .map((snapshot) => snapshot.feature.id);
        for (const polygonId of optimizeAfterIds) {
          pendingOptimizeRef.current.add(polygonId);
        }

        selectedPolygonIdRef.current = selectionAfter;
        onPolygonSelected?.(selectionAfter);

        const analysisPromises: Promise<any>[] = [];
        for (const snapshot of snapshots) {
          const feature = draw.get?.(snapshot.feature.id);
          if (feature?.geometry?.type === 'Polygon') {
            analysisPromises.push(analyzePolygon(snapshot.feature.id, feature));
          }
        }
        await Promise.allSettled(analysisPromises);

        await new Promise((resolve) => setTimeout(resolve, 0));
        return true;
      } catch (error) {
        onError?.(`Polygon operation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
        return false;
      } finally {
        suppressFlightLineEventsRef.current = prevSuppressEvents;
        suspendAutoAnalysisRef.current = false;
        suppressHistoryInvalidationRef.current = false;
        polygonOperationAffectedIdsRef.current = new Set();
        setPolygonOperationApplying(false);
        syncPolygonMergeState();
        if (!prevSuppressEvents) {
          onFlightLinesUpdated?.('__all__');
        }
      }
    }, [
      addPolygonSnapshotToDraw,
      analyzePolygon,
      cancelAnalysis,
      cancelPolygonMerge,
      debouncedAnalysisComplete,
      fitMapToRings,
      onError,
      onFlightLinesUpdated,
      onPolygonSelected,
      setPolygonOperationApplying,
      syncPolygonMergeState,
      syncProcessingPerimeterOverlay,
    ]);

    const applyPolygonOperationTransaction = useCallback(async (
      transaction: PolygonOperationTransaction,
      direction: 'forward' | 'backward',
    ) => {
      const snapshots = direction === 'forward'
        ? transaction.after.map((snapshot) => clonePolygonSnapshot(snapshot))
        : transaction.before.map((snapshot) => clonePolygonSnapshot(snapshot));
      const selectionAfter = direction === 'forward' ? transaction.selectionAfter : transaction.selectionBefore;
      return restorePolygonOperationSnapshots(
        snapshots,
        collectAffectedPolygonIds(transaction),
        selectionAfter,
      );
    }, [restorePolygonOperationSnapshots]);

    const storePolygonOperationTransaction = useCallback((transaction: PolygonOperationTransaction) => {
      replacePolygonOperationHistory(
        pushPolygonOperationTransaction(polygonOperationHistoryRef.current, {
          ...transaction,
          before: transaction.before.map((snapshot) => clonePolygonSnapshot(snapshot)),
          after: transaction.after.map((snapshot) => clonePolygonSnapshot(snapshot)),
        }),
      );
    }, [replacePolygonOperationHistory]);

    const canUndoPolygonOperation = useCallback(() => {
      return !isApplyingPolygonOperationRef.current && polygonOperationHistoryRef.current.undoStack.length > 0;
    }, []);

    const canRedoPolygonOperation = useCallback(() => {
      return !isApplyingPolygonOperationRef.current && polygonOperationHistoryRef.current.redoStack.length > 0;
    }, []);

    const undoPolygonOperation = useCallback(async () => {
      if (isApplyingPolygonOperationRef.current) return false;
      cancelPolygonMerge();
      const { history, transaction } = popUndoPolygonOperation(polygonOperationHistoryRef.current);
      if (!transaction) return false;
      const applied = await applyPolygonOperationTransaction(transaction, 'backward');
      if (!applied) return false;
      replacePolygonOperationHistory(history);
      return true;
    }, [applyPolygonOperationTransaction, cancelPolygonMerge, replacePolygonOperationHistory]);

    const redoPolygonOperation = useCallback(async () => {
      if (isApplyingPolygonOperationRef.current) return false;
      cancelPolygonMerge();
      const { history, transaction } = popRedoPolygonOperation(polygonOperationHistoryRef.current);
      if (!transaction) return false;
      const applied = await applyPolygonOperationTransaction(transaction, 'forward');
      if (!applied) return false;
      replacePolygonOperationHistory(history);
      return true;
    }, [applyPolygonOperationTransaction, cancelPolygonMerge, replacePolygonOperationHistory]);

    const startPolygonMerge = useCallback((polygonId: string) => {
      if (isApplyingPolygonOperationRef.current) return;
      const nextState = derivePolygonMergeState({
        features: getAllDrawPolygonFeatures(),
        primaryPolygonId: polygonId,
        selectedPolygonIds: [polygonId],
      });
      publishMergeState(nextState);
      if (nextState.warning && nextState.selectedPolygonIds.length <= 1 && nextState.eligiblePolygonIds.length === 0) {
        onError?.(nextState.warning, polygonId);
      }
      selectedPolygonIdRef.current = polygonId;
      onPolygonSelected?.(polygonId);
    }, [getAllDrawPolygonFeatures, onError, onPolygonSelected, publishMergeState]);

    const canStartPolygonMerge = useCallback((polygonId: string) => {
      if (isApplyingPolygonOperationRef.current) return false;
      const nextState = derivePolygonMergeState({
        features: getAllDrawPolygonFeatures(),
        primaryPolygonId: polygonId,
        selectedPolygonIds: [polygonId],
      });
      return nextState.mode === 'selecting' && nextState.eligiblePolygonIds.length > 0;
    }, [getAllDrawPolygonFeatures]);

    const togglePolygonMergeCandidate = useCallback((polygonId: string) => {
      if (isApplyingPolygonOperationRef.current) return;
      const current = mergeStateRef.current;
      if (current.mode !== 'selecting' || !current.primaryPolygonId || polygonId === current.primaryPolygonId) return;
      const isSelected = current.selectedPolygonIds.includes(polygonId);
      const isEligible = current.eligiblePolygonIds.includes(polygonId);
      if (!isSelected && !isEligible) return;
      const nextSelected = isSelected
        ? current.selectedPolygonIds.filter((candidateId) => candidateId !== polygonId)
        : [...current.selectedPolygonIds, polygonId];
      syncPolygonMergeState(current.primaryPolygonId, nextSelected);
      selectedPolygonIdRef.current = current.primaryPolygonId;
      onPolygonSelected?.(current.primaryPolygonId);
    }, [onPolygonSelected, syncPolygonMergeState]);

    const confirmPolygonMerge = useCallback(async (): Promise<{ mergedPolygonId: string | null; replaced: boolean }> => {
      if (isApplyingPolygonOperationRef.current) {
        return { mergedPolygonId: null, replaced: false };
      }
      const current = mergeStateRef.current;
      if (current.mode !== 'selecting' || !current.primaryPolygonId || !current.canConfirm || !current.previewRing) {
        if (current.warning) onError?.(current.warning, current.primaryPolygonId ?? undefined);
        return { mergedPolygonId: null, replaced: false };
      }

      const before = current.selectedPolygonIds
        .map((polygonId) => snapshotPolygonFeature(polygonId))
        .filter((snapshot): snapshot is PolygonSnapshot => snapshot !== null);
      const primarySnapshot = before.find((snapshot) => snapshot.feature.id === current.primaryPolygonId);
      if (before.length !== current.selectedPolygonIds.length || !primarySnapshot) {
        onError?.('Unable to collect all polygons for merge.', current.primaryPolygonId);
        return { mergedPolygonId: null, replaced: false };
      }

      const mergedPolygonId = createPolygonFeatureId();
      const mergedFeature = createPolygonFeatureSnapshot({
        id: mergedPolygonId,
        ring: current.previewRing,
        properties: {
          ...(primarySnapshot.feature.properties ?? {}),
          source: 'terrain-face-merge',
          mergeSourceIds: [...current.selectedPolygonIds],
        },
      });
      if (!mergedFeature) {
        onError?.('Unable to build merged polygon geometry.', current.primaryPolygonId);
        return { mergedPolygonId: null, replaced: false };
      }
      delete mergedFeature.properties.parentPolygonId;

      const transaction: PolygonOperationTransaction = {
        kind: 'merge',
        label: 'Merge Areas',
        before,
        after: [{
          feature: mergedFeature,
          params: primarySnapshot.params ? { ...primarySnapshot.params } : undefined,
        }],
        selectionBefore: current.primaryPolygonId,
        selectionAfter: mergedPolygonId,
      };

      const applied = await applyPolygonOperationTransaction(transaction, 'forward');
      if (!applied) {
        return { mergedPolygonId: null, replaced: false };
      }
      storePolygonOperationTransaction(transaction);
      publishMergeState(createIdlePolygonMergeState());
      return { mergedPolygonId, replaced: true };
    }, [
      applyPolygonOperationTransaction,
      createPolygonFeatureId,
      onError,
      publishMergeState,
      snapshotPolygonFeature,
      storePolygonOperationTransaction,
    ]);

    const editPolygonBoundary = useCallback((polygonId: string) => {
      const draw = drawRef.current as any;
      const feature = draw?.get?.(polygonId);
      if (!draw || feature?.geometry?.type !== 'Polygon') return;

      suppressSelectionDialogUntilRef.current = Date.now() + 1000;
      scheduleGuardedTimeout(() => onPolygonSelected?.(polygonId), 0);

      try {
        draw.changeMode('simple_select', { featureIds: [polygonId] });
      } catch {}

      scheduleGuardedTimeout(() => {
        try {
          draw.changeMode('direct_select', { featureId: polygonId });
        } catch {
          try {
            draw.changeMode('simple_select', { featureIds: [polygonId] });
          } catch {}
        }
      }, 0);
    }, [onPolygonSelected, scheduleGuardedTimeout]);

    const clearDrawSelectionForPan = useCallback(() => {
      const draw = drawRef.current as any;
      if (!draw) return;
      suppressNextEmptySelectionRef.current += 1;
      try {
        draw.changeMode('simple_select', { featureIds: [] });
        return;
      } catch {}
      try {
        draw.changeMode('simple_select');
      } catch {}
    }, []);

    const getClickedVertex = useCallback((map: MapboxMap, point: { x: number; y: number }) => {
      const features = map.queryRenderedFeatures(point as any);
      const vertex = features.find((feature: any) => {
        const meta = feature?.properties?.meta;
        const parent = feature?.properties?.parent;
        const coordPath = feature?.properties?.coord_path;
        return meta === 'vertex' && parent != null && coordPath != null;
      });
      if (!vertex) return null;
      const parentId = String(vertex.properties?.parent ?? '');
      const coordPath = String(vertex.properties?.coord_path ?? '');
      if (!parentId || !coordPath) return null;
      return { parentId, coordPath };
    }, []);

    const hitInteractiveMapFeature = useCallback((map: MapboxMap, point: { x: number; y: number }) => {
      const features = map.queryRenderedFeatures(point as any);
      return features.some((feature: any) => {
        const layerId = String(feature?.layer?.id ?? '');
        const meta = feature?.properties?.meta;
        if (meta === 'feature' || meta === 'vertex' || meta === 'midpoint') return true;
        return (
          layerId.startsWith('flight-lines-layer-') ||
          layerId.startsWith('flight-triggers-layer-') ||
          layerId.startsWith('flight-triggers-label-') ||
          layerId === 'selected-polygon-highlight-fill' ||
          layerId === 'selected-polygon-highlight-outer' ||
          layerId === 'selected-polygon-highlight-inner'
        );
      });
    }, []);

    const syncSelectedPolygonHighlight = useCallback(() => {
      const map = mapRef.current;
      const draw = drawRef.current as any;
      if (!map) return;
      const polygonId = selectedPolygonIdRef.current;
      const allPolygons: Array<{ polygonId: string; ring: [number, number][] }> = (draw?.getAll?.()?.features ?? [])
        .filter((feature: any) => feature?.geometry?.type === 'Polygon' && feature?.id)
        .map((feature: any) => ({
          polygonId: String(feature.id),
          ring: feature.geometry.coordinates?.[0] as [number, number][],
        }))
        .filter((polygon: { polygonId: string; ring: [number, number][] }) => Array.isArray(polygon.ring) && polygon.ring.length >= 4);
      if (!polygonId) {
        setNonSelectedPolygonDimMask(map, []);
        setSelectedPolygonHighlight(map, null);
        setFlightLineSelectionEmphasis(map, null, flightLinesVisibleRef.current);
        syncFlightLinesVisibility(flightLinesVisibleRef.current);
        return;
      }
      const mergePreviewRing =
        mergeStateRef.current.mode === 'selecting' &&
        mergeStateRef.current.primaryPolygonId === polygonId
          ? mergeStateRef.current.previewRing ?? undefined
          : undefined;
      const feature = draw?.get?.(polygonId);
      const ring = mergePreviewRing ?? (
        feature?.geometry?.type === 'Polygon'
          ? (feature.geometry.coordinates?.[0] as [number, number][] | undefined)
          : undefined
      );
      if (!ring || ring.length < 4) {
        setNonSelectedPolygonDimMask(map, []);
        setSelectedPolygonHighlight(map, null);
        setFlightLineSelectionEmphasis(map, null, flightLinesVisibleRef.current);
        syncFlightLinesVisibility(flightLinesVisibleRef.current);
        return;
      }
      setNonSelectedPolygonDimMask(map, allPolygons.filter((polygon) => polygon.polygonId !== polygonId));
      setSelectedPolygonHighlight(map, { polygonId, ring });
      setFlightLineSelectionEmphasis(map, polygonId, flightLinesVisibleRef.current);
      syncFlightLinesVisibility(flightLinesVisibleRef.current);
    }, [syncFlightLinesVisibility]);

    // ---------- Mapbox Draw handlers ----------
    const handleDrawCreate = useCallback((e: any) => {
      if (suspendAutoAnalysisRef.current) return;
      if (!suppressHistoryInvalidationRef.current) {
        cancelPolygonMerge();
        invalidatePolygonOperationHistory();
      }
      e.features.forEach((feature: any) => {
        if (feature.geometry.type === 'Polygon') {
          const polygonId = String(feature.id ?? '');
          const ring = feature.geometry.coordinates?.[0] as [number, number][] | undefined;
          if (polygonId && ring && ring.length >= 4) {
            scheduleGuardedTimeout(() => onPolygonSelected?.(polygonId), 0);
            scheduleGuardedTimeout(() => onRequestParams?.(polygonId, ring), 0);
          }
          analyzePolygon(feature.id, feature);
        }
      });
      syncSelectedPolygonHighlight();
    }, [analyzePolygon, cancelPolygonMerge, invalidatePolygonOperationHistory, onPolygonSelected, onRequestParams, scheduleGuardedTimeout, syncSelectedPolygonHighlight]);

    const handleDrawUpdate = useCallback((e: any) => {
      if (suspendAutoAnalysisRef.current) return;
      if (!suppressHistoryInvalidationRef.current) {
        cancelPolygonMerge();
        invalidatePolygonOperationHistory();
      }
      e.features.forEach((feature: any) => {
        if (feature.geometry.type === 'Polygon') {
          pendingGeometryRefreshRef.current.add(String(feature.id));
          analyzePolygon(feature.id, feature);
        }
      });
      syncSelectedPolygonHighlight();
    }, [analyzePolygon, cancelPolygonMerge, invalidatePolygonOperationHistory, syncSelectedPolygonHighlight]);

    const handleDrawDelete = useCallback((e: any) => {
      if (!suppressHistoryInvalidationRef.current) {
        cancelPolygonMerge();
        invalidatePolygonOperationHistory();
      }
      e.features.forEach((feature: any) => {
        if (feature.geometry.type === 'Polygon') {
          const polygonId = String(feature.id);
          const wasProgrammatic = pendingProgrammaticDeletesRef.current.delete(polygonId);
          splitPerfLog(polygonId, 'draw.delete event received', {
            wasProgrammatic,
            remainingPolygonIds: Array.isArray((drawRef.current as any)?.getAll?.()?.features)
              ? (drawRef.current as any).getAll().features
                  .filter((candidate: any) => candidate?.geometry?.type === 'Polygon')
                  .map((candidate: any) => String(candidate.id))
              : [],
          });
          cleanupPolygonState(polygonId);
        }
      });
      syncSelectedPolygonHighlight();
    }, [cancelPolygonMerge, cleanupPolygonState, invalidatePolygonOperationHistory, syncSelectedPolygonHighlight]);

    // ---------- Map init ----------
    const onMapLoad = useCallback(
      (map: MapboxMap, draw: MapboxDraw, overlay: MapboxOverlay) => {
        mapRef.current = map;
        drawRef.current = draw;
        deckOverlayRef.current = overlay;
        applyTerrainSourceToMap(map, terrainDemSourceTemplateRef.current, terrainSourceRef.current);
        syncFlightLinesVisibility(flightLinesVisibleRef.current);
        map.on('draw.create', handleDrawCreate);
        map.on('draw.update', handleDrawUpdate);
        map.on('draw.delete', handleDrawDelete);
        map.on('draw.create', syncProcessingPerimeterOverlay);
        map.on('draw.update', syncProcessingPerimeterOverlay);
        map.on('draw.delete', syncProcessingPerimeterOverlay);
        syncProcessingPerimeterOverlay();
        syncSelectedPolygonHighlight();
        if (processingPolygonIdsRef.current.size > 0) {
          startProcessingPerimeterAnimation();
        }
        map.on('mousedown', (e: any) => {
          try {
            const drawMode = (drawRef.current as any)?.getMode?.();
            if (drawMode !== 'direct_select') {
              vertexClickCandidateRef.current = null;
              return;
            }
            const hit = getClickedVertex(map, e.point);
            if (!hit) {
              vertexClickCandidateRef.current = null;
              return;
            }
            vertexClickCandidateRef.current = {
              ...hit,
              point: { x: e.point.x, y: e.point.y },
              timestampMs: Date.now(),
            };
          } catch {
            vertexClickCandidateRef.current = null;
          }
        });
        map.on('mouseup', (e: any) => {
          try {
            const candidate = vertexClickCandidateRef.current;
            vertexClickCandidateRef.current = null;
            const drawMode = (drawRef.current as any)?.getMode?.();
            if (!candidate || drawMode !== 'direct_select') return;

            const dx = e.point.x - candidate.point.x;
            const dy = e.point.y - candidate.point.y;
            const elapsedMs = Date.now() - candidate.timestampMs;
            if ((dx * dx) + (dy * dy) > 16 || elapsedMs > 450) return;

            const hit = getClickedVertex(map, e.point);
            if (!hit || hit.parentId !== candidate.parentId || hit.coordPath !== candidate.coordPath) return;

            const draw = drawRef.current as any;
            const feature = draw?.get?.(candidate.parentId);
            const ring =
              feature?.geometry?.type === 'Polygon'
                ? (feature.geometry.coordinates?.[0] as [number, number][] | undefined)
                : undefined;
            if (!ring || ring.length <= 4) return;

            scheduleGuardedTimeout(() => {
              try {
                draw?.trash?.();
              } catch {}
              syncSelectedPolygonHighlight();
            }, 0);
          } catch {}
        });
        map.on('click', (e: any) => {
          try {
            const drawMode = (drawRef.current as any)?.getMode?.();
            if (drawMode === 'draw_polygon' || drawMode === 'direct_select') return;
            if (hitInteractiveMapFeature(map, e.point)) return;
            if (mergeStateRef.current.mode === 'selecting') return;
            scheduleGuardedTimeout(() => onPolygonSelected?.(null), 0);
          } catch {}
        });
        // Open params dialog when user selects an existing polygon
        map.on('draw.selectionchange', (e: any) => {
          try {
            const feats = Array.isArray(e?.features) ? e.features : [];
            const f = feats.find((f: any) => f?.geometry?.type === 'Polygon');
            if (mergeStateRef.current.mode === 'selecting') {
              const primaryPolygonId = mergeStateRef.current.primaryPolygonId;
              if (!f) {
                if (suppressNextEmptySelectionRef.current > 0) {
                  suppressNextEmptySelectionRef.current -= 1;
                }
                if (primaryPolygonId) {
                  scheduleGuardedTimeout(() => onPolygonSelected?.(primaryPolygonId), 0);
                }
                return;
              }
              const pid = String(f.id ?? '');
              if (pid && primaryPolygonId) {
                scheduleGuardedTimeout(() => {
                  if (pid !== primaryPolygonId) {
                    togglePolygonMergeCandidate(pid);
                  } else {
                    onPolygonSelected?.(primaryPolygonId);
                  }
                }, 0);
                scheduleGuardedTimeout(() => clearDrawSelectionForPan(), 0);
              }
              return;
            }
            if (!f) {
              if (suppressNextEmptySelectionRef.current > 0) {
                suppressNextEmptySelectionRef.current -= 1;
                return;
              }
              scheduleGuardedTimeout(() => onPolygonSelected?.(null), 0);
              return;
            }
            const pid = f.id as string;
            const ring = (f.geometry?.coordinates?.[0]) as [number, number][] | undefined;
            if (pid && ring && ring.length >= 4) {
              const drawMode = (drawRef.current as any)?.getMode?.();
              const shouldSuppressDialog =
                drawMode === 'direct_select' || Date.now() < suppressSelectionDialogUntilRef.current;
              // Defer to avoid selection-change re-entrancy
              scheduleGuardedTimeout(() => onPolygonSelected?.(pid), 0);
              if (!shouldSuppressDialog) {
                // Keep normal polygon clicks lightweight: once the app has recorded the
                // selection, clear Draw's internal feature selection so left-drag pans
                // the map again instead of staying in feature-move mode.
                scheduleGuardedTimeout(() => clearDrawSelectionForPan(), 0);
              }
            }
          } catch {}
        });
      },
      [applyTerrainSourceToMap, clearDrawSelectionForPan, getClickedVertex, handleDrawCreate, handleDrawUpdate, handleDrawDelete, hitInteractiveMapFeature, onPolygonSelected, scheduleGuardedTimeout, syncFlightLinesVisibility, syncSelectedPolygonHighlight, togglePolygonMergeCandidate]
    );

    useEffect(() => {
      terrainDemSourceTemplateRef.current = terrainDemUrlTemplate;
      if (mapRef.current && mapRef.current.isStyleLoaded()) {
        applyTerrainSourceToMap(mapRef.current, terrainDemUrlTemplate, terrainSource);
      }
    }, [applyTerrainSourceToMap, terrainDemUrlTemplate, terrainSource]);

    useEffect(() => {
      selectedPolygonIdRef.current = selectedPolygonId;
      syncSelectedPolygonHighlight();
    }, [selectedPolygonId, syncSelectedPolygonHighlight]);

    useEffect(() => {
      syncSelectedPolygonHighlight();
    }, [mergeState, syncSelectedPolygonHighlight]);

    useEffect(() => {
      const onKeyDown = (event: KeyboardEvent) => {
        if (event.defaultPrevented) return;
        const target = event.target;
        if (target instanceof HTMLElement) {
          const tagName = target.tagName.toLowerCase();
          if (tagName === 'input' || tagName === 'textarea' || tagName === 'select' || target.isContentEditable) {
            return;
          }
        }
        const normalizedKey = event.key.toLowerCase();
        const drawMode = (drawRef.current as any)?.getMode?.();
        const isModifierPressed = event.metaKey || event.ctrlKey;
        const isUndoShortcut = isModifierPressed && !event.shiftKey && !event.altKey && normalizedKey === 'z';
        const isRedoShortcut = isModifierPressed && !event.altKey && (
          normalizedKey === 'y' ||
          (normalizedKey === 'z' && event.shiftKey)
        );
        if (isUndoShortcut || isRedoShortcut) {
          if (drawMode === 'draw_polygon' || drawMode === 'direct_select') return;
          if (isApplyingPolygonOperationRef.current) {
            event.preventDefault();
            return;
          }
          event.preventDefault();
          if (isUndoShortcut) {
            void undoPolygonOperation();
          } else {
            void redoPolygonOperation();
          }
          return;
        }
        if (event.key !== 'Escape') return;
        if (mergeStateRef.current.mode === 'selecting') {
          event.preventDefault();
          const primaryPolygonId = mergeStateRef.current.primaryPolygonId;
          cancelPolygonMerge();
          if (primaryPolygonId) {
            onPolygonSelected?.(primaryPolygonId);
          }
          return;
        }
        if (!selectedPolygonIdRef.current) return;
        if (drawMode === 'draw_polygon' || drawMode === 'direct_select') return;
        onPolygonSelected?.(null);
      };
      window.addEventListener('keydown', onKeyDown);
      return () => window.removeEventListener('keydown', onKeyDown);
    }, [cancelPolygonMerge, onPolygonSelected, redoPolygonOperation, undoPolygonOperation]);

    useMapInitialization({
      mapboxToken,
      center,
      zoom,
      mapContainer,
      onLoad: onMapLoad,
      onError: memoizedOnError,
    });

    // ---------- Draw utils ----------
    const addRingAsDrawFeature = useCallback((ring: [number, number][], name?: string, extraProps?: Record<string, any>): string | undefined => {
      const draw = drawRef.current as any;
      if (!draw) return;

      const normalizedRing = normalizeRingForGeometryOps(ring);
      if (!normalizedRing) return;

      const feature = {
        type: 'Feature',
        properties: { name: name || '', ...(extraProps || {}) },
        geometry: { type: 'Polygon', coordinates: [normalizedRing] },
      };

      const id = draw.add(feature);
      const featureId = Array.isArray(id) ? id[0] : id;

      // Only auto-analyze if not suspended (imports will analyze explicitly)
      if (!suspendAutoAnalysisRef.current) {
        const f = (draw.get as any)?.(featureId);
        if (f?.geometry?.type === 'Polygon') analyzePolygon(featureId, f);
      }
      return featureId as string;
    }, [analyzePolygon]);

    // ---------- KML import (unchanged behavior) ----------
    const importKmlFromText = useCallback(async (kmlText: string) => {
      try {
        cancelPolygonMerge();
        invalidatePolygonOperationHistory();
        const polygons = parseKmlPolygons(kmlText);
        let added = 0; const newIds: string[] = [];
        suspendAutoAnalysisRef.current = true;
        for (const p of polygons) {
          if (p.ring?.length >= 4) {
            const id = addRingAsDrawFeature(p.ring, p.name, { source: 'kml' });
            if (id) newIds.push(id);
            added++;
          }
        }

        if (added > 0 && mapRef.current) {
          const bounds = calculateKmlBounds(polygons.filter(p => p.ring?.length >= 4));
          if (bounds) {
            const padding = 0.001;
            const padded: [[number, number], [number, number]] = [
              [bounds.minLng - padding, bounds.minLat - padding],
              [bounds.maxLng + padding, bounds.maxLat + padding]
            ];
            mapRef.current.fitBounds(padded, { padding: 50, duration: 1000, maxZoom: 18 });
          }
        } else if (polygons.length > 0) {
          onError?.("KML contained polygons but none were valid (need at least 4 coordinates)");
        } else {
          onError?.("No valid polygons found in KML file");
        }

        // Run analyses after all added
        suspendAutoAnalysisRef.current = false;
        for (const pid of newIds) {
          const draw = drawRef.current as any;
            const f = draw?.get?.(pid);
            if (f?.geometry?.type === 'Polygon') analyzePolygon(pid, f);
        }
        return { added, total: polygons.length };
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to parse KML file";
        onError?.(message);
        return { added: 0, total: 0 };
      } finally {
        suspendAutoAnalysisRef.current = false;
      }
    }, [addRingAsDrawFeature, cancelPolygonMerge, invalidatePolygonOperationHistory, onError]);

    const handleKmlFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []).filter(f => /\.(kml|kmz)$/i.test(f.name));
      if (files.length === 0) {
        onError?.("Please select valid .kml or .kmz files");
        return;
      }
      for (const file of files) {
        try {
          let kmlText: string | null = null;
          if (/\.kmz$/i.test(file.name)) {
            const buf = await file.arrayBuffer();
            try {
              kmlText = await extractKmlFromKmz(buf);
            } catch (err) {
              onError?.(`Failed to extract KMZ ${file.name}: ${err instanceof Error ? err.message : 'Unknown error'}`);
              continue;
            }
          } else { // .kml
            kmlText = await file.text();
          }
          if (kmlText) {
            await importKmlFromText(kmlText);
          }
        } catch (error) {
          onError?.(`Failed to read file ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }
      if (kmlInputRef.current) kmlInputRef.current.value = '';
    }, [importKmlFromText, onError]);

    // ---------- Wingtra flightplan import ----------
    const importWingtraFromText = useCallback(async (json: string): Promise<{ added: number; total: number; areas: ImportedFlightplanArea[] }> => {
      const generation = resetGenerationRef.current;
      try {
        cancelPolygonMerge();
        invalidatePolygonOperationHistory();
        console.log(`📥 Importing Wingtra flightplan...`);
        const parsed = JSON.parse(json);
        setLastImportedFlightplan(parsed);
        const { importWingtraFlightPlan } = await loadWingtraConvertModule();
        const imported = importWingtraFlightPlan(parsed, { angleConvention: 'northCW' }); // change if you need eastCW
        const areasOut: ImportedFlightplanArea[] = [];

        if (!mapRef.current || !drawRef.current) {
          onError?.('Map is not ready yet');
          return { added: 0, total: imported.items.length, areas: [] };
        }

        console.log(`📦 Found ${imported.items.length} areas in flightplan`);

        // Suspend automatic per‑polygon side effects during batch
        suspendAutoAnalysisRef.current = true;
        suppressFlightLineEventsRef.current = true;
        const newRings: [number, number][][] = [];
        const newIds: string[] = [];

        // Batch state updates for better performance
        const polygonsToUpdate = new Map();
        const flightLinesToUpdate = new Map();

        // 1) Add features (no analysis yet), set params + overrides, draw file lines immediately
        for (const item of imported.items) {
          const id = addRingAsDrawFeature(item.ring, `Flightplan Area`, { source: 'wingtra' });
          if (!id) continue;
          newIds.push(id);
          newRings.push(item.ring as [number, number][]);
          const importedSpeedMps = Number.isFinite(item.speedMps) && (item.speedMps as number) > 0
            ? Math.max(0.1, item.speedMps as number)
            : undefined;

          const payloadKind = item.payloadKind ?? imported.payloadKind ?? DEFAULT_PAYLOAD_KIND;
          const lidarKey = item.lidarKey || imported.payloadLidarKey || DEFAULT_LIDAR.key;
          const cameraKey = item.cameraKey || imported.payloadCameraKey || 'SONY_RX1R2';
          const planeHardwareVersion = item.planeHardwareVersion ?? imported.planeHardwareVersion;
          let cameraYawOffsetDeg = 0;

          if (payloadKind === 'camera') {
            const cam = (CAMERA_REGISTRY as any)[cameraKey] || DEFAULT_CAMERA;
            const swathW = (cam.w_px * cam.sx_m * item.altitudeAGL) / cam.f_m;
            const swathH = (cam.h_px * cam.sy_m * item.altitudeAGL) / cam.f_m;
            const spacingFromW = swathW * (1 - (item.sideOverlap ?? 70) / 100);
            const spacingFromH = swathH * (1 - (item.sideOverlap ?? 70) / 100);
            const fileSpacing = item.lineSpacingM;
            cameraYawOffsetDeg =
              Number.isFinite(fileSpacing) && fileSpacing > 0
                ? (Math.abs(fileSpacing - spacingFromW) <= Math.abs(fileSpacing - spacingFromH) ? 0 : 90)
                : 0;
          }

          const polygonState = {
            params: {
              payloadKind,
              planeHardwareVersion,
              altitudeAGL: item.altitudeAGL,
              frontOverlap: item.frontOverlap,
              sideOverlap: item.sideOverlap,
              cameraKey: payloadKind === 'camera' ? cameraKey : undefined,
              lidarKey: payloadKind === 'lidar' ? lidarKey : undefined,
              triggerDistanceM: item.triggerDistanceM,
              cameraYawOffsetDeg,
              speedMps: importedSpeedMps,
              lidarReturnMode: payloadKind === 'lidar' ? item.lidarReturnMode : undefined,
              mappingFovDeg: payloadKind === 'lidar' ? item.mappingFovDeg : undefined,
              maxLidarRangeM: payloadKind === 'lidar' ? item.maxLidarRangeM : undefined,
              pointDensityPtsM2: payloadKind === 'lidar' ? item.pointDensityPtsM2 : undefined,
            },
            original: { bearingDeg: item.angleDeg, lineSpacingM: item.lineSpacingM },
            override: { bearingDeg: item.angleDeg, lineSpacingM: item.lineSpacingM, source: 'wingtra' as const }
          };
          polygonsToUpdate.set(id, polygonState);

          const lines = await buildFlightLinesForPolygonAsync({
            polygonId: id,
            ring: item.ring as [number, number][],
            bearingDeg: item.angleDeg,
            lineSpacingM: item.lineSpacingM,
            params: {
              payloadKind,
              planeHardwareVersion,
              altitudeAGL: item.altitudeAGL,
              frontOverlap: item.frontOverlap,
              sideOverlap: item.sideOverlap,
              cameraKey: payloadKind === 'camera' ? cameraKey : undefined,
              lidarKey: payloadKind === 'lidar' ? lidarKey : undefined,
              cameraYawOffsetDeg,
              speedMps: importedSpeedMps,
              lidarReturnMode: payloadKind === 'lidar' ? item.lidarReturnMode : undefined,
              mappingFovDeg: payloadKind === 'lidar' ? item.mappingFovDeg : undefined,
              maxLidarRangeM: payloadKind === 'lidar' ? item.maxLidarRangeM : undefined,
            },
            startedGeneration: generation,
          });
          if (!lines) continue;
          flightLinesToUpdate.set(id, { ...lines, altitudeAGL: item.altitudeAGL });

          areasOut.push({
            polygonId: id,
            params: {
              payloadKind,
              planeHardwareVersion,
              altitudeAGL: item.altitudeAGL,
              frontOverlap: item.frontOverlap,
              sideOverlap: item.sideOverlap,
              angleDeg: item.angleDeg,
              lineSpacingM: item.lineSpacingM,
              triggerDistanceM: item.triggerDistanceM,
              cameraKey: payloadKind === 'camera' ? cameraKey : undefined,
              lidarKey: payloadKind === 'lidar' ? lidarKey : undefined,
              cameraYawOffsetDeg,
              speedMps: importedSpeedMps,
              lidarReturnMode: payloadKind === 'lidar' ? item.lidarReturnMode : undefined,
              mappingFovDeg: payloadKind === 'lidar' ? item.mappingFovDeg : undefined,
              maxLidarRangeM: payloadKind === 'lidar' ? item.maxLidarRangeM : undefined,
              pointDensityPtsM2: payloadKind === 'lidar' ? item.pointDensityPtsM2 : undefined,
              source: 'wingtra'
            }
          });

        }

        setPolygonParams(prev => {
          const next = new Map(prev);
          for (const [id, state] of Array.from(polygonsToUpdate.entries())) next.set(id, state.params);
          polygonParamsRef.current = next;
          return next;
        });
        setImportedOriginals(prev => {
          const next = new Map(prev);
          for (const [id, state] of Array.from(polygonsToUpdate.entries())) next.set(id, state.original);
          importedOriginalsRef.current = next;
          return next;
        });
        setBearingOverrides(prev => {
          const next = new Map(prev);
          for (const [id, state] of Array.from(polygonsToUpdate.entries())) next.set(id, state.override);
          bearingOverridesRef.current = next;
          return next;
        });
        setPolygonFlightLines(prev => {
          const next = new Map(prev);
          for (const [id, lines] of Array.from(flightLinesToUpdate.entries())) next.set(id, lines);
          polygonFlightLinesRef.current = next;
          return next;
        });

        // 2) Fit map to imported rings
        fitMapToRings(newRings);

        // 3) Fetch tiles + build 3D paths (still before analysis so we can build paths early)
        for (let idx = 0; idx < newIds.length; idx++) {
          const polygonId = newIds[idx];
          const ring = newRings[idx];
          const z = calculateOptimalTerrainZoom({ coordinates: ring as any });
          const tiles = await fetchTilesForPolygon({ coordinates: ring as any }, z, mapboxToken, new AbortController().signal);
          setPolygonTiles(prev => {
            const next = new Map(prev);
            next.set(polygonId, tiles);
            polygonTilesRef.current = next;
            return next;
          });
          const flEntry = flightLinesToUpdate.get(polygonId);
          const lineSpacing = flEntry?.lineSpacing ?? imported.items[idx]?.lineSpacingM ?? 25;
          const altitudeAGL = polygonsToUpdate.get(polygonId)?.params.altitudeAGL ?? imported.items[idx]?.altitudeAGL ?? 100;
          if (deckOverlayRef.current && flEntry?.flightLines?.length) {
            const path3d = build3DFlightPath(
              flEntry,
              tiles,
              lineSpacing,
              { altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, preconnected: true },
            );
            update3DPathLayer(deckOverlayRef.current, polygonId, path3d, setDeckLayers);
          }
        }

        // 4) Re-enable auto-analysis and run analyses (collect promises)
        suspendAutoAnalysisRef.current = false;
        const analysisPromises: Promise<any>[] = [];
        for (const polygonId of newIds) {
          const draw = drawRef.current as any;
            const feature = draw?.get?.(polygonId);
            if (feature?.geometry?.type === 'Polygon') {
              const p = analyzePolygon(polygonId, feature);
              analysisPromises.push(p);
            }
        }
        console.log(`🧪 Launched ${analysisPromises.length} terrain analyses for imported areas (waiting to batch notify GSD)...`);
        await Promise.allSettled(analysisPromises);
        console.log(`🧪 All imported terrain analyses settled.`);

        // Ensure effects updating polygonResultsRef have flushed before emitting final results
        await new Promise(r => setTimeout(r, 0));
        if (!isAsyncGenerationStillValid(generation)) {
          return { added: 0, total: imported.items.length, areas: [] };
        }
        onAnalysisComplete?.(Array.from(polygonResultsRef.current.values()));

        // 5) Allow per‑polygon events again & emit a single aggregate update
        suppressFlightLineEventsRef.current = false;
        onFlightLinesUpdated?.('__all__');

        const firstImportedId = newIds[0];
        const draw = drawRef.current as any;
        if (firstImportedId && draw?.get?.(firstImportedId)) {
          try {
            draw.changeMode('simple_select', { featureIds: [firstImportedId] });
          } catch {}
          scheduleGuardedTimeout(() => onPolygonSelected?.(firstImportedId), 0);
        }

        console.log(`✅ Successfully imported ${newIds.length} areas with file bearings preserved. Use "Optimize" to get terrain-optimal directions.`);
        return { added: newIds.length, total: imported.items.length, areas: areasOut };
      } catch (e) {
        suppressFlightLineEventsRef.current = false;
        onError?.(`Failed to import flightplan: ${e instanceof Error ? e.message : 'Unknown error'}`);
        suspendAutoAnalysisRef.current = false;
        return { added: 0, total: 0, areas: [] };
      }
    }, [mapboxToken, addRingAsDrawFeature, cancelPolygonMerge, invalidatePolygonOperationHistory, polygonFlightLines, polygonParams, fitMapToRings, analyzePolygon, isAsyncGenerationStillValid, onFlightLinesUpdated, onError, buildFlightLinesForPolygonAsync]);

    const handleFlightplanFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (!files) return;
      for (const file of Array.from(files)) {
        try {
          const text = await file.text();
          lastImportedFlightplanNameRef.current = file.name;
          await importWingtraFromText(text);
        } catch (err) {
          onError?.(`Failed to read flightplan file ${file.name}: ${err instanceof Error ? err.message : 'Unknown error'}`);
        }
      }
      if (flightplanInputRef.current) flightplanInputRef.current.value = '';
    }, [importWingtraFromText, onError]);

    // ---------- Drag & drop (KML) ----------
    useEffect(() => {
      const el = mapContainer.current;
      if (!el) return;

      const onDragOver = (e: DragEvent) => {
        if (e.dataTransfer) {
          const hasKml = Array.from(e.dataTransfer.items || []).some((it) =>
            it.kind === 'file' && /\.(kml|kmz)$/i.test(it.type || it.getAsFile()?.name || '')
          );
          if (hasKml) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            setIsDraggingKml(true);
          }
        }
      };

      const onDragLeave = () => setIsDraggingKml(false);

      const onDrop = async (e: DragEvent) => {
        e.preventDefault();
        setIsDraggingKml(false);

        const files = Array.from(e.dataTransfer?.files || []).filter(f => /\.(kml|kmz)$/i.test(f.name));
        if (files.length === 0) {
          onError?.("No valid .kml or .kmz files found in drop");
          return;
        }

        for (const f of files) {
          try {
            let kmlText: string | null = null;
            if (/\.kmz$/i.test(f.name)) {
              const buf = await f.arrayBuffer();
              try {
                kmlText = await extractKmlFromKmz(buf);
              } catch (err) {
                onError?.(`Failed to extract KMZ ${f.name}: ${err instanceof Error ? err.message : 'Unknown error'}`);
                continue;
              }
            } else {
              kmlText = await f.text();
            }
            if (kmlText) {
              await importKmlFromText(kmlText);
            }
          } catch (error) {
            onError?.(`Failed to read file ${f.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
          }
        }
      };

      el.addEventListener('dragover', onDragOver);
      el.addEventListener('dragleave', onDragLeave);
      el.addEventListener('drop', onDrop);
      return () => {
        el.removeEventListener('dragover', onDragOver);
        el.removeEventListener('dragleave', onDragLeave);
        el.removeEventListener('drop', onDrop);
      };
    }, [importKmlFromText, onError]);

    // NEW: Apply same params to all queued polygons (bulk "Apply All")
    const applyParamsToAllPending = useCallback((params: PolygonParams) => {
      const queueSnapshot = [...pendingParamPolygons];
      if (queueSnapshot.length === 0) {
        bulkPresetParamsRef.current = params; // adopt for late arrivals
        return;
      }
      bulkApplyRef.current = true;
      bulkPresetParamsRef.current = params;
      setPolygonParams(prev => {
        const next = new Map(prev);
        for (const id of queueSnapshot) next.set(id, params);
        return next;
      });
      const prevSuppress = suppressFlightLineEventsRef.current;
      suppressFlightLineEventsRef.current = true;
      for (const pid of queueSnapshot) {
        if (polygonResultsRef.current.has(pid)) {
          void applyPolygonParams(pid, params, { skipEvent: true, skipQueue: true });
        }
      }
      setPendingParamPolygons([]);
      suppressFlightLineEventsRef.current = prevSuppress;
      bulkApplyRef.current = false;
      onFlightLinesUpdated?.('__all__');
    }, [pendingParamPolygons, onFlightLinesUpdated, applyPolygonParams]);

    // RESTORED: optimizePolygonDirection (terrain-optimal)
    const optimizePolygonDirection = useCallback((polygonId: string) => {
      clearPolygonOperationRedoStack();
      console.log(`🎯 Optimizing direction for polygon ${polygonId} - running exact global bearing search`);
      setBearingOverrides((prev) => {
        const next = new Map(prev);
        next.delete(polygonId);
        return next;
      });
      bearingOverridesRef.current = new Map(bearingOverridesRef.current);
      bearingOverridesRef.current.delete(polygonId);
      const currentParams = polygonParamsRef.current.get(polygonId) ?? { altitudeAGL: 100, frontOverlap: 80, sideOverlap: 70 };
      const params: PolygonParams = { ...currentParams, useCustomBearing: false };
      if ('customBearingDeg' in params) delete (params as any).customBearingDeg;

      const result = polygonResultsRef.current.get(polygonId);
      if (!result) {
        console.log(`⚡ No terrain analysis yet for polygon ${polygonId}, queueing optimization after analysis...`);
        pendingOptimizeRef.current.add(polygonId);
        setPolygonParams((prev) => {
          const next = new Map(prev);
          next.set(polygonId, params);
          return next;
        });
        const draw = drawRef.current as any;
        const f = draw?.get?.(polygonId);
        if (f?.geometry?.type === 'Polygon') analyzePolygon(polygonId, f);
        return;
      }

      const tiles = polygonTilesRef.current.get(polygonId) || [];
      const generation = resetGenerationRef.current;
      void withProcessingPolygon(polygonId, () =>
        runOptimizedBearingSearch(polygonId, params, result, tiles, generation),
      );
    }, [analyzePolygon, clearPolygonOperationRedoStack, runOptimizedBearingSearch, withProcessingPolygon]);

    // RESTORED: revertPolygonToImportedDirection
    const revertPolygonToImportedDirection = useCallback(async (polygonId: string) => {
      clearPolygonOperationRedoStack();
      console.log(`📁 Reverting polygon ${polygonId} to file direction (Wingtra bearing/spacing)`);
      const original = importedOriginals.get(polygonId);
      const res = polygonResults.get(polygonId);
      if (!original || !res || !mapRef.current) return;
      setBearingOverrides((prev) => {
        const next = new Map(prev);
        next.set(polygonId, { bearingDeg: original.bearingDeg, lineSpacingM: original.lineSpacingM, source: 'wingtra' });
        return next;
      });
      bearingOverridesRef.current = new Map(bearingOverridesRef.current);
      bearingOverridesRef.current.set(polygonId, { bearingDeg: original.bearingDeg, lineSpacingM: original.lineSpacingM, source: 'wingtra' });
      const currentParams = polygonParams.get(polygonId) ?? { altitudeAGL: 100, frontOverlap: 80, sideOverlap: 70 };
      const params: PolygonParams = { ...currentParams, useCustomBearing: false };
      if ('customBearingDeg' in params) delete (params as any).customBearingDeg;
      const generation = resetGenerationRef.current;
      const fl = await buildFlightLinesForPolygonAsync({
        polygonId,
        ring: res.polygon.coordinates as [number, number][],
        bearingDeg: original.bearingDeg,
        lineSpacingM: original.lineSpacingM,
        params,
        quality: res.result.fitQuality,
        startedGeneration: generation,
      });
      if (!fl) return;
      setPolygonFlightLines((prev) => {
        const next = new Map(prev);
        next.set(polygonId, { ...fl, altitudeAGL: params.altitudeAGL });
        return next;
      });
      const tiles = polygonTiles.get(polygonId) || [];
      if (deckOverlayRef.current && fl.flightLines.length > 0) {
        const path3d = build3DFlightPath(
          fl,
          tiles,
          fl.lineSpacing,
          { altitudeAGL: params.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, preconnected: true },
        );
        update3DPathLayer(deckOverlayRef.current, polygonId, path3d, setDeckLayers);
      }
      console.log(`✅ Restored file direction: ${original.bearingDeg}° bearing, ${original.lineSpacingM}m spacing`);
      onFlightLinesUpdated?.(polygonId);
    }, [buildFlightLinesForPolygonAsync, clearPolygonOperationRedoStack, importedOriginals, onFlightLinesUpdated, polygonParams, polygonResults, polygonTiles]);

    // RESTORED: runFullAnalysis
    const runFullAnalysis = useCallback((polygonId: string) => {
      clearPolygonOperationRedoStack();
      console.log(`🔄 Running full analysis for polygon ${polygonId} - clearing overrides and requesting fresh params`);
      setBearingOverrides((prev) => {
        const next = new Map(prev);
        next.delete(polygonId);
        return next;
      });
      setPolygonResults((prev) => {
        const next = new Map(prev);
        next.delete(polygonId);
        return next;
      });
      setPolygonParams((prev) => {
        const next = new Map(prev);
        next.delete(polygonId);
        return next;
      });
      if (mapRef.current) {
        removeFlightLinesForPolygon(mapRef.current, polygonId);
        removeTriggerPointsForPolygon(mapRef.current, polygonId);
      }
      if (deckOverlayRef.current) {
        remove3DPathLayer(deckOverlayRef.current, polygonId, setDeckLayers);
      }
      console.log(`⚡ Starting fresh terrain analysis for polygon ${polygonId}...`);
      const draw = drawRef.current as any;
      const f = draw?.get?.(polygonId);
      if (f?.geometry?.type === 'Polygon') {
        analyzePolygon(polygonId, f);
      }
    }, [analyzePolygon, clearPolygonOperationRedoStack]);

    const refreshTerrainForAllPolygons = useCallback(() => {
      console.log('🔄 Refreshing terrain analysis for all polygons using current terrain source');
      cancelAllAnalyses();
      const draw = drawRef.current as any;
      const features = draw?.getAll?.()?.features ?? [];
      for (const feature of features) {
        if (feature?.geometry?.type !== 'Polygon' || !feature.id) continue;
        analyzePolygon(String(feature.id), feature);
      }
    }, [analyzePolygon, cancelAllAnalyses]);

    const getTerrainPartitionContext = useCallback(async (polygonId: string) => {
      const draw = drawRef.current as any;
      const feature = draw?.get?.(polygonId);
      if (!feature?.geometry || feature.geometry.type !== 'Polygon') {
        onError?.('Polygon not found for terrain-face splitting.', polygonId);
        return null;
      }

      const ring = feature.geometry.coordinates?.[0] as [number, number][] | undefined;
      if (!Array.isArray(ring) || ring.length < 4) {
        onError?.('Polygon is invalid for terrain-face splitting.', polygonId);
        return null;
      }

      const params = sanitizePolygonParams(
        polygonParamsRef.current.get(polygonId) ?? { altitudeAGL: 100, frontOverlap: 80, sideOverlap: 70 },
      );
      const existingTiles = polygonTilesRef.current.get(polygonId);
      const tiles = existingTiles && existingTiles.length > 0
        ? existingTiles
        : await fetchTilesForPolygon(
            { coordinates: ring as any },
            calculateOptimalTerrainZoom({ coordinates: ring as any }),
            mapboxToken,
            new AbortController().signal,
          );

      if (!tiles || tiles.length === 0) {
        onError?.('Terrain tiles are required before the area can be auto-split.', polygonId);
        return null;
      }

      return {
        draw,
        ring,
        tiles,
        params: sanitizePolygonParams({
          ...params,
          useCustomBearing: false,
          customBearingDeg: undefined,
        }),
      };
    }, [mapboxToken, onError]);

    const getLocalTerrainPartitionSolutions = useCallback(async (
      ring: [number, number][],
      tiles: any[],
      params: PolygonParams,
    ): Promise<TerrainPartitionSolutionPreview[]> => {
      const { buildPartitionFrontier } = await loadTerrainPartitionGraphModule();
      const { solutions } = buildPartitionFrontier(ring, tiles, params, {
        tradeoffSamples: DEFAULT_PARTITION_TRADEOFF_SAMPLES,
      });
      return solutions
        .filter((solution) => solution.regions.length > 1)
        .map((solution) => ({
          signature: solution.signature,
          tradeoff: solution.tradeoff,
          regionCount: solution.partition.regionCount,
          totalMissionTimeSec: solution.partition.totalMissionTimeSec,
          normalizedQualityCost: solution.partition.normalizedQualityCost,
          weightedMeanMismatchDeg: solution.partition.weightedMeanMismatchDeg,
          hierarchyLevel: solution.hierarchyLevel,
          largestRegionFraction: solution.largestRegionFraction,
          meanConvexity: solution.meanConvexity,
          boundaryBreakAlignment: solution.boundaryBreakAlignment,
          isFirstPracticalSplit: solution.isFirstPracticalSplit,
          regions: solution.regions.map((region) => ({
            areaM2: region.objective.regularization.areaM2,
            bearingDeg: region.objective.bearingDeg,
            atomCount: region.atomIds.length,
            ring: region.ring,
            convexity: region.convexity,
            compactness: region.compactness,
            baseAltitudeAGL: params.altitudeAGL,
          })),
        }));
    }, []);

    type TerrainPartitionRegionApplication = {
      ring: [number, number][];
      bearingDeg?: number;
      baseAltitudeAGL?: number;
    };

    const getTerrainPartitionSolutions = useCallback(async (polygonId: string) => {
      const startedAt = splitPerfNow();
      const context = await getTerrainPartitionContext(polygonId);
	      if (!context) return [];
	      const { ring, tiles, params } = context;
      if (isTerrainPartitionBackendEnabled()) {
        try {
          const backendStartedAt = splitPerfNow();
          splitPerfLog(polygonId, 'requesting backend partition solutions', {
            payloadKind: isLidarParams(params) ? 'lidar' : 'camera',
          });
          const solutions = await solveTerrainPartitionWithBackend({
            polygonId,
            ring,
            payloadKind: isLidarParams(params) ? 'lidar' : 'camera',
            params,
            terrainSource,
            altitudeMode,
            minClearanceM,
            turnExtendM,
          });
          splitPerfLog(polygonId, 'backend partition solutions received', {
            totalMs: Math.round(splitPerfNow() - backendStartedAt),
            solutionCount: solutions.length,
            regionCounts: solutions.map((solution) => solution.regionCount),
          });
          console.log(`[terrain-split][${polygonId}] backend partition solutions received`, {
            totalMs: Math.round(splitPerfNow() - backendStartedAt),
            solutionCount: solutions.length,
            regionCounts: solutions.map((solution) => solution.regionCount),
            rankingSources: solutions.map((solution) => solution.rankingSource ?? 'surrogate'),
          });
          backendPartitionSolutionsRef.current.set(polygonId, solutions);
          const practicalSolutions = solutions.filter((solution) => solution.regions.length > 1);
          console.log(`[terrain-split][${polygonId}] backend partition practical solutions prepared`, {
            solutionCount: practicalSolutions.length,
            regionCounts: practicalSolutions.map((solution) => solution.regionCount),
            rankingSources: practicalSolutions.map((solution) => solution.rankingSource ?? 'surrogate'),
          });
          return practicalSolutions;
        } catch (error) {
          splitPerfLog(polygonId, 'backend partition solve failed; falling back to local solver', {
            totalMs: Math.round(splitPerfNow() - startedAt),
            error: error instanceof Error ? error.message : String(error),
          });
          console.warn('Terrain partition backend failed, falling back to local solver.', error);
        }
      }
      const localStartedAt = splitPerfNow();
      const local = await getLocalTerrainPartitionSolutions(ring, tiles, params);
      splitPerfLog(polygonId, 'local partition solutions computed', {
        totalMs: Math.round(splitPerfNow() - localStartedAt),
        solutionCount: local.length,
      });
      console.log(`[terrain-split][${polygonId}] local partition solutions computed`, {
        totalMs: Math.round(splitPerfNow() - localStartedAt),
        solutionCount: local.length,
        regionCounts: local.map((solution) => solution.regionCount),
        rankingSources: local.map((solution) => solution.rankingSource ?? 'surrogate'),
      });
      backendPartitionSolutionsRef.current.set(polygonId, local);
      return local;
    }, [altitudeMode, getLocalTerrainPartitionSolutions, getTerrainPartitionContext, minClearanceM, terrainSource, turnExtendM]);

    const applyTerrainPartitionRings = useCallback(async (
      polygonId: string,
      partitionRegions: TerrainPartitionRegionApplication[],
      inheritedParams: PolygonParams,
    ): Promise<{ createdIds: string[]; replaced: boolean }> => {
      const generation = resetGenerationRef.current;
      const startedAt = splitPerfNow();
      splitPerfLog(polygonId, 'begin applyTerrainPartitionRings', {
        requestedRegionCount: partitionRegions.length,
      });
      const draw = drawRef.current as any;
      if (!draw) {
        onError?.('Map is not ready for terrain-face splitting.', polygonId);
        return { createdIds: [], replaced: false };
      }

      let normalizedPartitionRegions: Array<TerrainPartitionRegionApplication & { ring: [number, number][] }>;
      let coverage: ReturnType<typeof computePartitionCoverageMetrics>;
      try {
        normalizedPartitionRegions = partitionRegions
          .map((region) => {
            const normalizedRing = normalizeRingForGeometryOps(region.ring);
            if (!normalizedRing) return null;
            return { ...region, ring: normalizedRing };
          })
          .filter((region): region is TerrainPartitionRegionApplication & { ring: [number, number][] } => region !== null);
        splitPerfLog(polygonId, 'normalized partition regions', {
          requestedRegionCount: partitionRegions.length,
          normalizedRegionCount: normalizedPartitionRegions.length,
          ringPointCounts: normalizedPartitionRegions.map((region) => region.ring.length),
        });

        coverage = computePartitionCoverageMetrics(
          (draw?.get?.(polygonId)?.geometry?.coordinates?.[0] as [number, number][] | undefined) ?? [],
          normalizedPartitionRegions.map((region) => region.ring),
        );
      } catch (error) {
        splitPerfLog(polygonId, 'applyTerrainPartitionRings preflight failed', {
          error: error instanceof Error ? error.message : String(error),
          requestedRegionCount: partitionRegions.length,
          regionRingPointCounts: partitionRegions.map((region) => Array.isArray(region.ring) ? region.ring.length : 0),
        });
        throw error;
      }
      splitPerfLog(polygonId, 'partition coverage metrics', coverage);

      if (normalizedPartitionRegions.length <= 1) {
        onError?.('No useful terrain-face split was found for this area.', polygonId);
        return { createdIds: [], replaced: false };
      }

      if (coverage.coverageRatio < 0.5 || coverage.overlapRatio > 0.35) {
        onError?.('Auto-split produced incomplete child coverage for this area and was rejected.', polygonId);
        return { createdIds: [], replaced: false };
      }

      const parentSnapshot = snapshotPolygonFeature(polygonId);
      if (!parentSnapshot) {
        onError?.('Auto-split source polygon is no longer available.', polygonId);
        return { createdIds: [], replaced: false };
      }

      const childSnapshots: PolygonSnapshot[] = [];
      for (const region of normalizedPartitionRegions) {
        const childPolygonId = createPolygonFeatureId();
        const feature = createPolygonFeatureSnapshot({
          id: childPolygonId,
          ring: region.ring,
          properties: {
            name: 'Terrain Face',
            source: 'terrain-face-split',
            parentPolygonId: polygonId,
          },
        });
        if (!feature) continue;
        const childParams = {
          ...inheritedParams,
          altitudeAGL: region.baseAltitudeAGL ?? inheritedParams.altitudeAGL,
        };
        childSnapshots.push({
          feature,
          params: childParams,
          override: Number.isFinite(region.bearingDeg)
            ? {
                bearingDeg: region.bearingDeg!,
                lineSpacingM: getLineSpacingForParams(childParams),
                source: 'partition',
              }
            : undefined,
        });
      }

      const createdIds = childSnapshots.map((snapshot) => snapshot.feature.id);
      splitPerfLog(polygonId, 'prepared child polygon snapshots', {
        createdIds,
      });

      if (createdIds.length <= 1) {
        onError?.('Auto-split did not produce enough valid child polygons.', polygonId);
        return { createdIds: [], replaced: false };
      }

      const transaction: PolygonOperationTransaction = {
        kind: 'split',
        label: 'Auto Split Area',
        before: [parentSnapshot],
        after: childSnapshots,
        selectionBefore: polygonId,
        selectionAfter: createdIds[0] ?? null,
      };

      const applied = await applyPolygonOperationTransaction(transaction, 'forward');
      if (!applied) {
        splitPerfLog(polygonId, 'applyTerrainPartitionRings failed during transaction apply', {
          totalMs: Math.round(splitPerfNow() - startedAt),
        });
        return { createdIds: [], replaced: false };
      }

      if (!isAsyncGenerationStillValid(generation)) {
        return { createdIds: [], replaced: false };
      }

      storePolygonOperationTransaction(transaction);
      splitPerfLog(polygonId, 'applyTerrainPartitionRings completed', {
        totalMs: Math.round(splitPerfNow() - startedAt),
        createdIds,
      });
      return { createdIds, replaced: true };
    }, [applyPolygonOperationTransaction, createPolygonFeatureId, isAsyncGenerationStillValid, onError, snapshotPolygonFeature, storePolygonOperationTransaction]);

    const refineTerrainPartitionPreview = useCallback(async (
      polygonId: string,
      solution: TerrainPartitionSolutionPreview,
    ): Promise<TerrainPartitionSolutionPreview> => {
      if (!solution?.regions?.length) return solution;
      const context = await getTerrainPartitionContext(polygonId);
      if (!context) return solution;
      const { params, tiles } = context;
      const refinedRegions = [...solution.regions];
      let changedCount = 0;

      for (let index = 0; index < solution.regions.length; index++) {
        const region = solution.regions[index];
        const scopeId = `${polygonId}:${solution.signature}:${index}`;
        const { best, evaluated, seedBearingDeg } = await searchExactBearingNearSeed({
          scopeId,
          ring: region.ring as [number, number][],
          params,
          tiles,
          seedBearingDeg: region.bearingDeg,
        });
        if (!best || !Number.isFinite(best.bearingDeg)) continue;
        if (Math.abs(best.bearingDeg - region.bearingDeg) <= 1e-6) continue;
        refinedRegions[index] = {
          ...region,
          bearingDeg: best.bearingDeg,
        };
        changedCount += 1;
        splitPerfLog(scopeId, 'refined partition region exact bearing', {
          seedBearingDeg,
          backendBearingDeg: region.bearingDeg,
          bestBearingDeg: best.bearingDeg,
          evaluatedBearingDegs: evaluated.map((candidate) => Math.round(candidate.bearingDeg * 10) / 10),
          bestExactCost: best.exactCost,
          metricKind: best.metricKind,
          diagnostics: best.diagnostics,
        });
      }

      if (changedCount === 0) return solution;
      splitPerfLog(polygonId, 'refined terrain partition preview bearings', {
        signature: solution.signature,
        changedCount,
        regionCount: solution.regions.length,
      });
      return {
        ...solution,
        regions: refinedRegions,
      };
    }, [getTerrainPartitionContext, searchExactBearingNearSeed]);

    const applyTerrainPartitionPreview = useCallback(async (
      polygonId: string,
      solution: TerrainPartitionSolutionPreview,
    ): Promise<{ createdIds: string[]; replaced: boolean }> => {
      const context = await getTerrainPartitionContext(polygonId);
      if (!context) return { createdIds: [], replaced: false };
      const { params } = context;
      if (!solution || solution.regions.length <= 1) {
        onError?.('Selected terrain partition is no longer valid for this area.', polygonId);
        return { createdIds: [], replaced: false };
      }
      return applyTerrainPartitionRings(
        polygonId,
        solution.regions.map((region) => ({
          ring: region.ring as [number, number][],
          bearingDeg: region.bearingDeg,
          baseAltitudeAGL: region.baseAltitudeAGL,
        })),
        params,
      );
    }, [applyTerrainPartitionRings, getTerrainPartitionContext, onError]);

    const applyTerrainPartitionSolution = useCallback(async (
      polygonId: string,
      signature: string,
    ): Promise<{ createdIds: string[]; replaced: boolean }> => {
      const cached = backendPartitionSolutionsRef.current.get(polygonId) ?? [];
      const solutions = cached.length > 0 ? cached : await getTerrainPartitionSolutions(polygonId);
      const selected = solutions.find((solution) => solution.signature === signature);
      if (!selected) {
        onError?.('Selected terrain partition is no longer valid for this area.', polygonId);
        return { createdIds: [], replaced: false };
      }
      return applyTerrainPartitionPreview(polygonId, selected);
    }, [applyTerrainPartitionPreview, getTerrainPartitionSolutions, onError]);

    const pickDefaultPartitionSolution = useCallback((solutions: TerrainPartitionSolutionPreview[]) => {
      const candidates = solutions.filter((solution) => solution.regions.length > 1);
      if (candidates.length === 0) return null;
      if (candidates.every((solution) => solution.rankingSource === 'backend-exact')) {
        return candidates[0];
      }
      const firstPractical = candidates.find((solution) => solution.isFirstPracticalSplit);
      if (firstPractical) return firstPractical;
      const bestTradeoffDistance = candidates.reduce(
        (best, candidate) => Math.min(best, Math.abs(candidate.tradeoff - DEFAULT_PARTITION_TARGET_TRADEOFF)),
        Infinity,
      );
      const tradeoffBucket = candidates.filter(
        (candidate) => Math.abs(Math.abs(candidate.tradeoff - DEFAULT_PARTITION_TARGET_TRADEOFF) - bestTradeoffDistance) <= 1e-9,
      );
      const minRegionCount = tradeoffBucket.reduce(
        (best, candidate) => Math.min(best, candidate.regionCount),
        Infinity,
      );
      return tradeoffBucket
        .filter((candidate) => candidate.regionCount === minRegionCount)
        .reduce((best, candidate) => (
          candidate.normalizedQualityCost < best.normalizedQualityCost ? candidate : best
        ));
    }, []);

    const autoSplitPolygonByTerrain = useCallback(async (
      polygonId: string,
      options?: { skipBackend?: boolean },
    ): Promise<{ createdIds: string[]; replaced: boolean }> => {
      const startedAt = splitPerfNow();
      splitPerfLog(polygonId, 'autoSplitPolygonByTerrain start');
      const context = await getTerrainPartitionContext(polygonId);
      if (!context) return { createdIds: [], replaced: false };
      const { ring, tiles, params } = context;
      if (!options?.skipBackend) {
        const solutions = await getTerrainPartitionSolutions(polygonId);
        splitPerfLog(polygonId, 'autoSplitPolygonByTerrain candidate solutions ready', {
          totalMs: Math.round(splitPerfNow() - startedAt),
          solutionCount: solutions.length,
        });
        const selected = pickDefaultPartitionSolution(solutions);
        if (selected) {
          splitPerfLog(polygonId, 'autoSplitPolygonByTerrain applying selected backend solution', {
            signature: selected.signature,
            regionCount: selected.regionCount,
            tradeoff: selected.tradeoff,
          });
          const applied = await applyTerrainPartitionRings(
            polygonId,
            selected.regions.map((region) => ({
              ring: region.ring as [number, number][],
              bearingDeg: region.bearingDeg,
              baseAltitudeAGL: region.baseAltitudeAGL,
            })),
            params,
          );
          if (applied.replaced) {
            splitPerfLog(polygonId, 'autoSplitPolygonByTerrain completed via backend solution', {
              totalMs: Math.round(splitPerfNow() - startedAt),
              createdIds: applied.createdIds,
            });
            return applied;
          }
        }
      }

      splitPerfLog(polygonId, 'autoSplitPolygonByTerrain falling back to legacy local partitioner');
      const { partitionPolygonByTerrainFaces } = await loadTerrainFacePartitionModule();
      const partition = partitionPolygonByTerrainFaces(ring, tiles, {
        ...params,
        useCustomBearing: false,
        customBearingDeg: undefined,
      }, {
        forceAtLeastOneSplit: true,
        candidateAngleStepDeg: 10,
        candidateOffsetFractions: [-0.42, -0.28, -0.14, 0, 0.14, 0.28, 0.42],
      });
      const fallbackResult = await applyTerrainPartitionRings(
        polygonId,
        partition.polygons.map((ring) => ({ ring })),
        params,
      );
      splitPerfLog(polygonId, 'autoSplitPolygonByTerrain completed via legacy fallback', {
        totalMs: Math.round(splitPerfNow() - startedAt),
        createdIds: fallbackResult.createdIds,
      });
      return fallbackResult;
    }, [applyTerrainPartitionRings, getTerrainPartitionContext, getTerrainPartitionSolutions, pickDefaultPartitionSolution]);

    const resetAllDrawingsState = useCallback(() => {
      resetGenerationRef.current += 1;
      plannedGeometryWorkerRef.current?.terminate();
      plannedGeometryWorkerRef.current = null;
      cancelAllGuardedTimeouts();
      cancelPolygonMerge();
      replacePolygonOperationHistory(clearPolygonOperationHistory());
      setProcessingPolygonIds([]);
      if (drawRef.current) drawRef.current.deleteAll();
      if (deckOverlayRef.current) {
        setDeckLayers([]);
        deckOverlayRef.current.setProps({ layers: [] });
      }
      if (mapRef.current) {
        clearAllFlightLines(mapRef.current);
        clearAllTriggerPoints(mapRef.current);
      }
      setPolygonResults(new Map());
      polygonResultsRef.current = new Map();
      setPolygonTiles(new Map());
      polygonTilesRef.current = new Map();
      setPolygonFlightLines(new Map());
      polygonFlightLinesRef.current = new Map();
      setPolygonParams(new Map());
      polygonParamsRef.current = new Map();
      setBearingOverrides(new Map());
      bearingOverridesRef.current = new Map();
      setImportedOriginals(new Map());
      importedOriginalsRef.current = new Map();
      setLastImportedFlightplan(null);
      lastImportedFlightplanNameRef.current = undefined;
      setPendingParamPolygons([]);
      pendingOptimizeRef.current.clear();
      pendingProgrammaticDeletesRef.current.clear();
      suppressSelectionDialogUntilRef.current = 0;

      cancelAllAnalyses();
      onClearGSD?.();
      onPolygonSelected?.(null);
      onAnalysisComplete?.([]);
    }, [cancelAllAnalyses, cancelAllGuardedTimeouts, cancelPolygonMerge, onAnalysisComplete, onClearGSD, onPolygonSelected, replacePolygonOperationHistory, setProcessingPolygonIds]);

	    React.useImperativeHandle(ref, () => ({
      clearAllDrawings: resetAllDrawingsState,
      clearPolygon: (polygonId: string) => {
        cancelPolygonMerge();
        invalidatePolygonOperationHistory();
        if (processingPolygonIdsRef.current.has(polygonId)) {
          setProcessingPolygonIds(Array.from(processingPolygonIdsRef.current).filter((id) => id !== polygonId));
        }
        const draw = drawRef.current as any;
        if (!draw) {
          cleanupPolygonState(polygonId);
          return;
        }
        deletePolygonFeature(polygonId);
      },
      editPolygonBoundary,
      setProcessingPolygonIds,
      autoSplitPolygonByTerrain,
      getTerrainPartitionSolutions,
      refineTerrainPartitionPreview,
      applyTerrainPartitionSolution,
      applyTerrainPartitionPreview,
      startPolygonMerge,
      cancelPolygonMerge,
      canStartPolygonMerge,
      togglePolygonMergeCandidate,
      confirmPolygonMerge,
      undoPolygonOperation,
      redoPolygonOperation,
      canUndoPolygonOperation,
      canRedoPolygonOperation,
      startPolygonDrawing: () => {
        if (drawRef.current) (drawRef.current as any).changeMode('draw_polygon');
      },
      getPolygonResults: () => Array.from(polygonResultsRef.current.values()),
      getMap: () => mapRef.current,
      refreshTerrainForAllPolygons,
      setTerrainDemSource: (tileUrlTemplate: string | null) => {
        const sameTemplate = terrainDemSourceTemplateRef.current === tileUrlTemplate;
        terrainDemSourceTemplateRef.current = tileUrlTemplate;
        if (mapRef.current && mapRef.current.isStyleLoaded()) {
          if (sameTemplate) {
            console.debug('[terrain-source] lightweight terrain reassert requested', {
              terrainMode: terrainSourceRef.current.mode,
              datasetId: terrainSourceRef.current.datasetId ?? null,
              tileUrlTemplate,
            });
            reassertTerrainDemSourceOnMap(mapRef.current, tileUrlTemplate);
            return;
          }
          applyTerrainSourceToMap(mapRef.current, tileUrlTemplate, terrainSourceRef.current);
        }
      },
      setFlightLinesVisible: (visible: boolean) => {
        syncFlightLinesVisibility(visible);
      },
      getPolygons: (): [number,number][][] => {
        const draw = drawRef.current;
        if (!draw) return [];
        const coll = draw.getAll();
        const rings: [number,number][][] = [];
        for (const f of coll.features) {
          if (f.geometry?.type === "Polygon" && Array.isArray(f.geometry.coordinates?.[0])) {
            rings.push(f.geometry.coordinates[0] as [number,number][]);
          }
        }
        return rings;
      },
      getPolygonsWithIds: (): { id?: string; ring: [number, number][] }[] => {
        const draw = drawRef.current;
        if (!draw) return [];
        const coll = draw.getAll();
        const polygonsWithIds: { id?: string; ring: [number, number][] }[] = [];
        for (const f of coll.features) {
          if (f.geometry?.type === "Polygon" && Array.isArray(f.geometry.coordinates?.[0])) {
            polygonsWithIds.push({
              id: f.id as string | undefined,
              ring: f.geometry.coordinates[0] as [number, number][]
            });
          }
        }
        return polygonsWithIds;
      },
      getFlightLines: () => polygonFlightLinesRef.current,
      getPolygonTiles: () => polygonTilesRef.current,
      addCameraPoints: (polygonId: string, positions: [number, number, number][]) => {
        if (deckOverlayRef.current) update3DCameraPointsLayer(deckOverlayRef.current, polygonId, positions, setDeckLayers);
      },
      removeCameraPoints: (polygonId: string) => {
        if (deckOverlayRef.current) remove3DCameraPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
      },
      applyPolygonParams: (polygonId: string, params: PolygonParams) => {
        clearPolygonOperationRedoStack();
        void applyPolygonParams(polygonId, params);
      },
      applyPolygonParamsBatch: (updates: Array<{ polygonId: string; params: PolygonParams }>) => {
        clearPolygonOperationRedoStack();
        applyPolygonParamsBatch(updates);
      },
      // expose bulk apply helper
      applyParamsToAllPending: (params: PolygonParams) => {
        clearPolygonOperationRedoStack();
        applyParamsToAllPending(params);
      },
      getPerPolygonParams: () => Object.fromEntries(polygonParamsRef.current),
      // Altitude strategy and clearance controls
      setAltitudeMode: (m: 'legacy' | 'min-clearance') => setAltitudeMode(m),
      getAltitudeMode: () => altitudeMode,
      setMinClearance: (m: number) => setMinClearanceM(Math.max(0, m)),
      getMinClearance: () => minClearanceM,
      setTurnExtend: (m: number) => setTurnExtendM(Math.max(0, m)),
      getTurnExtend: () => turnExtendM,

      openKmlFilePicker: () => {
        kmlInputRef.current?.click();
      },
      importKmlFromText,

      openFlightplanFilePicker: () => {
        flightplanInputRef.current?.click();
      },
      importWingtraFromText,

      optimizePolygonDirection,
      revertPolygonToImportedDirection: (polygonId: string) => {
        void revertPolygonToImportedDirection(polygonId);
      },
      runFullAnalysis,

	      getBearingOverrides: () => Object.fromEntries(bearingOverridesRef.current),
	      getImportedOriginals: () => Object.fromEntries(importedOriginalsRef.current),
	      getLastImportedFlightplanName: () => lastImportedFlightplanNameRef.current,
	      canExportWingtraFlightPlanDirectly: () => isWingtraFlightPlanTemplateExportReadyLocal(lastImportedFlightplan),
	      exportWingtraFlightPlan: async (config?: WingtraFreshExportConfig) => {
        const {
          areasFromState,
          exportToWingtraFlightPlan,
          replaceAreaItemsInWingtraFlightPlan,
          resolveFreshWingtraExportPayloadOptionFromAreas,
        } = await loadWingtraConvertModule();
        // Build area list from current state
        const polys: Array<{
          polygonId: string;
          ring: [number, number][];
          params: PolygonParams;
          bearingDeg: number;
          axialBearingDeg: number;
          lineSpacingM?: number;
          triggerDistanceM?: number;
        }> = [];
        polygonParams.forEach((params, pid) => {
          const res = polygonResults.get(pid);
          const collection = drawRef.current?.getAll();
          const feature = collection?.features.find(f=>f.id===pid && f.geometry?.type==='Polygon');
          const ring = res?.polygon.coordinates || (feature?.geometry as any)?.coordinates?.[0];
          if (!ring) return;
          const override = bearingOverrides.get(pid);
          const bearingDeg = normalizeBearing(override ? override.bearingDeg : (res?.result.contourDirDeg ?? 0)) ?? 0;
          const axialBearingDeg = normalizeAxialBearingDeg(bearingDeg);
          const lineSpacingM = override?.lineSpacingM || (polygonFlightLines.get(pid)?.lineSpacing);
          polys.push({
            polygonId: pid,
            ring: ring as any,
            params,
            bearingDeg,
            axialBearingDeg,
            lineSpacingM,
            triggerDistanceM: params.triggerDistanceM,
          });
        });
        const fallbackAreas = areasFromState(polys.map(({ ring, params, bearingDeg, lineSpacingM, triggerDistanceM }) => ({
          ring,
          params,
          bearingDeg,
          lineSpacingM,
          triggerDistanceM,
        })));
        const axialAreasByPolygonId = new Map(
          polys.map((poly) => [
            poly.polygonId,
            areasFromState([{
              ring: poly.ring,
              params: poly.params,
              bearingDeg: poly.axialBearingDeg,
              lineSpacingM: poly.lineSpacingM,
              triggerDistanceM: poly.triggerDistanceM,
            }])[0]!,
          ] as const),
        );
        let areas = fallbackAreas;
        if (areas.length === 0) {
          throw new Error('Draw or import at least one area before exporting a Wingtra flightplan.');
        }
        const exportSequenceMaxHeightAboveGroundM = resolveWingtraExportMaxHeightAboveGroundM(lastImportedFlightplan);
        const exportSequenceCruiseSpeedMps = resolveWingtraExportCruiseSpeedMps(lastImportedFlightplan);
        if (areas.length > 1 && isAreaSequenceBackendEnabled()) {
          try {
            const sequenceResult = await optimizeAreaSequenceWithBackend({
              areas: polys.map((poly) => ({
                polygonId: poly.polygonId,
                ring: poly.ring,
                bearingDeg: poly.axialBearingDeg,
                payloadKind: isLidarParams(poly.params) ? 'lidar' : 'camera',
                params: isLidarParams(poly.params) || (Number.isFinite(poly.params.speedMps) && (poly.params.speedMps as number) > 0)
                  ? poly.params
                  : { ...poly.params, speedMps: exportSequenceCruiseSpeedMps },
              })),
              terrainSource,
              altitudeMode,
              minClearanceM,
              turnExtendM,
              maxHeightAboveGroundM: exportSequenceMaxHeightAboveGroundM,
            });
            const optimizedAreas = [...sequenceResult.areas]
              .sort((left, right) => left.orderIndex - right.orderIndex)
              .map((choice) => {
                const baseArea = axialAreasByPolygonId.get(choice.polygonId);
                if (!baseArea) {
                  throw new Error(`Missing export area for polygon ${choice.polygonId}.`);
                }
                return {
                  ...baseArea,
                  angleDeg: choice.flipped
                    ? (normalizeBearing(baseArea.angleDeg + 180) ?? baseArea.angleDeg)
                    : baseArea.angleDeg,
                };
              });
            if (optimizedAreas.length === areas.length) {
              areas = optimizedAreas;
            }
          } catch (error) {
            console.warn('Wingtra export area sequencing failed; falling back to current order.', error);
          }
        }
        const payloadKind = config?.payloadKind ?? areas[0]?.payloadKind ?? 'camera';
        const resolvedPayloadOption =
          config?.payloadUniqueString
            ? {
                payloadKind: config.payloadKind,
                payloadUniqueString: config.payloadUniqueString,
                payloadName: config.payloadName ?? config.payloadUniqueString,
              }
            : resolveFreshWingtraExportPayloadOptionFromAreas(areas);
        if (!resolvedPayloadOption) {
          throw new Error(
            'Wingtra export requires all current areas to use the same analysis payload and a compatible drone version.',
          );
        }
        const canUseImportedTemplate = isWingtraFlightPlanTemplateExportReadyLocal(lastImportedFlightplan);
        let fp;
        if (canUseImportedTemplate) {
          // Deep clone original
            fp = JSON.parse(JSON.stringify(lastImportedFlightplan));
          // Replace only area items and preserve non-area mission items like takeoff, loiter, and landing.
          fp = replaceAreaItemsInWingtraFlightPlan(fp, areas, {
            payloadKind,
            payloadUniqueString: resolvedPayloadOption.payloadUniqueString,
            payloadName: resolvedPayloadOption.payloadName,
          });
          fp.flightPlan.payload = resolvedPayloadOption.payloadName;
          fp.flightPlan.payloadUniqueString = resolvedPayloadOption.payloadUniqueString;
          const hwVersion =
            resolvedPayloadOption.payloadUniqueString.endsWith('_v4') ? '4' : '5';
          fp.flightPlan.planeHardware = {
            ...(typeof fp.flightPlan.planeHardware === 'object' && fp.flightPlan.planeHardware ? fp.flightPlan.planeHardware : {}),
            hwVersion,
            displayName: hwVersion === '5' ? 'WingtraRay (any)' : 'WingtraOne (any)',
            isGenericPlane: true,
          };
          // Reset derived stats that may be stale
          fp.flightPlan.numberOfImages = 0;
          fp.flightPlan.totalArea = 0;
          fp.flightPlan.activeTotalArea = 0;
          fp.flightPlan.activeNumberOfImages = 0;
          fp.flightPlan.flownPercentage = 0;
          fp.flightPlan.resumeMissionIndex = 0;
          fp.flightPlan.resumeGridPointIndex = -1;
          fp.flightPlan.lastModifiedTime = Date.now();
        } else {
          fp = exportToWingtraFlightPlan(areas, {
            payloadKind,
            payloadUniqueString: resolvedPayloadOption.payloadUniqueString,
            payloadName: resolvedPayloadOption.payloadName,
          });
        }
        const json = JSON.stringify(fp, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        return { json, blob };
      },
    }), [
      polygonResults, polygonFlightLines, polygonTiles, polygonParams,
      cancelAllAnalyses, cancelAllGuardedTimeouts, applyPolygonParams, applyPolygonParamsBatch, cleanupPolygonState, clearPolygonOperationRedoStack, deletePolygonFeature, editPolygonBoundary, setProcessingPolygonIds, autoSplitPolygonByTerrain,
      getTerrainPartitionSolutions, refineTerrainPartitionPreview, applyTerrainPartitionSolution, applyTerrainPartitionPreview,
      bearingOverrides, importedOriginals,
      importKmlFromText, importWingtraFromText,
      optimizePolygonDirection, revertPolygonToImportedDirection, runFullAnalysis, refreshTerrainForAllPolygons, resetAllDrawingsState,
      syncFlightLinesVisibility,
      altitudeMode, minClearanceM, terrainSource, turnExtendM,
      lastImportedFlightplan
    ]);

    React.useEffect(() => {
      if (!shouldConsumeClearAllEpoch(lastHandledClearAllEpochRef.current, clearAllEpoch)) return;
      lastHandledClearAllEpochRef.current = clearAllEpoch;
      resetAllDrawingsState();
    }, [clearAllEpoch, resetAllDrawingsState]);

    React.useEffect(() => () => {
      plannedGeometryWorkerRef.current?.terminate();
      plannedGeometryWorkerRef.current = null;
      cancelAllGuardedTimeouts();
      stopProcessingPerimeterAnimation();
    }, [cancelAllGuardedTimeouts, stopProcessingPerimeterAnimation]);

    return (
      <div style={{ position: 'relative', width: '100%', height: '100%' }}>
        {/* ✅ EMPTY container that Mapbox owns */}
        <div ref={mapContainer} style={{ position: 'absolute', inset: 0 }} />

        {/* ✅ Siblings, not children of the Mapbox container */}
        <input
          ref={kmlInputRef}
          type="file"
          accept=".kml,.kmz,application/vnd.google-earth.kml+xml,application/vnd.google-earth.kmz"
          multiple
          onChange={handleKmlFileChange}
          style={{ display: 'none' }}
        />
        <input
          ref={flightplanInputRef}
          type="file"
          accept=".flightplan"
          multiple
          onChange={handleFlightplanFileChange}
          style={{ display: 'none' }}
        />

        {/* KML drag overlay */}
        {isDraggingKml && (
          <div
            style={{
              position: 'absolute', inset: 0, display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              background: 'rgba(59,130,246,0.08)', border: '2px dashed rgba(59,130,246,0.6)',
              zIndex: 10, pointerEvents: 'none',
            }}
          >
            <div
              style={{
                padding: '8px 12px', background: 'white', borderRadius: 6,
                boxShadow: '0 1px 3px rgba(0,0,0,0.15)', fontSize: 12, color: '#1f2937',
              }}
            >
              Drop <strong>.kml</strong>/<strong>.kmz</strong> file(s) to import areas
            </div>
          </div>
        )}
      </div>
    );
  }
);

MapFlightDirectionComponent.displayName = 'MapFlightDirection';

export const MapFlightDirection = React.memo(MapFlightDirectionComponent);
