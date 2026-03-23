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

import { setTerrainDemSourceOnMap, useMapInitialization } from './hooks/useMapInitialization';
import { usePolygonAnalysis } from './hooks/usePolygonAnalysis';
import {
  addFlightLinesForPolygon,
  animateProcessingPerimeter,
  removeFlightLinesForPolygon,
  clearAllFlightLines,
  removeTriggerPointsForPolygon,
  clearAllTriggerPoints,
  setProcessingPerimeterPolygons,
  generateFlightLinesForPolygon,
} from './utils/mapbox-layers';
import { update3DPathLayer, remove3DPathLayer, update3DCameraPointsLayer, remove3DCameraPointsLayer, update3DTriggerPointsLayer, remove3DTriggerPointsLayer } from './utils/deckgl-layers';
import { build3DFlightPath, calculateOptimalTerrainZoom, sampleCameraPositionsOnFlightPath, extendFlightLineForTurnRunout, queryMinMaxElevationAlongPolylineWGS84 } from './utils/geometry';
import { PolygonAnalysisResult, PolygonParams } from './types';
import { parseKmlPolygons, calculateKmlBounds, extractKmlFromKmz } from '@/utils/kml';
import { SONY_RX1R2, DJI_ZENMUSE_P1_24MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM, calculateGSD, forwardSpacingRotated, lineSpacingRotated } from '@/domain/camera';
import { DEFAULT_LIDAR, DEFAULT_LIDAR_MAX_RANGE_M, LIDAR_REGISTRY, getLidarMappingFovDeg, getLidarModel, lidarDeliverableDensity, lidarLineSpacing, lidarSinglePassDensity, lidarSwathWidth } from '@/domain/lidar';
import type { BearingOverride, MapFlightDirectionAPI, ImportedFlightplanArea, TerrainPartitionSolutionPreview } from './api';
import { fetchTilesForPolygon } from './utils/terrain';
import { partitionPolygonByTerrainFaces } from '@/utils/terrainFacePartition';
import { buildPartitionFrontier } from '@/utils/terrainPartitionGraph';
import { isTerrainPartitionBackendEnabled, solveTerrainPartitionWithBackend } from '@/services/terrainPartitionBackend';
import type { TerrainSourceSelection } from '@/terrain/types';
import { LidarDensityWorker, OverlapWorker, fetchTerrainRGBA, tilesCoveringPolygon } from '@/overlap/controller';
import type { GSDStats, LidarStripMeters, PoseMeters } from '@/overlap/types';
import { lngLatToMeters, tileMetersBounds } from '@/overlap/mercator';
// @ts-ignore Turf typings are inconsistent in this repo.
import * as turf from '@turf/turf';

// NEW: Wingtra import helpers
import { importWingtraFlightPlan } from '@/interop/wingtra/convert';
import { exportToWingtraFlightPlan, areasFromState } from '@/interop/wingtra/convert';

const CAMERA_REGISTRY: Record<string, any> = {
  SONY_RX1R2,
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
const EXACT_OPTIMIZE_ZOOM = 14;
const EXACT_MIN_OVERLAP_FOR_GSD = 3;
const EXACT_OPTIMIZE_TIME_WEIGHT = 0.1;
const TERRAIN_SPLIT_DEBUG = true;
const GEOMETRY_RING_EPSILON_DEG = 1e-7;

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

function statsTotalAreaM2(stats: GSDStats) {
  if (stats.totalAreaM2 && stats.totalAreaM2 > 0) return stats.totalAreaM2;
  return stats.histogram.reduce((sum, bin) => sum + (bin.areaM2 || 0), 0);
}

function sortedHistogramBins(stats: GSDStats) {
  return [...stats.histogram]
    .filter((bin) => (bin.areaM2 || 0) > 0)
    .sort((a, b) => a.bin - b.bin);
}

function histogramBinEdges(stats: GSDStats) {
  const bins = sortedHistogramBins(stats);
  if (bins.length === 0) return { bins, edges: [] as number[] };
  if (bins.length === 1) {
    const only = bins[0].bin;
    const lo = Number.isFinite(stats.min) ? Math.min(stats.min, only) : only;
    const hi = Number.isFinite(stats.max) ? Math.max(stats.max, only) : only;
    return { bins, edges: [lo, hi > lo ? hi : lo + 1] };
  }
  const centers = bins.map((bin) => bin.bin);
  const edges = new Array<number>(bins.length + 1);
  edges[0] = centers[0] - (centers[1] - centers[0]) * 0.5;
  for (let index = 1; index < centers.length; index++) {
    edges[index] = (centers[index - 1] + centers[index]) * 0.5;
  }
  edges[bins.length] = centers[bins.length - 1] + (centers[bins.length - 1] - centers[bins.length - 2]) * 0.5;
  return { bins, edges };
}

function histogramAreaBelow(stats: GSDStats, threshold: number) {
  const { bins, edges } = histogramBinEdges(stats);
  if (!Number.isFinite(threshold) || bins.length === 0) return 0;
  let areaBelow = 0;
  for (let index = 0; index < bins.length; index++) {
    const areaM2 = bins[index].areaM2 || 0;
    if (!(areaM2 > 0)) continue;
    const lower = edges[index];
    const upper = edges[index + 1];
    if (threshold <= lower) continue;
    if (threshold >= upper) {
      areaBelow += areaM2;
      continue;
    }
    const fraction = Math.max(0, Math.min(1, (threshold - lower) / Math.max(1e-9, upper - lower)));
    areaBelow += areaM2 * fraction;
  }
  return areaBelow;
}

function histogramQuantile(stats: GSDStats, q: number) {
  const { bins, edges } = histogramBinEdges(stats);
  const totalArea = statsTotalAreaM2(stats);
  if (!(totalArea > 0) || bins.length === 0) return 0;
  const target = Math.max(0, Math.min(1, q)) * totalArea;
  let cumulative = 0;
  for (let index = 0; index < bins.length; index++) {
    const areaM2 = bins[index].areaM2 || 0;
    cumulative += areaM2;
    if (cumulative >= target) {
      const previous = cumulative - areaM2;
      const fraction = areaM2 > 0 ? Math.max(0, Math.min(1, (target - previous) / areaM2)) : 0;
      const lower = edges[index];
      const upper = edges[index + 1];
      return lower + fraction * (upper - lower);
    }
  }
  return edges[edges.length - 1] ?? bins[bins.length - 1]?.bin ?? 0;
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

function scoreExactLidarStats(stats: GSDStats, params: PolygonParams) {
  const model = getLidarModel(params.lidarKey);
  const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
  const speedMps = params.speedMps ?? model.defaultSpeedMps;
  const returnMode = params.lidarReturnMode ?? 'single';
  const targetDensityPtsM2 = params.pointDensityPtsM2
    ?? lidarDeliverableDensity(model, params.altitudeAGL, params.sideOverlap, speedMps, returnMode, mappingFovDeg);
  const totalAreaM2 = Math.max(1, statsTotalAreaM2(stats));
  const holeThreshold = Math.max(5, targetDensityPtsM2 * 0.2);
  const weakThreshold = Math.max(holeThreshold + 1e-6, targetDensityPtsM2 * 0.7);
  const q10 = histogramQuantile(stats, 0.1);
  const q25 = histogramQuantile(stats, 0.25);
  const holeFraction = histogramAreaBelow(stats, holeThreshold) / totalAreaM2;
  const lowFraction = histogramAreaBelow(stats, weakThreshold) / totalAreaM2;
  const q10Deficit = Math.max(0, 1 - q10 / Math.max(1e-6, targetDensityPtsM2));
  const q25Deficit = Math.max(0, 1 - q25 / Math.max(1e-6, targetDensityPtsM2));
  const meanDeficit = Math.max(0, 1 - stats.mean / Math.max(1e-6, targetDensityPtsM2));
  const qualityCost =
    4.2 * holeFraction +
    2.4 * lowFraction +
    1.9 * q10Deficit +
    1.2 * q25Deficit +
    0.8 * meanDeficit;
  return {
    qualityCost,
    targetDensityPtsM2,
    holeFraction,
    lowFraction,
    q10,
    q25,
  };
}

function scoreExactCameraStats(stats: GSDStats, params: PolygonParams) {
  const cameraKey = params.cameraKey;
  const camera = cameraKey ? CAMERA_REGISTRY[cameraKey] || DEFAULT_CAMERA : DEFAULT_CAMERA;
  const targetGsdM = calculateGSD(camera, params.altitudeAGL);
  const totalAreaM2 = Math.max(1, statsTotalAreaM2(stats));
  const q75 = histogramQuantile(stats, 0.75);
  const q90 = histogramQuantile(stats, 0.9);
  const overTargetAreaFraction = Math.max(0, totalAreaM2 - histogramAreaBelow(stats, targetGsdM)) / totalAreaM2;
  const meanOvershoot = Math.max(0, stats.mean / Math.max(1e-6, targetGsdM) - 1);
  const q75Overshoot = Math.max(0, q75 / Math.max(1e-6, targetGsdM) - 1);
  const q90Overshoot = Math.max(0, q90 / Math.max(1e-6, targetGsdM) - 1);
  const maxOvershoot = Math.max(0, stats.max / Math.max(1e-6, targetGsdM) - 1);
  const qualityCost =
    1.85 * q90Overshoot +
    1.25 * overTargetAreaFraction +
    0.95 * meanOvershoot +
    0.55 * q75Overshoot +
    0.2 * maxOvershoot;
  return {
    qualityCost,
    targetGsdM,
    overTargetAreaFraction,
    q75,
    q90,
  };
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
  return {
    ...params,
    payloadKind,
    altitudeAGL: Math.max(1, Number.isFinite(params.altitudeAGL) ? params.altitudeAGL : DEFAULT_ALTITUDE_AGL),
    frontOverlap: isLidar ? 0 : clampNumber(params.frontOverlap, 0, 95, DEFAULT_FRONT_OVERLAP),
    sideOverlap: clampNumber(params.sideOverlap, 0, 95, DEFAULT_SIDE_OVERLAP),
    speedMps: isLidar ? Math.max(0.1, Number.isFinite(params.speedMps) ? params.speedMps! : DEFAULT_LIDAR.defaultSpeedMps) : undefined,
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
  center?: LngLatLike;
  zoom?: number;
  sampleStep?: number;
  terrainDemUrlTemplate?: string | null;
  terrainSource?: TerrainSourceSelection;

  onRequestParams?: (polygonId: string, ring: [number, number][]) => void;
  onAnalysisComplete?: (results: PolygonAnalysisResult[]) => void;
  onAnalysisStart?: (polygonId: string) => void;
  onError?: (error: string, polygonId?: string) => void;
  onFlightLinesUpdated?: (changed: string | '__all__') => void;
  onClearGSD?: () => void;
  onPolygonSelected?: (polygonId: string | null) => void;
}

export const MapFlightDirection = React.forwardRef<MapFlightDirectionAPI, Props>(
  (
    {
      mapboxToken,
      center = [8.54, 47.37],
      zoom = 13,
      sampleStep = 2,
      terrainDemUrlTemplate = null,
      terrainSource = { mode: 'mapbox', datasetId: null },
      onRequestParams,
      onAnalysisComplete,
      onAnalysisStart,
      onError,
      onFlightLinesUpdated,
      onClearGSD,
      onPolygonSelected,
    },
    ref
  ) => {
    const mapContainer = useRef<HTMLDivElement>(null);
    const mapRef = useRef<MapboxMap>();
    const drawRef = useRef<MapboxDraw>();
    const deckOverlayRef = useRef<MapboxOverlay>();
    const terrainDemSourceTemplateRef = useRef<string | null>(terrainDemUrlTemplate);

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
      Map<string, { flightLines: number[][][]; lineSpacing: number; altitudeAGL: number }>
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
    const pendingProgrammaticDeletesRef = React.useRef<Set<string>>(new Set());
    const suppressSelectionDialogUntilRef = React.useRef(0);
    const suppressNextEmptySelectionRef = React.useRef(0);
    // NEW: Altitude mode + minimum clearance configuration (global)
    const [altitudeMode, setAltitudeMode] = useState<'legacy' | 'min-clearance'>('legacy');
    const [minClearanceM, setMinClearanceM] = useState<number>(60);
    const [turnExtendM, setTurnExtendM] = useState<number>(96);

    React.useEffect(() => { polygonParamsRef.current = polygonParams; }, [polygonParams]);
    React.useEffect(() => { bearingOverridesRef.current = bearingOverrides; }, [bearingOverrides]);
    React.useEffect(() => { polygonTilesRef.current = polygonTiles; }, [polygonTiles]);
    React.useEffect(() => { polygonFlightLinesRef.current = polygonFlightLines; }, [polygonFlightLines]);
    React.useEffect(() => { polygonResultsRef.current = polygonResults; }, [polygonResults]);
    React.useEffect(() => { importedOriginalsRef.current = importedOriginals; }, [importedOriginals]);

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

    // Debounced callback to prevent React render conflicts when multiple analyses complete
    const debouncedAnalysisComplete = useCallback(() => {
      const timeoutId = setTimeout(() => {
        onAnalysisComplete?.(Array.from(polygonResultsRef.current.values()));
      }, 0);
      return timeoutId;
    }, [onAnalysisComplete]);

    const applyPolygonParams = useCallback((polygonId: string, params: PolygonParams, opts?: { skipEvent?: boolean; skipQueue?: boolean }) => {
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

      removeFlightLinesForPolygon(mapRef.current, polygonId);
      const fl = addFlightLinesForPolygon(
        mapRef.current,
        polygonId,
        res.polygon.coordinates,
        bearingDeg,
        spacing,
        res.result.fitQuality
      );

      const nextFlightLines = new Map(polygonFlightLinesRef.current);
      nextFlightLines.set(polygonId, { ...fl, altitudeAGL: safeParams.altitudeAGL });
      polygonFlightLinesRef.current = nextFlightLines;
      setPolygonFlightLines(nextFlightLines);

      if (!opts?.skipEvent && !suppressFlightLineEventsRef.current) {
        onFlightLinesUpdated?.(polygonId);
      }

      if (deckOverlayRef.current && fl.flightLines.length > 0) {
        const path3d = build3DFlightPath(fl.flightLines, tiles, fl.lineSpacing, { altitudeAGL: safeParams.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, turnExtendM });
        update3DPathLayer(deckOverlayRef.current, polygonId, path3d, setDeckLayers);
        const spacingForward = getForwardSpacingForParams(safeParams);
        if (spacingForward && spacingForward > 0) {
          const samples = sampleCameraPositionsOnFlightPath(path3d, spacingForward, { includeTurns: false });
          const zOffset = 1;
          const ring = res.polygon.coordinates as [number,number][];
          const inside = (lng:number,lat:number,ring:[number,number][]) => {
            let ins=false; for(let i=0,j=ring.length-1;i<ring.length;j=i++){
              const xi=ring[i][0], yi=ring[i][1], xj=ring[j][0], yj=ring[j][1];
              const intersect=((yi>lat)!==(yj>lat)) && (lng < (xj-xi)*(lat-yi)/(yj-yi)+xi); if(intersect) ins=!ins;
            } return ins;
          };
          const positions: [number, number, number][] = samples
            .filter(([lng,lat]) => inside(lng,lat,ring))
            .map(([lng,lat,alt]) => [lng,lat,alt + zOffset]);
          update3DTriggerPointsLayer(deckOverlayRef.current, polygonId, positions, setDeckLayers);
        } else {
          remove3DTriggerPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
        }
      }

      if (opts?.skipQueue) return;

      setPendingParamPolygons(prev => {
        const rest = prev.filter(id => id !== polygonId);
        if (!bulkApplyRef.current && rest.length > 0) {
          const nextId = rest[0];
          const nextRes = polygonResultsRef.current.get(nextId);
          if (nextRes) {
            setTimeout(() => onRequestParams?.(nextId, nextRes.polygon.coordinates), 0);
          } else {
            const draw = drawRef.current as any;
            const f = draw?.get?.(nextId);
            if (f?.geometry?.type === 'Polygon') setTimeout(() => onRequestParams?.(nextId, f.geometry.coordinates[0]), 0);
          }
        } else if (rest.length === 0) {
          bulkPresetParamsRef.current = null;
        }
        return rest;
      });
    }, [polygonResults, polygonTiles, onFlightLinesUpdated, altitudeMode, minClearanceM, turnExtendM]);

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
        applyPolygonParams(polygonId, params, { skipEvent: true, skipQueue: true });
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

    type ExactBearingCandidate = {
      bearingDeg: number;
      exactCost: number;
      qualityCost: number;
      missionTimeSec: number;
      normalizedTimeCost: number;
      metricKind: 'density' | 'gsd';
      stats: GSDStats;
      diagnostics: Record<string, number>;
    };

    type ExactBearingSearchResult = {
      best: ExactBearingCandidate | null;
      evaluated: ExactBearingCandidate[];
      seedBearingDeg: number;
      lineSpacingM: number;
      safeParams: PolygonParams;
    };

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

      const normalizedSeedBearingDeg = Number.isFinite(seedBearingDeg)
        ? normalizeAxialBearingDeg(seedBearingDeg)
        : 0;
      const lineSpacing = getLineSpacingForParams(safeParams);
      const photoSpacing = getForwardSpacingForParams(safeParams);
      const isLidar = isLidarParams(safeParams);
      const altitudeAGL = safeParams.altitudeAGL;
      const terrainTiles = tiles as any[];
      const exactQualityWeight = 1 - EXACT_OPTIMIZE_TIME_WEIGHT;
      const coarseOffsets = [-30, -20, -10, 0, 10, 20, 30];
      const globalCoarseBearings = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165];
      const refineStepsDeg = [8, 4, 2, 1];
      const minImprovement = 1e-4;
      const tileRefs = (() => {
        const seen = new Set<string>();
        const refs: Array<{ z: number; x: number; y: number }> = [];
        for (const tile of tilesCoveringPolygon({ ring }, EXACT_OPTIMIZE_ZOOM)) {
          const key = `${EXACT_OPTIMIZE_ZOOM}/${tile.x}/${tile.y}`;
          if (seen.has(key)) continue;
          seen.add(key);
          refs.push({ z: EXACT_OPTIMIZE_ZOOM, x: tile.x, y: tile.y });
        }
        return refs;
      })();

      await new Promise<void>((resolve) => {
        if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
          window.requestAnimationFrame(() => resolve());
          return;
        }
        setTimeout(resolve, 0);
      });

      const terrainTileCache = new Map<string, { width: number; height: number; data: Uint8ClampedArray }>();
      const normalizeTileRef = (tileRef: { z: number; x: number; y: number }) => {
        const tilesPerAxis = 1 << tileRef.z;
        const wrappedX = ((tileRef.x % tilesPerAxis) + tilesPerAxis) % tilesPerAxis;
        const clampedY = Math.max(0, Math.min(tilesPerAxis - 1, tileRef.y));
        return { z: tileRef.z, x: wrappedX, y: clampedY };
      };
      const getTile = async (tileRef: { z: number; x: number; y: number }) => {
        const normalizedRef = normalizeTileRef(tileRef);
        const cacheKey = `${normalizedRef.z}/${normalizedRef.x}/${normalizedRef.y}`;
        let tileData = terrainTileCache.get(cacheKey);
        if (!tileData) {
          const imgData = await fetchTerrainRGBA(normalizedRef.z, normalizedRef.x, normalizedRef.y, mapboxToken);
          tileData = {
            width: imgData.width,
            height: imgData.height,
            data: new Uint8ClampedArray(imgData.data),
          };
          terrainTileCache.set(cacheKey, tileData);
        }
        return {
          tile: {
            z: normalizedRef.z,
            x: normalizedRef.x,
            y: normalizedRef.y,
            size: tileData.width,
            data: new Uint8ClampedArray(tileData.data),
          },
        };
      };
      const getLidarTileWithHalo = async (tileRef: { z: number; x: number; y: number }, padTiles = 1) => {
        const center = await getTile(tileRef);
        if (padTiles <= 0) {
          return {
            tile: center.tile,
            demTile: {
              size: center.tile.size,
              padTiles: 0,
              data: center.tile.data,
            },
          };
        }
        const offsets: Array<{ dx: number; dy: number; tileRef: { z: number; x: number; y: number } }> = [];
        for (let dy = -padTiles; dy <= padTiles; dy++) {
          for (let dx = -padTiles; dx <= padTiles; dx++) {
            offsets.push({
              dx,
              dy,
              tileRef: normalizeTileRef({ z: tileRef.z, x: tileRef.x + dx, y: tileRef.y + dy }),
            });
          }
        }
        const neighborTiles = await Promise.all(offsets.map((entry) => getTile(entry.tileRef)));
        const tileSize = center.tile.size;
        const span = padTiles * 2 + 1;
        const demSize = tileSize * span;
        const demData = new Uint8ClampedArray(demSize * demSize * 4);
        for (let i = 0; i < offsets.length; i++) {
          const { dx, dy } = offsets[i];
          const srcTile = neighborTiles[i].tile;
          const offsetX = (dx + padTiles) * tileSize;
          const offsetY = (dy + padTiles) * tileSize;
          for (let row = 0; row < tileSize; row++) {
            const srcStart = row * tileSize * 4;
            const dstStart = ((offsetY + row) * demSize + offsetX) * 4;
            demData.set(srcTile.data.subarray(srcStart, srcStart + tileSize * 4), dstStart);
          }
        }
        return {
          tile: center.tile,
          demTile: { size: demSize, padTiles, data: demData },
        };
      };

      const estimateMissionTimeSec = (bearingDeg: number) => {
        const { flightLines } = generateFlightLinesForPolygon(ring, bearingDeg, lineSpacing);
        if (!flightLines.length) return 0;
        const path3d = build3DFlightPath(
          flightLines,
          terrainTiles,
          lineSpacing,
          { altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, turnExtendM },
        );
        const totalLengthM = path3dLengthMeters(path3d);
        const speedMps = isLidar
          ? (safeParams.speedMps ?? getLidarModel(safeParams.lidarKey).defaultSpeedMps)
          : 12;
        return totalLengthM / Math.max(0.1, speedMps);
      };

      const cameraWorker = !isLidar ? new OverlapWorker() : null;
      const lidarWorker = isLidar ? new LidarDensityWorker() : null;
      const exactCache = new Map<number, Promise<ExactBearingCandidate | null>>();

      const evaluateBearingExactly = async (bearingDeg: number): Promise<ExactBearingCandidate | null> => {
        const normalizedBearingDeg = normalizeAxialBearingDeg(bearingDeg);
        const cacheKey = Math.round(normalizedBearingDeg * 1000);
        if (exactCache.has(cacheKey)) {
          return exactCache.get(cacheKey)!;
        }
        const evaluationPromise = (async () => {
          if (isLidar) {
            const model = getLidarModel(safeParams.lidarKey);
            const mappingFovDeg = getLidarMappingFovDeg(model, safeParams.mappingFovDeg);
            const speedMps = safeParams.speedMps ?? model.defaultSpeedMps;
            const returnMode = safeParams.lidarReturnMode ?? 'single';
            const maxLidarRangeM = safeParams.maxLidarRangeM ?? model.defaultMaxRangeM ?? DEFAULT_LIDAR_MAX_RANGE_M;
            const frameRateHz = safeParams.lidarFrameRateHz ?? model.defaultFrameRateHz;
            const azimuthSectorCenterDeg = safeParams.lidarAzimuthSectorCenterDeg ?? model.defaultAzimuthSectorCenterDeg ?? 0;
            const boresightYawDeg = safeParams.lidarBoresightYawDeg ?? model.boresightYawDeg ?? 0;
            const boresightPitchDeg = safeParams.lidarBoresightPitchDeg ?? model.boresightPitchDeg ?? 0;
            const boresightRollDeg = safeParams.lidarBoresightRollDeg ?? model.boresightRollDeg ?? 0;
            const comparisonMode = safeParams.lidarComparisonMode ?? 'first-return';
            const densityPerPass = lidarSinglePassDensity(model, altitudeAGL, speedMps, returnMode, mappingFovDeg);
            const halfFovTan = Math.tan((mappingFovDeg * Math.PI) / 360);
            const strips: LidarStripMeters[] = [];
            const { flightLines } = generateFlightLinesForPolygon(ring, normalizedBearingDeg, lineSpacing);
            let passIndex = 0;
            for (let lineIndex = 0; lineIndex < flightLines.length; lineIndex++) {
              const sourceLine = flightLines[lineIndex];
              if (!Array.isArray(sourceLine) || sourceLine.length < 2) continue;
              const flownLine = lineIndex % 2 === 0 ? sourceLine : [...sourceLine].reverse();
              const activeSweepLine = extendFlightLineForTurnRunout(flownLine, turnExtendM);
              const sweepPath3d = build3DFlightPath(
                [activeSweepLine],
                terrainTiles,
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
                const terrainMin = queryMinMaxElevationAlongPolylineWGS84([[start[0], start[1]], [end[0], end[1]]], terrainTiles, 12).min;
                const maxSensorAltitude = Math.max(start[2], end[2]);
                const maxHalfWidth = Number.isFinite(terrainMin)
                  ? Math.max(lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2, Math.max(1, (maxSensorAltitude - terrainMin) * halfFovTan))
                  : lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2;
                strips.push({
                  id: `${scopeId}-line-${lineIndex}-seg-${i - 1}`,
                  polygonId: scopeId,
                  x1, y1, z1: start[2], x2, y2, z2: end[2],
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
            if (strips.length === 0 || !lidarWorker) return null;
            const perTileStats: GSDStats[] = [];
            for (const tileRef of tileRefs) {
              const tileStrips = strips.filter((strip) => lidarStripMayAffectTile(strip, tileRef));
              if (tileStrips.length === 0) continue;
              const { tile, demTile } = await getLidarTileWithHalo(tileRef, 1);
              const response = await lidarWorker.runTile({
                tile,
                demTile,
                polygons: [{ id: scopeId, ring }],
                strips: tileStrips,
                options: { clipInnerBufferM: 0 },
              } as any);
              const densityStats = response.perPolygon?.find((entry) => entry.polygonId === scopeId)?.densityStats;
              if (densityStats) perTileStats.push(densityStats);
            }
            if (perTileStats.length === 0) return null;
            const stats = aggregateMetricStats(perTileStats);
            const scored = scoreExactLidarStats(stats, safeParams);
            const missionTimeSec = estimateMissionTimeSec(normalizedBearingDeg);
            const normalizedTimeCost = missionTimeSec / 180;
            const exactCost = exactQualityWeight * scored.qualityCost + EXACT_OPTIMIZE_TIME_WEIGHT * normalizedTimeCost;
            const diagnostics: Record<string, number> = {
              qualityCost: scored.qualityCost,
              missionTimeSec,
              normalizedTimeCost,
              targetDensityPtsM2: scored.targetDensityPtsM2,
              holeFraction: scored.holeFraction,
              lowFraction: scored.lowFraction,
              q10: scored.q10,
              q25: scored.q25,
            };
            return {
              bearingDeg: normalizedBearingDeg,
              exactCost,
              qualityCost: scored.qualityCost,
              missionTimeSec,
              normalizedTimeCost,
              metricKind: 'density' as const,
              stats,
              diagnostics,
            };
          }

          if (!cameraWorker || !photoSpacing || !(photoSpacing > 0)) return null;
          const cameraKey = safeParams.cameraKey;
          const camera = cameraKey ? CAMERA_REGISTRY[cameraKey] || DEFAULT_CAMERA : DEFAULT_CAMERA;
          const yawOffset = safeParams.cameraYawOffsetDeg ?? 0;
          const normalizeDeg = (value: number) => ((value % 360) + 360) % 360;
          const poses: PoseMeters[] = [];
          const { flightLines } = generateFlightLinesForPolygon(ring, normalizedBearingDeg, lineSpacing);
          const path3d = build3DFlightPath(
            flightLines,
            terrainTiles,
            lineSpacing,
            { altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, turnExtendM },
          );
          const cameraPositions = sampleCameraPositionsOnFlightPath(path3d, photoSpacing, { includeTurns: false });
          const filtered = ring.length >= 3
            ? cameraPositions.filter(([lng, lat]) => pointInRing(lng, lat, ring))
            : cameraPositions;
          filtered.forEach(([lng, lat, altMSL, yawDeg], index) => {
            const [x, y] = lngLatToMeters(lng, lat);
            poses.push({
              id: `opt_pose_${scopeId}_${index}`,
              x,
              y,
              z: altMSL,
              omega_deg: 0,
              phi_deg: 0,
              kappa_deg: normalizeDeg(-yawDeg + yawOffset),
              polygonId: scopeId,
            });
          });
          if (poses.length === 0) return null;
          const perTileStats: GSDStats[] = [];
          for (const tileRef of tileRefs) {
            const { tile } = await getTile(tileRef);
            const response = await cameraWorker.runTile({
              tile,
              polygons: [{ id: scopeId, ring }],
              poses,
              cameras: [camera],
              poseCameraIndices: new Uint16Array(poses.length),
              camera: undefined,
              options: { clipInnerBufferM: 0, minOverlapForGsd: EXACT_MIN_OVERLAP_FOR_GSD },
            } as any);
            const gsdStats = response.perPolygon?.find((entry) => entry.polygonId === scopeId)?.gsdStats;
            if (gsdStats) perTileStats.push(gsdStats);
          }
          if (perTileStats.length === 0) return null;
          const stats = aggregateMetricStats(perTileStats);
          const scored = scoreExactCameraStats(stats, safeParams);
          const missionTimeSec = estimateMissionTimeSec(normalizedBearingDeg);
          const normalizedTimeCost = missionTimeSec / 180;
          const exactCost = exactQualityWeight * scored.qualityCost + EXACT_OPTIMIZE_TIME_WEIGHT * normalizedTimeCost;
          const diagnostics: Record<string, number> = {
            qualityCost: scored.qualityCost,
            missionTimeSec,
            normalizedTimeCost,
            targetGsdM: scored.targetGsdM,
            overTargetAreaFraction: scored.overTargetAreaFraction,
            q75: scored.q75,
            q90: scored.q90,
          };
          return {
            bearingDeg: normalizedBearingDeg,
            exactCost,
            qualityCost: scored.qualityCost,
            missionTimeSec,
            normalizedTimeCost,
            metricKind: 'gsd' as const,
            stats,
            diagnostics,
          };
        })();
        exactCache.set(cacheKey, evaluationPromise);
        return evaluationPromise;
      };

      const evaluateOffset = async (offsetDeg: number) => {
        if (Math.abs(offsetDeg) > 30 + 1e-6) return null;
        return evaluateBearingExactly(normalizedSeedBearingDeg + offsetDeg);
      };

      try {
        let best: ExactBearingCandidate | null = null;
        let bestOffset = 0;

        if (mode === 'global') {
          const coarseCandidates = Array.from(new Set([
            ...globalCoarseBearings,
            Math.round(normalizedSeedBearingDeg * 10) / 10,
          ]));
          for (const bearingDeg of coarseCandidates) {
            const candidate = await evaluateBearingExactly(bearingDeg);
            if (!candidate) continue;
            if (!best || candidate.exactCost < best.exactCost) {
              best = candidate;
            }
          }
        } else {
          best = await evaluateOffset(0);
          bestOffset = 0;
          for (const offsetDeg of coarseOffsets) {
            const candidate = await evaluateOffset(offsetDeg);
            if (!candidate) continue;
            if (!best || candidate.exactCost < best.exactCost) {
              best = candidate;
              bestOffset = offsetDeg;
            }
          }
        }

        if (best && Number.isFinite(best.bearingDeg)) {
          for (const stepDeg of refineStepsDeg) {
            let improved = true;
            while (improved) {
              improved = false;
              const currentBest: ExactBearingCandidate = best;
              const left: ExactBearingCandidate | null = mode === 'global'
                ? await evaluateBearingExactly(currentBest.bearingDeg - stepDeg)
                : await evaluateOffset(bestOffset - stepDeg);
              const right: ExactBearingCandidate | null = mode === 'global'
                ? await evaluateBearingExactly(currentBest.bearingDeg + stepDeg)
                : await evaluateOffset(bestOffset + stepDeg);
              const nextBest: { offsetDeg: number; candidate: ExactBearingCandidate } | null =
                [
                  { offsetDeg: mode === 'global' ? 0 : (bestOffset - stepDeg), candidate: left },
                  { offsetDeg: mode === 'global' ? 0 : (bestOffset + stepDeg), candidate: right },
                ]
                  .filter((value): value is { offsetDeg: number; candidate: ExactBearingCandidate } => value.candidate !== null)
                  .sort((a, b) => a.candidate.exactCost - b.candidate.exactCost)[0] ?? null;
              if (nextBest && nextBest.candidate.exactCost + minImprovement < currentBest.exactCost) {
                best = nextBest.candidate;
                if (mode !== 'global') {
                  bestOffset = nextBest.offsetDeg;
                }
                improved = true;
              }
            }
          }
        }

        const evaluated = (await Promise.all([...exactCache.values()]))
          .filter((value): value is ExactBearingCandidate => value !== null)
          .sort((left, right) => left.exactCost - right.exactCost);

        return {
          best: best ?? null,
          evaluated,
          seedBearingDeg: normalizedSeedBearingDeg,
          lineSpacingM: lineSpacing,
          safeParams,
        };
      } finally {
        cameraWorker?.terminate();
        lidarWorker?.terminate();
      }
    }, [altitudeMode, mapboxToken, minClearanceM, turnExtendM]);

    const runOptimizedBearingSearch = useCallback(async (
      polygonId: string,
      params: PolygonParams,
      result: PolygonAnalysisResult,
      tiles: any[],
    ) => {
      const ring = result.polygon.coordinates as [number, number][];
      const { best, evaluated, seedBearingDeg, lineSpacingM, safeParams } = await searchExactBearingNearSeed({
        scopeId: polygonId,
        ring,
        params,
        tiles,
        seedBearingDeg: result.result.contourDirDeg,
        mode: 'global',
      });

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
        applyPolygonParams(polygonId, safeParams, { skipQueue: true });
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
      applyPolygonParams(polygonId, safeParams, { skipQueue: true });
    }, [applyPolygonParams, searchExactBearingNearSeed]);

    // Rebuild 3D paths when altitude mode, minimum clearance, or turn extension changes
    useEffect(() => {
      if (!deckOverlayRef.current) return;
      const overlay = deckOverlayRef.current;
      // For each polygon with flight lines and tiles, rebuild path3D and update layer
      polygonFlightLines.forEach((fl, pid) => {
        const tiles = polygonTiles.get(pid) || [];
        if (!tiles || fl.flightLines.length === 0) return;
        const path3d = build3DFlightPath(fl.flightLines, tiles, fl.lineSpacing, { altitudeAGL: fl.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, turnExtendM });
        update3DPathLayer(overlay, pid, path3d, setDeckLayers);
      });
    }, [altitudeMode, minClearanceM, turnExtendM]);

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

    // ---------- analysis callbacks ----------
    const handleAnalysisResult = useCallback(
      (result: PolygonAnalysisResult, tiles: any[]) => {
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
            const next = [...prev, result.polygonId];
            if (next.length === 1) {
              // defer to avoid render phase state update warning
              setTimeout(() => onRequestParams?.(result.polygonId, result.polygon.coordinates), 0);
            }
            return next;
          });
          return;
        }

        if (!mapRef.current) return;

        // Use override if present (e.g., file direction), otherwise terrain-optimal
        const safeParams = sanitizePolygonParams(params);
        const bearingDeg = override ? override.bearingDeg : result.result.contourDirDeg;

        // Spacing: keep override spacing if present, otherwise recompute from params
        const spacing =
          override?.lineSpacingM ??
          getLineSpacingForParams(safeParams);

        // Remove existing flight lines first to avoid Mapbox layer conflicts
        removeFlightLinesForPolygon(mapRef.current, result.polygonId);

        const lines = addFlightLinesForPolygon(
          mapRef.current,
          result.polygonId,
          result.polygon.coordinates,
          bearingDeg,
          spacing,
          result.result.fitQuality
        );

        const nextFlightLines = new Map(polygonFlightLinesRef.current);
        nextFlightLines.set(result.polygonId, { ...lines, altitudeAGL: safeParams.altitudeAGL });
        polygonFlightLinesRef.current = nextFlightLines;
        setPolygonFlightLines(nextFlightLines);

        if (!suppressFlightLineEventsRef.current) {
          if (pendingGeometryRefreshRef.current.delete(result.polygonId)) {
            onFlightLinesUpdated?.('__all__');
          } else {
            onFlightLinesUpdated?.(result.polygonId);
          }
        }

        if (deckOverlayRef.current && lines.flightLines.length > 0) {
          const path3d = build3DFlightPath(lines.flightLines, tiles, lines.lineSpacing, { altitudeAGL: safeParams.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, turnExtendM });
          update3DPathLayer(deckOverlayRef.current, result.polygonId, path3d, setDeckLayers);
          const spacingForward = getForwardSpacingForParams(safeParams);
          if (spacingForward && spacingForward > 0) {
            // 3D trigger points sampled along the 3D path
            const samples = sampleCameraPositionsOnFlightPath(path3d, spacingForward, { includeTurns: false });
            const zOffset = 1; // lift triggers slightly above the path for visibility
            const ring = result.polygon.coordinates as [number,number][];
            const inside = (lng:number,lat:number,ring:[number,number][]) => {
              let ins=false; for(let i=0,j=ring.length-1;i<ring.length;j=i++){
                const xi=ring[i][0], yi=ring[i][1], xj=ring[j][0], yj=ring[j][1];
                const intersect=((yi>lat)!==(yj>lat)) && (lng < (xj-xi)*(lat-yi)/(yj-yi)+xi); if(intersect) ins=!ins;
              } return ins;
            };
            const positions: [number, number, number][] = samples
              .filter(([lng,lat]) => inside(lng,lat,ring))
              .map(([lng,lat,alt]) => [lng,lat,alt + zOffset]);
            update3DTriggerPointsLayer(deckOverlayRef.current, result.polygonId, positions, setDeckLayers);
          } else {
            remove3DTriggerPointsLayer(deckOverlayRef.current, result.polygonId, setDeckLayers);
          }
        }

        if (pendingOptimizeRef.current.has(result.polygonId)) {
          pendingOptimizeRef.current.delete(result.polygonId);
          const currentParams = polygonParamsRef.current.get(result.polygonId) ?? { altitudeAGL: 100, frontOverlap: 80, sideOverlap: 70 };
          const paramsToApply: PolygonParams = { ...currentParams, useCustomBearing: false };
          if ('customBearingDeg' in paramsToApply) delete (paramsToApply as any).customBearingDeg;
          void withProcessingPolygon(result.polygonId, () =>
            runOptimizedBearingSearch(result.polygonId, paramsToApply, result, tiles),
          );
        }
      },
      [debouncedAnalysisComplete, onFlightLinesUpdated, onRequestParams, altitudeMode, minClearanceM, runOptimizedBearingSearch, withProcessingPolygon]
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

      setTimeout(() => {
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
        onPolygonSelected?.(null);
        if (!suppressFlightLineEventsRef.current) {
          onFlightLinesUpdated?.('__all__');
        }
        try {
          const coll = (drawRef.current as any)?.getAll?.();
          const hasAny = Array.isArray(coll?.features) && coll.features.some((f: any) => f?.geometry?.type === 'Polygon');
          if (!hasAny) onClearGSD?.();
        } catch {}
      }, 0);
    }, [cancelAnalysis, debouncedAnalysisComplete, onClearGSD, onFlightLinesUpdated, onPolygonSelected]);

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
      setTimeout(() => {
        if (draw.get?.(polygonId)) {
          splitPerfLog(polygonId, 'deletePolygonFeature retrying after initial failure');
          attemptDelete();
        }
        if (pendingProgrammaticDeletesRef.current.delete(polygonId)) {
          cleanupPolygonState(polygonId);
        }
      }, 0);
    }, [cleanupPolygonState]);

    const editPolygonBoundary = useCallback((polygonId: string) => {
      const draw = drawRef.current as any;
      const feature = draw?.get?.(polygonId);
      if (!draw || feature?.geometry?.type !== 'Polygon') return;

      suppressSelectionDialogUntilRef.current = Date.now() + 1000;
      setTimeout(() => onPolygonSelected?.(polygonId), 0);

      try {
        draw.changeMode('simple_select', { featureIds: [polygonId] });
      } catch {}

      setTimeout(() => {
        try {
          draw.changeMode('direct_select', { featureId: polygonId });
        } catch {
          try {
            draw.changeMode('simple_select', { featureIds: [polygonId] });
          } catch {}
        }
      }, 0);
    }, [onPolygonSelected]);

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

    // ---------- Mapbox Draw handlers ----------
    const handleDrawCreate = useCallback((e: any) => {
      if (suspendAutoAnalysisRef.current) return;
      e.features.forEach((feature: any) => {
        if (feature.geometry.type === 'Polygon') {
          analyzePolygon(feature.id, feature);
        }
      });
    }, [analyzePolygon]);

    const handleDrawUpdate = useCallback((e: any) => {
      if (suspendAutoAnalysisRef.current) return;
      e.features.forEach((feature: any) => {
        if (feature.geometry.type === 'Polygon') {
          pendingGeometryRefreshRef.current.add(String(feature.id));
          analyzePolygon(feature.id, feature);
        }
      });
    }, [analyzePolygon]);

    const handleDrawDelete = useCallback((e: any) => {
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
    }, [cleanupPolygonState]);

    // ---------- Map init ----------
    const onMapLoad = useCallback(
      (map: MapboxMap, draw: MapboxDraw, overlay: MapboxOverlay) => {
        mapRef.current = map;
        drawRef.current = draw;
        deckOverlayRef.current = overlay;
        setTerrainDemSourceOnMap(map, terrainDemSourceTemplateRef.current);
        map.on('draw.create', handleDrawCreate);
        map.on('draw.update', handleDrawUpdate);
        map.on('draw.delete', handleDrawDelete);
        map.on('draw.create', syncProcessingPerimeterOverlay);
        map.on('draw.update', syncProcessingPerimeterOverlay);
        map.on('draw.delete', syncProcessingPerimeterOverlay);
        syncProcessingPerimeterOverlay();
        if (processingPolygonIdsRef.current.size > 0) {
          startProcessingPerimeterAnimation();
        }
        // Open params dialog when user selects an existing polygon
        map.on('draw.selectionchange', (e: any) => {
          try {
            const feats = Array.isArray(e?.features) ? e.features : [];
            const f = feats.find((f: any) => f?.geometry?.type === 'Polygon');
            if (!f) {
              if (suppressNextEmptySelectionRef.current > 0) {
                suppressNextEmptySelectionRef.current -= 1;
                return;
              }
              setTimeout(() => onPolygonSelected?.(null), 0);
              return;
            }
            const pid = f.id as string;
            const ring = (f.geometry?.coordinates?.[0]) as [number, number][] | undefined;
            if (pid && ring && ring.length >= 4) {
              const drawMode = (drawRef.current as any)?.getMode?.();
              const shouldSuppressDialog =
                drawMode === 'direct_select' || Date.now() < suppressSelectionDialogUntilRef.current;
              // Defer to avoid selection-change re-entrancy
              setTimeout(() => onPolygonSelected?.(pid), 0);
              if (!shouldSuppressDialog) {
                setTimeout(()=> onRequestParams?.(pid, ring), 0);
                // Keep normal polygon clicks lightweight: once the app has recorded the
                // selection, clear Draw's internal feature selection so left-drag pans
                // the map again instead of staying in feature-move mode.
                setTimeout(() => clearDrawSelectionForPan(), 0);
              }
            }
          } catch {}
        });
      },
      [clearDrawSelectionForPan, handleDrawCreate, handleDrawUpdate, handleDrawDelete, onRequestParams, onPolygonSelected]
    );

    useEffect(() => {
      terrainDemSourceTemplateRef.current = terrainDemUrlTemplate;
      if (mapRef.current && mapRef.current.isStyleLoaded()) {
        setTerrainDemSourceOnMap(mapRef.current, terrainDemUrlTemplate);
      }
    }, [terrainDemUrlTemplate]);

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
    }, [addRingAsDrawFeature, onError]);

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
      try {
        console.log(`📥 Importing Wingtra flightplan...`);
        const parsed = JSON.parse(json);
        setLastImportedFlightplan(parsed);
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

          const payloadKind = item.payloadKind ?? imported.payloadKind ?? DEFAULT_PAYLOAD_KIND;
          const lidarKey = item.lidarKey || imported.payloadLidarKey || DEFAULT_LIDAR.key;
          const cameraKey = item.cameraKey || imported.payloadCameraKey || 'SONY_RX1R2';
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
              altitudeAGL: item.altitudeAGL,
              frontOverlap: item.frontOverlap,
              sideOverlap: item.sideOverlap,
              cameraKey: payloadKind === 'camera' ? cameraKey : undefined,
              lidarKey: payloadKind === 'lidar' ? lidarKey : undefined,
              triggerDistanceM: item.triggerDistanceM,
              cameraYawOffsetDeg,
              speedMps: payloadKind === 'lidar' ? item.speedMps : undefined,
              lidarReturnMode: payloadKind === 'lidar' ? item.lidarReturnMode : undefined,
              mappingFovDeg: payloadKind === 'lidar' ? item.mappingFovDeg : undefined,
              maxLidarRangeM: payloadKind === 'lidar' ? item.maxLidarRangeM : undefined,
              pointDensityPtsM2: payloadKind === 'lidar' ? item.pointDensityPtsM2 : undefined,
            },
            original: { bearingDeg: item.angleDeg, lineSpacingM: item.lineSpacingM },
            override: { bearingDeg: item.angleDeg, lineSpacingM: item.lineSpacingM, source: 'wingtra' as const }
          };
          polygonsToUpdate.set(id, polygonState);

          const lines = addFlightLinesForPolygon(
            mapRef.current,
            id,
            item.ring as number[][],
            item.angleDeg,
            item.lineSpacingM,
            undefined
          );
          flightLinesToUpdate.set(id, { ...lines, altitudeAGL: item.altitudeAGL });

          areasOut.push({
            polygonId: id,
            params: {
              payloadKind,
              altitudeAGL: item.altitudeAGL,
              frontOverlap: item.frontOverlap,
              sideOverlap: item.sideOverlap,
              angleDeg: item.angleDeg,
              lineSpacingM: item.lineSpacingM,
              triggerDistanceM: item.triggerDistanceM,
              cameraKey: payloadKind === 'camera' ? cameraKey : undefined,
              lidarKey: payloadKind === 'lidar' ? lidarKey : undefined,
              cameraYawOffsetDeg,
              speedMps: payloadKind === 'lidar' ? item.speedMps : undefined,
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
            const path3d = build3DFlightPath(flEntry.flightLines, tiles, lineSpacing, { altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, turnExtendM });
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
        onAnalysisComplete?.(Array.from(polygonResultsRef.current.values()));

        // 5) Allow per‑polygon events again & emit a single aggregate update
        suppressFlightLineEventsRef.current = false;
        onFlightLinesUpdated?.('__all__');

        console.log(`✅ Successfully imported ${newIds.length} areas with file bearings preserved. Use "Optimize" to get terrain-optimal directions.`);
        return { added: newIds.length, total: imported.items.length, areas: areasOut };
      } catch (e) {
        suppressFlightLineEventsRef.current = false;
        onError?.(`Failed to import flightplan: ${e instanceof Error ? e.message : 'Unknown error'}`);
        suspendAutoAnalysisRef.current = false;
        return { added: 0, total: 0, areas: [] };
      }
    }, [mapboxToken, addRingAsDrawFeature, polygonFlightLines, polygonParams, fitMapToRings, analyzePolygon, onFlightLinesUpdated, onError]);

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
          applyPolygonParams(pid, params, { skipEvent: true, skipQueue: true });
        }
      }
      setPendingParamPolygons([]);
      suppressFlightLineEventsRef.current = prevSuppress;
      bulkApplyRef.current = false;
      onFlightLinesUpdated?.('__all__');
    }, [pendingParamPolygons, onFlightLinesUpdated, applyPolygonParams]);

    // RESTORED: optimizePolygonDirection (terrain-optimal)
    const optimizePolygonDirection = useCallback((polygonId: string) => {
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
      void withProcessingPolygon(polygonId, () =>
        runOptimizedBearingSearch(polygonId, params, result, tiles),
      );
    }, [analyzePolygon, runOptimizedBearingSearch, withProcessingPolygon]);

    // RESTORED: revertPolygonToImportedDirection
    const revertPolygonToImportedDirection = useCallback((polygonId: string) => {
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
      removeFlightLinesForPolygon(mapRef.current, polygonId);
      const fl = addFlightLinesForPolygon(
        mapRef.current,
        polygonId,
        res.polygon.coordinates,
        original.bearingDeg,
        original.lineSpacingM,
        res.result.fitQuality
      );
      setPolygonFlightLines((prev) => {
        const next = new Map(prev);
        next.set(polygonId, { ...fl, altitudeAGL: params.altitudeAGL });
        return next;
      });
      const tiles = polygonTiles.get(polygonId) || [];
      if (deckOverlayRef.current && fl.flightLines.length > 0) {
        const path3d = build3DFlightPath(fl.flightLines, tiles, fl.lineSpacing, { altitudeAGL: params.altitudeAGL, mode: altitudeMode, minClearance: minClearanceM, turnExtendM });
        update3DPathLayer(deckOverlayRef.current, polygonId, path3d, setDeckLayers);
      }
      console.log(`✅ Restored file direction: ${original.bearingDeg}° bearing, ${original.lineSpacingM}m spacing`);
      onFlightLinesUpdated?.(polygonId);
    }, [importedOriginals, polygonResults, polygonParams, polygonTiles, onFlightLinesUpdated]);

    // RESTORED: runFullAnalysis
    const runFullAnalysis = useCallback((polygonId: string) => {
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
    }, [analyzePolygon]);

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

    const getLocalTerrainPartitionSolutions = useCallback((
      ring: [number, number][],
      tiles: any[],
      params: PolygonParams,
    ): TerrainPartitionSolutionPreview[] => {
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
          backendPartitionSolutionsRef.current.set(polygonId, solutions);
          return solutions.filter((solution) => solution.regions.length > 1);
        } catch (error) {
          splitPerfLog(polygonId, 'backend partition solve failed; falling back to local solver', {
            totalMs: Math.round(splitPerfNow() - startedAt),
            error: error instanceof Error ? error.message : String(error),
          });
          console.warn('Terrain partition backend failed, falling back to local solver.', error);
        }
      }
      const localStartedAt = splitPerfNow();
      const local = getLocalTerrainPartitionSolutions(ring, tiles, params);
      splitPerfLog(polygonId, 'local partition solutions computed', {
        totalMs: Math.round(splitPerfNow() - localStartedAt),
        solutionCount: local.length,
      });
      backendPartitionSolutionsRef.current.set(polygonId, local);
      return local;
    }, [altitudeMode, getLocalTerrainPartitionSolutions, getTerrainPartitionContext, minClearanceM, terrainSource, turnExtendM]);

    const applyTerrainPartitionRings = useCallback(async (
      polygonId: string,
      partitionRegions: TerrainPartitionRegionApplication[],
      inheritedParams: PolygonParams,
    ): Promise<{ createdIds: string[]; replaced: boolean }> => {
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

      suspendAutoAnalysisRef.current = true;
      const prevSuppress = suppressFlightLineEventsRef.current;
      suppressFlightLineEventsRef.current = true;
      const createdIds: string[] = [];

      try {
        const createStartedAt = splitPerfNow();
        const createdRegions: Array<{ id: string; region: TerrainPartitionRegionApplication & { ring: [number, number][] } }> = [];
        for (const region of normalizedPartitionRegions) {
          const id = addRingAsDrawFeature(region.ring, 'Terrain Face', { source: 'terrain-face-split', parentPolygonId: polygonId });
          if (id) {
            createdIds.push(id);
            createdRegions.push({ id, region });
          }
        }
        splitPerfLog(polygonId, 'child polygons created', {
          totalMs: Math.round(splitPerfNow() - createStartedAt),
          createdIds,
        });

        if (createdIds.length <= 1) {
          onError?.('Auto-split did not produce enough valid child polygons.', polygonId);
          return { createdIds: [], replaced: false };
        }

        const nextParams = new Map(polygonParamsRef.current);
        nextParams.delete(polygonId);
        createdRegions.forEach(({ id, region }) => {
          nextParams.set(id, {
            ...inheritedParams,
            altitudeAGL: region.baseAltitudeAGL ?? inheritedParams.altitudeAGL,
          });
        });
        polygonParamsRef.current = nextParams;
        setPolygonParams(new Map(nextParams));

        const nextOverrides = new Map(bearingOverridesRef.current);
        nextOverrides.delete(polygonId);
        createdRegions.forEach(({ id, region }) => {
          if (!Number.isFinite(region.bearingDeg)) return;
          const childParams = nextParams.get(id) ?? inheritedParams;
          nextOverrides.set(id, {
            bearingDeg: region.bearingDeg!,
            lineSpacingM: getLineSpacingForParams(childParams),
            source: 'partition',
          });
        });
        bearingOverridesRef.current = nextOverrides;
        setBearingOverrides(new Map(nextOverrides));

        setImportedOriginals((prev) => {
          const next = new Map(prev);
          next.delete(polygonId);
          importedOriginalsRef.current = next;
          return next;
        });
        setPolygonResults((prev) => {
          if (!prev.has(polygonId)) return prev;
          const next = new Map(prev);
          next.delete(polygonId);
          polygonResultsRef.current = next;
          debouncedAnalysisComplete();
          return next;
        });
        if (polygonTilesRef.current.has(polygonId)) {
          const nextTiles = new Map(polygonTilesRef.current);
          nextTiles.delete(polygonId);
          polygonTilesRef.current = nextTiles;
          setPolygonTiles(nextTiles);
        }
        if (polygonFlightLinesRef.current.has(polygonId)) {
          const nextFlightLines = new Map(polygonFlightLinesRef.current);
          nextFlightLines.delete(polygonId);
          polygonFlightLinesRef.current = nextFlightLines;
          setPolygonFlightLines(nextFlightLines);
        }
        setPendingParamPolygons((prev) => prev.filter((id) => id !== polygonId));
        backendPartitionSolutionsRef.current.delete(polygonId);
        createdIds.forEach((id) => backendPartitionSolutionsRef.current.delete(id));

        if (mapRef.current) {
          removeFlightLinesForPolygon(mapRef.current, polygonId);
          removeTriggerPointsForPolygon(mapRef.current, polygonId);
        }
        if (deckOverlayRef.current) {
          remove3DPathLayer(deckOverlayRef.current, polygonId, setDeckLayers);
          remove3DTriggerPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
          remove3DCameraPointsLayer(deckOverlayRef.current, polygonId, setDeckLayers);
        }

        deletePolygonFeature(polygonId);

        fitMapToRings(normalizedPartitionRegions.map((region) => region.ring));

        suspendAutoAnalysisRef.current = false;
        const analyzeStartedAt = splitPerfNow();
        const analysisPromises: Promise<any>[] = [];
        for (const childId of createdIds) {
          const childFeature = draw?.get?.(childId);
          if (childFeature?.geometry?.type === 'Polygon') {
            analysisPromises.push(analyzePolygon(childId, childFeature));
          }
        }
        await Promise.allSettled(analysisPromises);
        splitPerfLog(polygonId, 'child analyzePolygon calls settled', {
          totalMs: Math.round(splitPerfNow() - analyzeStartedAt),
          analyzedChildCount: analysisPromises.length,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        for (const { id, region } of createdRegions) {
          applyPolygonParams(id, {
            ...inheritedParams,
            altitudeAGL: region.baseAltitudeAGL ?? inheritedParams.altitudeAGL,
          }, { skipEvent: true, skipQueue: true });
        }

        onPolygonSelected?.(createdIds[0] ?? null);
        splitPerfLog(polygonId, 'applyTerrainPartitionRings completed', {
          totalMs: Math.round(splitPerfNow() - startedAt),
          createdIds,
        });
        return { createdIds, replaced: true };
      } catch (error) {
        for (const childId of createdIds) {
          try {
            pendingProgrammaticDeletesRef.current.add(childId);
            draw?.delete?.(childId);
          } catch {}
        }
        splitPerfLog(polygonId, 'applyTerrainPartitionRings failed', {
          totalMs: Math.round(splitPerfNow() - startedAt),
          error: error instanceof Error ? error.message : String(error),
        });
        onError?.(`Terrain-face split failed: ${error instanceof Error ? error.message : 'Unknown error'}`, polygonId);
        return { createdIds: [], replaced: false };
      } finally {
        suspendAutoAnalysisRef.current = false;
        suppressFlightLineEventsRef.current = prevSuppress;
        if (!prevSuppress) {
          onFlightLinesUpdated?.('__all__');
        }
      }
    }, [addRingAsDrawFeature, analyzePolygon, applyPolygonParams, deletePolygonFeature, fitMapToRings, onError, onFlightLinesUpdated, onPolygonSelected]);

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

	    React.useImperativeHandle(ref, () => ({
      clearAllDrawings: () => {
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
        setPendingParamPolygons([]);
        pendingProgrammaticDeletesRef.current.clear();
        suppressSelectionDialogUntilRef.current = 0;

        cancelAllAnalyses();
        onClearGSD?.();
        onPolygonSelected?.(null);
        // Notify parent that results are cleared
        onAnalysisComplete?.([]);
      },
      clearPolygon: (polygonId: string) => {
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
      startPolygonDrawing: () => {
        if (drawRef.current) (drawRef.current as any).changeMode('draw_polygon');
      },
      getPolygonResults: () => Array.from(polygonResultsRef.current.values()),
      getMap: () => mapRef.current,
      refreshTerrainForAllPolygons,
      setTerrainDemSource: (tileUrlTemplate: string | null) => {
        terrainDemSourceTemplateRef.current = tileUrlTemplate;
        if (mapRef.current && mapRef.current.isStyleLoaded()) {
          setTerrainDemSourceOnMap(mapRef.current, tileUrlTemplate);
        }
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
      applyPolygonParams,
      applyPolygonParamsBatch,
      // expose bulk apply helper
      applyParamsToAllPending,
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
      revertPolygonToImportedDirection,
      runFullAnalysis,

	      getBearingOverrides: () => Object.fromEntries(bearingOverridesRef.current),
	      getImportedOriginals: () => Object.fromEntries(importedOriginalsRef.current),
	      getLastImportedFlightplanName: () => lastImportedFlightplanNameRef.current,
	      exportWingtraFlightPlan: () => {
        // Build area list from current state
        const polys: Array<{ ring:[number,number][]; params: PolygonParams; bearingDeg:number; lineSpacingM?:number; triggerDistanceM?:number }> = [];
        polygonParams.forEach((params, pid) => {
          const res = polygonResults.get(pid);
          const collection = drawRef.current?.getAll();
          const feature = collection?.features.find(f=>f.id===pid && f.geometry?.type==='Polygon');
          const ring = res?.polygon.coordinates || (feature?.geometry as any)?.coordinates?.[0];
          if (!ring) return;
          const override = bearingOverrides.get(pid);
          const bearingDeg = override ? override.bearingDeg : (res?.result.contourDirDeg ?? 0);
          const lineSpacingM = override?.lineSpacingM || (polygonFlightLines.get(pid)?.lineSpacing);
          polys.push({ ring: ring as any, params, bearingDeg, lineSpacingM, triggerDistanceM: params.triggerDistanceM });
        });
        const areas = areasFromState(polys);
        const payloadKind = areas[0]?.payloadKind ?? 'camera';
        let fp;
        if (lastImportedFlightplan) {
          // Deep clone original
            fp = JSON.parse(JSON.stringify(lastImportedFlightplan));
          // Replace flightPlan.items only (preserve metadata/stats; some tools may recalc them)
          fp.flightPlan.items = exportToWingtraFlightPlan(areas, { payloadKind }).flightPlan.items;
          // Optionally update payload fields if camera changed (skipped for now)
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
          fp = exportToWingtraFlightPlan(areas, { payloadKind });
        }
        const json = JSON.stringify(fp, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        return { json, blob };
      },
    }), [
      polygonResults, polygonFlightLines, polygonTiles, polygonParams,
      cancelAllAnalyses, applyPolygonParams, applyPolygonParamsBatch, cleanupPolygonState, deletePolygonFeature, editPolygonBoundary, setProcessingPolygonIds, autoSplitPolygonByTerrain,
      getTerrainPartitionSolutions, refineTerrainPartitionPreview, applyTerrainPartitionSolution, applyTerrainPartitionPreview,
      bearingOverrides, importedOriginals,
      importKmlFromText, importWingtraFromText,
      optimizePolygonDirection, revertPolygonToImportedDirection, runFullAnalysis, refreshTerrainForAllPolygons,
      lastImportedFlightplan
    ]);

    React.useEffect(() => () => {
      stopProcessingPerimeterAnimation();
    }, [stopProcessingPerimeterAnimation]);

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
          accept=".flightplan,.json,application/json"
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
