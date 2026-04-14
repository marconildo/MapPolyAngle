import React, { useCallback, useMemo, useRef, useState } from "react";
import type mapboxgl from "mapbox-gl";
import { addOrUpdateTileOverlay, clearAllOverlays, clearRunOverlays } from "@/overlap/overlay";
import type { CameraModel, PoseMeters, PolygonLngLatWithId, GSDStats, PolygonTileStats, LidarStripMeters, OverlayTileResult, TileResult } from "@/overlap/types";
import { lngLatToMeters, tileMetersBounds } from "@/overlap/mercator";
import { metersToLngLat } from "@/services/Projection";
import { SONY_RX1R2, SONY_RX1R3, SONY_A6100_20MM, DJI_ZENMUSE_P1_24MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM, forwardSpacingRotated } from "@/domain/camera";
import { DEFAULT_LIDAR_MAX_RANGE_M, getLidarMappingFovDeg, getLidarModel, lidarDeliverableDensity, lidarSinglePassDensity, lidarSwathWidth } from "@/domain/lidar";
import { isPointInRing, sampleCameraPositionsOnPlannedFlightGeometry, build3DFlightPath, groupFlightLinesForTraversal, queryMinMaxElevationAlongPolylineWGS84 } from "@/components/MapFlightDirection/utils/geometry";
import { generatePlannedFlightGeometryForPolygon } from "@/flight/plannedGeometry";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { toast } from "@/hooks/use-toast";
import type { BearingOverride, MapFlightDirectionAPI, PolygonHistoryState, PolygonMergeState, TerrainPartitionSolutionPreview } from "@/components/MapFlightDirection/api";
import type { FlightParams, LidarReturnMode, TerrainTile } from "@/domain/types";
import { extractPoses, wgs84ToWebMercator, extractCameraModel } from "@/utils/djiGeotags";
import type { PolygonAnalysisResult } from "@/components/MapFlightDirection/types";
import type { ExactPartitionPreview as SharedExactPartitionPreview } from "@/overlap/exact-region";
import { createCoveragePanelResetState } from "@/state/clearAllState";
import { shouldRunAsyncGeneration } from "@/state/asyncUpdateGuard";
// Turf types may be unresolved if TS can't find bundled types; cast as any.
// @ts-ignore
import * as turf from '@turf/turf';

type Props = {
  mapRef: React.RefObject<MapFlightDirectionAPI>;
  mapboxToken: string;
  clearAllEpoch?: number;
  /** Provide per‑polygon params (altitude/front/side) so we can compute per‑polygon photoSpacing. */
  getPerPolygonParams?: () => Record<string, FlightParams>;
  onEditPolygonParams?: (polygonId: string) => void;
  onAutoRun?: (autoRunFn: (opts?: { polygonId?: string; reason?: 'lines'|'spacing'|'alt'|'manual' }) => void) => void;
  onClearExposed?: (clearFn: () => void) => void;
  // NEW: expose a method so parent (header) can trigger DJI / Wingtra pose JSON import
  onExposePoseImporter?: (openImporter: (mode?: 'dji' | 'wingtra') => void) => void;
  // NEW: report pose import count to parent so parent can enable panel when only poses exist
  onPosesImported?: (count: number) => void;
  polygonAnalyses: PolygonAnalysisResult[];
  overrides: Record<string, BearingOverride>;
  importedOriginals: Record<string, { bearingDeg: number; lineSpacingM: number }>;
  mergeState: PolygonMergeState;
  historyState: PolygonHistoryState;
  selectedPolygonId?: string | null;
  onSelectPolygon?: (id: string | null) => void;
};

type MetricKind = 'gsd' | 'density';

type PolygonMetricSummary = {
  polygonId: string;
  metricKind: MetricKind;
  stats: GSDStats;
  areaAcres: number;
  sampleCount: number;
  sampleLabel: string;
  sourceLabel: string;
};

type OverallMetricStats = {
  gsd: GSDStats | null;
  density: GSDStats | null;
};

type OverlayMetricRange = {
  min: number;
  max: number;
};

type OverlayScaleRanges = Partial<Record<MetricKind, OverlayMetricRange>>;

type ExactPartitionPreview = SharedExactPartitionPreview;
type CoverageAutoRunModule = typeof import("@/overlap/coverageAutoRun");
type ExactRegionModule = typeof import("@/overlap/exact-region");
type ExactBrowserRuntimeModule = typeof import("@/overlap/exactBrowserRuntime");
type MetricAggregationModule = typeof import("@/overlap/metricAggregation");
type OverlapControllerModule = typeof import("@/overlap/controller");

let coverageAutoRunModulePromise: Promise<CoverageAutoRunModule> | null = null;
let exactRegionModulePromise: Promise<ExactRegionModule> | null = null;
let exactBrowserRuntimeModulePromise: Promise<ExactBrowserRuntimeModule> | null = null;
let metricAggregationModulePromise: Promise<MetricAggregationModule> | null = null;
let overlapControllerModulePromise: Promise<OverlapControllerModule> | null = null;

function loadCoverageAutoRunModule(): Promise<CoverageAutoRunModule> {
  if (!coverageAutoRunModulePromise) {
    coverageAutoRunModulePromise = import("@/overlap/coverageAutoRun");
  }
  return coverageAutoRunModulePromise;
}

function loadExactRegionModule(): Promise<ExactRegionModule> {
  if (!exactRegionModulePromise) {
    exactRegionModulePromise = import("@/overlap/exact-region");
  }
  return exactRegionModulePromise;
}

function loadExactBrowserRuntimeModule(): Promise<ExactBrowserRuntimeModule> {
  if (!exactBrowserRuntimeModulePromise) {
    exactBrowserRuntimeModulePromise = import("@/overlap/exactBrowserRuntime");
  }
  return exactBrowserRuntimeModulePromise;
}

function loadMetricAggregationModule(): Promise<MetricAggregationModule> {
  if (!metricAggregationModulePromise) {
    metricAggregationModulePromise = import("@/overlap/metricAggregation");
  }
  return metricAggregationModulePromise;
}

function loadOverlapControllerModule(): Promise<OverlapControllerModule> {
  if (!overlapControllerModulePromise) {
    overlapControllerModulePromise = import("@/overlap/controller");
  }
  return overlapControllerModulePromise;
}

const TERRAIN_SPLIT_DEBUG = true;
const HEATMAP_GRADIENT_GSD = "linear-gradient(90deg, rgb(0 0 255) 0%, rgb(0 255 255) 25%, rgb(0 255 0) 50%, rgb(255 255 0) 75%, rgb(255 0 0) 100%)";
const HEATMAP_GRADIENT_DENSITY = "linear-gradient(90deg, rgb(255 0 0) 0%, rgb(255 255 0) 25%, rgb(0 255 0) 50%, rgb(0 255 255) 75%, rgb(0 0 255) 100%)";
const OVERLAY_SCALE_LOWER_QUANTILE = 0.05;
const OVERLAY_SCALE_UPPER_QUANTILE = 0.95;
const CARD_SUMMARY_LOWER_QUANTILE = 0.05;
const CARD_SUMMARY_UPPER_QUANTILE = 0.95;
const DENSITY_OVERLAY_MAX = 100;

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
    if (index === 0 && bins[index].bin === 0) {
      if (threshold > 0) areaBelow += areaM2;
      continue;
    }
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
      if (index === 0 && bins[index].bin === 0) return 0;
      const previous = cumulative - areaM2;
      const fraction = areaM2 > 0 ? Math.max(0, Math.min(1, (target - previous) / areaM2)) : 0;
      const lower = edges[index];
      const upper = edges[index + 1];
      return lower + fraction * (upper - lower);
    }
  }
  return edges[edges.length - 1] ?? bins[bins.length - 1]?.bin ?? 0;
}

function lidarComparisonLabel(mode?: 'first-return' | 'all-returns') {
  return mode === 'all-returns' ? 'All returns' : 'First return';
}

// Helper function to calculate polygon area in acres
function calculatePolygonAreaAcres(ring: [number, number][]): number {
  if (ring.length < 3) return 0;

  // Use spherical excess formula for accurate area calculation
  // This is more accurate than the planar shoelace approximation, especially at scale
  const R = 6371008.8; // mean Earth radius in meters
  let sum = 0;

  for (let i = 0; i < ring.length; i++) {
    const [λ1, φ1] = ring[i];
    const [λ2, φ2] = ring[(i + 1) % ring.length];
    const lon1 = λ1 * Math.PI / 180;
    const lon2 = λ2 * Math.PI / 180;
    const lat1 = φ1 * Math.PI / 180;
    const lat2 = φ2 * Math.PI / 180;
    sum += (lon2 - lon1) * (2 + Math.sin(lat1) + Math.sin(lat2));
  }

  const areaSquareMeters = Math.abs(sum) * R * R / 2;

  // Convert to acres (1 acre = 4046.8564224 square meters)
  return areaSquareMeters / 4046.8564224;
}

function ringsRoughlyEqual(a: [number, number][], b: [number, number][], toleranceDeg = 1e-7) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i][0] - b[i][0]) > toleranceDeg || Math.abs(a[i][1] - b[i][1]) > toleranceDeg) {
      return false;
    }
  }
  return true;
}
function lidarStripMayAffectTile(
  strip: LidarStripMeters,
  tileRef: { z: number; x: number; y: number }
) {
  const bounds = tileMetersBounds(tileRef.z, tileRef.x, tileRef.y);
  const reachPadM = Math.max(
    strip.halfWidthM ?? 0,
    typeof strip.maxRangeM === "number" && Number.isFinite(strip.maxRangeM) ? strip.maxRangeM : 0
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
export function OverlapGSDPanel({ mapRef, mapboxToken, clearAllEpoch = 0, getPerPolygonParams, onEditPolygonParams, onAutoRun, onClearExposed, onExposePoseImporter, onPosesImported, polygonAnalyses, overrides, importedOriginals: _importedOriginals, mergeState, historyState, selectedPolygonId: controlledSelectedId, onSelectPolygon }: Props) {
  const CAMERA_REGISTRY: Record<string, CameraModel> = useMemo(()=>({
    SONY_RX1R2,
    SONY_RX1R3,
    SONY_A6100_20MM,
    DJI_ZENMUSE_P1_24MM,
    ILX_LR1_INSPECT_85MM,
    MAP61_17MM,
    RGB61_24MM,
  }),[]);

  // Global camera override JSON (optional). If blank, we'll use per‑polygon camera selections.
  const [cameraText, setCameraText] = useState(JSON.stringify(SONY_RX1R2, null, 2));
  const [useOverrideCamera, setUseOverrideCamera] = useState(false);
  const [altitude] = useState(100); // AGL in meters
  const [frontOverlap] = useState(80); // percentage
  const [sideOverlap] = useState(70); // percentage
  const [zoom] = useState(14);
  const [opacity] = useState(0.85);
  const [showOverlap] = useState(false); // Changed default to false
  const [showGsd, setShowGsd] = useState(true);
  const [showFlightLines, setShowFlightLines] = useState(true);
  const [running, setRunning] = useState(false);
  const [autoGenerate, setAutoGenerate] = useState(true);
  const [showFlightParameters, setShowFlightParameters] = useState(false);
  const [showCameraPoints, setShowCameraPoints] = useState(false); // Changed default to false
  const [overallStats, setOverallStats] = useState<OverallMetricStats>({ gsd: null, density: null });
  const [overlayScaleLocked, setOverlayScaleLocked] = useState(true);
  const [lockedOverlayRanges, setLockedOverlayRanges] = useState<OverlayScaleRanges>({});
  const [perPolygonStats, setPerPolygonStats] = useState<Map<string, PolygonMetricSummary>>(new Map());
  const [internalSelectedId, setInternalSelectedId] = useState<string | null>(null);
  const [splittingPolygonIds, setSplittingPolygonIds] = useState<Record<string, true>>({});
  const [partitionOptionsByPolygon, setPartitionOptionsByPolygon] = useState<Record<string, TerrainPartitionSolutionPreview[]>>({});
  const [partitionSelectionByPolygon, setPartitionSelectionByPolygon] = useState<Record<string, number>>({});
  const [, setLoadingPartitionOptionsIds] = useState<Record<string, true>>({});
  const [, setApplyingPartitionIds] = useState<Record<string, true>>({});
  const [, setExactPartitionPreviewByKey] = useState<Record<string, ExactPartitionPreview>>({});
  const isControlled = controlledSelectedId !== undefined;
  const activeSelectedId = isControlled ? (controlledSelectedId ?? null) : internalSelectedId;
  const itemRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const splitPerfSeqRef = useRef(0);

  const setSelection = useCallback((id: string | null) => {
    if (onSelectPolygon) {
      onSelectPolygon(id);
    } else {
      setInternalSelectedId(id);
    }
  }, [onSelectPolygon]);

  // NEW: poses-only mode state
  const poseFileRef = useRef<HTMLInputElement>(null);
  const [poseImportKind, setPoseImportKind] = useState<'auto' | 'dji' | 'wingtra'>('auto');
  const [importedPoses, setImportedPoses] = useState<PoseMeters[]>([]);
  const lastAutoComputedImportedPosesRef = useRef<PoseMeters[] | null>(null);
  const redrawAnalysisOverlaysRef = useRef<(() => void) | null>(null);
  const poseAreaRingRef = useRef<[number,number][]>([]);
  // Remember previous polygon rings so we can re-render tiles a moved polygon USED to cover
  const prevPolygonRingsRef = useRef<Map<string, [number, number][]>>(new Map());

  // Single global runId to avoid stacked overlays - Option B improvement
  const globalRunIdRef = useRef<string | null>(null);
  // Per-polygon, per-tile stats cache for correct cross-polygon crediting - Option B core feature
  const perPolyTileStatsRef = useRef<Map<string, Map<string, PolygonTileStats>>>(new Map());
  const cameraTileResultsRef = useRef<Map<string, OverlayTileResult>>(new Map());
  const lidarTileResultsRef = useRef<Map<string, OverlayTileResult>>(new Map());
  // Cache raw tile data (width, height, and cloned pixel data) to avoid ArrayBuffer transfer issues
  const tileCacheRef = useRef<Map<string, { width: number; height: number; data: Uint8ClampedArray }>>(new Map());
  const autoTriesRef = useRef(0);
  const autoRunTimeoutRef = useRef<number | null>(null);
  const deferredComputeTimeoutRef = useRef<number | null>(null);
  const computeSeqRef = useRef(0); // increment to invalidate in-flight computations
  const runningRef = useRef(false);
  const showGsdRef = useRef(showGsd);
  const showOverlapRef = useRef(showOverlap);
  const overlayScaleLockedRef = useRef(overlayScaleLocked);
  const lockedOverlayRangesRef = useRef<OverlayScaleRanges>(lockedOverlayRanges);
  const pendingComputeRef = useRef<{ polygonId?: string; suppressMapNotReadyToast?: boolean; generation?: number } | null>(null);
  const resetGenerationRef = useRef(0);
  const lastHandledClearAllEpochRef = useRef(clearAllEpoch);
  const guardedTimeoutsRef = useRef<Set<number>>(new Set());
  const suppressAutoRunUntilRef = useRef(0);
  const [clipInnerBufferM] = useState(0);
  const [maxTiltDeg, setMaxTiltDeg] = useState(30); // NEW: max allowable camera tilt (deg from vertical)
  const [minOverlapForGsd, setMinOverlapForGsd] = useState(3); // Minimum image overlap to consider GSD valid
  const minOverlapForGsdRef = useRef(minOverlapForGsd);
  React.useEffect(() => {
    minOverlapForGsdRef.current = minOverlapForGsd;
  }, [minOverlapForGsd]);
  React.useEffect(() => {
    runningRef.current = running;
  }, [running]);

  React.useEffect(() => {
    showGsdRef.current = showGsd;
  }, [showGsd]);

  React.useEffect(() => {
    showOverlapRef.current = showOverlap;
  }, [showOverlap]);

  React.useEffect(() => {
    overlayScaleLockedRef.current = overlayScaleLocked;
  }, [overlayScaleLocked]);

  React.useEffect(() => {
    lockedOverlayRangesRef.current = lockedOverlayRanges;
  }, [lockedOverlayRanges]);

  React.useEffect(() => {
    mapRef.current?.setFlightLinesVisible?.(showFlightLines);
  }, [mapRef, showFlightLines]);

  const cancelGuardedTimeout = useCallback((timeoutId: number | null) => {
    if (timeoutId === null) return;
    guardedTimeoutsRef.current.delete(timeoutId);
    window.clearTimeout(timeoutId);
  }, []);

  const cancelAllGuardedTimeouts = useCallback(() => {
    for (const timeoutId of guardedTimeoutsRef.current) {
      window.clearTimeout(timeoutId);
    }
    guardedTimeoutsRef.current.clear();
  }, []);

  const scheduleGuardedTimeout = useCallback((task: () => void, delayMs = 0, generation = resetGenerationRef.current) => {
    const timeoutId = window.setTimeout(() => {
      guardedTimeoutsRef.current.delete(timeoutId);
      if (!shouldRunAsyncGeneration(generation, resetGenerationRef.current)) return;
      task();
    }, delayMs);
    guardedTimeoutsRef.current.add(timeoutId);
    return timeoutId;
  }, []);
  // NEW: altitude strategy & min clearance & turn extension (synced with map API if available)
  const [altitudeModeUI, setAltitudeModeUI] = useState<'legacy' | 'min-clearance'>('legacy');
  const [minClearanceUI, setMinClearanceUI] = useState<number>(60);
  const [turnExtendUI, setTurnExtendUI] = useState<number>(96);

  // Sync initial values from map API
  React.useEffect(() => {
    const api = mapRef.current;
    const mode = (api as any)?.getAltitudeMode ? (api as any).getAltitudeMode() : 'legacy';
    const minc = (api as any)?.getMinClearance ? (api as any).getMinClearance() : 60;
    const ext = (api as any)?.getTurnExtend ? (api as any).getTurnExtend() : 96;
    setAltitudeModeUI(mode);
    setMinClearanceUI(minc);
    setTurnExtendUI(ext);
  }, [mapRef]);

  React.useEffect(() => {
    const mapApi = mapRef.current;
    mapApi?.setProcessingPolygonIds?.(Object.keys(splittingPolygonIds));
    return () => {
      mapApi?.setProcessingPolygonIds?.([]);
    };
  }, [mapRef, splittingPolygonIds]);

  const getMergedParamsMap = useCallback(() => {
    const externalParams = getPerPolygonParams?.() ?? {};
    const internalParams = mapRef.current?.getPerPolygonParams?.() ?? {};
    return { ...externalParams, ...internalParams };
  }, [getPerPolygonParams, mapRef]);

  // Helper function to generate user-friendly polygon names
  const getPolygonDisplayName = useCallback((polygonId: string): string => {
    if (polygonId === '__POSES__') return 'Imported Poses Area';
    const api = mapRef.current;
    if (!api?.getPolygonsWithIds) return 'Unknown';
    const polygons = api.getPolygonsWithIds();
    const index = polygons.findIndex((p: any) => (p.id || 'unknown') === polygonId);
    return index >= 0 ? `Area ${index + 1}` : 'Unknown';
  }, [mapRef]);

  const applyTerrainPartitionOption = useCallback(async (
    polygonId: string,
    overrideIndex?: number,
    overrideSolution?: TerrainPartitionSolutionPreview,
  ) => {
    const api = mapRef.current;
    const solutions = partitionOptionsByPolygon[polygonId] ?? [];
    const selectedIndex = overrideIndex ?? partitionSelectionByPolygon[polygonId] ?? 0;
    const selected = overrideSolution ?? solutions[selectedIndex];
    if ((!api?.applyTerrainPartitionSolution && !api?.applyTerrainPartitionPreview) || !selected) {
      return { replaced: false, createdIds: [] as string[] };
    }
    const startedAt = splitPerfNow();
    splitPerfLog(polygonId, 'applyTerrainPartitionOption start', {
      signature: selected.signature,
      regionCount: selected.regionCount,
    });
    setApplyingPartitionIds((prev) => ({ ...prev, [polygonId]: true }));
    suppressAutoRunUntilRef.current = Date.now() + 5000;
    try {
      const result = api.applyTerrainPartitionPreview
        ? await api.applyTerrainPartitionPreview(polygonId, selected)
        : await api.applyTerrainPartitionSolution(polygonId, selected.signature);
      splitPerfLog(polygonId, 'applyTerrainPartitionOption finished', {
        totalMs: Math.round(splitPerfNow() - startedAt),
        result,
      });
      if (result?.replaced) {
        resetComputedAnalysisState();
        setSelection(result.createdIds[0] ?? null);
        rerunAnalysisForCreatedPolygons(result.createdIds);
        setPartitionOptionsByPolygon((prev) => {
          const next = { ...prev };
          delete next[polygonId];
          return next;
        });
        setPartitionSelectionByPolygon((prev) => {
          const next = { ...prev };
          delete next[polygonId];
          return next;
        });
        setExactPartitionPreviewByKey((prev) => {
          const next = { ...prev };
          Object.keys(next).forEach((key) => {
            if (key.startsWith(`${polygonId}:`)) delete next[key];
          });
          return next;
        });
        return result;
      } else {
        toast({
          title: 'Partition not applied',
          description: 'The selected partition could not be applied to this area.',
          variant: 'destructive',
        });
        suppressAutoRunUntilRef.current = 0;
        return { replaced: false, createdIds: [] as string[] };
      }
    } catch (error) {
      splitPerfLog(polygonId, 'applyTerrainPartitionOption threw', {
        totalMs: Math.round(splitPerfNow() - startedAt),
        error: error instanceof Error ? error.message : String(error),
      });
      suppressAutoRunUntilRef.current = 0;
      toast({
        title: 'Partition apply failed',
        description: error instanceof Error ? error.message : 'Unable to apply the selected partition.',
        variant: 'destructive',
      });
      return { replaced: false, createdIds: [] as string[] };
    } finally {
      setApplyingPartitionIds((prev) => {
        if (!prev[polygonId]) return prev;
        const next = { ...prev };
        delete next[polygonId];
        return next;
      });
    }
  }, [mapRef, partitionOptionsByPolygon, partitionSelectionByPolygon]);

  // Helper function to highlight polygon on map
  const highlightPolygon = useCallback((polygonId: string) => {
    const api = mapRef.current;
    const map = api?.getMap?.();
    if (!map) return;

    if (polygonId === '__POSES__' && poseAreaRingRef.current?.length >= 4) {
      const ring = poseAreaRingRef.current;
      const lngs = ring.map(c=>c[0]);
      const lats = ring.map(c=>c[1]);
      map.fitBounds([[Math.min(...lngs), Math.min(...lats)],[Math.max(...lngs), Math.max(...lats)]], { padding:50, duration:1000, maxZoom:16 });
      return;
    }
    if (!api?.getPolygonsWithIds) return;
    const polygons = api.getPolygonsWithIds();
    const targetPolygon = polygons.find((p: any) => (p.id || 'unknown') === polygonId);
    if (targetPolygon && targetPolygon.ring?.length >= 4) {
      const lngs = targetPolygon.ring.map((coord: [number, number]) => coord[0]);
      const lats = targetPolygon.ring.map((coord: [number, number]) => coord[1]);
      map.fitBounds([[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]], { padding:50, duration:1000, maxZoom:16 });
    }
  }, [mapRef]);

  const matchCameraKeyFromName = useCallback((name?: string | null): string | null => {
    if (!name) return null;
    const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "");
    const stripVersionSuffix = (s: string) => normalize(s).replace(/v\\d+$/g, "");
    const target = normalize(name);
    const targetStem = stripVersionSuffix(name);
    for (const [key, cam] of Object.entries(CAMERA_REGISTRY)) {
      const names = cam.names || [];
      if (names.some(n => {
        const t = normalize(n);
        const ts = stripVersionSuffix(n);
        if (t === target) return true;
        if (target.includes(t) || t.includes(target)) return true;
        if (targetStem && ts && (targetStem === ts || targetStem.includes(ts) || ts.includes(targetStem))) return true;
        return false;
      })) {
        return key;
      }
    }
    return null;
  }, [CAMERA_REGISTRY]);

  const toNumber = (v: any): number | undefined => {
    if (typeof v === 'number') return v;
    if (typeof v === 'string') {
      const n = parseFloat(v);
      return Number.isFinite(n) ? n : undefined;
    }
    return undefined;
  };

  const radMaybeDegToDeg = (v: number | undefined): number | undefined => {
    if (!Number.isFinite(v)) return undefined;
    const abs = Math.abs(v as number);
    // Wingtra yaw/pitch/roll appear to be in radians; convert when value is within a 2π range
    if (abs <= Math.PI * 2 + 1e-3) return (v as number) * 180 / Math.PI;
    return v as number;
  };

  const parseWingtraGeotags = useCallback((payload: any): { poses: PoseMeters[]; cameraKey: string | null; camera: CameraModel | null; sourceLabel: string } | null => {
    if (!payload || !Array.isArray(payload.flights)) return null;
    const flights = payload.flights;
    const pickByName = (name: string) => flights.find((f: any) => String(f?.name || '').toLowerCase() === name);
    const chosen = pickByName('processedforward') || pickByName('raw') || flights[0];
    if (!chosen || !Array.isArray(chosen.geotag)) return null;

    const poses: PoseMeters[] = [];
    chosen.geotag.forEach((g: any, idx: number) => {
      const coord = g.coordinate;
      if (!Array.isArray(coord) || coord.length < 2) return;
      const lat = toNumber(coord[0]);
      const lon = toNumber(coord[1]);
      const alt = toNumber(coord[2]) ?? 0;
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      const { x, y } = wgs84ToWebMercator(lat as number, lon as number);
      const yawDeg = radMaybeDegToDeg(toNumber(g.yaw));
      const pitchDeg = radMaybeDegToDeg(toNumber(g.pitch));
      const rollDeg = radMaybeDegToDeg(toNumber(g.roll));
      poses.push({
        id: (g.sequence ?? idx)?.toString?.() ?? `pose_${idx}`,
        x,
        y,
        z: alt,
        omega_deg: rollDeg ?? 0,
        phi_deg: pitchDeg ?? 0,
        kappa_deg: yawDeg ?? 0,
      } as PoseMeters);
    });

    if (!poses.length) return null;
    const cameraKey = matchCameraKeyFromName(typeof payload.model === 'string' ? payload.model : undefined);
    const camera = cameraKey ? CAMERA_REGISTRY[cameraKey] : null;
    const sourceLabel = String(chosen.name || (pickByName('processedforward') ? 'ProcessedForward' : 'Raw'));
    return { poses, cameraKey, camera, sourceLabel };
  }, [CAMERA_REGISTRY, matchCameraKeyFromName]);

  const applyImportedPoses = useCallback((posesMeters: PoseMeters[], camera: CameraModel | null, cameraKey: string | null, sourceLabel: string) => {
    setImportedPoses(posesMeters);
    if (posesMeters.length) {
      poseAreaRingRef.current = []; // force rebuild in compute via AOI function
    }
    if (camera) {
      setCameraText(JSON.stringify(camera, null, 2));
      setUseOverrideCamera(true);
    }
    setAutoGenerate(false);
    setShowCameraPoints(true);
    const cameraMsg = camera ? (cameraKey ? ` using ${cameraKey} camera.` : ' with camera intrinsics.') : '.';
    toast({ title: "Imported poses", description: `${posesMeters.length} camera poses loaded (${sourceLabel})${cameraMsg}` });
    onPosesImported?.(posesMeters.length);
    scheduleGuardedTimeout(() => {
      if (poseAreaRingRef.current.length < 4) return;
      const api = mapRef.current;
      const map = api?.getMap?.();
      if (!map) return;
      const ring = poseAreaRingRef.current;
      const lngs = ring.map((c) => c[0]);
      const lats = ring.map((c) => c[1]);
      map.fitBounds(
        [[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]],
        { padding: 50, duration: 800, maxZoom: 16 },
      );
      map.once('idle', () => {
        redrawAnalysisOverlaysRef.current?.();
      });
    }, 30);
  }, [mapRef, onPosesImported, scheduleGuardedTimeout]);

  const parseCameraOverride = useCallback((): CameraModel | null => {
    if (!useOverrideCamera) return null;
    try { const obj = JSON.parse(cameraText); return obj as CameraModel; } catch { return null; }
  }, [cameraText, useOverrideCamera]);

  const effectiveCameraForPolygon = useCallback((polygonId: string, paramsMap: any): CameraModel => {
    const p = paramsMap[polygonId];
    if (p?.cameraKey && CAMERA_REGISTRY[p.cameraKey]) return CAMERA_REGISTRY[p.cameraKey];
    const override = parseCameraOverride();
    if (override) return override;
    return SONY_RX1R2; // fallback
  }, [parseCameraOverride, CAMERA_REGISTRY]);

  const isLidarPayload = useCallback((polygonId: string, paramsMap: any): boolean => {
    if (polygonId === '__POSES__') return false;
    return (paramsMap?.[polygonId]?.payloadKind ?? 'camera') === 'lidar';
  }, []);

  // Helper: per‑polygon spacing using that polygon's selected camera
  const photoSpacingFor = useCallback((polygonId: string, altitudeAGL: number, frontOverlap: number, paramsMap: any): number => {
    const p = paramsMap?.[polygonId];
    const explicit = p?.triggerDistanceM;
    if (Number.isFinite(explicit) && (explicit as number) > 0) {
      return explicit as number;
    }
    const cam = effectiveCameraForPolygon(polygonId, paramsMap);
    const yawOffset = p?.cameraYawOffsetDeg ?? 0;
    const rotate90 = Math.round((((yawOffset % 180) + 180) % 180)) === 90;
    return forwardSpacingRotated(cam, altitudeAGL, frontOverlap, rotate90);
  }, [effectiveCameraForPolygon]);

  const lineSpacingFor = useCallback((polygonId: string, altitudeAGL: number, sideOverlap: number, paramsMap: any): number => {
    const p = paramsMap?.[polygonId];
    if ((p?.payloadKind ?? 'camera') === 'lidar') {
      const model = getLidarModel(p?.lidarKey);
      const mappingFovDeg = getLidarMappingFovDeg(model, p?.mappingFovDeg);
      return lidarSwathWidth(altitudeAGL, mappingFovDeg) * (1 - sideOverlap / 100);
    }
    const cam = effectiveCameraForPolygon(polygonId, paramsMap);
    const yawOffset = p?.cameraYawOffsetDeg ?? 0;
    const rotate90 = Math.round((((yawOffset % 180) + 180) % 180)) === 90;
    const groundWidth = rotate90
      ? (cam.h_px * cam.sy_m * altitudeAGL) / cam.f_m
      : (cam.w_px * cam.sx_m * altitudeAGL) / cam.f_m;
    return groundWidth * (1 - sideOverlap / 100);
  }, [effectiveCameraForPolygon]);

  // Accurate aggregation with a higher-fidelity histogram for scoring.
  const tailAreaAcres = 1; // trim per side (acres)

  const overlayRangeForStats = useCallback((stats: GSDStats | null, metricKind: MetricKind) => {
    if (!stats || !(stats.count > 0) || !Number.isFinite(stats.min) || !Number.isFinite(stats.max)) return null;
    let min = histogramQuantile(stats, OVERLAY_SCALE_LOWER_QUANTILE);
    let max = Math.max(min + 1e-6, histogramQuantile(stats, OVERLAY_SCALE_UPPER_QUANTILE));

    if (metricKind === 'density') {
      max = Math.min(DENSITY_OVERLAY_MAX, max);
      if (!(min < max)) {
        min = Math.max(0, Math.min(stats.min, max - 1e-6));
      }
      if (!(min < max)) {
        min = 0;
      }
    }

    return { min, max: Math.max(min + 1e-6, max) };
  }, []);

  const computeOverlayRanges = useCallback((statsSet: OverallMetricStats): OverlayScaleRanges => {
    const ranges: OverlayScaleRanges = {};
    const gsdRange = overlayRangeForStats(statsSet.gsd, 'gsd');
    const densityRange = overlayRangeForStats(statsSet.density, 'density');
    if (gsdRange) ranges.gsd = gsdRange;
    if (densityRange) ranges.density = densityRange;
    return ranges;
  }, [overlayRangeForStats]);

  const mergeLockedOverlayRanges = useCallback((currentRanges: OverlayScaleRanges, nextStats: OverallMetricStats) => {
    const computedRanges = computeOverlayRanges(nextStats);
    const nextRanges: OverlayScaleRanges = { ...currentRanges };
    let changed = false;
    (['gsd', 'density'] as const).forEach((metricKind) => {
      const currentRange = nextRanges[metricKind];
      const computedRange = computedRanges[metricKind];
      if (!currentRange && computedRange) {
        nextRanges[metricKind] = computedRange;
        changed = true;
      }
    });
    return {
      changed,
      ranges: changed ? nextRanges : currentRanges,
    };
  }, [computeOverlayRanges]);

  const resolveOverlayRanges = useCallback((statsSet: OverallMetricStats, options?: {
    lockEnabled?: boolean;
    lockedRanges?: OverlayScaleRanges;
  }) => {
    const autoRanges = computeOverlayRanges(statsSet);
    const lockEnabled = options?.lockEnabled ?? overlayScaleLockedRef.current;
    const lockedRanges = options?.lockedRanges ?? lockedOverlayRangesRef.current;
    return {
      gsd: lockEnabled ? (lockedRanges.gsd ?? autoRanges.gsd ?? null) : (autoRanges.gsd ?? null),
      density: lockEnabled ? (lockedRanges.density ?? autoRanges.density ?? null) : (autoRanges.density ?? null),
    };
  }, [computeOverlayRanges]);

  const setOverlaySelectionEmphasis = useCallback((
    map: mapboxgl.Map,
    runId: string,
    selectedPolygonId: string | null,
    resetUnmatchedLayers: boolean,
  ) => {
    const selectedTileKeys = selectedPolygonId
      ? new Set(perPolyTileStatsRef.current.get(selectedPolygonId)?.keys() ?? [])
      : null;
    const shouldEmphasizeSelection = !!selectedPolygonId && !!selectedTileKeys && selectedTileKeys.size > 0;
    const layers = map.getStyle?.().layers ?? [];
    for (const layer of layers) {
      const layerId = String(layer?.id ?? '');
      if (!layerId.startsWith(`ogsd-${runId}-`)) continue;
      const match = layerId.match(/^ogsd-[^-]+-(?:overlap|pass|gsd|density)-(\d+)-(\d+)-(\d+)$/);
      if (!match) {
        if (!resetUnmatchedLayers) continue;
        try {
          map.setPaintProperty(layerId, 'raster-opacity', opacity);
        } catch {}
        continue;
      }

      const [, z, x, y] = match;
      const cacheKey = `${z}/${x}/${y}`;
      const isSelectedTile = shouldEmphasizeSelection ? selectedTileKeys!.has(cacheKey) : true;
      const rasterOpacity = shouldEmphasizeSelection
        ? (isSelectedTile ? opacity : Math.min(0.2, opacity * 0.24))
        : opacity;
      try {
        map.setPaintProperty(layerId, 'raster-opacity', rasterOpacity);
      } catch {}
    }
  }, [opacity]);

  const redrawAnalysisOverlays = useCallback((statsOverride?: OverallMetricStats, rangesOverride?: OverlayScaleRanges) => {
    const map = mapRef.current?.getMap?.();
    const runId = globalRunIdRef.current;
    if (!map || !map.isStyleLoaded?.() || !runId) return;

    clearRunOverlays(map, runId);

    const nextStats = statsOverride ?? overallStats;
    const { gsd: gsdRange, density: densityRange } = resolveOverlayRanges(nextStats, {
      lockedRanges: rangesOverride,
    });

    const showOverlapNow = showOverlapRef.current;
    const showGsdNow = showGsdRef.current;

    for (const result of cameraTileResultsRef.current.values()) {
      if (showOverlapNow) addOrUpdateTileOverlay(map, result, { kind: "overlap", runId, opacity });
      if (showGsdNow) {
        addOrUpdateTileOverlay(map, result, {
          kind: "gsd",
          runId,
          opacity,
          gsdMin: gsdRange?.min,
          gsdMax: gsdRange?.max,
        });
      }
    }

    for (const result of lidarTileResultsRef.current.values()) {
      if (showOverlapNow) addOrUpdateTileOverlay(map, result, { kind: "pass", runId, opacity });
      if (showGsdNow) {
        addOrUpdateTileOverlay(map, result, {
          kind: "density",
          runId,
          opacity,
          densityMin: densityRange?.min,
          densityMax: densityRange?.max,
        });
      }
    }
    setOverlaySelectionEmphasis(map, runId, activeSelectedId, false);
  }, [activeSelectedId, mapRef, overallStats, resolveOverlayRanges, setOverlaySelectionEmphasis]);

  const applyOverlaySelectionEmphasis = useCallback((selectedPolygonId: string | null) => {
    const map = mapRef.current?.getMap?.();
    const runId = globalRunIdRef.current;
    if (!map || !runId || !map.isStyleLoaded?.()) return;
    setOverlaySelectionEmphasis(map, runId, selectedPolygonId, true);
  }, [mapRef, setOverlaySelectionEmphasis]);

  React.useEffect(() => {
    redrawAnalysisOverlaysRef.current = () => {
      redrawAnalysisOverlays();
    };
  }, [redrawAnalysisOverlays]);

  React.useEffect(() => {
    applyOverlaySelectionEmphasis(activeSelectedId);
  }, [activeSelectedId, applyOverlaySelectionEmphasis, perPolygonStats]);

  const handleShowAnalysisOverlayChange = useCallback((checked: boolean) => {
    showGsdRef.current = checked;
    setShowGsd(checked);
    redrawAnalysisOverlays();
  }, [redrawAnalysisOverlays]);

  const toOverlayTileResult = useCallback((result: TileResult): OverlayTileResult => ({
    z: result.z,
    x: result.x,
    y: result.y,
    size: result.size,
    maxOverlap: result.maxOverlap,
    overlap: result.overlap,
    gsdMin: result.gsdMin,
    density: result.density,
  }), []);

  // Re-bin the stored histogram for display so the charts stay readable while
  // the underlying stats remain detailed enough for scoring.
  const convertHistogramToArea = useCallback((stats: GSDStats, metricKind: MetricKind): { bin: number; areaM2: number; isZeroBucket?: boolean }[] => {
    if (!stats || !stats.histogram.length) return [];
    const bins = sortedHistogramBins(stats);
    const hasExactZeroBucket = metricKind === 'density' && bins[0]?.bin === 0 && (bins[0]?.areaM2 || 0) > 0;
    const zeroBucket = hasExactZeroBucket ? { bin: 0, areaM2: bins[0].areaM2 || 0, isZeroBucket: true } : null;
    const positiveBins = hasExactZeroBucket ? bins.slice(1) : bins;

    if (positiveBins.length === 0) return zeroBucket ? [zeroBucket] : [];

    const maxDisplayBins = hasExactZeroBucket ? 7 : 8;
    if (positiveBins.length <= maxDisplayBins) {
      const direct = positiveBins.map((bin) => ({ bin: bin.bin, areaM2: bin.areaM2 || 0 }));
      return zeroBucket ? [zeroBucket, ...direct] : direct;
    }

    const min = positiveBins[0].bin;
    const max = positiveBins[positiveBins.length - 1].bin;
    const span = Math.max(1e-6, max - min);
    const displayBins = maxDisplayBins;
    const binSize = span / displayBins;
    const compact = new Array<{ bin: number; areaM2: number }>(displayBins);
    for (let index = 0; index < displayBins; index++) {
      compact[index] = { bin: min + (index + 0.5) * binSize, areaM2: 0 };
    }
    for (const bin of positiveBins) {
      let index = Math.floor((bin.bin - min) / binSize);
      if (index < 0) index = 0;
      if (index >= displayBins) index = displayBins - 1;
      compact[index].areaM2 += bin.areaM2 || 0;
    }
    const compactBins = compact.filter((bin) => bin.areaM2 > 0);
    return zeroBucket ? [zeroBucket, ...compactBins] : compactBins;
  }, []);

  const ACRE_M2 = 4046.8564224;

  // Generate poses from existing flight lines using 3D paths
  const generatePosesFromFlightLines = useCallback((): PoseMeters[] => {
    const api = mapRef.current;
    if (!api?.getFlightLines || !api?.getPolygonTiles) return [];

    const paramsMap = getMergedParamsMap();

    const flightLinesMap = api.getFlightLines();
    const tilesMap = api.getPolygonTiles();
    const poses: PoseMeters[] = [];
    let poseId = 0;

    for (const [polygonId, lineData] of Array.from(flightLinesMap.entries())) {
      const { flightLines, lineSpacing, altitudeAGL } = lineData;
      const tiles = tilesMap.get(polygonId) || [];
      if (flightLines.length === 0 || tiles.length === 0) continue;

      const p = (paramsMap as any)[polygonId];
      if ((p?.payloadKind ?? 'camera') === 'lidar') continue;
      const altForThisPoly = p?.altitudeAGL ?? altitudeAGL ?? 100;
      const front = p?.frontOverlap ?? 80;
      const spacingForward = photoSpacingFor(polygonId, altForThisPoly, front, paramsMap);
      const yawOffset = p?.cameraYawOffsetDeg ?? 0;

      const mode = (api as any)?.getAltitudeMode ? (api as any).getAltitudeMode() : 'legacy';
      const minClr = (api as any)?.getMinClearance ? (api as any).getMinClearance() : 60;
      const path3D = build3DFlightPath(
        lineData,
        tiles,
        lineSpacing,
        { altitudeAGL: altForThisPoly, mode, minClearance: minClr, preconnected: true },
      );

      const polys = api.getPolygonsWithIds?.() || [];
      const ring = (polys.find((pp:any)=> (pp.id||'unknown')===polygonId)?.ring) as [number,number][] | undefined;
      const filtered = sampleCameraPositionsOnPlannedFlightGeometry(lineData, path3D, spacingForward)
        .filter(([lng, lat]) => !ring || ring.length < 3 || isPointInRing(lng, lat, ring));

      const normalizeDeg = (d: number) => ((d % 360) + 360) % 360;
      filtered.forEach(([lng, lat, altMSL, yawDeg]) => {
        const [x, y] = lngLatToMeters(lng, lat);
        // Align camera so image height (y-axis) is along flight direction; width is cross-track.
        // yawDeg is bearing CW from North; kappa in our math is CCW about +Z.
        const kappaDeg = normalizeDeg(-yawDeg + yawOffset);
        poses.push({
          id: `photo_${poseId++}`,
          x, y, z: altMSL,
          omega_deg: 0,
          phi_deg: 0,
          kappa_deg: kappaDeg,
          polygonId // tag pose with polygon for per‑camera assignment
        });
      });
    }
    return poses;
  }, [getMergedParamsMap, mapRef, photoSpacingFor]);

  const parsePosesMeters = useCallback((): PoseMeters[] | null => {
    const api = mapRef.current;
    const fl = api?.getFlightLines?.();
    const haveLines = !!fl && Array.from(fl.values()).some((v: any) => v.flightLines && v.flightLines.length > 0);
    const generated = haveLines ? generatePosesFromFlightLines() : [];
    // Always keep imported poses; add generated ones when lines exist
    const base: PoseMeters[] = [ ...(importedPoses || []), ...(generated || []) ];
    if (!base) return [];
    // NEW: filter poses by tilt (sqrt(omega^2 + phi^2) ~ small-angle off-nadir approximation)
    const filtered = maxTiltDeg >= 0 ? base.filter(p => {
      const tilt = Math.sqrt((p.omega_deg||0)*(p.omega_deg||0) + (p.phi_deg||0)*(p.phi_deg||0));
      return tilt <= maxTiltDeg;
    }) : base;
    return filtered;
  }, [generatePosesFromFlightLines, importedPoses, maxTiltDeg, mapRef]);

  const getPolygons = useCallback((): PolygonLngLatWithId[] => {
    const api = mapRef.current;
    if (!api?.getPolygonsWithIds) return [];
    return api.getPolygonsWithIds(); // returns { id?: string; ring: [number, number][] }[]
  }, [mapRef]);

  const buildLidarStrips = useCallback((paramsMap: Record<string, any>, polygonFilter?: Set<string>): {
    strips: LidarStripMeters[];
  } => {
    const api = mapRef.current;
    const flightLinesMap = api?.getFlightLines?.();
    const tilesMap = api?.getPolygonTiles?.();
    if (!flightLinesMap || !tilesMap) return { strips: [] };

    const strips: LidarStripMeters[] = [];
    let globalPassIndex = 0;
    const altitudeMode = (api as any)?.getAltitudeMode ? (api as any).getAltitudeMode() : 'legacy';
    const minClearance = (api as any)?.getMinClearance ? (api as any).getMinClearance() : 60;
    for (const [polygonId, lineData] of Array.from(flightLinesMap.entries())) {
      if (polygonFilter && !polygonFilter.has(polygonId)) continue;
      if (!isLidarPayload(polygonId, paramsMap)) continue;
      const params = paramsMap[polygonId] ?? {};
      const tiles = tilesMap.get(polygonId) || [];
      const model = getLidarModel(params.lidarKey);
      const altitudeAGL = params.altitudeAGL ?? lineData.altitudeAGL ?? altitude;
      const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
      const speedMps = params.speedMps ?? model.defaultSpeedMps;
      const returnMode: LidarReturnMode = params.lidarReturnMode ?? 'single';
      const maxLidarRangeM = params.maxLidarRangeM ?? model.defaultMaxRangeM ?? DEFAULT_LIDAR_MAX_RANGE_M;
      const frameRateHz = params.lidarFrameRateHz ?? model.defaultFrameRateHz;
      const azimuthSectorCenterDeg = params.lidarAzimuthSectorCenterDeg ?? model.defaultAzimuthSectorCenterDeg ?? 0;
      const boresightYawDeg = params.lidarBoresightYawDeg ?? model.boresightYawDeg ?? 0;
      const boresightPitchDeg = params.lidarBoresightPitchDeg ?? model.boresightPitchDeg ?? 0;
      const boresightRollDeg = params.lidarBoresightRollDeg ?? model.boresightRollDeg ?? 0;
      const comparisonMode = params.lidarComparisonMode ?? 'first-return';
      const swathWidth = lidarSwathWidth(altitudeAGL, mappingFovDeg);
      const densityPerPass = lidarSinglePassDensity(model, altitudeAGL, speedMps, returnMode, mappingFovDeg);
      const halfFovTan = Math.tan((mappingFovDeg * Math.PI) / 360);
      const effectivePointRate = model.effectivePointRates[returnMode];
      if (!(swathWidth > 0) || !(densityPerPass > 0)) continue;

      const sweeps = lineData.sweepLines?.length
        ? lineData.sweepLines
        : groupFlightLinesForTraversal(lineData.flightLines ?? [], lineData.lineSpacing, lineData.sweepIndices)
          .map((sweep) => {
            const orderedFragments = sweep.directionForward ? sweep.fragments : [...sweep.fragments].reverse();
            const orientedFragments = orderedFragments.map((fragment) => sweep.directionForward ? fragment : [...fragment].reverse());
            const mergedLine: [number, number][] = [];
            orientedFragments.forEach((fragment, fragmentIndex) => {
              if (fragmentIndex === 0) {
                mergedLine.push(...fragment as [number, number][]);
                return;
              }
              const typedFragment = fragment as [number, number][];
              const previous = mergedLine[mergedLine.length - 1];
              const first = typedFragment[0];
              if (!previous || previous[0] !== first[0] || previous[1] !== first[1]) {
                mergedLine.push(first);
              }
              mergedLine.push(...typedFragment.slice(1));
            });
            return mergedLine;
          });
      for (let lineIndex = 0; lineIndex < sweeps.length; lineIndex++) {
        const passIndex = globalPassIndex++;
        const activeSweepLine = sweeps[lineIndex];
        if (!Array.isArray(activeSweepLine) || activeSweepLine.length < 2) continue;
        const sweepPath3d = build3DFlightPath(
          [activeSweepLine],
          tiles,
          lineData.lineSpacing,
          { altitudeAGL, mode: altitudeMode, minClearance, turnExtendM: 0 }
        )[0];
        if (!Array.isArray(sweepPath3d) || sweepPath3d.length < 2) continue;

        for (let i = 1; i < sweepPath3d.length; i++) {
          const start = sweepPath3d[i - 1];
          const end = sweepPath3d[i];
          if (!Array.isArray(start) || !Array.isArray(end) || start.length < 3 || end.length < 3) continue;
          const [x1, y1] = lngLatToMeters(start[0], start[1]);
          const [x2, y2] = lngLatToMeters(end[0], end[1]);
          const terrainMin = tiles.length > 0
            ? queryMinMaxElevationAlongPolylineWGS84([[start[0], start[1]], [end[0], end[1]]], tiles, 12).min
            : Number.NaN;
          const maxSensorAltitude = Math.max(start[2], end[2]);
          const maxHalfWidth = Number.isFinite(terrainMin)
            ? Math.max(swathWidth / 2, Math.max(1, (maxSensorAltitude - terrainMin) * halfFovTan))
            : swathWidth / 2;
          strips.push({
            id: `${polygonId}-line-${lineIndex}-seg-${i - 1}`,
            polygonId,
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
            effectivePointRate,
            halfFovTan,
            maxRangeM: maxLidarRangeM,
            passIndex,
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

    return { strips };
  }, [altitude, isLidarPayload, mapRef]);

  const evaluatePartitionOptionExact = useCallback(async (
    polygonId: string,
    solution: TerrainPartitionSolutionPreview,
  ): Promise<ExactPartitionPreview> => {
    const api = mapRef.current;
    const paramsMap = getMergedParamsMap();
    const params = paramsMap[polygonId];
    if (!params) throw new Error('Missing flight parameters for this polygon.');

    const parentTiles = (api?.getPolygonTiles?.().get(polygonId) ?? []) as TerrainTile[];
    if (!parentTiles.length) throw new Error('Terrain tiles are not available yet. Run analysis on this polygon first.');
    const [
      { aggregateMetricStats },
      { fetchTerrainRGBA, tilesCoveringPolygon, LidarDensityWorker, OverlapWorker },
    ] = await Promise.all([
      loadMetricAggregationModule(),
      loadOverlapControllerModule(),
    ]);

    const altitudeMode = (api as any)?.getAltitudeMode ? (api as any).getAltitudeMode() : altitudeModeUI;
    const minClearance = (api as any)?.getMinClearance ? (api as any).getMinClearance() : minClearanceUI;
    const refinedSolution = api?.refineTerrainPartitionPreview
      ? await api.refineTerrainPartitionPreview(polygonId, solution)
      : solution;
    const virtualPolygons = refinedSolution.regions.map((region, index) => ({
      id: `${polygonId}::${index}`,
      ring: region.ring,
      bearingDeg: region.bearingDeg,
    }));

    const allTileRefs = (() => {
      const seen = new Set<string>();
      const refs: Array<{ z: number; x: number; y: number }> = [];
      for (const polygon of virtualPolygons) {
        for (const tile of tilesCoveringPolygon({ ring: polygon.ring }, zoom)) {
          const key = `${zoom}/${tile.x}/${tile.y}`;
          if (seen.has(key)) continue;
          seen.add(key);
          refs.push({ z: zoom, x: tile.x, y: tile.y });
        }
      }
      return refs;
    })();

    const getTile = async (tileRef: { z: number; x: number; y: number }) => {
      const cacheKey = `${tileRef.z}/${tileRef.x}/${tileRef.y}`;
      let tileData = tileCacheRef.current.get(cacheKey);
      if (!tileData) {
        const imgData = await fetchTerrainRGBA(tileRef.z, tileRef.x, tileRef.y, mapboxToken);
        tileData = {
          width: imgData.width,
          height: imgData.height,
          data: new Uint8ClampedArray(imgData.data),
        };
        tileCacheRef.current.set(cacheKey, tileData);
      }
      return {
        cacheKey,
        tile: { z: tileRef.z, x: tileRef.x, y: tileRef.y, size: tileData.width, data: new Uint8ClampedArray(tileData.data) },
      };
    };

    const normalizeTileRef = (tileRef: { z: number; x: number; y: number }) => {
      const tilesPerAxis = 1 << tileRef.z;
      const wrappedX = ((tileRef.x % tilesPerAxis) + tilesPerAxis) % tilesPerAxis;
      const clampedY = Math.max(0, Math.min(tilesPerAxis - 1, tileRef.y));
      return { z: tileRef.z, x: wrappedX, y: clampedY };
    };

    const getLidarTileWithHalo = async (tileRef: { z: number; x: number; y: number }, padTiles = 1) => {
      const center = await getTile(tileRef);
      if (padTiles <= 0) {
        return {
          cacheKey: center.cacheKey,
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
        cacheKey: center.cacheKey,
        tile: center.tile,
        demTile: { size: demSize, padTiles, data: demData },
      };
    };

    const pointInRing = (lng: number, lat: number, ring: [number, number][]) => {
      let inside = false;
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i];
        const [xj, yj] = ring[j];
        const intersect = ((yi > lat) !== (yj > lat)) &&
          (lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi);
        if (intersect) inside = !inside;
      }
      return inside;
    };

    if (isLidarPayload(polygonId, paramsMap)) {
      const model = getLidarModel(params.lidarKey);
      const altitudeAGL = params.altitudeAGL ?? altitude;
      const lineSpacing = lineSpacingFor(polygonId, altitudeAGL, params.sideOverlap ?? sideOverlap, paramsMap);
      const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
      const speedMps = params.speedMps ?? model.defaultSpeedMps;
      const returnMode = params.lidarReturnMode ?? 'single';
      const maxLidarRangeM = params.maxLidarRangeM ?? model.defaultMaxRangeM ?? DEFAULT_LIDAR_MAX_RANGE_M;
      const frameRateHz = params.lidarFrameRateHz ?? model.defaultFrameRateHz;
      const azimuthSectorCenterDeg = params.lidarAzimuthSectorCenterDeg ?? model.defaultAzimuthSectorCenterDeg ?? 0;
      const boresightYawDeg = params.lidarBoresightYawDeg ?? model.boresightYawDeg ?? 0;
      const boresightPitchDeg = params.lidarBoresightPitchDeg ?? model.boresightPitchDeg ?? 0;
      const boresightRollDeg = params.lidarBoresightRollDeg ?? model.boresightRollDeg ?? 0;
      const comparisonMode = params.lidarComparisonMode ?? 'first-return';
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
            parentTiles,
            lineSpacing,
            { altitudeAGL, mode: altitudeMode, minClearance, turnExtendM: 0 },
          )[0];
          if (!Array.isArray(sweepPath3d) || sweepPath3d.length < 2) continue;
          const localPassIndex = passIndex++;
          for (let i = 1; i < sweepPath3d.length; i++) {
            const start = sweepPath3d[i - 1];
            const end = sweepPath3d[i];
            if (!Array.isArray(start) || !Array.isArray(end) || start.length < 3 || end.length < 3) continue;
            const [x1, y1] = lngLatToMeters(start[0], start[1]);
            const [x2, y2] = lngLatToMeters(end[0], end[1]);
            const terrainMin = queryMinMaxElevationAlongPolylineWGS84([[start[0], start[1]], [end[0], end[1]]], parentTiles, 12).min;
            const maxSensorAltitude = Math.max(start[2], end[2]);
            const maxHalfWidth = Number.isFinite(terrainMin)
              ? Math.max(lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2, Math.max(1, (maxSensorAltitude - terrainMin) * halfFovTan))
              : lidarSwathWidth(altitudeAGL, mappingFovDeg) / 2;
            strips.push({
              id: `${region.id}-line-${lineIndex}-seg-${i - 1}`,
              polygonId: region.id,
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
      }

      const worker = new LidarDensityWorker();
      try {
        const perRegionStats = new Map<string, GSDStats[]>();
        for (const tileRef of allTileRefs) {
          const tileStrips = strips.filter((strip) => lidarStripMayAffectTile(strip, tileRef));
          if (tileStrips.length === 0) continue;
          const { demTile, tile } = await getLidarTileWithHalo(tileRef, 1);
          const res = await worker.runTile({
            tile,
            demTile,
            polygons: virtualPolygons.map(({ id, ring }) => ({ id, ring })),
            strips: tileStrips,
            options: { clipInnerBufferM },
          } as any);
          (res.perPolygon ?? []).forEach((polyStats) => {
            if (!polyStats.densityStats) return;
            const list = perRegionStats.get(polyStats.polygonId) ?? [];
            list.push(polyStats.densityStats);
            perRegionStats.set(polyStats.polygonId, list);
          });
        }
        const regionSummaries = Array.from(perRegionStats.values())
          .map((statsList) => aggregateMetricStats(statsList))
          .filter((stats) => stats.count > 0);
        if (regionSummaries.length === 0) throw new Error('No lidar density preview could be computed for this partition.');
        return {
          solution: refinedSolution,
          metricKind: 'density',
          stats: aggregateMetricStats(regionSummaries),
          regionStats: regionSummaries,
          regionCount: refinedSolution.regionCount,
          sampleCount: new Set(strips.map((strip) => strip.passIndex ?? -1)).size,
          sampleLabel: 'Flight lines',
        };
      } finally {
        worker.terminate();
      }
    }

    const camera = effectiveCameraForPolygon(polygonId, paramsMap);
    const altitudeAGL = params.altitudeAGL ?? altitude;
    const photoSpacing = photoSpacingFor(polygonId, altitudeAGL, params.frontOverlap ?? frontOverlap, paramsMap);
    const lineSpacing = lineSpacingFor(polygonId, altitudeAGL, params.sideOverlap ?? sideOverlap, paramsMap);
    const yawOffset = params.cameraYawOffsetDeg ?? 0;
    const normalizeDeg = (d: number) => ((d % 360) + 360) % 360;
    const poses: PoseMeters[] = [];
    let poseId = 0;

    for (const region of virtualPolygons) {
      const geometry = generatePlannedFlightGeometryForPolygon(region.ring, region.bearingDeg, lineSpacing, params);
      const path3d = build3DFlightPath(
        geometry,
        parentTiles,
        lineSpacing,
        { altitudeAGL, mode: altitudeMode, minClearance, preconnected: true },
      );
      const filtered = sampleCameraPositionsOnPlannedFlightGeometry(geometry, path3d, photoSpacing)
        .filter(([lng, lat]) => region.ring.length < 3 || isPointInRing(lng, lat, region.ring));
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

    if (poses.length === 0) throw new Error('No camera poses could be generated for this partition.');

    const worker = new OverlapWorker();
    try {
      const perRegionStats = new Map<string, GSDStats[]>();
      for (const tileRef of allTileRefs) {
        const { tile } = await getTile(tileRef);
        const res = await worker.runTile({
          tile,
          polygons: virtualPolygons.map(({ id, ring }) => ({ id, ring })),
          poses,
          cameras: [camera],
          poseCameraIndices: new Uint16Array(poses.length),
          camera: undefined,
          options: { clipInnerBufferM, minOverlapForGsd: minOverlapForGsdRef.current },
        } as any);
        (res.perPolygon ?? []).forEach((polyStats) => {
          if (!polyStats.gsdStats) return;
          const list = perRegionStats.get(polyStats.polygonId) ?? [];
          list.push(polyStats.gsdStats);
          perRegionStats.set(polyStats.polygonId, list);
        });
      }
      const regionSummaries = Array.from(perRegionStats.values())
        .map((statsList) => aggregateMetricStats(statsList))
        .filter((stats) => stats.count > 0);
      if (regionSummaries.length === 0) throw new Error('No camera GSD preview could be computed for this partition.');
      return {
        solution: refinedSolution,
        metricKind: 'gsd',
        stats: aggregateMetricStats(regionSummaries),
        regionStats: regionSummaries,
        regionCount: refinedSolution.regionCount,
        sampleCount: poses.length,
        sampleLabel: 'Images',
      };
    } finally {
      worker.terminate();
    }
  }, [
    altitude,
    altitudeModeUI,
    clipInnerBufferM,
    effectiveCameraForPolygon,
    frontOverlap,
    getMergedParamsMap,
    isLidarPayload,
    lineSpacingFor,
    mapRef,
    mapboxToken,
    minClearanceUI,
    minOverlapForGsdRef,
    photoSpacingFor,
    sideOverlap,
    turnExtendUI,
    zoom,
  ]);

  const scoreLidarPartitionPreview = useCallback((
    polygonId: string,
    solution: TerrainPartitionSolutionPreview,
    preview: ExactPartitionPreview,
    fastestMissionTimeSec: number,
  ) => {
    const params = getMergedParamsMap()[polygonId];
    if (!params || preview.metricKind !== 'density') {
      return {
        score: solution.normalizedQualityCost,
        holeFraction: 0,
        lowFraction: 0,
        worstRegionHoleFraction: 0,
        worstRegionLowFraction: 0,
      };
    }
    const model = getLidarModel(params.lidarKey);
    const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
    const speedMps = params.speedMps ?? model.defaultSpeedMps;
    const returnMode = params.lidarReturnMode ?? 'single';
    const targetDensityPtsM2 = params.pointDensityPtsM2
      ?? lidarDeliverableDensity(
        model,
        params.altitudeAGL ?? altitude,
        params.sideOverlap ?? sideOverlap,
        speedMps,
        returnMode,
        mappingFovDeg,
      );
    const totalAreaM2 = Math.max(1, statsTotalAreaM2(preview.stats));
    const holeThreshold = Math.max(5, targetDensityPtsM2 * 0.2);
    const weakThreshold = Math.max(holeThreshold + 1e-6, targetDensityPtsM2 * 0.7);
    const q10 = histogramQuantile(preview.stats, 0.1);
    const q25 = histogramQuantile(preview.stats, 0.25);
    const holeFraction = histogramAreaBelow(preview.stats, holeThreshold) / totalAreaM2;
    const lowFraction = histogramAreaBelow(preview.stats, weakThreshold) / totalAreaM2;
    const q10Deficit = Math.max(0, 1 - q10 / Math.max(1e-6, targetDensityPtsM2));
    const q25Deficit = Math.max(0, 1 - q25 / Math.max(1e-6, targetDensityPtsM2));
    const meanDeficit = Math.max(0, 1 - preview.stats.mean / Math.max(1e-6, targetDensityPtsM2));
    const regionStats = preview.regionStats.length > 0 ? preview.regionStats : [preview.stats];
    const regionSignals = regionStats.map((stats) => {
      const regionAreaM2 = Math.max(1, statsTotalAreaM2(stats));
      const regionQ10 = histogramQuantile(stats, 0.1);
      return {
        holeFraction: histogramAreaBelow(stats, holeThreshold) / regionAreaM2,
        lowFraction: histogramAreaBelow(stats, weakThreshold) / regionAreaM2,
        q10Deficit: Math.max(0, 1 - regionQ10 / Math.max(1e-6, targetDensityPtsM2)),
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
    const regionPenalty = Math.max(0, solution.regionCount - 1) * 0.035;
    const score =
      4.8 * worstRegionHoleFraction +
      2.9 * worstRegionLowFraction +
      2.1 * worstRegionQ10Deficit +
      0.9 * worstRegionMeanDeficit +
      4.2 * holeFraction +
      2.4 * lowFraction +
      1.9 * q10Deficit +
      1.2 * q25Deficit +
      0.8 * meanDeficit +
      0.18 * relativeTimePenalty +
      regionPenalty;
    return {
      score,
      holeFraction,
      lowFraction,
      worstRegionHoleFraction,
      worstRegionLowFraction,
    };
  }, [altitude, getMergedParamsMap, sideOverlap]);

  const scoreCameraPartitionPreview = useCallback((
    solution: TerrainPartitionSolutionPreview,
    preview: ExactPartitionPreview,
    fastestMissionTimeSec: number,
  ) => {
    if (preview.metricKind !== 'gsd') {
      return { score: solution.normalizedQualityCost };
    }
    const overallMeanCm = preview.stats.mean * 100;
    const overallQ75Cm = histogramQuantile(preview.stats, 0.75) * 100;
    const overallQ90Cm = histogramQuantile(preview.stats, 0.9) * 100;
    const overallMaxCm = preview.stats.max * 100;
    const regionStats = preview.regionStats.length > 0 ? preview.regionStats : [preview.stats];
    const worstRegionMeanCm = Math.max(...regionStats.map((stats) => stats.mean * 100));
    const worstRegionQ90Cm = Math.max(...regionStats.map((stats) => histogramQuantile(stats, 0.9) * 100));
    const worstRegionMaxCm = Math.max(...regionStats.map((stats) => stats.max * 100));
    const relativeTimePenalty = fastestMissionTimeSec > 0
      ? Math.max(0, solution.totalMissionTimeSec / fastestMissionTimeSec - 1)
      : 0;
    const regionPenalty = Math.max(0, solution.regionCount - 1) * 0.12;
    const score =
      2.6 * worstRegionQ90Cm +
      2.0 * worstRegionMeanCm +
      1.2 * worstRegionMaxCm +
      1.0 * overallQ90Cm +
      0.7 * overallQ75Cm +
      0.4 * overallMeanCm +
      0.25 * overallMaxCm +
      0.35 * relativeTimePenalty +
      regionPenalty;
    return { score };
  }, []);

  const rerankPartitionSolutionsWithExactRegion = useCallback(async (
    polygonId: string,
    solutions: TerrainPartitionSolutionPreview[],
  ) => {
    const api = mapRef.current;
    const paramsMap = getMergedParamsMap();
    const params = paramsMap[polygonId];
    if (!params || solutions.length === 0) {
      return { bestIndex: 0, solutions };
    }
    const parentRing = api?.getPolygonsWithIds?.().find((polygon) => polygon.id === polygonId)?.ring;
    const parentTiles = (api?.getPolygonTiles?.().get(polygonId) ?? []) as TerrainTile[];
    if (!parentRing || parentRing.length < 3 || !parentTiles.length) {
      return { bestIndex: 0, solutions };
    }
    const altitudeMode = (api as any)?.getAltitudeMode ? (api as any).getAltitudeMode() : altitudeModeUI;
    const minClearance = (api as any)?.getMinClearance ? (api as any).getMinClearance() : minClearanceUI;
    const turnExtend = (api as any)?.getTurnExtend ? Math.max(0, (api as any).getTurnExtend()) : turnExtendUI;
    const [{ rerankPartitionSolutionsExact }, { withBrowserExactRegionRuntime }] = await Promise.all([
      loadExactRegionModule(),
      loadExactBrowserRuntimeModule(),
    ]);
    const exact = await withBrowserExactRegionRuntime(mapboxToken, (runtime) => rerankPartitionSolutionsExact(runtime, {
      scopeId: polygonId,
      polygonId,
      ring: parentRing,
      params,
      altitudeMode,
      minClearanceM: minClearance,
      turnExtendM: turnExtend,
      exactOptimizeZoom: zoom,
      timeWeight: 0.1,
      clipInnerBufferM,
      minOverlapForGsd: minOverlapForGsdRef.current,
      geometryTiles: parentTiles,
      solutions,
      rankingSource: 'frontend-exact',
    }));
    setExactPartitionPreviewByKey((prev) => {
      const next = { ...prev };
      Object.entries(exact.previewsBySignature).forEach(([signature, preview]) => {
        next[`${polygonId}:${signature}`] = preview;
      });
      return next;
    });
    return { bestIndex: exact.bestIndex, solutions: exact.solutions };
  }, [
    altitudeModeUI,
    clipInnerBufferM,
    getMergedParamsMap,
    mapRef,
    mapboxToken,
    minClearanceUI,
    minOverlapForGsdRef,
    turnExtendUI,
    zoom,
  ]);

  const chooseBestExactLidarPartitionIndex = useCallback(async (
    polygonId: string,
    solutions: TerrainPartitionSolutionPreview[],
  ) => {
    const startedAt = splitPerfNow();
    if (solutions.length === 0) return { bestIndex: 0, solutions };
    const { bestIndex, solutions: refinedSolutions } = await rerankPartitionSolutionsWithExactRegion(polygonId, solutions);
    splitPerfLog(polygonId, 'finished exact lidar partition ranking', {
      totalMs: Math.round(splitPerfNow() - startedAt),
      solutionCount: solutions.length,
      bestIndex,
    });
    return { bestIndex, solutions: refinedSolutions };
  }, [rerankPartitionSolutionsWithExactRegion]);

  const chooseBestExactCameraPartitionIndex = useCallback(async (
    polygonId: string,
    solutions: TerrainPartitionSolutionPreview[],
  ) => {
    const startedAt = splitPerfNow();
    if (solutions.length === 0) return { bestIndex: 0, solutions };
    const { bestIndex, solutions: refinedSolutions } = await rerankPartitionSolutionsWithExactRegion(polygonId, solutions);
    splitPerfLog(polygonId, 'finished exact camera partition ranking', {
      totalMs: Math.round(splitPerfNow() - startedAt),
      solutionCount: solutions.length,
      bestIndex,
    });
    return { bestIndex, solutions: refinedSolutions };
  }, [rerankPartitionSolutionsWithExactRegion]);

  const loadTerrainPartitionOptions = useCallback(async (
    polygonId: string,
    options?: {
      showEmptyToast?: boolean;
      showErrorToast?: boolean;
    },
  ) => {
    const api = mapRef.current;
    if (!api?.getTerrainPartitionSolutions) return { solutions: [], defaultIndex: 0 };
    const startedAt = splitPerfNow();
    const showEmptyToast = options?.showEmptyToast ?? true;
    const showErrorToast = options?.showErrorToast ?? true;
    setLoadingPartitionOptionsIds((prev) => ({ ...prev, [polygonId]: true }));
    try {
      splitPerfLog(polygonId, 'loading terrain partition options');
      const solutions = await api.getTerrainPartitionSolutions(polygonId);
      splitPerfLog(polygonId, 'terrain partition options fetched', {
        totalMs: Math.round(splitPerfNow() - startedAt),
        solutionCount: solutions.length,
        regionCounts: solutions.map((solution) => solution.regionCount),
      });
      const firstPracticalIndex = Math.max(
        0,
        solutions.findIndex((solution) => solution.isFirstPracticalSplit),
      );
      const currentIndex = partitionSelectionByPolygon[polygonId];
      let selectedIndex = Number.isInteger(currentIndex) && currentIndex! >= 0 && currentIndex! < solutions.length
        ? currentIndex!
        : firstPracticalIndex;
      setExactPartitionPreviewByKey((prev) => {
        const next = { ...prev };
        Object.keys(next).forEach((key) => {
          if (key.startsWith(`${polygonId}:`)) delete next[key];
        });
        return next;
      });
      let preparedSolutions = solutions;
      if (solutions.length > 1) {
        const backendExactPrepared = preparedSolutions.every((solution) => solution.rankingSource === 'backend-exact');
        console.log(`[terrain-split][${polygonId}] preparing partition options`, {
          incomingSolutionCount: preparedSolutions.length,
          incomingRankingSources: preparedSolutions.map((solution) => solution.rankingSource ?? 'surrogate'),
          backendExactPrepared,
        });
        if (backendExactPrepared) {
          selectedIndex = 0;
          console.log(`[terrain-split][${polygonId}] using backend-exact partition ranking`, {
            selectedIndex,
            solutionCount: preparedSolutions.length,
          });
        } else {
          if (isLidarPayload(polygonId, getMergedParamsMap())) {
            console.log(`[terrain-split][${polygonId}] running frontend exact rerank for lidar partition options`, {
              solutionCount: preparedSolutions.length,
            });
            const exact = await chooseBestExactLidarPartitionIndex(polygonId, preparedSolutions);
            selectedIndex = exact.bestIndex;
            preparedSolutions = exact.solutions;
            console.log(`[terrain-split][${polygonId}] frontend exact rerank finished for lidar partition options`, {
              selectedIndex,
              solutionCount: preparedSolutions.length,
              rankingSources: preparedSolutions.map((solution) => solution.rankingSource ?? 'surrogate'),
            });
          } else {
            console.log(`[terrain-split][${polygonId}] running frontend exact rerank for camera partition options`, {
              solutionCount: preparedSolutions.length,
            });
            const exact = await chooseBestExactCameraPartitionIndex(polygonId, preparedSolutions);
            selectedIndex = exact.bestIndex;
            preparedSolutions = exact.solutions;
            console.log(`[terrain-split][${polygonId}] frontend exact rerank finished for camera partition options`, {
              selectedIndex,
              solutionCount: preparedSolutions.length,
              rankingSources: preparedSolutions.map((solution) => solution.rankingSource ?? 'surrogate'),
            });
          }
        }
      } else {
        console.log(`[terrain-split][${polygonId}] partition options do not require exact rerank`, {
          solutionCount: preparedSolutions.length,
          rankingSources: preparedSolutions.map((solution) => solution.rankingSource ?? 'surrogate'),
        });
      }
      splitPerfLog(polygonId, 'terrain partition options prepared for UI', {
        totalMs: Math.round(splitPerfNow() - startedAt),
        selectedIndex,
      });
      setPartitionOptionsByPolygon((prev) => ({ ...prev, [polygonId]: preparedSolutions }));
      setPartitionSelectionByPolygon((prev) => ({ ...prev, [polygonId]: selectedIndex }));
      if (solutions.length === 0 && showEmptyToast) {
        toast({
          title: 'No partition options',
          description: 'This area does not currently produce more than one practical terrain partition.',
          variant: 'destructive',
        });
      }
      return { solutions: preparedSolutions, defaultIndex: selectedIndex };
    } catch (error) {
      if (showErrorToast) {
        toast({
          title: 'Partition planning failed',
          description: error instanceof Error ? error.message : 'Unable to compute terrain partition options.',
          variant: 'destructive',
        });
      }
      return { solutions: [], defaultIndex: 0 };
    } finally {
      setLoadingPartitionOptionsIds((prev) => {
        if (!prev[polygonId]) return prev;
        const next = { ...prev };
        delete next[polygonId];
        return next;
      });
    }
  }, [
    chooseBestExactCameraPartitionIndex,
    chooseBestExactLidarPartitionIndex,
    getMergedParamsMap,
    isLidarPayload,
    mapRef,
    partitionSelectionByPolygon,
  ]);

  const combinedPolygons = useMemo(() => {
    const polygonOrdering = getPolygons().map((p) => p.id || 'unknown');
    const livePolygonIds = new Set(polygonOrdering);
    const analysisOrdering = polygonAnalyses.map((analysis) => analysis.polygonId);
    const poseOnlyIds = analysisOrdering.filter((id) => id === '__POSES__');
    const order = polygonOrdering.length > 0 ? [...polygonOrdering, ...poseOnlyIds.filter((id) => !polygonOrdering.includes(id))] : analysisOrdering;
    const map = new Map<string, { analysis?: PolygonAnalysisResult; stats?: PolygonMetricSummary }>();

    polygonAnalyses.forEach((analysis) => {
      map.set(analysis.polygonId, { analysis, stats: perPolygonStats.get(analysis.polygonId) });
    });

    perPolygonStats.forEach((stats, polygonId) => {
      if (map.has(polygonId)) {
        map.set(polygonId, { analysis: map.get(polygonId)?.analysis, stats });
      } else {
        map.set(polygonId, { stats });
      }
    });

    const orderedIds = [
      ...order,
      ...analysisOrdering.filter((id) => id === '__POSES__' && !order.includes(id)),
    ];

    return orderedIds
      .map((polygonId, index) => ({ polygonId, analysis: map.get(polygonId)?.analysis, stats: map.get(polygonId)?.stats, sortIndex: index }))
      .filter(({ polygonId, analysis, stats }) => (
        polygonId === '__POSES__' || livePolygonIds.has(polygonId)
      ) && (analysis || stats))
      .sort((a, b) => a.sortIndex - b.sortIndex);
  }, [polygonAnalyses, perPolygonStats, getPolygons]);

  React.useEffect(() => {
    splitPerfLog('__panel__', 'combinedPolygons recalculated', {
      livePolygonIds: getPolygons().map((polygon) => polygon.id || 'unknown'),
      analysisIds: polygonAnalyses.map((analysis) => analysis.polygonId),
      statIds: Array.from(perPolygonStats.keys()),
      combinedIds: combinedPolygons.map((polygon) => polygon.polygonId),
    });
  }, [combinedPolygons, getPolygons, perPolygonStats, polygonAnalyses]);

  React.useEffect(() => {
    if (combinedPolygons.length === 0) {
      if (activeSelectedId) setSelection(null);
      return;
    }

    const hasActiveSelection = !!activeSelectedId && combinedPolygons.some((item) => item.polygonId === activeSelectedId);
    if (hasActiveSelection) return;

    if (activeSelectedId) {
      setSelection(null);
    }
  }, [combinedPolygons, activeSelectedId, setSelection]);

  React.useEffect(() => {
    if (!activeSelectedId) return;
    const node = itemRefs.current.get(activeSelectedId);
    if (node && node.scrollIntoView) {
      node.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeSelectedId]);

  /**
   * Compute payload-aware coverage analysis for either:
   *  - a single polygon (opts.polygonId provided), or
   *  - all polygons (default)
   */
  const compute = useCallback(async (opts?: { polygonId?: string; suppressMapNotReadyToast?: boolean; generation?: number }) => {
    const generation = opts?.generation ?? resetGenerationRef.current;
    if (!shouldRunAsyncGeneration(generation, resetGenerationRef.current)) {
      splitPerfLog(opts?.polygonId ?? '__all__', 'dropping stale coverage compute before start', {
        generation,
        currentGeneration: resetGenerationRef.current,
      });
      return;
    }
    if (runningRef.current) {
      const nextPending = pendingComputeRef.current;
      pendingComputeRef.current = {
        polygonId: nextPending?.polygonId === undefined || opts?.polygonId === undefined
          ? undefined
          : (nextPending?.polygonId ?? opts?.polygonId),
        suppressMapNotReadyToast: Boolean(
          nextPending?.suppressMapNotReadyToast || opts?.suppressMapNotReadyToast
        ),
        generation,
      };
      splitPerfLog(opts?.polygonId ?? '__all__', 'coverage compute queued while another run is active', {
        queued: pendingComputeRef.current,
      });
      return;
    }
    const api = mapRef.current;
    const paramsMap = getMergedParamsMap();
    const overrideCam = parseCameraOverride();

    const perPolyCam: Record<string, CameraModel> = {};
    Object.entries(paramsMap).forEach(([pid, p]: any) => {
      if (p?.cameraKey && CAMERA_REGISTRY[p.cameraKey]) perPolyCam[pid] = CAMERA_REGISTRY[p.cameraKey];
    });

    const identifyCameraKey = (cam: CameraModel | null): string | null => {
      if (!cam) return null;
      for (const [key, model] of Object.entries(CAMERA_REGISTRY)) {
        if (model.w_px === cam.w_px && model.h_px === cam.h_px && Math.abs(model.f_m - cam.f_m) / model.f_m < 0.02) {
          return key;
        }
      }
      return null;
    };

    const overrideCamKey = overrideCam ? identifyCameraKey(overrideCam) : null;
    const poses = parsePosesMeters() ?? [];
    let camerasArr: CameraModel[] | undefined;
    let poseIdxArr: Uint16Array | undefined;
    if (poses.length > 0) {
      const camObjToIndex = new Map<CameraModel, number>();
      const cams: CameraModel[] = [];
      const indices = new Uint16Array(poses.length);
      for (let i = 0; i < poses.length; i++) {
        const pose = poses[i];
        let cam: CameraModel | undefined;
        if (pose.polygonId && perPolyCam[pose.polygonId]) {
          cam = perPolyCam[pose.polygonId];
        } else if (!pose.polygonId && importedPoses.length > 0) {
          try {
            const parsed = JSON.parse(cameraText) as CameraModel;
            if (parsed && typeof parsed.f_m === 'number') cam = parsed;
          } catch {}
        } else if (overrideCam && typeof overrideCam === 'object' && 'f_m' in overrideCam) {
          cam = overrideCam;
        }
        if (!cam) cam = SONY_RX1R2;
        if (!camObjToIndex.has(cam)) {
          camObjToIndex.set(cam, cams.length);
          cams.push(cam);
        }
        indices[i] = camObjToIndex.get(cam)!;
      }
      camerasArr = cams.length > 0 ? cams : undefined;
      poseIdxArr = cams.length > 1 ? indices : new Uint16Array(poses.length);
    }

    let allPolygons = getPolygons();
    if (poses.length > 0 && importedPoses.length > 0) {
      const buildPosesAOIRing = (posesIn: PoseMeters[]): [number, number][] => {
        if (!posesIn || posesIn.length < 3) return [];
        const maxPoints = 2000;
        const step = Math.max(1, Math.floor(posesIn.length / maxPoints));
        const pts = posesIn
          .filter((_, i) => i % step === 0)
          .map((pose) => {
            const [lng, lat] = metersToLngLat(pose.x, pose.y);
            return turf.point([lng, lat]);
          });
        const fc = turf.featureCollection(pts);
        const hull = turf.convex(fc);
        if (!hull) return [];
        const geom: any = (hull as any).geometry;
        return geom.type === 'Polygon' ? geom.coordinates[0] : geom.coordinates[0][0];
      };
      const ring = poseAreaRingRef.current.length ? poseAreaRingRef.current : buildPosesAOIRing(importedPoses);
      poseAreaRingRef.current = ring;
      if (ring.length >= 4) allPolygons = [...allPolygons, { id: '__POSES__', ring }];
    }

    if (allPolygons.length === 0) {
      toast({ variant: 'destructive', title: 'Missing inputs', description: 'Draw or import an area first.' });
      return;
    }

    const cameraPolygons = allPolygons.filter((polygon) => !isLidarPayload(polygon.id || 'unknown', paramsMap));
    const lidarPolygons = allPolygons.filter((polygon) => isLidarPayload(polygon.id || 'unknown', paramsMap));
    const requestedTargetPolygonId = opts?.polygonId;
    const currentTargetPolygon = requestedTargetPolygonId
      ? allPolygons.find((polygon) => (polygon.id || 'unknown') === requestedTargetPolygonId)
      : undefined;
    const previousTargetRing = requestedTargetPolygonId
      ? prevPolygonRingsRef.current.get(requestedTargetPolygonId)
      : undefined;
    const targetPolygonMoved = !!(
      requestedTargetPolygonId &&
      currentTargetPolygon?.ring &&
      previousTargetRing &&
      !ringsRoughlyEqual(currentTargetPolygon.ring, previousTargetRing)
    );
    const targetPolygonId = targetPolygonMoved ? undefined : requestedTargetPolygonId;
    const targetIsCamera = targetPolygonId ? cameraPolygons.some((polygon) => (polygon.id || 'unknown') === targetPolygonId) : cameraPolygons.length > 0;
    const targetIsLidar = targetPolygonId ? lidarPolygons.some((polygon) => (polygon.id || 'unknown') === targetPolygonId) : lidarPolygons.length > 0;
    const canRunCamera = targetIsCamera && poses.length > 0;
    const canRunLidar = targetIsLidar;

    if (!canRunCamera && !canRunLidar) {
      toast({ variant: 'destructive', title: 'Missing inputs', description: 'Provide poses or generate flight lines before running analysis.' });
      return;
    }

    const map: mapboxgl.Map | undefined = mapRef.current?.getMap?.();
    if (!map || !map.isStyleLoaded?.()) {
      if (opts?.suppressMapNotReadyToast) {
        if (deferredComputeTimeoutRef.current !== null) {
          cancelGuardedTimeout(deferredComputeTimeoutRef.current);
        }
        splitPerfLog(opts?.polygonId ?? '__all__', 'coverage compute deferred because map is not ready', {
          polygonId: opts?.polygonId,
        });
        deferredComputeTimeoutRef.current = scheduleGuardedTimeout(() => {
          deferredComputeTimeoutRef.current = null;
          compute({ ...opts, generation });
        }, 200, generation);
      } else {
        toast({
          variant: "destructive",
          title: "Map not ready",
          description: "Please wait for the map to load completely."
        });
      }
      return;
    }

    const computeStartedAt = splitPerfNow();
    const mySeq = ++computeSeqRef.current;
    const scope = opts?.polygonId ?? '__all__';
    const [
      { aggregateMetricStats, aggregateOverallMetricStats },
      { fetchTerrainRGBA, tilesCoveringPolygon, LidarDensityWorker, OverlapWorker },
    ] = await Promise.all([
      loadMetricAggregationModule(),
      loadOverlapControllerModule(),
    ]);
    splitPerfLog(scope, 'coverage compute start', {
      seq: mySeq,
      polygonId: opts?.polygonId,
    });

    const now = Date.now();
    if (!opts?.polygonId || targetPolygonMoved) {
      clearAllOverlays(map);
      globalRunIdRef.current = `${now}`;
      perPolyTileStatsRef.current.clear();
      cameraTileResultsRef.current.clear();
      lidarTileResultsRef.current.clear();
    }
    const runId = globalRunIdRef.current ?? `${now}`;
    if (!globalRunIdRef.current) globalRunIdRef.current = runId;

    const polygonMap = new Map(allPolygons.map((polygon) => [polygon.id || 'unknown', polygon] as const));
    const collectTiles = (sourcePolygons: PolygonLngLatWithId[]): { z: number; x: number; y: number }[] => {
      const seen = new Set<string>();
      const tiles: { z: number; x: number; y: number }[] = [];
      for (const poly of sourcePolygons) {
        for (const tile of tilesCoveringPolygon(poly, zoom)) {
          const key = `${zoom}/${tile.x}/${tile.y}`;
          if (seen.has(key)) continue;
          seen.add(key);
          tiles.push({ z: zoom, x: tile.x, y: tile.y });
        }
      }
      if (targetPolygonId && sourcePolygons.some((polygon) => (polygon.id || 'unknown') === targetPolygonId)) {
        const prevRing = prevPolygonRingsRef.current.get(targetPolygonId);
        if (prevRing && prevRing.length >= 4) {
          const prevPolygon = { id: targetPolygonId, ring: prevRing } as PolygonLngLatWithId;
          for (const tile of tilesCoveringPolygon(prevPolygon, zoom)) {
            const key = `${zoom}/${tile.x}/${tile.y}`;
            if (seen.has(key)) continue;
            seen.add(key);
            tiles.push({ z: zoom, x: tile.x, y: tile.y });
          }
        }
      }
      return tiles;
    };

    const buildNeededTileKeys = (polygons: PolygonLngLatWithId[]) => {
      const keys = new Set<string>();
      for (const polygon of polygons) {
        for (const tile of tilesCoveringPolygon(polygon, zoom)) {
          keys.add(`${zoom}/${tile.x}/${tile.y}`);
        }
      }
      return keys;
    };

    const pruneOverlaysByKinds = (kinds: Array<'overlap' | 'pass' | 'gsd' | 'density'>, neededTileKeys: Set<string>) => {
      const styleLayers = map.getStyle()?.layers ?? [];
      for (const layer of styleLayers) {
        const id = layer.id;
        if (!id.startsWith(`ogsd-${runId}-`)) continue;
        const parts = id.split('-');
        if (parts.length < 6) continue;
        const kind = parts[2] as 'overlap' | 'pass' | 'gsd' | 'density';
        if (!kinds.includes(kind)) continue;
        const zStr = parts[parts.length - 3];
        const xStr = parts[parts.length - 2];
        const yStr = parts[parts.length - 1];
        const key = `${zStr}/${xStr}/${yStr}`;
        if (neededTileKeys.has(key)) continue;
        try {
          if (map.getLayer(id)) map.removeLayer(id);
          if (map.getSource(id)) map.removeSource(id);
        } catch {}
      }
    };

    const pruneCachedTileResults = (cache: Map<string, OverlayTileResult>, neededTileKeys: Set<string>) => {
      for (const key of Array.from(cache.keys())) {
        if (!neededTileKeys.has(key)) cache.delete(key);
      }
    };

    const getTile = async (tileRef: { z: number; x: number; y: number }) => {
      const cacheKey = `${tileRef.z}/${tileRef.x}/${tileRef.y}`;
      let tileData = tileCacheRef.current.get(cacheKey);
      if (!tileData) {
        const imgData = await fetchTerrainRGBA(tileRef.z, tileRef.x, tileRef.y, mapboxToken);
        tileData = {
          width: imgData.width,
          height: imgData.height,
          data: new Uint8ClampedArray(imgData.data),
        };
        tileCacheRef.current.set(cacheKey, tileData);
        const maxTiles = 256;
        if (tileCacheRef.current.size > maxTiles) {
          const firstKey = tileCacheRef.current.keys().next().value;
          if (firstKey) tileCacheRef.current.delete(firstKey);
        }
      }
      return {
        cacheKey,
        tile: { z: tileRef.z, x: tileRef.x, y: tileRef.y, size: tileData.width, data: new Uint8ClampedArray(tileData.data) },
      };
    };

    const normalizeTileRef = (tileRef: { z: number; x: number; y: number }) => {
      const tilesPerAxis = 1 << tileRef.z;
      const wrappedX = ((tileRef.x % tilesPerAxis) + tilesPerAxis) % tilesPerAxis;
      const clampedY = Math.max(0, Math.min(tilesPerAxis - 1, tileRef.y));
      return { z: tileRef.z, x: wrappedX, y: clampedY };
    };

    const getLidarTileWithHalo = async (tileRef: { z: number; x: number; y: number }, padTiles = 1) => {
      const center = await getTile(tileRef);
      if (padTiles <= 0) {
        return {
          cacheKey: center.cacheKey,
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
        cacheKey: center.cacheKey,
        tile: center.tile,
        demTile: {
          size: demSize,
          padTiles,
          data: demData,
        },
      };
    };

    const upsertTileStats = (cacheKey: string, stats: PolygonTileStats[] | undefined) => {
      if (!stats) return;
      for (const polyStats of stats) {
        if (!perPolyTileStatsRef.current.has(polyStats.polygonId)) {
          perPolyTileStatsRef.current.set(polyStats.polygonId, new Map());
        }
        perPolyTileStatsRef.current.get(polyStats.polygonId)!.set(cacheKey, polyStats);
      }
    };

    runningRef.current = true;
    setRunning(true);
    autoTriesRef.current = 0;

    const cameraWorker = canRunCamera ? new OverlapWorker() : null;
    const lidarWorker = canRunLidar ? new LidarDensityWorker() : null;
    try {
      if (cameraWorker && camerasArr && cameraPolygons.length > 0) {
        const cameraSourcePolygons = targetPolygonId
          ? cameraPolygons.filter((polygon) => (polygon.id || 'unknown') === targetPolygonId)
          : cameraPolygons;
        const cameraTiles = collectTiles(cameraSourcePolygons);
        for (const tileRef of cameraTiles) {
          const { cacheKey, tile } = await getTile(tileRef);
          const res = await cameraWorker.runTile({
            tile,
            polygons: cameraPolygons,
            poses,
            cameras: camerasArr,
            poseCameraIndices: poseIdxArr,
            camera: undefined,
            options: { clipInnerBufferM, minOverlapForGsd: minOverlapForGsdRef.current },
          } as any);
          if (mySeq !== computeSeqRef.current) break;
          cameraTileResultsRef.current.set(cacheKey, toOverlayTileResult(res));
          upsertTileStats(cacheKey, res.perPolygon);
        }
      }

      if (lidarWorker && lidarPolygons.length > 0) {
        const lidarSourcePolygons = targetPolygonId
          ? lidarPolygons.filter((polygon) => (polygon.id || 'unknown') === targetPolygonId)
          : lidarPolygons;
        const lidarTiles = collectTiles(lidarSourcePolygons);
        const { strips: lidarStrips } = buildLidarStrips(paramsMap);
        for (const tileRef of lidarTiles) {
          const tileStrips = lidarStrips.filter((strip) => lidarStripMayAffectTile(strip, tileRef));
          if (tileStrips.length === 0) continue;
          const { cacheKey, tile, demTile } = await getLidarTileWithHalo(tileRef, 1);
          const res = await lidarWorker.runTile({
            tile,
            demTile,
            // Lidar density overlays are tile-global and additive across overlapping polygons.
            // Even during a targeted recompute, evaluate the affected tiles against the full lidar
            // polygon set so we do not overwrite a shared tile with one-polygon-only density.
            polygons: lidarPolygons,
            strips: tileStrips,
            options: { clipInnerBufferM },
          } as any);
          if (mySeq !== computeSeqRef.current) break;
          lidarTileResultsRef.current.set(cacheKey, toOverlayTileResult(res));
          upsertTileStats(cacheKey, res.perPolygon);
        }
      }

      if (mySeq !== computeSeqRef.current || !shouldRunAsyncGeneration(generation, resetGenerationRef.current)) return;

      const neededCameraTileKeys = buildNeededTileKeys(cameraPolygons);
      const neededLidarTileKeys = buildNeededTileKeys(lidarPolygons);
      pruneOverlaysByKinds(['overlap', 'gsd'], neededCameraTileKeys);
      pruneOverlaysByKinds(['pass', 'density'], neededLidarTileKeys);
      pruneCachedTileResults(cameraTileResultsRef.current, neededCameraTileKeys);
      pruneCachedTileResults(lidarTileResultsRef.current, neededLidarTileKeys);

      const emptyPolygonIds: string[] = [];
      perPolyTileStatsRef.current.forEach((tileMap, polygonId) => {
        const neededKeys = isLidarPayload(polygonId, paramsMap) ? neededLidarTileKeys : neededCameraTileKeys;
        for (const key of Array.from(tileMap.keys())) {
          if (!neededKeys.has(key)) tileMap.delete(key);
        }
        if (tileMap.size === 0 || !polygonMap.has(polygonId)) emptyPolygonIds.push(polygonId);
      });
      emptyPolygonIds.forEach((polygonId) => perPolyTileStatsRef.current.delete(polygonId));

      const nextPerPolygon = new Map<string, PolygonMetricSummary>();
      const overallMetricGroups: Array<{ metricKind: MetricKind; tileStats: GSDStats[] }> = [];

      perPolyTileStatsRef.current.forEach((polygonTileStatsMap, polygonId) => {
        const polygon = polygonMap.get(polygonId);
        if (!polygon) return;
        const areaAcres = calculatePolygonAreaAcres(polygon.ring);
        const allTileStats = Array.from(polygonTileStatsMap.values());
        const isLidarPolygon = isLidarPayload(polygonId, paramsMap);

        if (isLidarPolygon) {
          const densityStatList = allTileStats.map((stats) => stats.densityStats).filter(Boolean) as GSDStats[];
          const aggregatedDensityStats = aggregateMetricStats(densityStatList);
          if (!(aggregatedDensityStats.count > 0)) return;
          const uniqueLineIds = new Set<number>();
          for (const stats of allTileStats) {
            const hitLineIds = stats.hitLineIds;
            if (!hitLineIds) continue;
            for (let i = 0; i < hitLineIds.length; i++) uniqueLineIds.add(hitLineIds[i]);
          }
          const params = (paramsMap as any)[polygonId];
          const model = getLidarModel(params?.lidarKey);
          const comparisonLabel = lidarComparisonLabel(params?.lidarComparisonMode);
          nextPerPolygon.set(polygonId, {
            polygonId,
            metricKind: 'density',
            stats: aggregatedDensityStats,
            areaAcres,
            sampleCount: uniqueLineIds.size,
            sampleLabel: 'Flight lines',
            sourceLabel: `${model.key} · ${comparisonLabel}`,
          });
          overallMetricGroups.push({ metricKind: 'density', tileStats: densityStatList });
          return;
        }

        const gsdStatList = allTileStats.map((stats) => stats.gsdStats).filter(Boolean) as GSDStats[];
        const aggregatedGsdStats = aggregateMetricStats(gsdStatList);
        if (!(aggregatedGsdStats.count > 0)) return;
        const uniquePoseIds = new Set<number>();
        for (const stats of allTileStats) {
          const hitPoseIds = stats.hitPoseIds;
          if (!hitPoseIds) continue;
          for (let i = 0; i < hitPoseIds.length; i++) uniquePoseIds.add(hitPoseIds[i]);
        }
        const params = (paramsMap as any)[polygonId];
        let cameraLabel = 'SONY_RX1R2';
        if (params?.cameraKey && CAMERA_REGISTRY[params.cameraKey]) {
          cameraLabel = params.cameraKey;
        } else if (overrideCam) {
          cameraLabel = overrideCamKey ? `Override (${overrideCamKey})` : 'Override';
        }
        nextPerPolygon.set(polygonId, {
          polygonId,
          metricKind: 'gsd',
          stats: aggregatedGsdStats,
          areaAcres,
          sampleCount: uniquePoseIds.size,
          sampleLabel: 'Images',
          sourceLabel: cameraLabel,
        });
        overallMetricGroups.push({ metricKind: 'gsd', tileStats: gsdStatList });
      });

      const nextOverallStats = aggregateOverallMetricStats(overallMetricGroups, tailAreaAcres);
      let nextLockedRanges = lockedOverlayRangesRef.current;
      if (overlayScaleLockedRef.current) {
        const mergedLockedRanges = mergeLockedOverlayRanges(lockedOverlayRangesRef.current, nextOverallStats);
        if (mergedLockedRanges.changed) {
          nextLockedRanges = mergedLockedRanges.ranges;
          lockedOverlayRangesRef.current = nextLockedRanges;
          setLockedOverlayRanges(nextLockedRanges);
        }
      }

      setPerPolygonStats(nextPerPolygon);
      setOverallStats(nextOverallStats);
      redrawAnalysisOverlays(nextOverallStats, nextLockedRanges);

      if (showCameraPoints && poses.length > 0) {
        const importedOnly = poses.filter((pose) => !pose.polygonId);
        const cameraPositions: [number, number, number][] = importedOnly.map((pose) => {
          const [lng, lat] = metersToLngLat(pose.x, pose.y);
          return [lng, lat, pose.z];
        });
        if (api?.addCameraPoints) api.addCameraPoints('__POSES__', cameraPositions);
      }

      prevPolygonRingsRef.current = new Map(getPolygons().map((polygon) => [polygon.id || 'unknown', polygon.ring] as const));
    } finally {
      cameraWorker?.terminate();
      lidarWorker?.terminate();
      runningRef.current = false;
      setRunning(false);
      splitPerfLog(scope, 'coverage compute end', {
        seq: mySeq,
        totalMs: Math.round(splitPerfNow() - computeStartedAt),
      });
      const pending = pendingComputeRef.current;
      if (pending) {
        pendingComputeRef.current = null;
        scheduleGuardedTimeout(() => {
          compute(pending);
        }, 0, pending.generation ?? generation);
      }
    }
  }, [CAMERA_REGISTRY, buildLidarStrips, cameraText, cancelGuardedTimeout, clipInnerBufferM, getMergedParamsMap, getPolygons, importedPoses, isLidarPayload, mapRef, mapboxToken, parseCameraOverride, parsePosesMeters, redrawAnalysisOverlays, scheduleGuardedTimeout, showCameraPoints, toOverlayTileResult, zoom]);

  // Auto-run function that can be called externally
  const autoRun = useCallback(async (opts?: { polygonId?: string; reason?: 'lines'|'spacing'|'alt'|'manual' }) => {
    if (suppressAutoRunUntilRef.current > Date.now()) return;
    const api = mapRef.current;
    const map = api?.getMap?.();
    const ready = !!map?.isStyleLoaded?.();
    const poses = parsePosesMeters();
    const paramsMap = getMergedParamsMap();

    const rings: [number, number][][] = api?.getPolygons?.() ?? [];
    const fl = api?.getFlightLines?.();
    const haveLines = !!fl && (
      opts?.polygonId
        ? !!fl.get(opts.polygonId) && fl.get(opts.polygonId)!.flightLines.length > 0
        : Array.from(fl.values()).some((v: any) => v.flightLines.length > 0)
    );
    const havePolys = opts?.polygonId ? (api?.getPolygonsWithIds?.() ?? []).some((p:any)=>p.id===opts.polygonId) : (rings.length > 0);
    const relevantIds = opts?.polygonId
      ? [opts.polygonId]
      : (api?.getPolygonsWithIds?.() ?? []).map((polygon: any) => polygon.id || 'unknown');
    const haveLidarPolys = relevantIds.some((polygonId) => isLidarPayload(polygonId, paramsMap));
    const { planCoverageAutoRun } = await loadCoverageAutoRunModule();
    const plan = planCoverageAutoRun({
      request: opts,
      nowMs: Date.now(),
      suppressAutoRunUntilMs: suppressAutoRunUntilRef.current,
      autoGenerate,
      importedPosesCount: importedPoses.length,
      ready,
      havePolys,
      haveLines,
      haveLidarPolys,
      posesCount: poses?.length ?? 0,
      retryCount: autoTriesRef.current,
    });

    autoTriesRef.current = plan.nextRetryCount;

    if (plan.kind === 'compute') {
      // Always defer one tick to allow React state updates (lines/tiles) to flush.
      const generation = resetGenerationRef.current;
      scheduleGuardedTimeout(() => compute({ ...plan.computeRequest, generation }), 0, generation);
      return;
    }

    if (plan.kind === 'retry') {
      if (autoRunTimeoutRef.current !== null) cancelGuardedTimeout(autoRunTimeoutRef.current);
      const generation = resetGenerationRef.current;
      autoRunTimeoutRef.current = scheduleGuardedTimeout(() => {
        if (autoGenerate || importedPoses.length > 0) autoRun(opts);
      }, plan.delayMs, generation);
    }
  }, [autoGenerate, cancelGuardedTimeout, importedPoses, compute, getMergedParamsMap, isLidarPayload, mapRef, parsePosesMeters, scheduleGuardedTimeout]);

  function rerunAnalysisForCreatedPolygons(createdIds: string[]) {
    const deadlineMs = 8000;
    const startedAt = Date.now();
    const scope = createdIds[0] ?? `split-${++splitPerfSeqRef.current}`;
    const generation = resetGenerationRef.current;
    splitPerfLog(scope, 'waiting for child flight lines before full raster recompute', {
      createdIds,
    });
    const attempt = () => {
      if (!shouldRunAsyncGeneration(generation, resetGenerationRef.current)) return;
      const api = mapRef.current;
      const lines = api?.getFlightLines?.();
      const tiles = api?.getPolygonTiles?.();
      const polygonIds = new Set((api?.getPolygonsWithIds?.() ?? []).map((polygon) => polygon.id || 'unknown'));
      const paramsByPolygon = api?.getPerPolygonParams?.() ?? {};
      const resultsByPolygon = new Set((api?.getPolygonResults?.() ?? []).map((result) => result.polygonId));
      const havePolygons = createdIds.every((id) => polygonIds.has(id));
      const haveLines = !!lines && createdIds.every((id) => {
        const entry = lines.get(id);
        return !!entry && Array.isArray(entry.flightLines) && entry.flightLines.length > 0;
      });
      const haveTiles = !!tiles && createdIds.every((id) => {
        const entry = tiles.get(id);
        return Array.isArray(entry) && entry.length > 0;
      });
      const haveParams = createdIds.every((id) => !!paramsByPolygon[id]);
      const haveResults = createdIds.every((id) => resultsByPolygon.has(id));
      const ready = havePolygons && haveLines && haveTiles && haveParams && haveResults;
      if ((ready || Date.now() - startedAt >= deadlineMs) && !runningRef.current) {
        suppressAutoRunUntilRef.current = 0;
        splitPerfLog(scope, 'triggering post-split auto-run for created polygons', {
          ready,
          waitMs: Date.now() - startedAt,
          havePolygons,
          haveLines,
          haveTiles,
          haveParams,
          haveResults,
          createdIds,
        });
        autoRun({ reason: 'lines' });
        return;
      }
      scheduleGuardedTimeout(attempt, 150, generation);
    };
    scheduleGuardedTimeout(attempt, 0, generation);
  }

  // Provide autoRun function to parent component - register immediately and on changes
  React.useEffect(() => {
    onAutoRun?.(autoRun);
  }, [autoRun, onAutoRun]);

  const clear = useCallback(() => {
    const map: any = mapRef.current?.getMap?.();
    const clearedState = createCoveragePanelResetState(Date.now());
    resetGenerationRef.current += 1;
    computeSeqRef.current += 1;
    pendingComputeRef.current = null;
    runningRef.current = clearedState.running;
    autoTriesRef.current = clearedState.autoRetryCount;
    setRunning(clearedState.running);
    cancelAllGuardedTimeouts();
    if (autoRunTimeoutRef.current) { autoRunTimeoutRef.current = null; }
    if (deferredComputeTimeoutRef.current !== null) {
      deferredComputeTimeoutRef.current = null;
    }
    if (map) {
      clearAllOverlays(map);
      const api = mapRef.current;
      if (api?.removeCameraPoints) {
        api.removeCameraPoints('__ALL__');
        api.removeCameraPoints('__POSES__');
      }
    }
    perPolyTileStatsRef.current.clear();
    cameraTileResultsRef.current.clear();
    lidarTileResultsRef.current.clear();
    tileCacheRef.current.clear();
    prevPolygonRingsRef.current = new Map();
    globalRunIdRef.current = clearedState.runId;
    poseAreaRingRef.current = clearedState.poseAreaRing;
    setImportedPoses(clearedState.importedPoses as PoseMeters[]);
    lockedOverlayRangesRef.current = {};
    setLockedOverlayRanges({});
    setOverallStats(clearedState.overallStats);
    setPerPolygonStats(clearedState.perPolygonStats as Map<string, PolygonMetricSummary>);
    setPartitionOptionsByPolygon(clearedState.partitionOptionsByPolygon as Record<string, TerrainPartitionSolutionPreview[]>);
    setPartitionSelectionByPolygon(clearedState.partitionSelectionByPolygon);
    setLoadingPartitionOptionsIds(clearedState.loadingPartitionOptionsIds);
    setApplyingPartitionIds(clearedState.applyingPartitionIds);
    setExactPartitionPreviewByKey(clearedState.exactPartitionPreviewByKey as Record<string, ExactPartitionPreview>);
    setSplittingPolygonIds(clearedState.splittingPolygonIds);
    setSelection(clearedState.selectedPolygonId);
    onPosesImported?.(0);
  }, [mapRef, onPosesImported]);

  const resetComputedAnalysisState = useCallback(() => {
    const map: any = mapRef.current?.getMap?.();
    const clearedState = createCoveragePanelResetState(Date.now());
    resetGenerationRef.current += 1;
    computeSeqRef.current += 1;
    pendingComputeRef.current = null;
    runningRef.current = clearedState.running;
    autoTriesRef.current = clearedState.autoRetryCount;
    setRunning(clearedState.running);
    cancelAllGuardedTimeouts();
    if (autoRunTimeoutRef.current) {
      autoRunTimeoutRef.current = null;
    }
    if (deferredComputeTimeoutRef.current !== null) {
      deferredComputeTimeoutRef.current = null;
    }
    if (map) {
      clearAllOverlays(map);
    }
    perPolyTileStatsRef.current.clear();
    cameraTileResultsRef.current.clear();
    lidarTileResultsRef.current.clear();
    tileCacheRef.current.clear();
    prevPolygonRingsRef.current = new Map();
    globalRunIdRef.current = clearedState.runId;
    setOverallStats(clearedState.overallStats);
    setPerPolygonStats(clearedState.perPolygonStats as Map<string, PolygonMetricSummary>);
    if (mapRef.current?.removeCameraPoints) {
      mapRef.current.removeCameraPoints('__ALL__');
      mapRef.current.removeCameraPoints('__POSES__');
    }
  }, [cancelAllGuardedTimeouts, mapRef]);

  React.useEffect(() => {
    const api = mapRef.current;
    const map: any = api?.getMap?.();
    if (!map?.isStyleLoaded?.()) return;
    const polygonCount = api?.getPolygonsWithIds?.().length ?? 0;
    if (polygonCount !== 0 || importedPoses.length !== 0) return;

    const hasOverlayLayers = (map.getStyle?.().layers ?? []).some((layer: any) => String(layer?.id || '').startsWith('ogsd-'));
    const hasOverlaySources = Object.keys(map.getStyle?.().sources ?? {}).some((id) => id.startsWith('ogsd-'));
    const hasAnalysisState = perPolyTileStatsRef.current.size > 0 || perPolygonStats.size > 0 || !!overallStats.gsd || !!overallStats.density;
    if (hasOverlayLayers || hasOverlaySources || hasAnalysisState) {
      clear();
    }
  }, [clear, importedPoses.length, mapRef, overallStats.density, overallStats.gsd, perPolygonStats, polygonAnalyses.length]);

  // Provide clear function to parent so header and Map can invoke it
  React.useEffect(() => {
    onClearExposed?.(clear);
  }, [clear, onClearExposed]);

  React.useEffect(() => {
    if (clearAllEpoch === lastHandledClearAllEpochRef.current) return;
    lastHandledClearAllEpochRef.current = clearAllEpoch;
    clear();
  }, [clear, clearAllEpoch]);

  const handlePoseFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const text = evt.target?.result as string;
        const obj = JSON.parse(text);

        // 1) Wingtra geotags (preferred when user selected Wingtra or auto-detected)
        let wingtraResult: ReturnType<typeof parseWingtraGeotags> | null = null;
        if (poseImportKind !== 'dji') {
          wingtraResult = parseWingtraGeotags(obj);
        }

        if (wingtraResult) {
          applyImportedPoses(wingtraResult.poses, wingtraResult.camera, wingtraResult.cameraKey, wingtraResult.sourceLabel || 'Wingtra');
          return;
        }

        // 2) DJI OPF input_cameras.json
        const posesWgs = extractPoses(obj);
        const djiCam = extractCameraModel(obj);
        let matchedRegistryKey: string | null = null;
        if (djiCam) {
          // Try to match against known cameras (same dimensions + ~5% focal length tolerance)
          for (const [key, cam] of Object.entries(CAMERA_REGISTRY)) {
            if (cam.w_px === djiCam.w_px && cam.h_px === djiCam.h_px) {
              const relErr = Math.abs(cam.f_m - djiCam.f_m) / cam.f_m;
              if (relErr < 0.05) { matchedRegistryKey = key; break; }
            }
          }
          const camPayload = matchedRegistryKey ? CAMERA_REGISTRY[matchedRegistryKey] : djiCam;
          setCameraText(JSON.stringify(camPayload, null, 2));
          // Auto‑enable override so imported intrinsics are actually used
          setUseOverrideCamera(true);
        }
        const posesMeters: PoseMeters[] = posesWgs.map((p, i) => {
          const { x, y } = wgs84ToWebMercator(p.lat, p.lon);
          return { id: p.id ?? `pose_${i}`, x, y, z: p.alt ?? 0, omega_deg: p.roll ?? 0, phi_deg: p.pitch ?? 0, kappa_deg: p.yaw ?? 0 } as PoseMeters;
        });
        applyImportedPoses(posesMeters, djiCam ?? null, matchedRegistryKey, 'DJI');
      } catch {
        toast({ variant: "destructive", title: "Invalid file", description: "Unable to parse Wingtra geotags or DJI OPF input_cameras.json" });
        onPosesImported?.(0);
      }
    };
    reader.readAsText(file);
    // Allow selecting the same file again by resetting the input element
    e.target.value = "";
  }, [applyImportedPoses, parseWingtraGeotags, poseImportKind, CAMERA_REGISTRY, onPosesImported]);

  React.useEffect(()=>{
    if (onExposePoseImporter) {
      onExposePoseImporter((mode?: 'dji' | 'wingtra') => {
        setPoseImportKind(mode ?? 'auto');
        poseFileRef.current?.click();
      });
    }
  }, [onExposePoseImporter]);

  // Auto-compute when imported poses arrive (poses-only mode)
  React.useEffect(()=>{
    if (autoGenerate || importedPoses.length === 0) {
      lastAutoComputedImportedPosesRef.current = null;
      return;
    }
    if (lastAutoComputedImportedPosesRef.current === importedPoses) return;
    lastAutoComputedImportedPosesRef.current = importedPoses;

    const generation = resetGenerationRef.current;
    const attempt = () => {
      if (!shouldRunAsyncGeneration(generation, resetGenerationRef.current)) return;
      const map = mapRef.current?.getMap?.();
      if (map?.isStyleLoaded?.()) compute({ generation });
      else scheduleGuardedTimeout(attempt, 200, generation);
    };
    attempt();
  }, [autoGenerate, compute, importedPoses, mapRef, scheduleGuardedTimeout]);

  const formatMetricValue = useCallback((metricKind: MetricKind, value: number, precision = 1) => {
    if (metricKind === 'density') return `${value.toFixed(precision)} pts/m²`;
    return `${(value * 100).toFixed(precision)} cm/px`;
  }, []);

  const metricSummaryValues = useCallback((stats: GSDStats) => {
    if (!(stats.count > 0)) {
      return { low: 0, mean: 0, high: 0 };
    }
    return {
      low: histogramQuantile(stats, CARD_SUMMARY_LOWER_QUANTILE),
      mean: stats.mean,
      high: histogramQuantile(stats, CARD_SUMMARY_UPPER_QUANTILE),
    };
  }, []);

  const metricValueColorClass = useCallback((metricKind: MetricKind, statKind: "min" | "mean" | "max") => {
    if (metricKind === "density") {
      if (statKind === "min") return "text-red-600";
      if (statKind === "mean") return "text-green-600";
      return "text-blue-600";
    }
    if (statKind === "mean") return "text-blue-600";
    return statKind === "min" ? "text-green-600" : "text-red-600";
  }, []);

  const metricLabels = useCallback((metricKind: MetricKind) => {
    if (metricKind === 'density') {
      return {
        title: 'Predicted Point Density',
        min: 'P5 density',
        mean: 'Mean density',
        max: 'P95 density',
        xAxis: 'Density (pts/m²)',
        tooltipLabel: 'Predicted density',
      };
    }
    return {
      title: 'GSD',
      min: 'P5 GSD',
      mean: 'Mean GSD',
      max: 'P95 GSD',
      xAxis: 'GSD (cm/px)',
      tooltipLabel: 'GSD',
    };
  }, []);

  const overallCards = useMemo(() => {
    const cards: Array<{ metricKind: MetricKind; stats: GSDStats }> = [];
    if (overallStats.gsd?.count) cards.push({ metricKind: 'gsd', stats: overallStats.gsd });
    if (overallStats.density?.count) cards.push({ metricKind: 'density', stats: overallStats.density });
    return cards;
  }, [overallStats]);

  const overlayLegends = useMemo(() => {
    const legends: Array<{
      metricKind: MetricKind;
      title: string;
      leftValue: number;
      rightValue: number;
    }> = [];

    const { gsd: gsdRange, density: densityRange } = resolveOverlayRanges(overallStats, {
      lockEnabled: overlayScaleLocked,
      lockedRanges: lockedOverlayRanges,
    });

    if (showGsd && gsdRange) {
      legends.push({
        metricKind: 'gsd',
        title: `GSD scale${overlayScaleLocked ? ' (locked)' : ' (p5-p95)'}`,
        leftValue: gsdRange.min,
        rightValue: gsdRange.max,
      });
    }

    if (showGsd && densityRange) {
      legends.push({
        metricKind: 'density',
        title: `Density scale${overlayScaleLocked ? ' (locked)' : ' (p5-p95)'}`,
        leftValue: densityRange.min,
        rightValue: densityRange.max,
      });
    }

    return legends;
  }, [lockedOverlayRanges, overallStats, overlayScaleLocked, resolveOverlayRanges, showGsd]);

  const handleOverlayScaleLockedChange = useCallback((checked: boolean) => {
    if (checked && !overlayScaleLockedRef.current) {
      const mergedLockedRanges = mergeLockedOverlayRanges(lockedOverlayRangesRef.current, overallStats);
      if (mergedLockedRanges.changed) {
        lockedOverlayRangesRef.current = mergedLockedRanges.ranges;
        setLockedOverlayRanges(mergedLockedRanges.ranges);
      }
    }
    overlayScaleLockedRef.current = checked;
    setOverlayScaleLocked(checked);
    redrawAnalysisOverlays(undefined, checked ? lockedOverlayRangesRef.current : undefined);
  }, [mergeLockedOverlayRanges, overallStats, redrawAnalysisOverlays]);

  React.useEffect(() => {
    redrawAnalysisOverlays();
  }, [redrawAnalysisOverlays, showGsd, showOverlap]);

  const displayParamsMap = getMergedParamsMap();
  const lidarPolygonIds = (mapRef.current?.getPolygonsWithIds?.() ?? [])
    .map((polygon) => polygon.id || 'unknown')
    .filter((polygonId) => isLidarPayload(polygonId, displayParamsMap));
  const lidarRangeValues: number[] = lidarPolygonIds.map((polygonId) => {
    const value = displayParamsMap[polygonId]?.maxLidarRangeM;
    return typeof value === 'number' && Number.isFinite(value) ? value : DEFAULT_LIDAR_MAX_RANGE_M;
  });
  const firstLidarRangeValue = lidarRangeValues[0] ?? DEFAULT_LIDAR_MAX_RANGE_M;
  const lidarRangeMixed = lidarRangeValues.length > 1 && lidarRangeValues.some((value) => Math.abs(value - firstLidarRangeValue) > 1e-6);
  const lidarRangeSharedValue = lidarRangeValues.length > 0 && !lidarRangeMixed ? String(firstLidarRangeValue) : '';
  const [bulkLidarRangeInput, setBulkLidarRangeInput] = useState<string>(String(DEFAULT_LIDAR_MAX_RANGE_M));

  const lidarPolygonIdsKey = lidarPolygonIds.join('|');

  React.useEffect(() => {
    if (lidarPolygonIds.length === 0) {
      setBulkLidarRangeInput(String(DEFAULT_LIDAR_MAX_RANGE_M));
      return;
    }
    setBulkLidarRangeInput(lidarRangeSharedValue);
  }, [lidarPolygonIdsKey, lidarRangeSharedValue]);

  const applyBulkLidarRange = useCallback((rawValue?: string) => {
    if (lidarPolygonIds.length === 0) return;
    const nextRange = parseFloat(rawValue ?? bulkLidarRangeInput);
    if (!(nextRange > 0)) {
      toast({
        variant: 'destructive',
        title: 'Invalid lidar range',
        description: 'Enter a max lidar range greater than 0 meters.',
      });
      setBulkLidarRangeInput(lidarRangeSharedValue);
      return;
    }

    const api = mapRef.current;
    const paramsMap = getMergedParamsMap();
    const updates = lidarPolygonIds
      .map((polygonId) => {
        const currentParams = paramsMap[polygonId];
        if (!currentParams) return null;
        return {
          polygonId,
          params: {
            ...currentParams,
            maxLidarRangeM: nextRange,
          },
        };
      })
      .filter((update): update is { polygonId: string; params: any } => update !== null);
    if (updates.length === 0) return;

    if (api?.applyPolygonParamsBatch) {
      api.applyPolygonParamsBatch(updates);
    } else {
      for (const update of updates) {
        api?.applyPolygonParams?.(update.polygonId, update.params);
      }
    }
    setBulkLidarRangeInput(String(nextRange));
  }, [bulkLidarRangeInput, getMergedParamsMap, lidarPolygonIds, lidarRangeSharedValue, mapRef]);

  return (
    <div className="backdrop-blur-md bg-white/95 rounded-md border p-3 space-y-3">
      <input
        ref={poseFileRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={handlePoseFileChange}
      />
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-900">Coverage Analysis</h3>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        <label className="text-xs col-span-1">
          <input type="checkbox" checked={showGsd} onChange={e=>handleShowAnalysisOverlayChange(e.target.checked)} className="mr-2" />
          <span className="font-medium">Show analysis overlay</span>
        </label>
        <label className="text-xs col-span-1">
          <input type="checkbox" checked={showFlightLines} onChange={e=>setShowFlightLines(e.target.checked)} className="mr-2" />
          <span className="font-medium">Show flight lines</span>
        </label>
      </div>

      <div className="space-y-2">
        {combinedPolygons.length === 0 ? (
          <Card>
            <CardContent className="py-6 text-center text-xs text-gray-500">
              No polygons analyzed yet.
            </CardContent>
          </Card>
        ) : (
          combinedPolygons.map(({ polygonId, analysis, stats }) => {
            const displayName = getPolygonDisplayName(polygonId);
            const overrideInfo = overrides?.[polygonId];
            const directionSource = overrideInfo?.source === 'user'
              ? 'Custom'
              : overrideInfo?.source === 'optimized'
                ? 'Optimized'
              : overrideInfo?.source === 'partition'
                ? 'Split'
              : overrideInfo?.source === 'wingtra'
                ? 'File'
                : 'Terrain';
            const directionDeg = (overrideInfo?.bearingDeg ?? analysis?.result?.contourDirDeg ?? 0).toFixed(1);
            const metricKind = stats?.metricKind ?? (isLidarPayload(polygonId, displayParamsMap) ? 'density' : 'gsd');
            const metricStats = stats?.stats;
            const labels = metricLabels(metricKind);
            const areaAcres = stats?.areaAcres ?? 0;
            const isSelected = activeSelectedId === polygonId;
            const isPoseArea = polygonId === '__POSES__';
            const isMergeMode = mergeState.mode === 'selecting';
            const isMergePrimary = isMergeMode && mergeState.primaryPolygonId === polygonId;
            const isPolygonOperationApplying = historyState.isApplyingOperation;
            const disableStructuralActions = isMergeMode || isPolygonOperationApplying;
            const canStartMerge = !isPoseArea && !disableStructuralActions && !!mapRef.current?.canStartPolygonMerge?.(polygonId);

            return (
              <Card
                key={polygonId}
                ref={(node) => {
                  if (node) {
                    itemRefs.current.set(polygonId, node);
                  } else {
                    itemRefs.current.delete(polygonId);
                  }
                }}
                className={`mt-2 transition-shadow border-l-4 ${isSelected ? 'border-l-blue-500 shadow-lg ring-1 ring-blue-400' : 'border-l-transparent hover:shadow-md'}`}
              >
                <CardContent className={`p-3 ${isSelected ? 'space-y-3' : ''}`}>
                  <button
                    type="button"
                    className="flex w-full items-center justify-between gap-3 text-left"
                    onClick={() => {
                      if (isMergeMode || isPolygonOperationApplying) return;
                      if (isSelected) {
                        setSelection(null);
                        return;
                      }
                      setSelection(polygonId);
                      highlightPolygon(polygonId);
                    }}
                  >
                    <div className="min-w-0 text-sm font-medium text-gray-900">{displayName}</div>
                    <div className={`grid shrink-0 items-center gap-x-1.5 ${isSelected ? 'grid-cols-[max-content_3.75rem_max-content]' : 'grid-cols-[max-content_3.75rem]'}`}>
                      <span className="text-[11px] font-medium text-blue-900 justify-self-end">Flight Direction</span>
                      <span className="w-[3.75rem] text-right font-mono tabular-nums text-sm font-bold text-blue-700">{directionDeg}°</span>
                      {isSelected && (
                        <Badge variant="outline" className="text-[10px] uppercase tracking-wide">{directionSource}</Badge>
                      )}
                    </div>
                  </button>

                  {isSelected && (
                    <>
                      <div className="flex flex-nowrap items-center gap-1 overflow-x-auto pb-1">
                        {!isPoseArea && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-6 shrink-0 whitespace-nowrap px-1.5 text-[10px]"
                            disabled={disableStructuralActions}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelection(polygonId);
                              onEditPolygonParams?.(polygonId);
                            }}
                            title="Edit payload settings for this area"
                          >
                            Edit Payload
                          </Button>
                        )}

                        {!isPoseArea && (
                          <Button
                            size="sm"
                            variant="secondary"
                            className="h-6 shrink-0 whitespace-nowrap border border-input bg-background px-1.5 text-[10px] hover:bg-accent hover:text-accent-foreground"
                            disabled={disableStructuralActions}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelection(polygonId);
                              highlightPolygon(polygonId);
                              mapRef.current?.editPolygonBoundary?.(polygonId);
                            }}
                            title="Edit area boundary on the map"
                          >
                            Edit Area
                          </Button>
                        )}

                        {!isPoseArea && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-6 shrink-0 whitespace-nowrap px-1.5 text-[10px]"
                            disabled={!!splittingPolygonIds[polygonId] || disableStructuralActions}
                            onClick={async (e) => {
                              e.stopPropagation();
                              setSelection(polygonId);
                              setSplittingPolygonIds((prev) => ({ ...prev, [polygonId]: true }));
                              const splitRunId = `${polygonId}:${++splitPerfSeqRef.current}`;
                              const startedAt = splitPerfNow();
                              splitPerfLog(splitRunId, 'auto split button clicked', {
                                polygonId,
                                payloadKind: isLidarPayload(polygonId, getMergedParamsMap()) ? 'lidar' : 'camera',
                              });
                              suppressAutoRunUntilRef.current = Date.now() + 5000;
                              try {
                                let result: { replaced: boolean; createdIds: string[] } | undefined;
                                let handledPostApply = false;
                                const api = mapRef.current;
                                if (api?.getTerrainPartitionSolutions) {
                                  const { solutions, defaultIndex } = await loadTerrainPartitionOptions(polygonId, {
                                    showEmptyToast: false,
                                    showErrorToast: false,
                                  }) ?? { solutions: [], defaultIndex: 0 };
                                  if (solutions.length > 0) {
                                    result = await applyTerrainPartitionOption(polygonId, defaultIndex, solutions[defaultIndex]);
                                    handledPostApply = !!result?.replaced;
                                  }
                                }
                                if (!result?.replaced) {
                                  result = await api?.autoSplitPolygonByTerrain?.(polygonId, { skipBackend: true });
                                }
                                splitPerfLog(splitRunId, 'auto split action finished', {
                                  totalMs: Math.round(splitPerfNow() - startedAt),
                                  handledPostApply,
                                  result,
                                });
                                if (result?.replaced && result.createdIds.length > 1) {
                                  if (!handledPostApply) {
                                    resetComputedAnalysisState();
                                    setSelection(result.createdIds[0] ?? null);
                                    rerunAnalysisForCreatedPolygons(result.createdIds);
                                  }
                                } else {
                                  toast({
                                    variant: 'destructive',
                                    title: 'No split created',
                                    description: 'No useful terrain-face split was found for this area with the current rules.',
                                  });
                                  suppressAutoRunUntilRef.current = 0;
                                }
                              } finally {
                                setSplittingPolygonIds((prev) => {
                                  if (!prev[polygonId]) return prev;
                                  const next = { ...prev };
                                  delete next[polygonId];
                                  return next;
                                });
                              }
                          }}
                            title="Auto split this area into a few terrain-aligned faces"
                          >
                            {!!splittingPolygonIds[polygonId] ? 'Auto Split' : 'Auto Split'}
                          </Button>
                        )}

                          <Button
                            size="sm"
                            variant="outline"
                            className="h-6 shrink-0 whitespace-nowrap px-1.5 text-[10px]"
                          disabled={isPoseArea || disableStructuralActions}
                          onClick={(e) => {
                            e.stopPropagation();
                            mapRef.current?.optimizePolygonDirection?.(polygonId);
                            scheduleGuardedTimeout(() => setSelection(polygonId), 0);
                          }}
                          title="Automatically choose the terrain-optimal direction"
                        >
                          Auto Direction
                        </Button>

                        {!isPoseArea && !isMergeMode && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-6 shrink-0 whitespace-nowrap border-blue-300 px-1.5 text-[10px] text-blue-700 hover:bg-blue-50 hover:text-blue-800"
                            disabled={!canStartMerge}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelection(polygonId);
                              highlightPolygon(polygonId);
                              mapRef.current?.startPolygonMerge?.(polygonId);
                            }}
                            title="Merge this autosplit area with touching autosplit neighbors"
                          >
                            Merge
                          </Button>
                        )}

                        {!isPoseArea && isMergePrimary && (
                          <>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-6 shrink-0 whitespace-nowrap border-blue-300 px-1.5 text-[10px] text-blue-700 hover:bg-blue-50 hover:text-blue-800"
                              disabled={isPolygonOperationApplying || !mergeState.canConfirm}
                              onClick={async (e) => {
                                e.stopPropagation();
                                const result = await mapRef.current?.confirmPolygonMerge?.();
                                if (result?.replaced) {
                                  resetComputedAnalysisState();
                                  setSelection(result.mergedPolygonId ?? null);
                                }
                              }}
                              title="Merge the selected autosplit polygons"
                            >
                              Merge {mergeState.selectedPolygonIds.length}
                            </Button>

                            <Button
                              size="sm"
                              variant="outline"
                              className="h-6 shrink-0 whitespace-nowrap px-1.5 text-[10px]"
                              disabled={isPolygonOperationApplying}
                              onClick={(e) => {
                                e.stopPropagation();
                                mapRef.current?.cancelPolygonMerge?.();
                                setSelection(polygonId);
                              }}
                              title="Cancel merge mode"
                            >
                              Cancel
                            </Button>
                          </>
                        )}

                        {!isPoseArea && (
                          <Button
                            size="sm"
                            variant="outline"
                            className="ml-auto h-6 shrink-0 whitespace-nowrap border-red-300 px-1.5 text-[10px] text-red-600 hover:bg-red-50 hover:text-red-700"
                            disabled={disableStructuralActions}
                            onClick={(e) => {
                              e.stopPropagation();
                              mapRef.current?.clearPolygon?.(polygonId);
                              scheduleGuardedTimeout(() => setSelection(null), 0);
                            }}
                            title="Delete polygon"
                          >
                            Delete
                          </Button>
                        )}
                      </div>

                      {isMergePrimary && (
                        <div className="rounded-md border border-blue-200 bg-blue-50 px-2 py-1.5 text-[11px] text-blue-800">
                          Select touching autosplit polygons on the map to add them to this merge preview.
                          {mergeState.warning ? ` ${mergeState.warning}` : ''}
                        </div>
                      )}

                      {metricStats ? (() => {
                        const displayStats = metricSummaryValues(metricStats);
                        return (
                        <div className="space-y-2">
                          <div className="grid grid-cols-4 gap-3 text-xs">
                            <div className="text-center">
                              <div className={`font-medium ${metricValueColorClass(metricKind, 'min')}`}>{formatMetricValue(metricKind, displayStats.low, metricKind === 'density' ? 0 : 1)}</div>
                              <div className="text-gray-500">{labels.min}</div>
                            </div>
                            <div className="text-center">
                              <div className={`font-medium ${metricValueColorClass(metricKind, 'mean')}`}>{formatMetricValue(metricKind, displayStats.mean, metricKind === 'density' ? 1 : 2)}</div>
                              <div className="text-gray-500">{labels.mean}</div>
                            </div>
                            <div className="text-center">
                              <div className={`font-medium ${metricValueColorClass(metricKind, 'max')}`}>{formatMetricValue(metricKind, displayStats.high, metricKind === 'density' ? 0 : 1)}</div>
                              <div className="text-gray-500">{labels.max}</div>
                            </div>
                            <div className="text-center">
                              <div className="font-medium text-gray-900">{areaAcres.toFixed(2)} acres</div>
                              <div className="text-gray-500">Area</div>
                            </div>
                          </div>
                        </div>
                        );
                      })() : (
                        <div className="text-xs text-gray-500">
                          {metricKind === 'density'
                            ? 'Point density analysis will appear after lidar flight lines are generated.'
                            : 'GSD analysis will appear after camera flight lines are generated.'}
                        </div>
                      )}
                    </>
                  )}
                </CardContent>
              </Card>
            );
          })
        )}
      </div>

      {overallCards.map(({ metricKind, stats }) => {
        const labels = metricLabels(metricKind);
        const displayStats = metricSummaryValues(stats);
        const totalAreaM2 = statsTotalAreaM2(stats);
        return (
          <Card className="mt-2" key={metricKind}>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Overall {labels.title} Analysis</CardTitle>
              <CardDescription className="text-xs">Cumulative {labels.title.toLowerCase()} statistics for {stats.count.toLocaleString()} pixels, with p5/p95 summary tails.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-3 gap-4 text-xs">
                <div className="text-center"><div className={`font-medium ${metricValueColorClass(metricKind, 'min')}`}>{formatMetricValue(metricKind, displayStats.low, metricKind === 'density' ? 0 : 1)}</div><div className="text-gray-500">{labels.min}</div></div>
                <div className="text-center"><div className={`font-medium ${metricValueColorClass(metricKind, 'mean')}`}>{formatMetricValue(metricKind, displayStats.mean, metricKind === 'density' ? 1 : 2)}</div><div className="text-gray-500">{labels.mean}</div></div>
                <div className="text-center"><div className={`font-medium ${metricValueColorClass(metricKind, 'max')}`}>{formatMetricValue(metricKind, displayStats.high, metricKind === 'density' ? 0 : 1)}</div><div className="text-gray-500">{labels.max}</div></div>
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={convertHistogramToArea(stats, metricKind).map(bin => ({ metric: metricKind === 'density' ? (bin.isZeroBucket ? '0' : bin.bin.toFixed(0)) : (bin.bin * 100).toFixed(1), metricLabel: metricKind === 'density' ? (bin.isZeroBucket ? 'Holes / 0 pts/m²' : `${bin.bin.toFixed(0)} pts/m²`) : `${(bin.bin * 100).toFixed(1)} cm/px`, areaM2: bin.areaM2, areaAcres: bin.areaM2 / ACRE_M2, isZeroBucket: !!bin.isZeroBucket }))}
                    margin={{ top: 4, right: 8, bottom: 18 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis
                      dataKey="metric"
                      tick={{ fontSize: 10 }}
                      height={28}
                      label={{ value: labels.xAxis, position: 'bottom', offset: 6, style: { fontSize: '10px' } }}
                    />
                    <YAxis tick={{ fontSize: 10 }} tickFormatter={(v:number)=> (v/ACRE_M2).toFixed(2)} label={{ value: 'Area (acres)', angle: -90, position: 'insideLeft', style: { fontSize: '10px' } }} />
                    <Tooltip
                      formatter={(value)=>{
                        const m2 = value as number;
                        const acres = m2 / ACRE_M2;
                        const areaPct = totalAreaM2 > 0 ? (m2 / totalAreaM2) * 100 : 0;
                        return [`${acres.toFixed(2)} acres (${areaPct.toFixed(1)}%)`, 'Area'];
                      }}
                      labelFormatter={(_, payload) => {
                        const first = payload?.[0]?.payload as { metricLabel?: string } | undefined;
                        return `${labels.tooltipLabel}: ${first?.metricLabel ?? ''}`;
                      }}
                      labelStyle={{ fontSize: '11px' }}
                      contentStyle={{ fontSize: '11px' }}
                    />
                    <Bar dataKey="areaM2" fill="#3b82f6" stroke="#1e40af" strokeWidth={0.5} radius={[1,1,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        );
      })}

      {overlayLegends.length > 0 && (
        <div className="rounded-md border border-gray-200 bg-gray-50/80 p-2 space-y-2">
          <div className="flex items-center justify-between gap-3 text-[11px] text-gray-500">
            <label className="flex items-center gap-2 text-gray-700">
              <input
                type="checkbox"
                className="h-3.5 w-3.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                checked={overlayScaleLocked}
                onChange={(event) => handleOverlayScaleLockedChange(event.target.checked)}
              />
              <span className="font-medium">Lock color scale</span>
            </label>
            <span>{overlayScaleLocked ? 'Fixed from first evaluation' : 'Auto-rescales to current run'}</span>
          </div>
          {overlayLegends.map((legend) => (
            <div key={legend.metricKind} className="space-y-1">
              <div className="flex items-center justify-between text-[11px] text-gray-600">
                <span>{legend.title}</span>
                <span>{overlayScaleLocked ? 'Locked' : 'Current run'}</span>
              </div>
              <div
                className="h-2 rounded-sm border border-gray-200"
                style={{ background: legend.metricKind === 'density' ? HEATMAP_GRADIENT_DENSITY : HEATMAP_GRADIENT_GSD }}
              />
              <div className="flex items-center justify-between text-[11px] text-gray-600">
                <span>{formatMetricValue(legend.metricKind, legend.leftValue)}</span>
                <span>{formatMetricValue(legend.metricKind, legend.rightValue)}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 gap-2">
        <div className="overflow-hidden rounded-md border border-gray-200 bg-gray-50/80">
          <button
            type="button"
            onClick={() => setShowFlightParameters((current) => !current)}
            className="flex w-full items-center justify-between px-3 py-2 text-left"
          >
            <span className="text-xs font-medium text-gray-700">Flight Parameters</span>
            <span className="text-[11px] text-gray-500">{showFlightParameters ? 'Hide' : 'Show'}</span>
          </button>
          {showFlightParameters && (
            <div className="space-y-2 border-t border-gray-200 px-3 py-2">
              <label className="text-xs text-gray-600 block">
                Altitude mode
                <select
                  className="w-full border rounded px-2 py-1 text-xs mt-1"
                  value={altitudeModeUI}
                  onChange={(e)=>{
                    const m = (e.target.value as 'legacy'|'min-clearance');
                    setAltitudeModeUI(m);
                    const api = mapRef.current as any;
                    if (api?.setAltitudeMode) api.setAltitudeMode(m);
                    scheduleGuardedTimeout(() => { compute(); }, 0);
                  }}
                >
                  <option value="legacy">Legacy (highest ground + AGL)</option>
                  <option value="min-clearance">Min-clearance (lowest + AGL; enforce clearance)</option>
                </select>
              </label>
              <label className="text-xs text-gray-600 block">Min clearance (m)
                <input
                  className="w-full border rounded px-2 py-1 text-xs"
                  type="number"
                  min={0}
                  value={minClearanceUI}
                  onChange={(e)=>{
                    const v = Math.max(0, parseFloat(e.target.value||'60'));
                    setMinClearanceUI(v);
                    const api = mapRef.current as any;
                    if (api?.setMinClearance) api.setMinClearance(v);
                    scheduleGuardedTimeout(() => { compute(); }, 0);
                  }}
                />
              </label>
              <label className="text-xs text-gray-600 block">Turn extend (m)
                <input
                  className="w-full border rounded px-2 py-1 text-xs"
                  type="number"
                  min={0}
                  value={turnExtendUI}
                  onChange={(e)=>{
                    const v = Math.max(0, parseFloat(e.target.value||'96'));
                    setTurnExtendUI(v);
                    const api = mapRef.current as any;
                    if (api?.setTurnExtend) api.setTurnExtend(v);
                    scheduleGuardedTimeout(() => { compute(); }, 0);
                  }}
                />
              </label>
              {lidarPolygonIds.length > 0 && (
                <label className="text-xs text-gray-600 block">
                  Max lidar range for all areas (m)
                  <input
                    className="w-full border rounded px-2 py-1 text-xs"
                    type="number"
                    min={1}
                    step={1}
                    value={bulkLidarRangeInput}
                    placeholder={lidarRangeMixed ? 'Mixed' : undefined}
                    onChange={(e) => setBulkLidarRangeInput(e.target.value)}
                    onBlur={(e) => applyBulkLidarRange(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        applyBulkLidarRange((e.target as HTMLInputElement).value);
                      }
                    }}
                  />
                  <span className="block mt-1 text-[11px] text-gray-500">
                    {lidarRangeMixed
                      ? `Different values are set across ${lidarPolygonIds.length} lidar areas. Enter one value and press Enter or click away to apply it to all.`
                      : `Applies to all ${lidarPolygonIds.length} lidar area${lidarPolygonIds.length === 1 ? '' : 's'}.`}
                  </span>
                </label>
              )}
              <label className="text-xs text-gray-600 block">Max tilt (deg)<input className="w-full border rounded px-2 py-1 text-xs" type="number" min={0} max={90} value={maxTiltDeg} onChange={e=>setMaxTiltDeg(Math.max(0, Math.min(90, parseFloat(e.target.value||'10'))))} /></label>
              <label className="text-xs text-gray-600 block">
                Min overlap for GSD (images)
                <input
                  className="w-full border rounded px-2 py-1 text-xs"
                  type="number"
                  min={1}
                  max={10}
                  value={minOverlapForGsd}
                  onChange={(e)=>{
                    const v = Math.max(1, Math.min(10, Math.round(parseFloat(e.target.value || '3'))));
                    minOverlapForGsdRef.current = v;
                    setMinOverlapForGsd(v);
                    scheduleGuardedTimeout(() => { compute(); }, 0);
                  }}
                />
              </label>
              {autoGenerate && <div className="text-xs text-gray-500">{parsePosesMeters()?.length || 0} poses generated</div>}
            </div>
          )}
        </div>
      </div>

      <div className="flex gap-2 items-center">
        <button onClick={() => compute()} disabled={running} className="h-8 px-2 rounded bg-blue-600 text-white text-xs disabled:opacity-50">{running ? 'Computing…' : 'Recompute Analysis'}</button>
      </div>

      <p className="text-[11px] text-gray-500">Automatic coverage analysis runs when polygons are created or flight parameters change.</p>
    </div>
  );
}

export default React.memo(OverlapGSDPanel);
