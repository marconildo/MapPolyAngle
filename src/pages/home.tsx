import React, { Suspense, lazy, useState, useRef, useCallback, useMemo } from 'react';
import type { BearingOverride, MapFlightDirectionAPI } from '@/components/MapFlightDirection/api';
import type { PolygonAnalysisResult } from '@/components/MapFlightDirection/types';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from '@/components/ui/drawer';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { useIsMobile } from '@/hooks/use-mobile';
import { Map, Trash2, AlertCircle, Upload, Download, SlidersHorizontal } from 'lucide-react';
import type { PolygonParams } from '@/components/MapFlightDirection/types';
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator } from '@/components/ui/dropdown-menu';
import { toast } from "@/hooks/use-toast";
import {
  activateRememberedTerrainSource,
  getTerrainDemUrlTemplateForCurrentSource,
  getTerrainSourceState,
  initializeTerrainSourceState,
  loadTerrainSourceFromFile,
  subscribeTerrainSource,
} from '@/terrain/terrainSource';
import {
  clearDsmFootprintPolygon,
  setDsmFootprintPolygon,
  setImageryOverlayOnMap,
} from '@/components/MapFlightDirection/utils/mapbox-layers';
import {
  clearActiveImageryOverlay,
  getActiveImageryOverlay,
  getImageryOverlayState,
  loadImageryOverlayFromFile,
  subscribeImageryOverlay,
} from '@/terrain/imageryOverlay';
import type { GeoTiffSourceDescriptor, ImageryOverlayState, TerrainSourceState } from '@/terrain/types';
import { createHomeClearAllState } from '@/state/clearAllState';

const MapFlightDirection = lazy(async () => {
  const mod = await import('@/components/MapFlightDirection');
  return { default: mod.MapFlightDirection };
});

const OverlapGSDPanel = lazy(() => import('@/components/OverlapGSDPanel'));
const PolygonParamsDialog = lazy(() => import('@/components/PolygonParamsDialog'));

function DeferredPanelFallback() {
  return (
    <Card className="backdrop-blur-md bg-white/95">
      <CardContent className="p-6">
        <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
          <LoadingSpinner size="sm" />
          <span>Loading analysis tools…</span>
        </div>
      </CardContent>
    </Card>
  );
}

function DeferredMapFallback() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
      <div className="flex items-center gap-3 rounded-lg bg-white/90 px-4 py-3 text-sm text-gray-700 shadow-sm">
        <LoadingSpinner size="sm" />
        <span>Loading map workspace…</span>
      </div>
    </div>
  );
}

function downloadFlightplanBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  window.setTimeout(() => {
    URL.revokeObjectURL(url);
    anchor.remove();
  }, 1000);
}

function stripFlightplanExtension(filename: string | undefined): string {
  if (!filename) return 'exported';
  return filename.replace(/\.flightplan$/i, '') || 'exported';
}

function normalizeFlightplanFilename(name: string | undefined): string {
  const base = (name ?? '')
    .trim()
    .replace(/\.flightplan$/i, '')
    .replace(/[\\/:"*?<>|]+/g, '-')
    .replace(/\s+/g, ' ');
  return `${base.length > 0 ? base : 'exported'}.flightplan`;
}

export default function Home() {
  const isMobile = useIsMobile();
  const imageryOverlayImportEnabled = false;
  const mapRef = useRef<MapFlightDirectionAPI>(null);
  const dsmInputRef = useRef<HTMLInputElement>(null);
  const imageryInputRef = useRef<HTMLInputElement>(null);
  const previousTerrainSourceKeyRef = useRef<string | undefined>(undefined);
  const pendingCoverageAutoRunRef = useRef<number | null>(null);

  const [polygonResults, setPolygonResults] = useState<PolygonAnalysisResult[]>([]);
  const [analyzingPolygons, setAnalyzingPolygons] = useState<Set<string>>(new Set());

  // NEW: per‑polygon flight parameters
  const [paramsByPolygon, setParamsByPolygon] = useState<Record<string, PolygonParams>>({});
  const [paramsDialog, setParamsDialog] = useState<{ open: boolean; polygonId: string | null }>({ open: false, polygonId: null });

  // Imported/or override state (queried from Map component)
  const [importedOriginals, setImportedOriginals] = useState<Record<string, { bearingDeg: number; lineSpacingM: number }>>({});
  const [overrides, setOverrides] = useState<Record<string, BearingOverride>>({});
  const [selectedPolygonId, setSelectedPolygonId] = useState<string | null>(null);
  const [terrainSourceState, setTerrainSourceState] = useState<TerrainSourceState>(() => getTerrainSourceState());
  const [imageryOverlayState, setImageryOverlayState] = useState<ImageryOverlayState>(() => getImageryOverlayState());
  const [showTerrainSource, setShowTerrainSource] = useState(false);
  const [mobileAnalysisOpen, setMobileAnalysisOpen] = useState(false);
  const [importUiState, setImportUiState] = useState<null | {
    operationId: number;
    kind: 'terrain' | 'imagery';
    phase: 'uploading' | 'applying';
    targetKey: string | null;
  }>(null);
  const [exportNameDialogOpen, setExportNameDialogOpen] = useState(false);
  const [exportNameDraft, setExportNameDraft] = useState('exported');
  // NEW: track imported pose count
  const [importedPoseCount, setImportedPoseCount] = useState(0);
  const [clearAllEpoch, setClearAllEpoch] = useState(0);

  // Auto-run GSD analysis when flight lines are updated (already wired)
  const autoRunGSDRef = useRef<((opts?: { polygonId?: string; reason?: 'lines'|'spacing'|'alt'|'manual' }) => void) | null>(null);
  const clearGSDRef = useRef<(() => void) | null>(null);
  // NEW: ref to open pose JSON importer (DJI or Wingtra) inside OverlapGSDPanel
  const openDJIImporterRef = useRef<((mode?: 'dji' | 'wingtra') => void) | null>(null);
  const importOperationIdRef = useRef(0);
  const lastReadyTerrainKeyRef = useRef<string | null>(null);

  const scheduleTerrainSourceReapplyAfterViewportSettle = useCallback((targetKey: string) => {
    const map = mapRef.current?.getMap?.();
    if (!map) return;

    const reapply = () => {
      const currentState = getTerrainSourceState();
      const currentKey = `${currentState.source.mode}:${currentState.source.datasetId ?? ''}`;
      if (currentKey !== targetKey) {
        console.debug('[terrain-source] skipping post-fit terrain reapply because terrain source changed', {
          targetKey,
          currentKey,
        });
        return;
      }
      console.log('[terrain-source] reapplying terrain source after viewport settle', {
        targetKey,
      });
      mapRef.current?.setTerrainDemSource(getTerrainDemUrlTemplateForCurrentSource());
    };

    window.setTimeout(() => {
      if (map.isMoving()) {
        map.once('idle', reapply);
        return;
      }
      reapply();
    }, 0);
  }, []);

  const cancelPendingCoverageAutoRun = useCallback(() => {
    if (pendingCoverageAutoRunRef.current !== null) {
      window.clearTimeout(pendingCoverageAutoRunRef.current);
      pendingCoverageAutoRunRef.current = null;
    }
  }, []);

  const sampleStep = 1;
  const mapboxToken = useMemo(() =>
    import.meta.env.VITE_MAPBOX_TOKEN ||
    import.meta.env.VITE_MAPBOX_ACCESS_TOKEN ||
    "", []
  );

  // Use useMemo to prevent center from being recreated on every render
  const center = useMemo<[number, number]>(() => [8.54, 47.37], []);
  const initialZoom = useMemo(() => 13, []);

  // Expose mapRef to window for console testing
  React.useEffect(() => {
    const timer = window.setTimeout(() => {
      if (mapRef.current) {
        (window as any).mapApi = mapRef.current;
        console.log('🔧 Map API available as window.mapApi for console testing');
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  React.useEffect(() => {
    const unsubscribeTerrain = subscribeTerrainSource(() => {
      setTerrainSourceState(getTerrainSourceState());
    });
    const unsubscribeImagery = subscribeImageryOverlay(() => {
      setImageryOverlayState(getImageryOverlayState());
    });
    void initializeTerrainSourceState().catch((error) => {
      toast({
        variant: 'destructive',
        title: 'DSM library load failed',
        description: error instanceof Error ? error.message : String(error),
      });
    });
    return () => {
      cancelPendingCoverageAutoRun();
      unsubscribeTerrain();
      unsubscribeImagery();
    };
  }, [cancelPendingCoverageAutoRun]);

  React.useEffect(() => {
    const map = mapRef.current?.getMap?.();
    if (!map) return;

    const syncOverlay = () => {
      if (terrainSourceState.descriptor) {
        setDsmFootprintPolygon(map, terrainSourceState.descriptor.id, terrainSourceState.descriptor.footprintRingLngLat);
      } else {
        clearDsmFootprintPolygon(map);
      }
    };

    if (map.isStyleLoaded()) {
      syncOverlay();
      return;
    }

    map.once('load', syncOverlay);
    return () => {
      map.off('load', syncOverlay);
    };
  }, [terrainSourceState.descriptor]);

  React.useEffect(() => {
    const map = mapRef.current?.getMap?.();
    if (!map) return;

    const syncOverlay = () => {
      const activeOverlay = getActiveImageryOverlay();
      setImageryOverlayOnMap(
        map,
        activeOverlay
          ? {
              url: activeOverlay.imageUrl,
              coordinates: activeOverlay.coordinates,
            }
          : null,
      );
    };

    if (map.isStyleLoaded()) {
      syncOverlay();
      return;
    }

    map.once('load', syncOverlay);
    return () => {
      map.off('load', syncOverlay);
    };
  }, [imageryOverlayState.descriptor]);

  React.useEffect(() => {
    if (terrainSourceState.isLoading) return;
    const currentKey = `${terrainSourceState.source.mode}:${terrainSourceState.source.datasetId ?? ''}`;
    const previousKey = previousTerrainSourceKeyRef.current;
    previousTerrainSourceKeyRef.current = currentKey;
    if (previousKey === undefined || previousKey === currentKey) return;

    clearGSDRef.current?.();
    mapRef.current?.refreshTerrainForAllPolygons?.();
  }, [terrainSourceState.isLoading, terrainSourceState.source.datasetId, terrainSourceState.source.mode]);

  React.useEffect(() => {
    if (!importUiState || importUiState.kind !== 'terrain' || importUiState.phase !== 'applying') {
      return;
    }
    const { operationId, targetKey } = importUiState;
    const timeoutId = window.setTimeout(() => {
      setImportUiState((current) => {
        if (!current || current.operationId !== operationId || current.phase !== 'applying') {
          return current;
        }
        console.warn('[terrain-source] clearing stuck terrain apply overlay after timeout', {
          operationId,
          targetKey,
          currentTerrainKey: `${terrainSourceState.source.mode}:${terrainSourceState.source.datasetId ?? ''}`,
          lastReadyTerrainKey: lastReadyTerrainKeyRef.current,
        });
        return null;
      });
    }, 12000);
    return () => window.clearTimeout(timeoutId);
  }, [importUiState, terrainSourceState.source.datasetId, terrainSourceState.source.mode]);

  // Memoize handlers to prevent unnecessary re-renders
  const handleAnalysisStart = useCallback((polygonId: string) => {
    setAnalyzingPolygons(prev => new Set(prev).add(polygonId));
  }, []);

  const handleAnalysisComplete = useCallback((results: PolygonAnalysisResult[]) => {
    console.log('[home] handleAnalysisComplete', {
      resultIds: results.map((result) => result.polygonId),
    });
    setPolygonResults(results);
    setAnalyzingPolygons(new Set()); // Clear all analyzing states
    const api = mapRef.current;
    if (api) {
      const imported = api.getImportedOriginals?.() ?? {};
      const overrides = api.getBearingOverrides?.() ?? {};
      setImportedOriginals(imported);
      setOverrides(overrides);
      const mapParams = api.getPerPolygonParams?.() ?? {};
      setParamsByPolygon(mapParams as any);
      console.log('[home] synced map state after analysis complete', {
        importedOriginalIds: Object.keys(imported),
        overrideIds: Object.keys(overrides),
        polygonParamIds: Object.keys(mapParams),
      });
    }
  }, []);

  const handleError = useCallback((error: string, polygonId?: string) => {
    if (polygonId) {
      setAnalyzingPolygons(prev => {
        const newSet = new Set(prev);
        newSet.delete(polygonId);
        return newSet;
      });
    }
    toast({
      variant: 'destructive',
      title: 'Action failed',
      description: error,
    });
  }, []);

  const handleRequestParams = useCallback((polygonId: string) => {
    setParamsDialog({ open: true, polygonId });
  }, []);

  const handleEditPolygonParams = useCallback((polygonId: string) => {
    setSelectedPolygonId(polygonId);
    setParamsDialog({ open: true, polygonId });
  }, []);

  const handleApplyParams = useCallback((params: PolygonParams) => {
    const polygonId = paramsDialog.polygonId!;
    mapRef.current?.applyPolygonParams?.(polygonId, params);
    const updated = mapRef.current?.getPerPolygonParams?.() || {};
    setParamsByPolygon(updated as any);
    setParamsDialog({ open: false, polygonId: null });
  }, [paramsDialog.polygonId]);

  const handleCloseParams = useCallback(() => {
    setParamsDialog({ open: false, polygonId: null });
  }, []);

  const handleFlightLinesUpdated = useCallback((which: string | '__all__') => {
    const api = mapRef.current;
    const polygons = api?.getPolygonsWithIds?.() ?? [];

    if (api) {
      setImportedOriginals(api.getImportedOriginals?.() ?? {});
      setOverrides(api.getBearingOverrides?.() ?? {});
      const mapParams = api.getPerPolygonParams?.() ?? {};
      setParamsByPolygon(mapParams as any);
    }

    if (polygons.length === 0 && importedPoseCount === 0) {
      cancelPendingCoverageAutoRun();
      clearGSDRef.current?.();
      setSelectedPolygonId(null);
      return;
    }

    cancelPendingCoverageAutoRun();
    pendingCoverageAutoRunRef.current = window.setTimeout(() => {
      pendingCoverageAutoRunRef.current = null;
      const latestApi = mapRef.current;
      const latestPolygons = latestApi?.getPolygonsWithIds?.() ?? [];
      if (latestPolygons.length === 0 && importedPoseCount === 0) return;
      if (autoRunGSDRef.current) {
        if (which === '__all__') autoRunGSDRef.current({ reason: 'spacing' });
        else autoRunGSDRef.current({ polygonId: which, reason: 'lines' });
      }
    }, 0);
  }, [cancelPendingCoverageAutoRun, importedPoseCount]);

  // Handler to receive the auto-run function from OverlapGSDPanel
  const handleAutoRunReceived = useCallback((autoRunFn: (opts?: { polygonId?: string; reason?: 'lines'|'spacing'|'alt'|'manual' }) => void) => {
    autoRunGSDRef.current = autoRunFn;
    // Don't call immediately—only when MapFlightDirection tells us something changed
  }, []);

  // Handler to receive the clear function from OverlapGSDPanel
  const handleClearReceived = useCallback((clearFn: () => void) => {
    clearGSDRef.current = clearFn;
  }, []);

  // Also refresh overrides when results change
  React.useEffect(() => {
    if (mapRef.current) {
      setImportedOriginals(mapRef.current.getImportedOriginals?.() ?? {});
      setOverrides(mapRef.current.getBearingOverrides?.() ?? {});
      const mapParams = mapRef.current.getPerPolygonParams?.() ?? {};
      setParamsByPolygon(mapParams as any);
    }
  }, [polygonResults.length]);

  const clearAllDrawings = useCallback(() => {
    cancelPendingCoverageAutoRun();
    const clearedState = createHomeClearAllState();
    setClearAllEpoch((prev) => prev + 1);
    setPolygonResults(clearedState.polygonResults as PolygonAnalysisResult[]);
    setAnalyzingPolygons(clearedState.analyzingPolygons);
    setParamsByPolygon(clearedState.paramsByPolygon as Record<string, PolygonParams>);
    setParamsDialog(clearedState.paramsDialog);
    setImportedOriginals(clearedState.importedOriginals as Record<string, { bearingDeg: number; lineSpacingM: number }>);
    setOverrides(clearedState.overrides as Record<string, BearingOverride>);
    setImportedPoseCount(clearedState.importedPoseCount);
    setSelectedPolygonId(clearedState.selectedPolygonId);
  }, [cancelPendingCoverageAutoRun]);

  const fitMapToGeoTiffDescriptor = useCallback((descriptor: Pick<GeoTiffSourceDescriptor, 'footprintLngLat'>) => {
    const map = mapRef.current?.getMap?.();
    if (!descriptor || !map) return;
    map.fitBounds(
      [
        [descriptor.footprintLngLat.minLng, descriptor.footprintLngLat.minLat],
        [descriptor.footprintLngLat.maxLng, descriptor.footprintLngLat.maxLat],
      ],
      { padding: 60, duration: 900, maxZoom: 18 }
    );
  }, []);

  const zoomToDsm = useCallback(() => {
    if (!terrainSourceState.descriptor) return;
    fitMapToGeoTiffDescriptor(terrainSourceState.descriptor);
  }, [terrainSourceState.descriptor, fitMapToGeoTiffDescriptor]);

  const zoomToRememberedDsm = useCallback(() => {
    if (!terrainSourceState.rememberedDescriptor) return;
    fitMapToGeoTiffDescriptor(terrainSourceState.rememberedDescriptor);
  }, [fitMapToGeoTiffDescriptor, terrainSourceState.rememberedDescriptor]);

  const zoomToImageryOverlay = useCallback(() => {
    if (!imageryOverlayState.descriptor) return;
    fitMapToGeoTiffDescriptor(imageryOverlayState.descriptor);
  }, [fitMapToGeoTiffDescriptor, imageryOverlayState.descriptor]);

  const handleOpenDsmPicker = useCallback(() => {
    dsmInputRef.current?.click();
  }, []);

  const handleOpenImageryPicker = useCallback(() => {
    imageryInputRef.current?.click();
  }, []);

  const handleDsmFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const operationId = importOperationIdRef.current + 1;
    importOperationIdRef.current = operationId;
    const priorTerrainKey = `${terrainSourceState.source.mode}:${terrainSourceState.source.datasetId ?? ''}`;
    setImportUiState({ operationId, kind: 'terrain', phase: 'uploading', targetKey: null });
    try {
      const descriptor = await loadTerrainSourceFromFile(file, {
        onProgressPhase: (phase) => {
          if (phase === 'validating') return;
          setImportUiState((current) => (
            current?.operationId === operationId
              ? { operationId, kind: 'terrain', phase, targetKey: null }
              : current
          ));
        },
      });
      const targetKey = `blended:${descriptor.id}`;
      const shouldWaitForApply = priorTerrainKey !== targetKey && lastReadyTerrainKeyRef.current !== targetKey;
      setImportUiState((current) => {
        if (!current || current.operationId !== operationId) return current;
        return shouldWaitForApply ? { operationId, kind: 'terrain', phase: 'applying', targetKey } : null;
      });
      if (shouldWaitForApply) {
        mapRef.current?.setTerrainDemSource(getTerrainDemUrlTemplateForCurrentSource());
      }
      fitMapToGeoTiffDescriptor(descriptor);
      setShowTerrainSource(true);
      scheduleTerrainSourceReapplyAfterViewportSettle(targetKey);
      console.log('[terrain-source] DSM upload/import completed; waiting for map terrain apply', {
        operationId,
        descriptorId: descriptor.id,
        priorTerrainKey,
        targetKey,
        lastReadyTerrainKey: lastReadyTerrainKeyRef.current,
      });
    } catch (error) {
      setImportUiState((current) => (current?.operationId === operationId ? null : current));
      toast({
        variant: 'destructive',
        title: 'Terrain source load failed',
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      if (dsmInputRef.current) dsmInputRef.current.value = '';
    }
  }, [
    fitMapToGeoTiffDescriptor,
    scheduleTerrainSourceReapplyAfterViewportSettle,
    terrainSourceState.source.datasetId,
    terrainSourceState.source.mode,
  ]);

  const handleApplyRememberedTerrainSource = useCallback(async () => {
    const rememberedDescriptor = terrainSourceState.rememberedDescriptor;
    if (!rememberedDescriptor) return;
    const operationId = importOperationIdRef.current + 1;
    importOperationIdRef.current = operationId;
    const priorTerrainKey = `${terrainSourceState.source.mode}:${terrainSourceState.source.datasetId ?? ''}`;
    try {
      const descriptor = await activateRememberedTerrainSource();
      const targetKey = `blended:${descriptor.id}`;
      const shouldWaitForApply = priorTerrainKey !== targetKey && lastReadyTerrainKeyRef.current !== targetKey;
      setImportUiState(
        shouldWaitForApply
          ? { operationId, kind: 'terrain', phase: 'applying', targetKey }
          : null,
      );
      if (shouldWaitForApply) {
        mapRef.current?.setTerrainDemSource(getTerrainDemUrlTemplateForCurrentSource());
      }
      fitMapToGeoTiffDescriptor(descriptor);
      setShowTerrainSource(true);
      scheduleTerrainSourceReapplyAfterViewportSettle(targetKey);
      console.log('[terrain-source] remembered DSM activated; waiting for map terrain apply', {
        operationId,
        descriptorId: descriptor.id,
        priorTerrainKey,
        targetKey,
        lastReadyTerrainKey: lastReadyTerrainKeyRef.current,
      });
    } catch (error) {
      setImportUiState(null);
      toast({
        variant: 'destructive',
        title: 'Terrain source load failed',
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  }, [
    fitMapToGeoTiffDescriptor,
    scheduleTerrainSourceReapplyAfterViewportSettle,
    terrainSourceState.rememberedDescriptor,
    terrainSourceState.source.datasetId,
    terrainSourceState.source.mode,
  ]);

  const handleImageryFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const operationId = importOperationIdRef.current + 1;
    importOperationIdRef.current = operationId;
    setImportUiState({ operationId, kind: 'imagery', phase: 'uploading', targetKey: null });
    try {
      const descriptor = await loadImageryOverlayFromFile(file);
      fitMapToGeoTiffDescriptor(descriptor);
      setShowTerrainSource(true);
      setImportUiState((current) => (current?.operationId === operationId ? null : current));
    } catch (error) {
      setImportUiState((current) => (current?.operationId === operationId ? null : current));
      toast({
        variant: 'destructive',
        title: 'Imagery overlay load failed',
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      if (imageryInputRef.current) imageryInputRef.current.value = '';
    }
  }, [fitMapToGeoTiffDescriptor]);

  const formatBytes = useCallback((bytes: number) => {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIdx = 0;
    while (value >= 1024 && unitIdx < units.length - 1) {
      value /= 1024;
      unitIdx += 1;
    }
    return `${value >= 100 || unitIdx === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIdx]}`;
  }, []);

  const handleExportWingtra = useCallback(() => {
    const api = mapRef.current;
    if (!api?.exportWingtraFlightPlan) return;

    const suggestedName = stripFlightplanExtension(api.getLastImportedFlightplanName?.());
    setExportNameDraft(suggestedName);
    setExportNameDialogOpen(true);
  }, []);

  const handleConfirmWingtraExport = useCallback(() => {
    const api = mapRef.current;
    if (!api?.exportWingtraFlightPlan) return;

    try {
      const { blob } = api.exportWingtraFlightPlan();
      downloadFlightplanBlob(blob, normalizeFlightplanFilename(exportNameDraft));
      setExportNameDialogOpen(false);
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'Export failed',
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }, [exportNameDraft]);

  if (!mapboxToken) {
    return (
      <div className="min-h-screen w-full flex items-center justify-center bg-gray-50">
        <Card className="w-full max-w-md mx-4">
          <CardContent className="pt-6">
            <div className="flex mb-4 gap-2">
              <AlertCircle className="h-8 w-8 text-red-500" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Missing Mapbox Token</h1>
                <p className="text-sm text-gray-600 mt-2">
                  Please set your Mapbox access token in the environment variables:
                </p>
                <code className="block bg-gray-100 p-2 rounded text-xs mt-2">
                  VITE_MAPBOX_TOKEN=your_token_here
                </code>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const isAnalyzing = analyzingPolygons.size > 0;
  const hasResults = polygonResults.length > 0;
  const hasImportedPolygons = Object.keys(importedOriginals).length > 0;
  const hasPolygonsToAnalyze = hasResults || hasImportedPolygons;
  const panelEnabled = hasPolygonsToAnalyze || importedPoseCount>0; // enable if poses-only
  const terrainDescriptor = terrainSourceState.descriptor;
  const rememberedTerrainDescriptor = terrainSourceState.rememberedDescriptor;
  const imageryDescriptor = imageryOverlayState.descriptor;
  const terrainAnalysisDisabled = terrainSourceState.isLoading || importUiState !== null;
  const importUiMessage = importUiState?.kind === 'imagery'
    ? 'Loading imagery overlay…'
    : importUiState?.phase === 'uploading'
      ? 'Loading DSM…'
      : 'Applying DSM terrain to map…';
  const importUiSubmessage = importUiState?.kind === 'imagery'
    ? 'Please wait while the ortho overlay is prepared.'
    : importUiState?.phase === 'uploading'
      ? 'Please wait while the source model is prepared.'
      : 'Please wait while the map refreshes with the new elevation.';
  const collapsedTerrainSourceTitle = terrainDescriptor
    ? `Terrain source: ${terrainDescriptor.name.length > 9 ? `${terrainDescriptor.name.slice(0, 9)}...` : terrainDescriptor.name}`
    : rememberedTerrainDescriptor
      ? `Terrain source: ${rememberedTerrainDescriptor.name.length > 9 ? `${rememberedTerrainDescriptor.name.slice(0, 9)}...` : rememberedTerrainDescriptor.name}`
      : imageryOverlayImportEnabled && imageryDescriptor
      ? `Imagery overlay: ${imageryDescriptor.name.length > 9 ? `${imageryDescriptor.name.slice(0, 9)}...` : imageryDescriptor.name}`
      : 'Terrain source';
  const headerButtonClassName = isMobile
    ? 'h-9 flex-1 min-w-[5.75rem] justify-center px-3'
    : 'h-8 px-2 whitespace-nowrap';
  const mobileAnalysisSummary = terrainAnalysisDisabled
    ? importUiMessage
    : panelEnabled
      ? 'Open terrain, overlap & GSD controls'
      : 'Import or draw polygons to start analysis';
  const analysisPanelBody = (
    <>
      <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50/70">
        <button
          type="button"
          onClick={() => setShowTerrainSource((current) => !current)}
          className="flex w-full items-center justify-between px-3 py-2 text-left"
        >
          <span className="text-xs font-medium text-slate-900">{showTerrainSource ? 'Terrain source' : collapsedTerrainSourceTitle}</span>
          <span className="text-[11px] text-slate-500">{showTerrainSource ? 'Hide' : 'Show'}</span>
        </button>
        {showTerrainSource && (
          <div className="space-y-3 border-t border-slate-200 px-3 py-3">
            <div className="space-y-2">
              <div className="text-[11px] font-medium text-slate-700">Terrain source</div>
              {terrainDescriptor ? (
                <div className="space-y-1 text-[11px] text-slate-600">
                  <div className="font-medium text-slate-800">{terrainDescriptor.name}</div>
                  <div>
                    {terrainDescriptor.width.toLocaleString()} x {terrainDescriptor.height.toLocaleString()} px · {formatBytes(terrainDescriptor.fileSizeBytes)}
                  </div>
                  <div>{terrainDescriptor.sourceCrsLabel}</div>
                  {(terrainDescriptor.nativeResolutionXM || terrainDescriptor.nativeResolutionYM) && (
                    <div>
                      Native resolution: {(terrainDescriptor.nativeResolutionXM ?? 0).toFixed(2)} m × {(terrainDescriptor.nativeResolutionYM ?? 0).toFixed(2)} m
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-[11px] text-slate-500">
                  Import a GeoTIFF terrain source to use custom elevation.
                </div>
              )}
              {terrainSourceState.error && (
                <div className="mt-1 text-[11px] text-red-600">{terrainSourceState.error}</div>
              )}
              {terrainDescriptor && (
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="h-7 px-2 text-[11px]" onClick={zoomToDsm}>
                    Zoom
                  </Button>
                </div>
              )}
              {!terrainDescriptor && rememberedTerrainDescriptor && (
                <div className="space-y-2 text-[11px] text-slate-600">
                  <div className="font-medium text-slate-800">{rememberedTerrainDescriptor.name}</div>
                  <div>
                    {rememberedTerrainDescriptor.width.toLocaleString()} x {rememberedTerrainDescriptor.height.toLocaleString()} px · {formatBytes(rememberedTerrainDescriptor.fileSizeBytes)}
                  </div>
                  <div>{rememberedTerrainDescriptor.sourceCrsLabel}</div>
                  {(rememberedTerrainDescriptor.nativeResolutionXM || rememberedTerrainDescriptor.nativeResolutionYM) && (
                    <div>
                      Native resolution: {(rememberedTerrainDescriptor.nativeResolutionXM ?? 0).toFixed(2)} m × {(rememberedTerrainDescriptor.nativeResolutionYM ?? 0).toFixed(2)} m
                    </div>
                  )}
                  <div className="text-[11px] text-slate-500">
                    Saved locally. Click Load to apply this DSM terrain.
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-[11px]"
                      onClick={handleApplyRememberedTerrainSource}
                      disabled={terrainAnalysisDisabled || !terrainSourceState.backendEnabled}
                    >
                      Load
                    </Button>
                    <Button size="sm" variant="outline" className="h-7 px-2 text-[11px]" onClick={zoomToRememberedDsm}>
                      Zoom
                    </Button>
                  </div>
                </div>
              )}
            </div>

            {imageryOverlayImportEnabled ? (
              <div className="border-t border-slate-200 pt-3 space-y-2">
                <div className="text-[11px] font-medium text-slate-700">Imagery overlay</div>
                {imageryDescriptor ? (
                  <div className="space-y-1 text-[11px] text-slate-600">
                    <div className="font-medium text-slate-800">{imageryDescriptor.name}</div>
                    <div>
                      {imageryDescriptor.width.toLocaleString()} x {imageryDescriptor.height.toLocaleString()} px · {formatBytes(imageryDescriptor.fileSizeBytes)}
                    </div>
                    <div>{imageryDescriptor.sourceCrsLabel}</div>
                  </div>
                ) : (
                  <div className="text-[11px] text-slate-500">
                    Import a georeferenced GeoTIFF ortho to view it as an imagery overlay.
                  </div>
                )}
                {imageryOverlayState.error && (
                  <div className="mt-1 text-[11px] text-red-600">{imageryOverlayState.error}</div>
                )}
                {imageryDescriptor && (
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" className="h-7 px-2 text-[11px]" onClick={zoomToImageryOverlay}>
                      Zoom
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 px-2 text-[11px]"
                      onClick={() => clearActiveImageryOverlay()}
                    >
                      Clear
                    </Button>
                  </div>
                )}
              </div>
            ) : null}
          </div>
        )}
      </div>

      {isAnalyzing && (
        <div className="flex items-center justify-center py-4 border-b mb-3">
          <div className="text-center">
            <LoadingSpinner size="sm" className="mx-auto mb-2" />
            <p className="text-xs text-gray-600">
              Analyzing {analyzingPolygons.size} polygon{analyzingPolygons.size !== 1 ? 's' : ''}...
            </p>
          </div>
        </div>
      )}

      {!isAnalyzing && !hasResults && (
        <div className="text-center py-6 text-gray-500">
          <p className="text-xs">Import flightplan or draw polygons to start analysis</p>
        </div>
      )}

      <div className={panelEnabled && !terrainAnalysisDisabled ? '' : 'opacity-50 pointer-events-none'}>
        <Suspense fallback={<DeferredPanelFallback />}>
          <OverlapGSDPanel
            mapRef={mapRef}
            mapboxToken={mapboxToken}
            clearAllEpoch={clearAllEpoch}
            getPerPolygonParams={() => paramsByPolygon}
            onEditPolygonParams={handleEditPolygonParams}
            onAutoRun={handleAutoRunReceived}
            onClearExposed={handleClearReceived}
            onExposePoseImporter={(fn)=>{ openDJIImporterRef.current = fn; }}
            onPosesImported={(c)=> setImportedPoseCount(c)}
            polygonAnalyses={polygonResults}
            overrides={overrides}
            importedOriginals={importedOriginals}
            selectedPolygonId={selectedPolygonId}
            onSelectPolygon={setSelectedPolygonId}
          />
        </Suspense>
      </div>
    </>
  );

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header (compact) */}
      <header className="bg-white/95 backdrop-blur border-b border-gray-200 px-3 md:px-4 py-2 z-50">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 bg-blue-600 rounded-md flex items-center justify-center">
              <Map className="w-4 h-4 text-white" />
            </div>
            <div className="leading-tight">
              <h1 className="text-sm md:text-base font-semibold text-gray-900 tracking-tight">
                Flight Plan Analyser
              </h1>
              <p className="hidden md:block text-[11px] text-gray-500">
                Terrain‑aware flight planning &amp; GSD analysis
              </p>
            </div>
          </div>
          <div className="flex w-full flex-wrap items-center gap-2 md:w-auto md:justify-end">
            {/* Consolidated Import dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="sm" variant="outline" className={headerButtonClassName}>
                  <Upload className="w-3 h-3 mr-1" /> Import ▾
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuLabel>Import</DropdownMenuLabel>
                <DropdownMenuItem onSelect={() => mapRef.current?.openFlightplanFilePicker?.()}>
                  Wingtra Flightplan (.flightplan)
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => mapRef.current?.openKmlFilePicker?.()}>
                  KML Polygons (.kml)
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={handleOpenDsmPicker}>
                  Terrain source (.tif/.tiff)
                </DropdownMenuItem>
                {imageryOverlayImportEnabled ? (
                  <DropdownMenuItem onSelect={handleOpenImageryPicker}>
                    Imagery overlay (.tif/.tiff)
                  </DropdownMenuItem>
                ) : null}
                <DropdownMenuItem onSelect={() => openDJIImporterRef.current?.('dji')}>
                  DJI Camera JSON (input_cameras.json)
                </DropdownMenuItem>
                <DropdownMenuItem onSelect={() => openDJIImporterRef.current?.('wingtra')}>
                  Wingtra Geotags (.json)
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Consolidated Export dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="sm" variant="outline" className={headerButtonClassName} title="Export data">
                  <Download className="w-3 h-3 mr-1" /> Export ▾
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuLabel>Export</DropdownMenuLabel>
                <DropdownMenuItem onSelect={handleExportWingtra}>
                  Wingtra Flightplan
                </DropdownMenuItem>
                {/* Future export targets */}
                <DropdownMenuSeparator />
                <DropdownMenuItem disabled>
                  CSV Report (soon)
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <Button
              size="sm"
              variant="outline"
              className={headerButtonClassName}
              onClick={clearAllDrawings}
            >
              <Trash2 className="w-3 h-3 mr-1" />
              Clear All
            </Button>
          </div>
        </div>
      </header>

      <div className="flex-1 relative">
        {importUiState && (
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/20 backdrop-blur-[1px]">
            <div className="rounded-xl border border-slate-200 bg-white/95 px-5 py-4 shadow-lg">
              <div className="flex items-center gap-3">
                <LoadingSpinner size="sm" />
                <div>
                  <div className="text-sm font-medium text-slate-900">{importUiMessage}</div>
                  <div className="text-xs text-slate-600">{importUiSubmessage}</div>
                </div>
              </div>
            </div>
          </div>
        )}

        <Dialog open={exportNameDialogOpen} onOpenChange={setExportNameDialogOpen}>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Export Wingtra Flightplan</DialogTitle>
            </DialogHeader>
            <div className="space-y-2">
              <Label htmlFor="flightplan-export-name">Flightplan name</Label>
              <Input
                id="flightplan-export-name"
                value={exportNameDraft}
                onChange={(event) => setExportNameDraft(event.target.value)}
                placeholder="exported"
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    handleConfirmWingtraExport();
                  }
                }}
              />
              <div className="text-xs text-slate-500">
                Saved as <span className="font-mono">{normalizeFlightplanFilename(exportNameDraft)}</span>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setExportNameDialogOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleConfirmWingtraExport}>
                Export
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <input
          ref={dsmInputRef}
          type="file"
          accept=".tif,.tiff,image/tiff"
          onChange={handleDsmFileChange}
          style={{ display: 'none' }}
        />
        {imageryOverlayImportEnabled ? (
          <input
            ref={imageryInputRef}
            type="file"
            accept=".tif,.tiff,image/tiff"
            onChange={handleImageryFileChange}
            style={{ display: 'none' }}
          />
        ) : null}
        {/* PER‑POLYGON PARAMS DIALOG */}
        <Suspense fallback={null}>
        {(() => {
          const pid = paramsDialog.polygonId || "";
          const mapParams = mapRef.current?.getPerPolygonParams?.() || {} as any;
          const current = mapParams[pid] || paramsByPolygon[pid] || {} as any;
          return (
        <PolygonParamsDialog
          open={paramsDialog.open}
          polygonId={paramsDialog.polygonId}
          onClose={handleCloseParams}
          onSubmit={handleApplyParams}
          onSubmitAll={(params) => {
            mapRef.current?.applyParamsToAllPending?.(params);
            // Refresh local cache from source of truth
            const updated = mapRef.current?.getPerPolygonParams?.() || {};
            setParamsByPolygon(updated as any);
            setParamsDialog({ open: false, polygonId: null });
          }}
          defaults={{
            payloadKind: current.payloadKind ?? 'camera',
            planeHardwareVersion: current.planeHardwareVersion,
            altitudeAGL: current.altitudeAGL ?? 100,
            frontOverlap: current.frontOverlap ?? ((current.payloadKind ?? 'camera') === 'lidar' ? 0 : 70),
            sideOverlap: current.sideOverlap ?? 70,
            cameraKey: current.cameraKey ?? 'MAP61_17MM',
            lidarKey: current.lidarKey,
            cameraYawOffsetDeg: current.cameraYawOffsetDeg ?? 0,
            speedMps: current.speedMps,
            lidarReturnMode: current.lidarReturnMode,
            mappingFovDeg: current.mappingFovDeg,
            maxLidarRangeM: current.maxLidarRangeM,
            pointDensityPtsM2: current.pointDensityPtsM2,
            useCustomBearing: current.useCustomBearing ?? false,
            customBearingDeg: current.customBearingDeg ?? undefined,
          }}
        />); })()}
        </Suspense>

        {isMobile && (
          <Drawer open={mobileAnalysisOpen} onOpenChange={setMobileAnalysisOpen}>
            <DrawerContent className="h-[50vh] max-h-[50vh] rounded-t-3xl border-t border-slate-200 bg-white/98">
              <DrawerHeader className="pb-2">
                <DrawerTitle className="text-base">Analysis</DrawerTitle>
                <DrawerDescription>
                  Terrain source, overlap and GSD controls for the current workspace.
                </DrawerDescription>
              </DrawerHeader>
              <div className="overflow-y-auto overscroll-contain px-4 pb-[max(1.5rem,env(safe-area-inset-bottom))]">
                <div className="space-y-3">
                  {analysisPanelBody}
                </div>
              </div>
            </DrawerContent>
          </Drawer>
        )}

        {isMobile && !mobileAnalysisOpen && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 z-40 px-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
            <button
              type="button"
              aria-label="Open analysis panel"
              className="pointer-events-auto mx-auto block w-full max-w-sm rounded-[22px] border border-slate-200 bg-white/96 px-4 pb-4 pt-2 text-left shadow-[0_-10px_30px_rgba(15,23,42,0.08)] backdrop-blur focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 touch-manipulation"
              onClick={() => setMobileAnalysisOpen(true)}
            >
              <div className="mx-auto mb-3 h-1.5 w-12 rounded-full bg-slate-200" />
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-sm font-semibold text-slate-900">Analysis</div>
                  <p className="mt-1 text-xs text-slate-600">{mobileAnalysisSummary}</p>
                </div>
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-slate-100 text-slate-700">
                  <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
                </div>
              </div>
            </button>
          </div>
        )}

        {/* Right Side Panel - Combined Controls and Instructions - Hidden on mobile */}
        {!isMobile && (
          <div className="absolute top-2 right-2 z-40 w-[500px] max-w-[90vw] max-h-[calc(100vh-120px)] overflow-y-auto">

          {/* Unified Analysis Panel */}
          <Card className="backdrop-blur-md bg-white/95">
            <CardContent className="p-3 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-gray-900">Analysis</h3>
              </div>
              {analysisPanelBody}
            </CardContent>
          </Card>
        </div>
        )}

        {/* Map Container */}
        <Suspense fallback={<DeferredMapFallback />}>
          <MapFlightDirection
            ref={mapRef}
            mapboxToken={mapboxToken}
            clearAllEpoch={clearAllEpoch}
            center={center}
            zoom={initialZoom}
            sampleStep={sampleStep}
            terrainDemUrlTemplate={getTerrainDemUrlTemplateForCurrentSource()}
            terrainSource={terrainSourceState.source}
            onTerrainSourceReady={(readyTerrainSource) => {
              const readyKey = `${readyTerrainSource.mode}:${readyTerrainSource.datasetId ?? ''}`;
              lastReadyTerrainKeyRef.current = readyKey;
              console.log('[terrain-source] map reported terrain source ready', {
                readyKey,
                currentImportUiState: importUiState,
              });
              setImportUiState((current) => {
                if (!current || current.phase !== 'applying') return current;
                return current.targetKey === readyKey ? null : current;
              });
            }}
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onError={handleError}
            onRequestParams={handleRequestParams}
            onFlightLinesUpdated={handleFlightLinesUpdated}
            onClearGSD={() => clearGSDRef.current?.()}
            onPolygonSelected={setSelectedPolygonId}
            selectedPolygonId={selectedPolygonId}
          />
        </Suspense>
      </div>
    </div>
  );
}
