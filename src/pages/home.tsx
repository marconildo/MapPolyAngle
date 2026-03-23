import React, { Suspense, lazy, useState, useRef, useCallback, useMemo } from 'react';
import type { BearingOverride, MapFlightDirectionAPI } from '@/components/MapFlightDirection/api';
import type { PolygonAnalysisResult } from '@/components/MapFlightDirection/types';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { useIsMobile } from '@/hooks/use-mobile';
import { Map, Trash2, AlertCircle, Upload, Download } from 'lucide-react';
import type { PolygonParams } from '@/components/MapFlightDirection/types';
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator } from '@/components/ui/dropdown-menu';
import { toast } from "@/hooks/use-toast";
import {
  clearTerrainSourceSelection,
  getTerrainDemUrlTemplateForCurrentSource,
  getTerrainSourceState,
  initializeTerrainSourceState,
  loadTerrainSourceFromFile,
  refreshTerrainSourceDatasets,
  selectTerrainSourceDataset,
  setTerrainSourceMode,
  subscribeTerrainSource,
} from '@/terrain/terrainSource';
import { clearDsmFootprintPolygon, setDsmFootprintPolygon } from '@/components/MapFlightDirection/utils/mapbox-layers';
import type { TerrainSourceState } from '@/terrain/types';
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

export default function Home() {
  const isMobile = useIsMobile();
  const mapRef = useRef<MapFlightDirectionAPI>(null);
  const dsmInputRef = useRef<HTMLInputElement>(null);
  const previousTerrainSourceKeyRef = useRef<string | undefined>(undefined);

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
  // NEW: track imported pose count
  const [importedPoseCount, setImportedPoseCount] = useState(0);
  const [clearAllEpoch, setClearAllEpoch] = useState(0);

  // Auto-run GSD analysis when flight lines are updated (already wired)
  const autoRunGSDRef = useRef<((opts?: { polygonId?: string; reason?: 'lines'|'spacing'|'alt'|'manual' }) => void) | null>(null);
  const clearGSDRef = useRef<(() => void) | null>(null);
  // NEW: ref to open pose JSON importer (DJI or Wingtra) inside OverlapGSDPanel
  const openDJIImporterRef = useRef<((mode?: 'dji' | 'wingtra') => void) | null>(null);

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
    const unsubscribe = subscribeTerrainSource(() => {
      setTerrainSourceState(getTerrainSourceState());
    });
    void initializeTerrainSourceState().catch((error) => {
      toast({
        variant: 'destructive',
        title: 'DSM library load failed',
        description: error instanceof Error ? error.message : String(error),
      });
    });
    return () => {
      unsubscribe();
    };
  }, []);

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
    if (terrainSourceState.isLoading) return;
    const currentKey = `${terrainSourceState.source.mode}:${terrainSourceState.source.datasetId ?? ''}`;
    const previousKey = previousTerrainSourceKeyRef.current;
    previousTerrainSourceKeyRef.current = currentKey;
    if (previousKey === undefined || previousKey === currentKey) return;

    clearGSDRef.current?.();
    mapRef.current?.refreshTerrainForAllPolygons?.();
  }, [terrainSourceState.isLoading, terrainSourceState.source.datasetId, terrainSourceState.source.mode]);

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

  // MapFlightDirection now calls us to request params per polygon
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
      clearGSDRef.current?.();
      setSelectedPolygonId(null);
      return;
    }

    if (autoRunGSDRef.current) {
      if (which === '__all__') autoRunGSDRef.current({ reason: 'spacing' });
      else autoRunGSDRef.current({ polygonId: which, reason: 'lines' });
    }
  }, [importedPoseCount]);

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
  }, []);

  const fitMapToDsmDescriptor = useCallback((descriptor: NonNullable<TerrainSourceState['descriptor']>) => {
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
    fitMapToDsmDescriptor(terrainSourceState.descriptor);
  }, [terrainSourceState.descriptor, fitMapToDsmDescriptor]);

  const handleOpenDsmPicker = useCallback(() => {
    dsmInputRef.current?.click();
  }, []);

  const handleDsmFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const descriptor = await loadTerrainSourceFromFile(file);
      fitMapToDsmDescriptor(descriptor);
      toast({
        title: 'DSM loaded',
        description: `${descriptor.name} is active as the blended terrain source for mesh, analysis, and autosplit.`,
      });
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'DSM load failed',
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      if (dsmInputRef.current) dsmInputRef.current.value = '';
    }
  }, [fitMapToDsmDescriptor]);

  const handleClearDsm = useCallback(async () => {
    clearTerrainSourceSelection();
    toast({
      title: 'DSM cleared',
      description: 'Terrain source is back to Mapbox only.',
    });
  }, []);

  const handleRefreshDsmLibrary = useCallback(async () => {
    try {
      await refreshTerrainSourceDatasets();
      toast({
        title: 'DSM library refreshed',
        description: 'Available backend DSM datasets have been updated.',
      });
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'DSM refresh failed',
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }, []);

  const handleSelectExistingDsm = useCallback(async (event: React.ChangeEvent<HTMLSelectElement>) => {
    const datasetId = event.target.value;
    if (!datasetId) return;
    try {
      const descriptor = await selectTerrainSourceDataset(datasetId, 'blended');
      fitMapToDsmDescriptor(descriptor);
      toast({
        title: 'DSM selected',
        description: `${descriptor.name} is active as the blended terrain source for mesh, analysis, and autosplit.`,
      });
    } catch (error) {
      toast({
        variant: 'destructive',
        title: 'DSM selection failed',
        description: error instanceof Error ? error.message : String(error),
      });
    }
  }, [fitMapToDsmDescriptor]);

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

  // helper to export Wingtra flight plan
  const handleExportWingtra = useCallback(() => {
    const api = mapRef.current; if (!api?.exportWingtraFlightPlan) return;
    const { blob } = api.exportWingtraFlightPlan();
    const original = api.getLastImportedFlightplanName?.();
    const fn = (original && /\.flightplan$/.test(original)) ? original.replace(/\.flightplan$/, '-exported.flightplan') : 'exported.flightplan';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = fn; document.body.appendChild(a); a.click();
    setTimeout(()=>{ URL.revokeObjectURL(url); a.remove(); }, 1000);
  }, []);

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
  const availableTerrainDatasets = terrainSourceState.datasets;
  const terrainMode = terrainSourceState.source.mode;
  const terrainCoverageRatio = terrainDescriptor?.validCoverageRatio ?? null;
  const terrainCoverageWarning = terrainCoverageRatio != null && terrainCoverageRatio < 0.8;
  const terrainAnalysisDisabled = terrainSourceState.isLoading;

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header (compact) */}
      <header className="bg-white/95 backdrop-blur border-b border-gray-200 px-3 md:px-4 py-2 z-50">
        <div className="flex items-center justify-between">
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
          <div className="flex items-center gap-2">
            {/* Consolidated Import dropdown */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button size="sm" variant="outline" className="h-8 px-2 whitespace-nowrap">
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
                  Surface model (.tif/.tiff)
                </DropdownMenuItem>
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
                <Button size="sm" variant="outline" className="h-8 px-2 whitespace-nowrap" title="Export data">
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
              className="h-8 px-2 whitespace-nowrap"
              onClick={clearAllDrawings}
            >
              <Trash2 className="w-3 h-3 mr-1" />
              Clear All
            </Button>
          </div>
        </div>
      </header>

      <div className="flex-1 relative">
        <input
          ref={dsmInputRef}
          type="file"
          accept=".tif,.tiff,image/tiff"
          onChange={handleDsmFileChange}
          style={{ display: 'none' }}
        />
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

        {/* Right Side Panel - Combined Controls and Instructions - Hidden on mobile */}
        {!isMobile && (
          <div className="absolute top-2 right-2 z-40 w-[500px] max-w-[90vw] max-h-[calc(100vh-120px)] overflow-y-auto">

          {/* Unified Analysis Panel */}
          <Card className="backdrop-blur-md bg-white/95">
            <CardContent className="p-3 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-medium text-gray-900">Analysis</h3>
              </div>

              <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-xs font-medium text-slate-900">Terrain source</div>
                    <div className="mt-2 flex gap-2">
                      <Button
                        size="sm"
                        variant={terrainMode === 'mapbox' ? 'default' : 'outline'}
                        className="h-7 px-2 text-[11px]"
                        onClick={() => setTerrainSourceMode('mapbox')}
                        disabled={terrainSourceState.isLoading}
                      >
                        Mapbox
                      </Button>
                      <Button
                        size="sm"
                        variant={terrainMode === 'blended' ? 'default' : 'outline'}
                        className="h-7 px-2 text-[11px]"
                        onClick={() => setTerrainSourceMode('blended')}
                        disabled={terrainSourceState.isLoading || !terrainDescriptor}
                      >
                        Blended DSM
                      </Button>
                    </div>
                    {terrainDescriptor ? (
                      <div className="mt-1 space-y-1 text-[11px] text-slate-600">
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
                        {terrainCoverageRatio != null && (
                          <div>
                            Valid DSM coverage: {(terrainCoverageRatio * 100).toFixed(1)}%
                          </div>
                        )}
                        <div className="text-slate-500">
                          {terrainMode === 'blended'
                            ? 'Mesh, polygon analysis, GSD, lidar density, and autosplit use the blended DSM source.'
                            : 'Mapbox terrain is active. The uploaded DSM is kept available for switching back.'}
                        </div>
                        {terrainCoverageWarning && (
                          <div className="text-amber-700">
                            Large invalid regions will fall back to Mapbox terrain outside the valid DSM coverage.
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="mt-1 text-[11px] text-slate-500">
                        {availableTerrainDatasets.length > 0
                          ? 'Using Mapbox terrain. Select a saved DSM below or import a new GeoTIFF to enable a blended DSM terrain source.'
                          : 'Using Mapbox terrain. Import a GeoTIFF DSM to enable a blended DSM terrain source.'}
                      </div>
                    )}
                    {terrainSourceState.backendEnabled && (
                      <div className="mt-2 space-y-1">
                        <div className="flex items-center justify-between gap-2">
                          <div className="text-[11px] font-medium text-slate-700">
                            Saved DSMs {terrainSourceState.isDatasetListLoading ? '…' : `(${availableTerrainDatasets.length})`}
                          </div>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-6 px-2 text-[10px]"
                            onClick={handleRefreshDsmLibrary}
                            disabled={terrainSourceState.isLoading || terrainSourceState.isDatasetListLoading}
                          >
                            Refresh
                          </Button>
                        </div>
                        <select
                          className="h-8 w-full rounded-md border border-slate-200 bg-white px-2 text-[11px] text-slate-700"
                          value={terrainDescriptor?.id ?? ''}
                          onChange={handleSelectExistingDsm}
                          disabled={terrainSourceState.isLoading || terrainSourceState.isDatasetListLoading || availableTerrainDatasets.length === 0}
                        >
                          <option value="">
                            {availableTerrainDatasets.length === 0 ? 'No DSMs uploaded on this backend yet' : 'Select a saved DSM'}
                          </option>
                          {availableTerrainDatasets.map((dataset) => (
                            <option key={dataset.id} value={dataset.id}>
                              {dataset.name}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}
                    {terrainSourceState.error && (
                      <div className="mt-1 text-[11px] text-red-600">{terrainSourceState.error}</div>
                    )}
                  </div>
                  <div className="flex shrink-0 gap-2">
                    <Button size="sm" variant="outline" className="h-7 px-2 text-[11px]" onClick={handleOpenDsmPicker} disabled={terrainSourceState.isLoading}>
                      {terrainSourceState.isLoading ? 'Loading…' : terrainDescriptor ? 'Replace' : 'Load DSM'}
                    </Button>
                    {terrainDescriptor && (
                      <>
                        <Button size="sm" variant="outline" className="h-7 px-2 text-[11px]" onClick={zoomToDsm}>
                          Zoom
                        </Button>
                        <Button size="sm" variant="ghost" className="h-7 px-2 text-[11px]" onClick={handleClearDsm}>
                          Clear
                        </Button>
                      </>
                    )}
                  </div>
                </div>
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
                  <p className="text-xs">Draw polygons to start analysis</p>
                  <p className="text-xs mt-1 text-gray-400">Support for multiple areas!</p>
                </div>
              )}

              {/* Multiple Polygon Results */}

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
            onAnalysisStart={handleAnalysisStart}
            onAnalysisComplete={handleAnalysisComplete}
            onError={handleError}
            onRequestParams={handleRequestParams}
            onFlightLinesUpdated={handleFlightLinesUpdated}
            onClearGSD={() => clearGSDRef.current?.()}
            onPolygonSelected={setSelectedPolygonId}
          />
        </Suspense>
      </div>
    </div>
  );
}
