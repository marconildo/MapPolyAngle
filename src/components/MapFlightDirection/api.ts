/**
 * Formal API interface for the MapFlightDirection component.
 * This provides type safety for the imperative ref API used by consumers.
 */

import type { Map as MapboxMap } from 'mapbox-gl';
import type {
  FlightParams,
  PlannedFlightGeometry,
  PayloadKind,
} from '@/domain/types';
import type { TerrainPartitionSolutionPreview } from '@/terrain-partition/types';
import type { PolygonAnalysisResult } from './types';

export type { TerrainPartitionSolutionPreview } from '@/terrain-partition/types';

export interface PolygonWithId {
  id?: string;
  ring: [number, number][];
}

export type BearingOverrideSource = 'wingtra' | 'user' | 'partition' | 'optimized';

export interface BearingOverride {
  bearingDeg: number;
  lineSpacingM?: number;
  source: BearingOverrideSource;
}

export interface ImportedFlightplanArea {
  polygonId: string;
  params: FlightParams & {
    angleDeg: number;
    lineSpacingM: number;
    triggerDistanceM: number;
    source: 'wingtra';
  };
}

export interface WingtraFreshExportConfig {
  payloadKind: PayloadKind;
  payloadUniqueString: string;
  payloadName?: string;
}

export interface MapFlightDirectionAPI {
  // Core map operations
  clearAllDrawings(): void;
  clearPolygon(polygonId: string): void;
  editPolygonBoundary(polygonId: string): void;
  setProcessingPolygonIds(polygonIds: string[]): void;
  autoSplitPolygonByTerrain(
    polygonId: string,
    options?: { skipBackend?: boolean }
  ): Promise<{ createdIds: string[]; replaced: boolean }>;
  getTerrainPartitionSolutions(polygonId: string): Promise<TerrainPartitionSolutionPreview[]>;
  refineTerrainPartitionPreview(
    polygonId: string,
    solution: TerrainPartitionSolutionPreview,
  ): Promise<TerrainPartitionSolutionPreview>;
  applyTerrainPartitionSolution(polygonId: string, signature: string): Promise<{ createdIds: string[]; replaced: boolean }>;
  applyTerrainPartitionPreview(
    polygonId: string,
    solution: TerrainPartitionSolutionPreview,
  ): Promise<{ createdIds: string[]; replaced: boolean }>;
  startPolygonDrawing(): void;
  getMap(): MapboxMap | undefined;

  // Polygon management
  getPolygons(): [number, number][][]; // legacy format for backward compatibility
  getPolygonsWithIds(): PolygonWithId[];
  getPolygonResults(): PolygonAnalysisResult[];
  getPolygonTiles(): Map<string, any[]>; // Keep as any[] for now to match current implementation
  refreshTerrainForAllPolygons(): void;
  setTerrainDemSource(tileUrlTemplate: string | null): void;
  setFlightLinesVisible(visible: boolean): void;

  // Flight planning
  applyPolygonParams(polygonId: string, params: FlightParams): void;
  applyPolygonParamsBatch(updates: Array<{ polygonId: string; params: FlightParams }>): void;
  applyParamsToAllPending(params: FlightParams): void; // bulk apply same params to queued polygons
  getFlightLines(): Map<string, PlannedFlightGeometry & { altitudeAGL: number }>;
  getPerPolygonParams(): Record<string, FlightParams>;

  // Altitude strategy and clearance controls
  setAltitudeMode(mode: 'legacy' | 'min-clearance'): void;
  getAltitudeMode(): 'legacy' | 'min-clearance';
  setMinClearance(meters: number): void;
  getMinClearance(): number;
  setTurnExtend(meters: number): void;
  getTurnExtend(): number;

  // 3D visualization
  addCameraPoints(polygonId: string, positions: [number, number, number][]): void;
  removeCameraPoints(polygonId: string): void;

  // KML import
  openKmlFilePicker(): void;
  importKmlFromText(kml: string): Promise<{ added: number; total: number }>;

  // Wingtra flightplan import
  openFlightplanFilePicker(): void;
  importWingtraFromText(json: string): Promise<{ added: number; total: number; areas: ImportedFlightplanArea[] }>;

  // Overrides & optimization
  optimizePolygonDirection(polygonId: string): void;                 // drop override → use terrain-optimal
  revertPolygonToImportedDirection(polygonId: string): void;         // re-apply file heading/spacing
  runFullAnalysis(polygonId: string): void;                          // run complete analysis pipeline (as if manually drawn)
  getBearingOverrides(): Record<string, BearingOverride>;
  getImportedOriginals(): Record<string, { bearingDeg: number; lineSpacingM: number }>;
  getLastImportedFlightplanName(): string | undefined;
  canExportWingtraFlightPlanDirectly(): boolean;

  // Export current (possibly optimized/edited) plan as Wingtra .flightplan JSON
  exportWingtraFlightPlan(config?: WingtraFreshExportConfig): { json: string; blob: Blob };
}
