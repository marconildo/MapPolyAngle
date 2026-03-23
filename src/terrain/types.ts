export type Bounds = { minX: number; minY: number; maxX: number; maxY: number };
export type LngLatBounds = { minLng: number; minLat: number; maxLng: number; maxLat: number };

export interface DsmSourceDescriptor {
  id: string;
  name: string;
  fileSizeBytes: number;
  width: number;
  height: number;
  sourceBounds: Bounds;
  footprint3857: Bounds;
  footprintLngLat: LngLatBounds;
  footprintRingLngLat: [number, number][];
  sourceCrsCode: string | null;
  sourceCrsLabel: string;
  sourceProj4: string;
  horizontalUnits: string | null;
  verticalScaleToMeters: number;
  noDataValue: number | null;
  nativeResolutionXM?: number | null;
  nativeResolutionYM?: number | null;
  validCoverageRatio?: number | null;
  loadedAtIso: string;
}

export type TerrainSourceMode = 'mapbox' | 'blended';

export interface TerrainSourceSelection {
  mode: TerrainSourceMode;
  datasetId?: string | null;
}

export interface TerrainSourceState {
  source: TerrainSourceSelection;
  descriptor: DsmSourceDescriptor | null;
  datasets: DsmSourceDescriptor[];
  isLoading: boolean;
  isDatasetListLoading: boolean;
  error: string | null;
  backendEnabled: boolean;
}
