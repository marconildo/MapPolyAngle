export type Bounds = { minX: number; minY: number; maxX: number; maxY: number };
export type LngLatBounds = { minLng: number; minLat: number; maxLng: number; maxLat: number };

export interface GeoTiffSourceDescriptor {
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
  loadedAtIso: string;
}

export interface DsmSourceDescriptor extends GeoTiffSourceDescriptor {
  verticalScaleToMeters: number;
  noDataValue: number | null;
  nativeResolutionXM?: number | null;
  nativeResolutionYM?: number | null;
  validCoverageRatio?: number | null;
}

export interface ImageryOverlayDescriptor extends GeoTiffSourceDescriptor {}

export type TerrainSourceMode = 'mapbox' | 'blended';

export interface TerrainSourceSelection {
  mode: TerrainSourceMode;
  datasetId?: string | null;
}

export interface TerrainSourceState {
  source: TerrainSourceSelection;
  descriptor: DsmSourceDescriptor | null;
  isLoading: boolean;
  error: string | null;
  backendEnabled: boolean;
}

export interface ImageryOverlayState {
  descriptor: ImageryOverlayDescriptor | null;
  isLoading: boolean;
  error: string | null;
}
