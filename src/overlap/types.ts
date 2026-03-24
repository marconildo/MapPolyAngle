import type { LidarComparisonMode, LidarReturnMode } from "@/domain/types";

export type CameraModel = {
  f_m: number;      // focal length (meters)
  sx_m: number;     // pixel pitch x (meters)
  sy_m: number;     // pixel pitch y (meters)
  w_px: number;
  h_px: number;
  cx_px?: number;   // default: w/2
  cy_px?: number;   // default: h/2
  names?: string[];
};

export type PoseMeters = {
  id?: string;
  x: number;        // EPSG:3857 meters
  y: number;        // EPSG:3857 meters
  z: number;        // meters ASL (if AGL handled beforehand)
  omega_deg: number;
  phi_deg: number;
  kappa_deg: number;
  // OPTIONAL: source polygon association (enables per‑polygon camera selection)
  polygonId?: string;
};

export type TileRGBA = {
  z: number; x: number; y: number;
  size: number;             // 256 or 512
  data: Uint8ClampedArray;  // RGBA (Terrain-RGB)
};

export type PaddedDemTileRGBA = {
  size: number;
  padTiles: number;
  data: Uint8ClampedArray;
};

export type GSDStats = {
  min: number;
  max: number;
  mean: number;
  count: number;            // number of DEM ground pixels contributing
  totalAreaM2?: number;      // precise summed ground area (m^2)
  histogram: { bin: number; count: number; areaM2?: number }[]; // per-bin area in m^2
};

export type DensityStats = GSDStats;

export type PolygonLngLat = { ring: [number, number][] }; // single-ring polygon

// Allow passing polygon IDs so we can report per‑polygon stats deterministically
export type PolygonLngLatWithId = { id?: string; ring: [number, number][] };

// Per‑polygon statistics for a single tile
export type PolygonTileStats = {
  polygonId: string;
  activePixelCount: number;
  gsdStats?: GSDStats;
  densityStats?: DensityStats;
  /** Set of global pose indices (in the 'poses' array) that saw this polygon in this tile */
  hitPoseIds?: Uint32Array;
  /** Set of global strip indices (in the 'strips' array) that intersected this polygon in this tile */
  hitLineIds?: Uint32Array;
};

export type TileResult = {
  z: number; x: number; y: number; size: number;
  maxOverlap: number;
  minGsd: number;   // finite min over tile or +Inf if none
  overlap: Uint16Array;
  gsdMin: Float32Array;
  gsdStats?: GSDStats; // Per-tile GSD statistics
  density?: Float32Array;
  maxDensity?: number;
  densityStats?: DensityStats;
};

export type OverlayTileResult = {
  z: number;
  x: number;
  y: number;
  size: number;
  maxOverlap?: number;
  overlap?: Uint16Array;
  gsdMin?: Float32Array;
  density?: Float32Array;
};

export type LidarStripMeters = {
  id?: string;
  polygonId?: string;
  x1: number;
  y1: number;
  z1?: number;
  x2: number;
  y2: number;
  z2?: number;
  plannedAltitudeAGL?: number;
  halfWidthM: number;
  densityPerPass: number;
  speedMps?: number;
  effectivePointRate?: number;
  halfFovTan?: number;
  maxRangeM?: number;
  passIndex?: number;
  frameRateHz?: number;
  nativeHorizontalFovDeg?: number;
  mappingFovDeg?: number;
  verticalAnglesDeg?: number[];
  returnMode?: LidarReturnMode;
  comparisonMode?: LidarComparisonMode;
  azimuthSectorCenterDeg?: number;
  boresightYawDeg?: number;
  boresightPitchDeg?: number;
  boresightRollDeg?: number;
};

export type WorkerIn = {
  tile: TileRGBA;
  polygons: PolygonLngLatWithId[];  // mask to these polygons
  poses: PoseMeters[];        // pre-converted to EPSG:3857 meters
  /** Legacy single-camera (still honored if multi-camera arrays not provided). */
  camera?: CameraModel;
  /** Optional multi-camera support: list of unique cameras. */
  cameras?: CameraModel[];
  /** For each pose, index into `cameras` (same length as poses). */
  poseCameraIndices?: Uint16Array;
  options?: {
    gsdMaxForPalette?: number; // meters per pixel for visualization clamp
    /** Optional: stop counting once overlap reaches this number (per pixel). Default: Infinity */
    maxOverlapNeeded?: number;
    /** Minimum number of overlapping images required for GSD to be considered valid. Default: 4 */
    minOverlapForGsd?: number;
    /** Optional: spatial grid resolution for pose indexing per-tile. Default: 8 */
    gridSize?: number;
    /**
     * Optional: clip (erode) interior edge of polygon masks by this many meters BEFORE any per‑pixel computation.
     * Applied as a morphological erosion on each polygon mask independently, then recombined into the union mask.
     * Use to discard boundary pixels influenced by partial coverage / DEM edge artifacts.
     */
    clipInnerBufferM?: number;
  };
};

export type WorkerOut = TileResult & {
  /** Per‑polygon stats for this tile (empty if no active pixels fell in any polygon) */
  perPolygon?: PolygonTileStats[];
};

export type LidarWorkerIn = {
  tile: TileRGBA;
  demTile?: PaddedDemTileRGBA;
  polygons: PolygonLngLatWithId[];
  strips: LidarStripMeters[];
  options?: {
    clipInnerBufferM?: number;
  };
};

export type LidarWorkerOut = TileResult & {
  perPolygon?: PolygonTileStats[];
};
