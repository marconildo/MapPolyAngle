/**
 * Shared domain types for the Flight Plan Analyser application.
 * These types are used across components to ensure consistency.
 */

export type LngLat = [number, number];

export interface PolygonRing {
  id: string;          // always defined
  ring: LngLat[];      // closed ring
}

export interface CameraModel {
  f_m: number;         // focal length in meters
  sx_m: number;        // pixel size x in meters
  sy_m: number;        // pixel size y in meters
  w_px: number;        // image width in pixels
  h_px: number;        // image height in pixels
  cx_px?: number;      // principal point x (optional, defaults to w_px/2)
  cy_px?: number;      // principal point y (optional, defaults to h_px/2)
  names?: string[];    // canonical & alias names (exact matches to Wingtra payload strings)
}

export type PayloadKind = 'camera' | 'lidar';
export type PlaneHardwareVersion = '4' | '5';
export type LidarReturnMode = 'single' | 'dual' | 'triple';
export type LidarComparisonMode = 'first-return' | 'all-returns';

export interface LidarModel {
  key: string;
  defaultSpeedMps: number;
  nativeHorizontalFovDeg: number;
  mappingHorizontalFovDeg: number;
  effectiveHorizontalFovDeg: number; // compatibility alias for the mapping sector
  verticalFovMinDeg: number;
  verticalFovMaxDeg: number;
  verticalResolutionDeg: number;
  verticalAnglesDeg: number[];
  defaultFrameRateHz: number;
  supportedFrameRatesHz: number[];
  nativePointRates: Record<LidarReturnMode, number>;
  effectivePointRates: Record<LidarReturnMode, number>;
  defaultAzimuthSectorCenterDeg?: number;
  boresightYawDeg?: number;
  boresightPitchDeg?: number;
  boresightRollDeg?: number;
  defaultMaxRangeM?: number;
  names?: string[];
}

export interface FlightParams {
  payloadKind?: PayloadKind; // defaults to 'camera' for legacy polygons
  planeHardwareVersion?: PlaneHardwareVersion; // WingtraOne(v4) vs WingtraRay(v5) for export compatibility
  altitudeAGL: number;  // altitude above ground level in meters
  frontOverlap: number; // front overlap percentage (0–95); 0 for lidar payloads
  sideOverlap: number;  // side overlap percentage (0–95)
  cameraKey?: string;   // optional camera identifier (maps to models in domain/camera)
  lidarKey?: string;    // optional lidar identifier (maps to models in domain/lidar)
  triggerDistanceM?: number; // optional: explicit trigger distance from import
  cameraYawOffsetDeg?: number; // optional: rotate camera about Z (e.g. 90 to swap width/height)
  speedMps?: number;    // optional cruise speed, used for lidar analysis and fixed-wing export sequencing
  lidarReturnMode?: LidarReturnMode; // lidar return mode used for density estimates
  mappingFovDeg?: number; // lidar mapping sector in degrees; Wingtra lidar defaults to 90
  lidarFrameRateHz?: number; // lidar spin / frame rate used for beam sampling
  lidarAzimuthSectorCenterDeg?: number; // center of the Wingtra mapping sector in sensor azimuth degrees
  lidarBoresightYawDeg?: number; // optional yaw boresight correction for scan geometry
  lidarBoresightPitchDeg?: number; // optional pitch boresight correction for scan geometry
  lidarBoresightRollDeg?: number; // optional roll boresight correction for scan geometry
  lidarComparisonMode?: LidarComparisonMode; // how predicted density should be compared to delivered LAS
  maxLidarRangeM?: number; // lidar slant range limit for valid returns; beyond this is treated as no-data
  pointDensityPtsM2?: number; // imported or computed lidar density estimate
  useCustomBearing?: boolean; // optional manual bearing flag
  customBearingDeg?: number;  // optional manual bearing degrees clockwise from north
}

export interface TerrainTile {
  z: number;           // zoom level
  x: number;           // tile x coordinate
  y: number;           // tile y coordinate
  width: number;       // tile width in pixels
  height: number;      // tile height in pixels
  data: Uint8ClampedArray; // terrain-rgb data
}

export interface FlightLines {
  lineSpacing: number; // spacing between flight lines in meters
  lines: LngLat[][];   // array of flight lines (each line is an array of LngLat points)
}

export interface PlannedTurnBlock {
  startSweep: LngLat;
  endSweep: LngLat;
  turnOff: LngLat;
  loiterCenter: LngLat;
  nextSweepStart: LngLat;
  loiterRadiusM: number;
  loiterDirection: 1 | -1;
  turnOffAcceptanceRadiusM: number;
  loiterEntryPoint?: LngLat;
  loiterExitPoint?: LngLat;
}

export interface PlannedFlightGeometry {
  lineSpacing: number;
  flightLines: LngLat[][]; // raw clipped sweep fragments kept for backward compatibility
  sweepIndices: number[];
  sweepLines: LngLat[][]; // ordered per-sweep lines after fragment grouping/merging
  gridPoints: LngLat[];
  leadInPoints: LngLat[];
  leadOutPoints: LngLat[];
  connectedLines: LngLat[][]; // sweep/turn segments in flown order
  turnaroundRadiusM: number;
  turnBlocks: PlannedTurnBlock[];
}

export interface PolygonAnalysisResult {
  polygonId: string;
  ring: LngLat[];
  result: {
    contourDirDeg: number;
    aspectDeg: number;
    samples: number;
    maxElevation?: number;
    rSquared?: number;
    rmse?: number;
    slopeMagnitude?: number;
    fitQuality?: 'excellent' | 'good' | 'fair' | 'poor';
  };
  terrainZoom: number;
}
