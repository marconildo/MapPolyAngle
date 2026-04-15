// src/interop/wingtra/types.ts

import type { FlightParams, LidarReturnMode, LngLat, PayloadKind, PlaneHardwareVersion } from "@/domain/types";

export type WingtraAngleConvention =
  // Most QGC/WingtraPilot area missions: clockwise from North (0° = North)
  | "northCW"
  // Some tools store angles clockwise from East (0° = East)
  | "eastCW";

export interface WingtraCameraBlock {
  pointDensity?: number;
  AltitudeOffset?: number;
  groundResolution?: number; // meters per pixel
  imageSideOverlap: number;  // percent
  imageFrontalOverlap: number; // percent
  cameraTriggerDistance?: number; // meters
}

export interface WingtraGridBlock {
  angle: number;               // degrees (see convention)
  spacing: number;             // line spacing (meters)
  altitude: number;            // AGL meters (terrainFollowing applies)
  multithreading?: boolean;
  turnAroundDistance?: number;
  turnAroundSideOffset?: number;
  safeRTHMaxSurveyAltitude?: number | null;
}

export interface WingtraAreaItem {
  type: "ComplexItem";
  complexItemType: "area";
  version?: number;
  terrainFollowing?: boolean;
  grid: WingtraGridBlock;
  camera: WingtraCameraBlock;
  polygon: number[][]; // NOTE: Wingtra uses [lat, lon]; our app uses [lng, lat]
  wasFlown?: boolean;
}

export interface WingtraFlightPlan {
  locked?: boolean;
  safety?: Record<string, unknown>;
  siteId?: string;
  version: number;
  fileType: "Plan";
  flightId?: string;
  geofence?: any;
  flightPlanOrigin?: {
    location: string;
    coordinate: number[];
  };
  groundStation?: string;
  flightPlan: {
    items: Array<WingtraAreaItem | Record<string, unknown>>;
    payload?: string;              // e.g. "RX1RII 42MP"
    payloadUniqueString?: string;  // e.g. "RX1R2_v4"
    version: number;
    numberOfImages?: number;
    totalArea?: number;
    // ... we keep the rest opaque
    [k: string]: unknown;
  };
  flightPlanHistory?: unknown[];
}

export interface ImportedArea {
  id: string;                 // synthetic id
  ring: LngLat[];             // [lng, lat] order (closed ring is OK but not required)
  payloadKind: PayloadKind;
  altitudeAGL: number;        // meters
  frontOverlap: number;       // %
  sideOverlap: number;        // %
  lineSpacingM: number;       // meters
  triggerDistanceM: number;   // meters
  angleDeg: number;           // direction of flight, 0..360
  terrainFollowing: boolean;
  cameraKey?: string;         // internal camera registry key (e.g. SONY_RX1R2, MAP61_17MM)
  lidarKey?: string;          // internal lidar registry key (e.g. WINGTRA_LIDAR_XT32M2X)
  planeHardwareVersion?: PlaneHardwareVersion;
  speedMps?: number;          // cruise speed carried through from the flightplan (or lidar model defaults)
  lidarReturnMode?: LidarReturnMode;
  mappingFovDeg?: number;
  maxLidarRangeM?: number;
  pointDensityPtsM2?: number;
  // carry through metadata that could be useful
  wingtraRaw?: WingtraAreaItem;
}

export interface ImportedWingtraPlan {
  items: ImportedArea[];
  payloadKind: PayloadKind;
  payloadName?: string;
  payloadKey?: string;      // payloadUniqueString
  payloadCameraKey?: string; // resolved internal camera key
  payloadLidarKey?: string; // resolved internal lidar key
  planeHardwareVersion?: PlaneHardwareVersion;
  meta: {
    version: number;
    fileType: "Plan";
    groundStation?: string;
  };
}

export interface ExportedArea extends FlightParams {
  ring: LngLat[];
  angleDeg: number;
  lineSpacingM?: number;
  terrainFollowing?: boolean;
}
