// src/interop/wingtra/convert.ts

import type { CameraModel, FlightParams, LidarModel, LidarReturnMode, LngLat, PayloadKind } from "@/domain/types";
import { forwardSpacing, lineSpacing as computeLineSpacing, calculateGSD, SONY_RX1R2, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM, DJI_ZENMUSE_P1_24MM } from "@/domain/camera";
import { DEFAULT_LIDAR, DEFAULT_LIDAR_MAX_RANGE_M, WINGTRA_LIDAR_XT32M2X, lidarDeliverableDensity, lidarLineSpacing } from "@/domain/lidar";
import type {
  ExportedArea,
  WingtraAngleConvention,
  WingtraAreaItem,
  WingtraFlightPlan,
  ImportedWingtraPlan,
  ImportedArea,
} from "./types";

// ---------------------------
// Small helpers
// ---------------------------
const toLngLat = (latlon: number[]): LngLat => [latlon[1], latlon[0]];
const toLatLon = (lnglat: LngLat): [number, number] => [lnglat[1], lnglat[0]];

const normalize360 = (a: number) => ((a % 360) + 360) % 360;
const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "");
const stripVersionSuffix = (s: string) => normalize(s).replace(/v\d+$/g, "");
const payloadMatches = (candidate: string, name: string) => {
  const cn = normalize(candidate);
  const nn = normalize(name);
  if (cn === nn) return true;
  const cs = stripVersionSuffix(candidate);
  const ns = stripVersionSuffix(name);
  if (cs === ns && cs.length > 0) return true;
  if (cn.includes(nn) || nn.includes(cn)) return true;
  if (cs && ns && (cs.includes(ns) || ns.includes(cs))) return true;
  return false;
};

function isWingtraFlightPlan(value: unknown): value is WingtraFlightPlan {
  if (!value || typeof value !== "object") return false;
  const maybePlan = (value as { flightPlan?: unknown }).flightPlan;
  if (!maybePlan || typeof maybePlan !== "object") return false;
  return Array.isArray((maybePlan as { items?: unknown }).items);
}

/**
 * Convert Wingtra "grid.angle" to "bearing° clockwise from North".
 * - If the JSON is already northCW, this is identity.
 * - If the JSON uses eastCW convention (0 = East), convert to northCW.
 */
export function wingtraAngleToBearing(
  wingtraAngle: number,
  convention: WingtraAngleConvention = "northCW"
): number {
  return convention === "northCW" ? normalize360(wingtraAngle) : normalize360(90 - wingtraAngle);
}

/**
 * Convert our "bearing° clockwise from North
 */
export function bearingToWingtraAngle(
  bearingDeg: number,
  convention: WingtraAngleConvention = "northCW"
): number {
  return convention === "northCW" ? normalize360(bearingDeg) : normalize360(90 - bearingDeg);
}

/** Try to map Wingtra payload to a camera. Extend as needed. */
export function resolveCameraFromWingtra(payloadName?: string, payloadKey?: string): CameraModel {
  return resolveCameraInfoFromWingtra(payloadName, payloadKey).camera;
}

// Simplified resolver: uses names arrays for exact matching
const CAMERA_LIST: Array<{ key:string; model: CameraModel }> = [
  { key: 'SONY_RX1R2', model: SONY_RX1R2 },
  { key: 'ILX_LR1_INSPECT_85MM', model: ILX_LR1_INSPECT_85MM },
  { key: 'MAP61_17MM', model: MAP61_17MM },
  { key: 'RGB61_24MM', model: RGB61_24MM },
  { key: 'DJI_ZENMUSE_P1_24MM', model: DJI_ZENMUSE_P1_24MM },
];

const LIDAR_LIST: Array<{ key: string; model: LidarModel }> = [
  { key: WINGTRA_LIDAR_XT32M2X.key, model: WINGTRA_LIDAR_XT32M2X },
];

export function resolveCameraInfoFromWingtra(payloadName?: string, payloadKey?: string): { camera: CameraModel; key: string } {
  const candidates = Array.from(new Set([payloadName, payloadKey].filter(Boolean))) as string[];

  // 1) Exact (case-sensitive) match against provided names
  for (const c of candidates) {
    for (const { key, model } of CAMERA_LIST) {
      if (model.names?.includes(c) || key === c) {
        return { camera: model, key };
      }
    }
  }

  // 2) Normalized exact match (case-insensitive, punctuation/whitespace removed)
  for (const c of candidates) {
    for (const { key, model } of CAMERA_LIST) {
      const names = [key, ...(model.names || [])];
      for (const n of names) {
        if (payloadMatches(c, n)) {
          return { camera: model, key };
        }
      }
    }
  }

  // 3) Fallback: try to match partials after stripping trailing version suffixes
  for (const c of candidates) {
    const cs = stripVersionSuffix(c);
    if (!cs) continue;
    for (const { key, model } of CAMERA_LIST) {
      const names = [key, ...(model.names || [])];
      for (const n of names) {
        const ns = stripVersionSuffix(n);
        if (cs.includes(ns) || ns.includes(cs)) {
          return { camera: model, key };
        }
      }
    }
  }
  return { camera: SONY_RX1R2, key: 'SONY_RX1R2' };
}

export function resolveLidarInfoFromWingtra(payloadName?: string, payloadKey?: string): { lidar: LidarModel; key: string } | null {
  const candidates = Array.from(new Set([payloadName, payloadKey].filter(Boolean))) as string[];

  for (const c of candidates) {
    for (const { key, model } of LIDAR_LIST) {
      if (model.names?.includes(c) || key === c) {
        return { lidar: model, key };
      }
    }
  }

  for (const c of candidates) {
    for (const { key, model } of LIDAR_LIST) {
      const names = [key, ...(model.names || [])];
      for (const n of names) {
        if (payloadMatches(c, n)) {
          return { lidar: model, key };
        }
      }
    }
  }

  return null;
}

type ResolvedPayloadInfo =
  | { payloadKind: 'camera'; camera: CameraModel; cameraKey: string }
  | { payloadKind: 'lidar'; lidar: LidarModel; lidarKey: string };

export function resolvePayloadInfoFromWingtra(payloadName?: string, payloadKey?: string): ResolvedPayloadInfo {
  const lidarInfo = resolveLidarInfoFromWingtra(payloadName, payloadKey);
  if (lidarInfo) {
    return { payloadKind: 'lidar', lidar: lidarInfo.lidar, lidarKey: lidarInfo.key };
  }

  const cameraInfo = resolveCameraInfoFromWingtra(payloadName, payloadKey);
  return { payloadKind: 'camera', camera: cameraInfo.camera, cameraKey: cameraInfo.key };
}

/** Deduce overlaps/spacing from an optical payload item and/or recompute from camera if needed. */
function readCameraItemParams(
  it: WingtraAreaItem,
  camera: CameraModel
): { altitudeAGL: number; frontOverlap: number; sideOverlap: number; lineSpacingM: number; triggerDistanceM: number } {
  const altitudeAGL = it.grid.altitude ?? 100;
  const frontOverlap = it.camera.imageFrontalOverlap ?? 70;
  const sideOverlap  = it.camera.imageSideOverlap ?? 70;

  // Prefer explicit values if present, otherwise recompute from camera model
  const triggerDistanceM =
    typeof it.camera.cameraTriggerDistance === "number"
      ? it.camera.cameraTriggerDistance
      : forwardSpacing(camera, altitudeAGL, frontOverlap);

  const lineSpacingM =
    typeof it.grid.spacing === "number"
      ? it.grid.spacing
      : computeLineSpacing(camera, altitudeAGL, sideOverlap);

  return { altitudeAGL, frontOverlap, sideOverlap, lineSpacingM, triggerDistanceM };
}

function readLidarItemParams(
  it: WingtraAreaItem,
  lidar: LidarModel,
  cruiseSpeedMps?: number
): {
  altitudeAGL: number;
  frontOverlap: number;
  sideOverlap: number;
  lineSpacingM: number;
  triggerDistanceM: number;
  speedMps: number;
  lidarReturnMode: LidarReturnMode;
  mappingFovDeg: number;
  maxLidarRangeM: number;
  pointDensityPtsM2: number;
} {
  const altitudeAGL = it.grid.altitude ?? 100;
  const frontOverlap = it.camera.imageFrontalOverlap ?? 0;
  const sideOverlap = it.camera.imageSideOverlap ?? 50;
  const speedMps = Number.isFinite(cruiseSpeedMps) && (cruiseSpeedMps as number) > 0
    ? (cruiseSpeedMps as number)
    : lidar.defaultSpeedMps;
  const lidarReturnMode: LidarReturnMode = 'single';
  const mappingFovDeg = lidar.effectiveHorizontalFovDeg;
  const maxLidarRangeM = DEFAULT_LIDAR_MAX_RANGE_M;
  const lineSpacingM =
    typeof it.grid.spacing === "number"
      ? it.grid.spacing
      : lidarLineSpacing(altitudeAGL, sideOverlap, mappingFovDeg);
  const pointDensityPtsM2 =
    typeof it.camera.pointDensity === "number"
      ? it.camera.pointDensity
      : lidarDeliverableDensity(lidar, altitudeAGL, sideOverlap, speedMps, lidarReturnMode, mappingFovDeg);

  return {
    altitudeAGL,
    frontOverlap,
    sideOverlap,
    lineSpacingM,
    triggerDistanceM: 0,
    speedMps,
    lidarReturnMode,
    mappingFovDeg,
    maxLidarRangeM,
    pointDensityPtsM2,
  };
}

// ---------------------------
// Import: Wingtra -> Internal
// ---------------------------
export function importWingtraFlightPlan(
  fp: WingtraFlightPlan,
  opts?: { angleConvention?: WingtraAngleConvention }
): ImportedWingtraPlan {
  if (!isWingtraFlightPlan(fp)) {
    const maybeGeotags = fp && typeof fp === "object" && Array.isArray((fp as { flights?: unknown }).flights);
    if (maybeGeotags) {
      throw new Error("This file looks like Wingtra geotag JSON, not a flightplan. Use Import > Wingtra Geotags (.json).");
    }
    throw new Error("Invalid Wingtra flightplan file.");
  }
  const angleConv = opts?.angleConvention ?? "northCW";
  const payloadName = fp.flightPlan.payload;
  const payloadKey  = (fp.flightPlan as any).payloadUniqueString as string | undefined;
  const cruiseSpeedMps = Number(fp.flightPlan.cruiseSpeed);
  const payloadInfo = resolvePayloadInfoFromWingtra(payloadName, payloadKey);

  const items: ImportedArea[] = [];
  let idx = 0;

  for (const raw of fp.flightPlan.items || []) {
    const it = raw as any;
    if (it?.type !== "ComplexItem" || it?.complexItemType !== "area") continue;

    const area = it as WingtraAreaItem;
    const angleDeg = wingtraAngleToBearing(area.grid.angle ?? 0, angleConv);
    // Polygon conversion: Wingtra uses [lat, lon]; app uses [lng, lat]
    const ring = (area.polygon || []).map(toLngLat);
    if (payloadInfo.payloadKind === 'lidar') {
      const params = readLidarItemParams(area, payloadInfo.lidar, cruiseSpeedMps);
      items.push({
        id: `wingtra-${idx++}`,
        ring,
        payloadKind: 'lidar',
        altitudeAGL: params.altitudeAGL,
        frontOverlap: params.frontOverlap,
        sideOverlap: params.sideOverlap,
        lineSpacingM: params.lineSpacingM,
        triggerDistanceM: params.triggerDistanceM,
        angleDeg,
        terrainFollowing: !!area.terrainFollowing,
        lidarKey: payloadInfo.lidarKey,
        speedMps: params.speedMps,
        lidarReturnMode: params.lidarReturnMode,
        mappingFovDeg: params.mappingFovDeg,
        maxLidarRangeM: params.maxLidarRangeM,
        pointDensityPtsM2: params.pointDensityPtsM2,
        wingtraRaw: area,
      });
    } else {
      const params = readCameraItemParams(area, payloadInfo.camera);
      items.push({
        id: `wingtra-${idx++}`,
        ring,
        payloadKind: 'camera',
        altitudeAGL: params.altitudeAGL,
        frontOverlap: params.frontOverlap,
        sideOverlap: params.sideOverlap,
        lineSpacingM: params.lineSpacingM,
        triggerDistanceM: params.triggerDistanceM,
        angleDeg,
        terrainFollowing: !!area.terrainFollowing,
        cameraKey: payloadInfo.cameraKey,
        wingtraRaw: area,
      });
    }
  }

  return {
    items,
    payloadKind: payloadInfo.payloadKind,
    payloadName,
    payloadKey,
    payloadCameraKey: payloadInfo.payloadKind === 'camera' ? payloadInfo.cameraKey : undefined,
    payloadLidarKey: payloadInfo.payloadKind === 'lidar' ? payloadInfo.lidarKey : undefined,
    meta: {
      version: fp.version,
      fileType: fp.fileType,
      groundStation: fp.groundStation as string | undefined,
    },
  };
}

// ---------------------------
// Export: Internal -> Wingtra
// ---------------------------
export interface ExportToWingtraOptions {
  payloadKind?: PayloadKind;
  angleConvention?: WingtraAngleConvention;
  terrainFollowing?: boolean;
  payloadName?: string;        // e.g. "RX1RII 42MP"
  payloadUniqueString?: string; // e.g. "RX1R2_v4"
  geofenceRadius?: number;     // optional convenience
  // Provide the camera used for spacing math (if you want us to (re)compute values)
  camera?: CameraModel;
  lidar?: LidarModel;
  lidarReturnMode?: LidarReturnMode;
  speedMps?: number;
  mappingFovDeg?: number;
  // Sprinkle in a few defaults to keep WingtraPilot happy:
  defaults?: {
    rthMode?: number;
    version?: number;
    maxGroundClearance?: number;
    minGroundClearance?: number;
    ceilingAboveTakeOff?: number;
    connectionLossTimeout?: number;
    minRTHHeightAboveHome?: number;
    hoverSpeed?: number;
    cruiseSpeed?: number;
  };
}

/**
 * Build a minimal-but-correct Wingtra flightplan object from internal areas.
 * You can pass a "template" later if you need to preserve extra top-level fields.
 */
export function exportToWingtraFlightPlan(
  areas: ExportedArea[],
  opts?: ExportToWingtraOptions
): WingtraFlightPlan {
  const angleConv = opts?.angleConvention ?? "northCW";
  const payloadKind = opts?.payloadKind ?? areas[0]?.payloadKind ?? 'camera';
  const camera = opts?.camera ?? SONY_RX1R2;
  const lidar = opts?.lidar ?? DEFAULT_LIDAR;

  // Optional safety defaults
  const safety = {
    rthMode: opts?.defaults?.rthMode ?? 0,
    version: 2,
    maxGroundClearance: opts?.defaults?.maxGroundClearance ?? 200,
    minGroundClearance: opts?.defaults?.minGroundClearance ?? 60,
    ceilingAboveTakeOff: opts?.defaults?.ceilingAboveTakeOff ?? 2000,
    connectionLossTimeout: opts?.defaults?.connectionLossTimeout ?? 60,
    minRTHHeightAboveHome: opts?.defaults?.minRTHHeightAboveHome ?? 60,
  };

  const items = areas.map((a) => {
    const areaPayloadKind = a.payloadKind ?? payloadKind;
    const spacing = typeof a.lineSpacingM === "number"
      ? a.lineSpacingM
      : areaPayloadKind === 'lidar'
        ? lidarLineSpacing(a.altitudeAGL, a.sideOverlap, a.mappingFovDeg ?? opts?.mappingFovDeg ?? lidar.effectiveHorizontalFovDeg)
        : computeLineSpacing(camera, a.altitudeAGL, a.sideOverlap);

    const trigger = areaPayloadKind === 'lidar'
      ? 0
      : typeof a.triggerDistanceM === "number"
        ? a.triggerDistanceM
        : forwardSpacing(camera, a.altitudeAGL, a.frontOverlap);

    const wingtraAngle = bearingToWingtraAngle(a.angleDeg, angleConv);

    const pointDensity = areaPayloadKind === 'lidar'
      ? (a.pointDensityPtsM2 ??
        lidarDeliverableDensity(
          lidar,
          a.altitudeAGL,
          a.sideOverlap,
          a.speedMps ?? opts?.speedMps ?? lidar.defaultSpeedMps,
          a.lidarReturnMode ?? opts?.lidarReturnMode ?? 'single',
          a.mappingFovDeg ?? opts?.mappingFovDeg ?? lidar.effectiveHorizontalFovDeg
        ))
      : undefined;

    const cameraBlock = {
      pointDensity,
      AltitudeOffset: 0,
      groundResolution: areaPayloadKind === 'lidar' ? pointDensity : calculateGSD(camera, a.altitudeAGL),
      imageSideOverlap: a.sideOverlap,
      imageFrontalOverlap: areaPayloadKind === 'lidar' ? 0 : a.frontOverlap,
      cameraTriggerDistance: trigger,
    };

    const gridBlock = {
      angle: wingtraAngle,
      spacing,
      altitude: a.altitudeAGL,
      multithreading: false,
      turnAroundDistance: 80,
      turnAroundSideOffset: 70,
      safeRTHMaxSurveyAltitude: null,
    };

    const polygonLatLon = a.ring.map(toLatLon);

    const areaItem: WingtraAreaItem = {
      type: "ComplexItem",
      complexItemType: "area",
      version: 3,
      terrainFollowing: a.terrainFollowing ?? true,
      grid: gridBlock,
      camera: cameraBlock,
      polygon: polygonLatLon,
      wasFlown: false,
    };

    return areaItem;
  });

  const fp: WingtraFlightPlan = {
    locked: false,
    safety,
    siteId: cryptoRandomUuid(),
    version: 1,
    fileType: "Plan",
    flightId: cryptoRandomUuid(),
    geofence: {
      version: 1,
      geofenceType: 0,
      geofenceRadius: opts?.geofenceRadius ?? 1200,
      terminationSettings: null,
    },
    flightPlan: {
      items,
      payload: opts?.payloadName ?? (payloadKind === 'lidar' ? "LIDAR" : "RX1RII 42MP"),
      payloadUniqueString: opts?.payloadUniqueString ?? (payloadKind === 'lidar' ? "LIDAR_v5" : "RX1R2_v4"),
      version: 6,
      gisItems: [],
      hoverSpeed: opts?.defaults?.hoverSpeed ?? 3,
      cruiseSpeed: opts?.defaults?.cruiseSpeed ?? 16,
      missionStatus: 0,
      planeHardware: {
        hwVersion: "4",
        vehicleId: -1,
        displayName: "WingtraOne (any)",
        isGenericPlane: true,
      },
      numberOfImages: 0,
      totalArea: 0,
      // the rest can be filled by WingtraPilot when opening
    } as any,
    groundStation: "WingtraPilot",
    flightPlanHistory: [],
  };

  return fp;
}

// Use crypto if available; otherwise fallback to a very simple ID.
function cryptoRandomUuid(): string {
  try {
    // @ts-ignore
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  } catch {}
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function areasFromState(polys: Array<{ring:[number,number][]; params:FlightParams; bearingDeg:number; lineSpacingM?:number; triggerDistanceM?:number }>): ExportedArea[] {
  return polys.map(p => ({
    ring: p.ring,
    payloadKind: p.params.payloadKind ?? 'camera',
    altitudeAGL: p.params.altitudeAGL,
    frontOverlap: p.params.frontOverlap,
    sideOverlap: p.params.sideOverlap,
    cameraKey: p.params.cameraKey,
    lidarKey: p.params.lidarKey,
    speedMps: p.params.speedMps,
    lidarReturnMode: p.params.lidarReturnMode,
    mappingFovDeg: p.params.mappingFovDeg,
    pointDensityPtsM2: p.params.pointDensityPtsM2,
    cameraYawOffsetDeg: p.params.cameraYawOffsetDeg,
    useCustomBearing: p.params.useCustomBearing,
    customBearingDeg: p.params.customBearingDeg,
    angleDeg: p.bearingDeg,
    lineSpacingM: p.lineSpacingM,
    triggerDistanceM: p.triggerDistanceM,
    terrainFollowing: true,
  }));
}
