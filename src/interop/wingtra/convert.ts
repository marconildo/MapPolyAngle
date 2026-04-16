// src/interop/wingtra/convert.ts

import type {
  CameraModel,
  FlightParams,
  LidarModel,
  LidarReturnMode,
  LngLat,
  PayloadKind,
  PlaneHardwareVersion,
} from "@/domain/types";
import { forwardSpacing, lineSpacing as computeLineSpacing, calculateGSD, SONY_RX1R2, SONY_RX1R3, SONY_A6100_20MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM, DJI_ZENMUSE_P1_24MM } from "@/domain/camera";
import { DEFAULT_LIDAR, DEFAULT_LIDAR_MAX_RANGE_M, WINGTRA_LIDAR_XT32M2X, lidarDeliverableDensity, lidarLineSpacing } from "@/domain/lidar";
import type {
  ExportedArea,
  WingtraAngleConvention,
  WingtraAreaItem,
  WingtraFlightPlan,
  ImportedWingtraPlan,
  ImportedArea,
} from "./types";

export type WingtraPlaneHardwareVersion = PlaneHardwareVersion;

export interface WingtraFreshExportPayloadOption {
  payloadKind: PayloadKind;
  payloadUniqueString: string;
  payloadName: string;
  planeHardwareVersion: WingtraPlaneHardwareVersion;
  label: string;
}

const WICTopLevelDefaults = {
  fileVersion: 1,
  flightPlanVersion: 6,
  groundStation: "WingtraPilot" as const,
  creationLocation: "",
  geofenceType: 0,
  geofenceRadius: 1200,
  safety: {
    rthMode: 0,
    version: 2,
    maxGroundClearance: 200,
    minGroundClearance: 60,
    ceilingAboveTakeOff: 2000,
    connectionLossTimeout: 60,
    minRTHHeightAboveHome: 60,
  },
  elevationData: {
    enabled: true,
    name: "SRTM",
    type: "Auto",
  },
  cruiseSpeed: 15.375008,
  hoverSpeed: 3,
  plannedHomePosition: [0, 0, 0] as [number, number, number],
  vehicleLastFlownCoordinate: [null, null] as [null, null],
};

export const WINGTRA_FRESH_EXPORT_PAYLOAD_OPTIONS: WingtraFreshExportPayloadOption[] = [
  {
    payloadKind: "camera",
    payloadUniqueString: "MAPSTARNadir_v5",
    payloadName: "MAPSTARNadir",
    planeHardwareVersion: "5",
    label: "MAPSTAR Nadir (WingtraRay)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "RX1R2_v4",
    payloadName: "RX1RII 42MP",
    planeHardwareVersion: "4",
    label: "RX1RII 42MP (WingtraOne)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "RX1R2_v5",
    payloadName: "RX1RII 42MP",
    planeHardwareVersion: "5",
    label: "RX1RII 42MP (WingtraRay)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "RX1R3_v5",
    payloadName: "SURVEY61",
    planeHardwareVersion: "5",
    label: "SURVEY61 (WingtraRay)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "A6100_v5",
    payloadName: "SURVEY24",
    planeHardwareVersion: "5",
    label: "SURVEY24 (WingtraRay)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "MAPSTARHighRes_v4",
    payloadName: "MAPSTARHighRes",
    planeHardwareVersion: "4",
    label: "MAPSTAR HighRes (WingtraOne)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "MAPSTARHighRes_v5",
    payloadName: "MAPSTARHighRes",
    planeHardwareVersion: "5",
    label: "MAPSTAR HighRes (WingtraRay)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "MAPSTAROblique_v4",
    payloadName: "MAPSTAROblique",
    planeHardwareVersion: "4",
    label: "MAPSTAR Oblique (WingtraOne)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "MAPSTAROblique_v5",
    payloadName: "MAPSTAROblique",
    planeHardwareVersion: "5",
    label: "MAPSTAR Oblique (WingtraRay)",
  },
  {
    payloadKind: "camera",
    payloadUniqueString: "RGB61_v4",
    payloadName: "RGB61",
    planeHardwareVersion: "4",
    label: "RGB61 (WingtraOne)",
  },
  {
    payloadKind: "lidar",
    payloadUniqueString: "LIDAR_v4",
    payloadName: "LIDAR",
    planeHardwareVersion: "4",
    label: "LIDAR (WingtraOne)",
  },
  {
    payloadKind: "lidar",
    payloadUniqueString: "LIDAR_v5",
    payloadName: "LIDAR",
    planeHardwareVersion: "5",
    label: "LIDAR (WingtraRay)",
  },
];

const CAMERA_KEY_TO_WINGTRA_PAYLOADS: Record<string, string[]> = {
  SONY_RX1R2: ["RX1R2_v4", "RX1R2_v5"],
  SONY_RX1R3: ["RX1R3_v5"],
  SONY_A6100_20MM: ["A6100_v5"],
  ILX_LR1_INSPECT_85MM: ["MAPSTARHighRes_v4", "MAPSTARHighRes_v5"],
  MAP61_17MM: ["MAPSTAROblique_v4", "MAPSTAROblique_v5"],
  RGB61_24MM: ["RGB61_v4"],
  DJI_ZENMUSE_P1_24MM: ["MAPSTARNadir_v5"],
};

const LIDAR_KEY_TO_WINGTRA_PAYLOADS: Record<string, string[]> = {
  WINGTRA_LIDAR_XT32M2X: ["LIDAR_v4", "LIDAR_v5"],
};

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

function isWingtraAreaItem(value: unknown): value is WingtraAreaItem {
  if (!value || typeof value !== "object") return false;
  const maybeItem = value as { type?: unknown; complexItemType?: unknown };
  return maybeItem.type === "ComplexItem" && maybeItem.complexItemType === "area";
}

function isWingtraTakeoffItem(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object") return false;
  const maybeItem = value as { type?: unknown; complexItemType?: unknown; simpleItemType?: unknown };
  return (
    (maybeItem.type === "ComplexItem" && maybeItem.complexItemType === "takeoff")
    || (maybeItem.type === "SimpleItem" && maybeItem.simpleItemType === "takeoff")
  );
}

function isWingtraLandingItem(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object") return false;
  const maybeItem = value as { type?: unknown; complexItemType?: unknown; simpleItemType?: unknown };
  return (
    (maybeItem.type === "ComplexItem" && maybeItem.complexItemType === "land")
    || (maybeItem.type === "SimpleItem" && maybeItem.simpleItemType === "landing")
  );
}

function isWingtraDroppableOptimizedSequenceItem(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object") return false;
  const maybeItem = value as { type?: unknown; complexItemType?: unknown; simpleItemType?: unknown };
  const itemType =
    (typeof maybeItem.complexItemType === "string" ? maybeItem.complexItemType : undefined)
    ?? (typeof maybeItem.simpleItemType === "string" ? maybeItem.simpleItemType : undefined);
  return itemType === "loiter" || itemType === "waypoint";
}

function readFiniteNumber(value: unknown): number | undefined {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function readWingtraMissionItemPoint(item: Record<string, unknown>): LngLat | undefined {
  const coordinate = item.coordinate;
  if (!Array.isArray(coordinate) || coordinate.length < 2) return undefined;
  const lat = readFiniteNumber(coordinate[0]);
  const lng = readFiniteNumber(coordinate[1]);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return undefined;
  if (Math.abs(lat!) < 1e-9 && Math.abs(lng!) < 1e-9) return undefined;
  if (lat! < -90 || lat! > 90 || lng! < -180 || lng! > 180) return undefined;
  return [lng!, lat!];
}

export interface WingtraSequenceEndpoint {
  point: LngLat;
  altitudeWgs84M: number;
  headingDeg?: number;
  loiterRadiusM?: number;
}

export interface WingtraSequenceEndpoints {
  startEndpoint?: WingtraSequenceEndpoint;
  endEndpoint?: WingtraSequenceEndpoint;
}

// Wingtra takeoff/landing file items store these altitudes relative to the
// takeoff/home altitude, not as raw AMSL/WGS84 values. This matches the
// WingtraCloud generator, which writes:
// - takeoff `exitAltitude = loiterExitAMSL - takeOffPosition[2]`
// - landing `loiterEntryAltitude = loiterEntryAMSL - takeOffAltitude`
// So we convert them back into approximate WGS84 altitudes using the stored
// planned home altitude here.
function resolveWingtraHomeAltitudeWgs84M(template: WingtraFlightPlan | null | undefined): number {
  const plannedHomePosition = template?.flightPlan?.plannedHomePosition;
  if (Array.isArray(plannedHomePosition) && plannedHomePosition.length >= 3) {
    const homeAltitude = readFiniteNumber(plannedHomePosition[2]);
    if (Number.isFinite(homeAltitude)) return homeAltitude!;
  }
  const originCoordinate = template?.flightPlanOrigin?.coordinate;
  if (Array.isArray(originCoordinate) && originCoordinate.length >= 3) {
    const originAltitude = readFiniteNumber(originCoordinate[2]);
    if (Number.isFinite(originAltitude)) return originAltitude!;
  }
  return 0;
}

function readWingtraTakeoffEndpoint(
  item: Record<string, unknown>,
  homeAltitudeWgs84M: number,
): WingtraSequenceEndpoint | undefined {
  const point = readWingtraMissionItemPoint(item);
  const relativeAltitudeM = readFiniteNumber(item.exitAltitude);
  const altitudeWgs84M = Number.isFinite(relativeAltitudeM) ? homeAltitudeWgs84M + relativeAltitudeM! : undefined;
  if (!point || !Number.isFinite(altitudeWgs84M)) return undefined;
  const headingDeg = readFiniteNumber(item.loiterHeading);
  const loiterRadiusM = readFiniteNumber(item.loiterRadius);
  if (!Number.isFinite(loiterRadiusM) || loiterRadiusM! <= 0) return undefined;
  return {
    point,
    altitudeWgs84M: altitudeWgs84M!,
    headingDeg: Number.isFinite(headingDeg) ? headingDeg : undefined,
    loiterRadiusM: loiterRadiusM!,
  };
}

function readWingtraLandingEndpoint(
  item: Record<string, unknown>,
  homeAltitudeWgs84M: number,
): WingtraSequenceEndpoint | undefined {
  const point = readWingtraMissionItemPoint(item);
  const relativeAltitudeM =
    readFiniteNumber(item.loiterEntryAltitude)
    ?? readFiniteNumber(item.transitionAltitude);
  const altitudeWgs84M = Number.isFinite(relativeAltitudeM) ? homeAltitudeWgs84M + relativeAltitudeM! : undefined;
  if (!point || !Number.isFinite(altitudeWgs84M)) return undefined;
  const headingDeg = readFiniteNumber(item.loiterHeading);
  const loiterRadiusM = readFiniteNumber(item.loiterRadius);
  if (!Number.isFinite(loiterRadiusM) || loiterRadiusM! <= 0) return undefined;
  return {
    point,
    altitudeWgs84M: altitudeWgs84M!,
    headingDeg: Number.isFinite(headingDeg) ? headingDeg : undefined,
    loiterRadiusM: loiterRadiusM!,
  };
}

export function extractWingtraSequenceEndpoints(
  template: WingtraFlightPlan | null | undefined,
): WingtraSequenceEndpoints {
  const items = Array.isArray(template?.flightPlan?.items) ? template.flightPlan.items : [];
  const takeoffItem = items.find((item) => isWingtraTakeoffItem(item));
  const landingItem = items.find((item) => isWingtraLandingItem(item));
  const homeAltitudeWgs84M = resolveWingtraHomeAltitudeWgs84M(template);
  return {
    startEndpoint: takeoffItem ? readWingtraTakeoffEndpoint(takeoffItem, homeAltitudeWgs84M) : undefined,
    endEndpoint: landingItem ? readWingtraLandingEndpoint(landingItem, homeAltitudeWgs84M) : undefined,
  };
}

export function getWingtraFreshExportPayloadOptions(
  payloadKind?: PayloadKind,
  planeHardwareVersion?: WingtraPlaneHardwareVersion,
): WingtraFreshExportPayloadOption[] {
  return WINGTRA_FRESH_EXPORT_PAYLOAD_OPTIONS.filter((option) => {
    if (payloadKind && option.payloadKind !== payloadKind) return false;
    if (planeHardwareVersion && option.planeHardwareVersion !== planeHardwareVersion) return false;
    return true;
  });
}

export function getWingtraFreshExportPayloadOption(
  payloadUniqueString: string | undefined,
): WingtraFreshExportPayloadOption | undefined {
  if (!payloadUniqueString) return undefined;
  return WINGTRA_FRESH_EXPORT_PAYLOAD_OPTIONS.find((option) => option.payloadUniqueString === payloadUniqueString);
}

export function getPlaneHardwareVersionFromWingtraPayloadUniqueString(
  payloadUniqueString: string | undefined,
): WingtraPlaneHardwareVersion | undefined {
  if (!payloadUniqueString) return undefined;
  const explicit = getWingtraFreshExportPayloadOption(payloadUniqueString)?.planeHardwareVersion;
  if (explicit) return explicit;
  if (payloadUniqueString.endsWith("_v4")) return "4";
  if (payloadUniqueString.endsWith("_v5")) return "5";
  return undefined;
}

export function getSuggestedFreshWingtraExportPayloadOption(
  areas: ExportedArea[],
): WingtraFreshExportPayloadOption {
  const firstArea = areas[0];
  if (!firstArea) {
    return getWingtraFreshExportPayloadOption("MAPSTARNadir_v5")!;
  }

  const payloadKind = firstArea.payloadKind ?? "camera";
  const payloadUniqueString =
    payloadKind === "lidar"
      ? (LIDAR_KEY_TO_WINGTRA_PAYLOADS[firstArea.lidarKey ?? DEFAULT_LIDAR.key] ?? ["LIDAR_v5"]).at(-1) ?? "LIDAR_v5"
      : (CAMERA_KEY_TO_WINGTRA_PAYLOADS[firstArea.cameraKey ?? ""] ?? ["MAPSTARNadir_v5"]).at(-1) ?? "MAPSTARNadir_v5";

  return (
    getWingtraFreshExportPayloadOption(payloadUniqueString) ??
    getWingtraFreshExportPayloadOptions(payloadKind)[0] ??
    getWingtraFreshExportPayloadOption("MAPSTARNadir_v5")!
  );
}

export interface WingtraFreshExportResolution {
  payloadKind: PayloadKind;
  payloadLabel: string;
  analysisKey: string;
  options: WingtraFreshExportPayloadOption[];
}

export function getCompatibleWingtraPayloadOptionsForAnalysisParams(
  params: Pick<FlightParams, "payloadKind" | "cameraKey" | "lidarKey">,
): WingtraFreshExportPayloadOption[] {
  const payloadKind = params.payloadKind ?? "camera";
  const payloadUniqueStrings =
    payloadKind === "lidar"
      ? LIDAR_KEY_TO_WINGTRA_PAYLOADS[params.lidarKey ?? DEFAULT_LIDAR.key] ?? []
      : CAMERA_KEY_TO_WINGTRA_PAYLOADS[params.cameraKey ?? ""] ?? [];

  return payloadUniqueStrings
    .map((payloadUniqueString) => getWingtraFreshExportPayloadOption(payloadUniqueString))
    .filter((option): option is WingtraFreshExportPayloadOption => !!option);
}

export function getPreferredWingtraPlaneHardwareVersionForAnalysisParams(
  params: Pick<FlightParams, "payloadKind" | "cameraKey" | "lidarKey">,
): WingtraPlaneHardwareVersion | undefined {
  return getCompatibleWingtraPayloadOptionsForAnalysisParams(params).at(-1)?.planeHardwareVersion;
}

export function resolveFreshWingtraExportOptionsFromAreas(
  areas: ExportedArea[],
): WingtraFreshExportResolution | undefined {
  const firstArea = areas[0];
  if (!firstArea) return undefined;

  const payloadKind = firstArea.payloadKind ?? "camera";
  const analysisKeys = new Set(
    areas.map((area) =>
      payloadKind === "lidar" ? area.lidarKey ?? DEFAULT_LIDAR.key : area.cameraKey ?? "DJI_ZENMUSE_P1_24MM",
    ),
  );

  if (analysisKeys.size !== 1) {
    return undefined;
  }

  const analysisKey = Array.from(analysisKeys)[0];
  const options = getCompatibleWingtraPayloadOptionsForAnalysisParams({
    payloadKind,
    cameraKey: payloadKind === "camera" ? analysisKey : undefined,
    lidarKey: payloadKind === "lidar" ? analysisKey : undefined,
  });

  if (options.length === 0) {
    return undefined;
  }

  return {
    payloadKind,
    analysisKey,
    payloadLabel: options[0].payloadName,
    options,
  };
}

export function resolveFreshWingtraExportPayloadOptionFromAreas(
  areas: ExportedArea[],
): WingtraFreshExportPayloadOption | undefined {
  const resolution = resolveFreshWingtraExportOptionsFromAreas(areas);
  if (!resolution) return undefined;

  const explicitVersions = new Set(
    areas
      .map((area) => area.planeHardwareVersion)
      .filter((version): version is WingtraPlaneHardwareVersion => version === "4" || version === "5"),
  );
  if (explicitVersions.size > 1) return undefined;

  const requestedVersion =
    explicitVersions.size === 1
      ? Array.from(explicitVersions)[0]
      : getPreferredWingtraPlaneHardwareVersionForAnalysisParams({
          payloadKind: resolution.payloadKind,
          cameraKey: resolution.payloadKind === "camera" ? resolution.analysisKey : undefined,
          lidarKey: resolution.payloadKind === "lidar" ? resolution.analysisKey : undefined,
        });

  return (
    resolution.options.find((option) => option.planeHardwareVersion === requestedVersion) ??
    resolution.options.at(-1) ??
    resolution.options[0]
  );
}

export function isWingtraFlightPlanTemplateExportReady(value: unknown): value is WingtraFlightPlan {
  if (!isWingtraFlightPlan(value)) return false;
  const payloadUniqueString = value.flightPlan?.payloadUniqueString;
  const planeHardwareVersion = (value.flightPlan as { planeHardware?: { hwVersion?: unknown } })?.planeHardware?.hwVersion;

  if (typeof payloadUniqueString !== "string" || payloadUniqueString.length === 0) return false;
  if (planeHardwareVersion !== "4" && planeHardwareVersion !== "5") return false;
  if (getPlaneHardwareVersionFromWingtraPayloadUniqueString(payloadUniqueString) !== planeHardwareVersion) return false;
  if (!value.geofence || typeof value.geofence !== "object") return false;
  if (!value.safety || typeof value.safety !== "object") return false;
  if (typeof (value.flightPlan as { creationTime?: unknown }).creationTime !== "number") return false;
  return true;
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
  { key: 'SONY_RX1R3', model: SONY_RX1R3 },
  { key: 'SONY_A6100_20MM', model: SONY_A6100_20MM },
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
  const planeHardwareVersion =
    ((fp.flightPlan as { planeHardware?: { hwVersion?: unknown } }).planeHardware?.hwVersion === "4" ||
    (fp.flightPlan as { planeHardware?: { hwVersion?: unknown } }).planeHardware?.hwVersion === "5"
      ? (fp.flightPlan as { planeHardware?: { hwVersion?: WingtraPlaneHardwareVersion } }).planeHardware?.hwVersion
      : undefined) ??
    getPlaneHardwareVersionFromWingtraPayloadUniqueString(payloadKey);
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
        planeHardwareVersion,
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
        planeHardwareVersion,
        speedMps: Number.isFinite(cruiseSpeedMps) && cruiseSpeedMps > 0 ? cruiseSpeedMps : undefined,
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
    planeHardwareVersion,
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
  const camera = opts?.camera ?? SONY_RX1R2;
  const lidar = opts?.lidar ?? DEFAULT_LIDAR;
  const selectedPayloadOption =
    getWingtraFreshExportPayloadOption(opts?.payloadUniqueString) ??
    getSuggestedFreshWingtraExportPayloadOption(areas);
  const selectedPayloadKind = selectedPayloadOption.payloadKind;
  const effectivePayloadKind = opts?.payloadKind ?? areas[0]?.payloadKind ?? selectedPayloadKind;
  const planeHardwareVersion =
    getPlaneHardwareVersionFromWingtraPayloadUniqueString(selectedPayloadOption.payloadUniqueString) ?? "5";
  const now = Date.now();

  // Optional safety defaults
  const safety = {
    rthMode: opts?.defaults?.rthMode ?? WICTopLevelDefaults.safety.rthMode,
    version: WICTopLevelDefaults.safety.version,
    maxGroundClearance: opts?.defaults?.maxGroundClearance ?? WICTopLevelDefaults.safety.maxGroundClearance,
    minGroundClearance: opts?.defaults?.minGroundClearance ?? WICTopLevelDefaults.safety.minGroundClearance,
    ceilingAboveTakeOff: opts?.defaults?.ceilingAboveTakeOff ?? WICTopLevelDefaults.safety.ceilingAboveTakeOff,
    connectionLossTimeout:
      opts?.defaults?.connectionLossTimeout ?? WICTopLevelDefaults.safety.connectionLossTimeout,
    minRTHHeightAboveHome:
      opts?.defaults?.minRTHHeightAboveHome ?? WICTopLevelDefaults.safety.minRTHHeightAboveHome,
  };

  const items = areas.map((a) => {
    const areaPayloadKind = a.payloadKind ?? effectivePayloadKind;
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
    version: WICTopLevelDefaults.fileVersion,
    fileType: "Plan",
    flightId: cryptoRandomUuid(),
    flightPlanOrigin: {
      location: WICTopLevelDefaults.creationLocation,
      coordinate: [...WICTopLevelDefaults.plannedHomePosition],
    },
    geofence: {
      version: 1,
      geofenceType: WICTopLevelDefaults.geofenceType,
      geofenceRadius: opts?.geofenceRadius ?? WICTopLevelDefaults.geofenceRadius,
      terminationSettings: null,
    },
    flightPlan: {
      items,
      payload: opts?.payloadName ?? selectedPayloadOption.payloadName,
      payloadUniqueString: selectedPayloadOption.payloadUniqueString,
      version: WICTopLevelDefaults.flightPlanVersion,
      gisItems: [],
      activeMaxTelemetryDistance: 0,
      activeNumberOfImages: 0,
      activeTotalArea: 0,
      activeTotalFlightCruiseDistance: 0,
      activeTotalFlightCruiseTime: 0,
      activeTotalFlightDistance: 0,
      activeTotalFlightHoverDistance: 0,
      activeTotalFlightHoverTime: 0,
      activeTotalFlightTime: 0,
      creationTime: now,
      hoverSpeed: opts?.defaults?.hoverSpeed ?? WICTopLevelDefaults.hoverSpeed,
      cruiseSpeed: opts?.defaults?.cruiseSpeed ?? WICTopLevelDefaults.cruiseSpeed,
      elevationData: { ...WICTopLevelDefaults.elevationData },
      flightNumber: 0,
      missionStatus: 0,
      flownPercentage: 0,
      planeHardware: {
        hwVersion: planeHardwareVersion,
        vehicleId: -1,
        displayName: planeHardwareVersion === "5" ? "WingtraRay (any)" : "WingtraOne (any)",
        isGenericPlane: true,
      },
      numberOfImages: 0,
      totalArea: 0,
      totalFlightCruiseDistance: 0,
      totalFlightCruiseTime: 0,
      totalFlightDistance: 0,
      totalFlightHoverDistance: 0,
      totalFlightHoverTime: 0,
      totalFlightTime: 0,
      maxTelemetryDistance: 0,
      lastModifiedTime: now,
      plannedHomePosition: [...WICTopLevelDefaults.plannedHomePosition],
      resumeMissionIndex: 0,
      resumeGridPointIndex: -1,
      vehicleLastFlownCoordinate: [...WICTopLevelDefaults.vehicleLastFlownCoordinate],
    } as any,
    groundStation: WICTopLevelDefaults.groundStation,
    flightPlanHistory: [],
  };

  return fp;
}

export function replaceAreaItemsInWingtraFlightPlan(
  template: WingtraFlightPlan,
  areas: ExportedArea[],
  opts?: ExportToWingtraOptions,
): WingtraFlightPlan {
  const exportedAreaItems = exportToWingtraFlightPlan(areas, opts).flightPlan.items;
  const originalItems = Array.isArray(template.flightPlan?.items) ? template.flightPlan.items : [];
  const mergedItems: Array<WingtraAreaItem | Record<string, unknown>> = [];
  let nextExportedAreaIndex = 0;
  let lastInsertedAreaIndex = -1;

  for (const item of originalItems) {
    if (!isWingtraAreaItem(item)) {
      mergedItems.push(item);
      continue;
    }

    if (nextExportedAreaIndex < exportedAreaItems.length) {
      mergedItems.push(exportedAreaItems[nextExportedAreaIndex] as WingtraAreaItem | Record<string, unknown>);
      lastInsertedAreaIndex = mergedItems.length - 1;
      nextExportedAreaIndex += 1;
    }
  }

  if (nextExportedAreaIndex < exportedAreaItems.length) {
    const remainingAreas = exportedAreaItems.slice(nextExportedAreaIndex) as Array<WingtraAreaItem | Record<string, unknown>>;
    const insertIndex = lastInsertedAreaIndex >= 0 ? lastInsertedAreaIndex + 1 : mergedItems.length;
    mergedItems.splice(insertIndex, 0, ...remainingAreas);
  }

  return {
    ...template,
    flightPlan: {
      ...template.flightPlan,
      items: mergedItems,
    },
  };
}

export function replaceAreaItemsInWingtraFlightPlanForOptimizedSequence(
  template: WingtraFlightPlan,
  areas: ExportedArea[],
  opts?: ExportToWingtraOptions,
): WingtraFlightPlan {
  const exportedAreaItems = exportToWingtraFlightPlan(areas, opts).flightPlan.items;
  const originalItems = Array.isArray(template.flightPlan?.items) ? template.flightPlan.items : [];
  const mergedItems: Array<WingtraAreaItem | Record<string, unknown>> = [];
  let nextExportedAreaIndex = 0;
  let lastInsertedAreaIndex = -1;

  for (const item of originalItems) {
    if (isWingtraDroppableOptimizedSequenceItem(item)) {
      continue;
    }
    if (!isWingtraAreaItem(item)) {
      mergedItems.push(item);
      continue;
    }
    if (nextExportedAreaIndex < exportedAreaItems.length) {
      mergedItems.push(exportedAreaItems[nextExportedAreaIndex] as WingtraAreaItem | Record<string, unknown>);
      lastInsertedAreaIndex = mergedItems.length - 1;
      nextExportedAreaIndex += 1;
    }
  }

  if (nextExportedAreaIndex < exportedAreaItems.length) {
    const remainingAreas = exportedAreaItems.slice(nextExportedAreaIndex) as Array<WingtraAreaItem | Record<string, unknown>>;
    const insertIndex = lastInsertedAreaIndex >= 0 ? lastInsertedAreaIndex + 1 : mergedItems.length;
    mergedItems.splice(insertIndex, 0, ...remainingAreas);
  }
  return {
    ...template,
    flightPlan: {
      ...template.flightPlan,
      items: mergedItems,
    },
  };
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
    planeHardwareVersion: p.params.planeHardwareVersion,
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
