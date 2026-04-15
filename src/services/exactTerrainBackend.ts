import type { FlightParams } from "@/domain/types";
import type { TerrainSourceSelection } from "@/terrain/types";
import { configuredBackendBaseUrl, normalizedConfiguredBackendBaseUrl } from "@/services/backendBaseUrl";

export interface ExactBearingBackendRequest {
  polygonId?: string;
  ring: [number, number][];
  payloadKind: "camera" | "lidar";
  params: FlightParams;
  terrainSource: TerrainSourceSelection;
  altitudeMode: "legacy" | "min-clearance";
  minClearanceM: number;
  turnExtendM: number; // deprecated compatibility field; backend keeps accepting it while ignoring it
  seedBearingDeg: number;
  mode?: "local" | "global";
  halfWindowDeg?: number;
}

export interface ExactBearingBackendResponse {
  bearingDeg: number | null;
  exactScore?: number | null;
  qualityCost?: number | null;
  missionTimeSec?: number | null;
  normalizedTimeCost?: number | null;
  metricKind?: "gsd" | "density" | null;
  seedBearingDeg: number;
  lineSpacingM?: number | null;
  diagnostics?: Record<string, number>;
}

export function isExactTerrainBackendEnabled(): boolean {
  const backendBaseUrl = configuredBackendBaseUrl();
  return typeof backendBaseUrl === "string" && backendBaseUrl.trim().length > 0;
}

export async function optimizeBearingWithBackend(
  request: ExactBearingBackendRequest,
): Promise<ExactBearingBackendResponse> {
  const backendBaseUrl = normalizedConfiguredBackendBaseUrl();
  if (!isExactTerrainBackendEnabled()) {
    throw new Error("Exact terrain backend is not configured.");
  }
  const response = await fetch(`${backendBaseUrl!}/v1/exact/optimize-bearing`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Backend exact optimize failed with status ${response.status}`);
  }
  return await response.json() as ExactBearingBackendResponse;
}
