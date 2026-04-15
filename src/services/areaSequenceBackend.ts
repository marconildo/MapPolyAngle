import type { FlightParams } from '@/domain/types';
import type { TerrainSourceSelection } from '@/terrain/types';

export interface MissionAreaSequenceBackendArea {
  polygonId: string;
  ring: [number, number][];
  bearingDeg: number;
  payloadKind: 'camera' | 'lidar';
  params: FlightParams;
}

export interface MissionAreaSequenceBackendRequest {
  areas: MissionAreaSequenceBackendArea[];
  terrainSource: TerrainSourceSelection;
  altitudeMode: 'legacy' | 'min-clearance';
  minClearanceM: number;
  turnExtendM: number;
  maxHeightAboveGroundM: number;
}

export interface MissionAreaSequenceBackendChoice {
  polygonId: string;
  orderIndex: number;
  flipped: boolean;
  bearingDeg: number;
}

interface MissionAreaSequenceBackendResponse {
  requestId: string;
  solveMode: 'exact-dp' | 'greedy-fallback';
  solvedExactly: boolean;
  areas: MissionAreaSequenceBackendChoice[];
}

function configuredBackendBaseUrl(): string | undefined {
  const fromImportMeta = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_TERRAIN_PARTITION_BACKEND_URL;
  const fromProcess = typeof process !== 'undefined' ? process.env.VITE_TERRAIN_PARTITION_BACKEND_URL : undefined;
  return [fromImportMeta, fromProcess].find(
    (value): value is string => typeof value === 'string' && value.trim().length > 0,
  );
}

export function isAreaSequenceBackendEnabled(): boolean {
  const backendBaseUrl = configuredBackendBaseUrl();
  return typeof backendBaseUrl === 'string' && backendBaseUrl.trim().length > 0;
}

export async function optimizeAreaSequenceWithBackend(
  request: MissionAreaSequenceBackendRequest,
): Promise<MissionAreaSequenceBackendResponse> {
  const backendBaseUrl = configuredBackendBaseUrl();
  if (!isAreaSequenceBackendEnabled()) {
    throw new Error('Area sequence backend is not configured.');
  }

  const response = await fetch(`${backendBaseUrl!.replace(/\/$/, '')}/v1/mission/optimize-area-sequence`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Area sequence backend request failed with status ${response.status}`);
  }

  const payload = await response.json() as MissionAreaSequenceBackendResponse;
  return {
    requestId: payload.requestId,
    solveMode: payload.solveMode,
    solvedExactly: payload.solvedExactly,
    areas: Array.isArray(payload.areas) ? payload.areas : [],
  };
}
