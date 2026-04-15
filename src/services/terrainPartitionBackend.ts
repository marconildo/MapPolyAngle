import type { FlightParams } from '@/domain/types';
import type { TerrainPartitionSolutionPreview } from '@/terrain-partition/types';
import type { TerrainSourceSelection } from '@/terrain/types';
import { configuredBackendBaseUrl, normalizedConfiguredBackendBaseUrl } from '@/services/backendBaseUrl';

export interface TerrainPartitionBackendRequest {
  polygonId?: string;
  ring: [number, number][];
  payloadKind: 'camera' | 'lidar';
  params: FlightParams;
  terrainSource: TerrainSourceSelection;
  altitudeMode: 'legacy' | 'min-clearance';
  minClearanceM: number;
  turnExtendM: number; // deprecated compatibility field; backend keeps accepting it while ignoring it
  tradeoff?: number;
  debug?: boolean;
}

interface TerrainPartitionBackendResponse {
  requestId: string;
  solutions: TerrainPartitionSolutionPreview[];
}

function configuredBackendDebugEnabled(): boolean {
  const fromImportMeta = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_TERRAIN_PARTITION_BACKEND_DEBUG;
  const fromProcess = typeof process !== 'undefined' ? process.env.VITE_TERRAIN_PARTITION_BACKEND_DEBUG : undefined;
  const configured = [fromImportMeta, fromProcess].find(
    (value): value is string => typeof value === 'string' && value.trim().length > 0
  );
  return typeof configured === 'string' && /^(1|true|yes|on)$/i.test(configured.trim());
}

export function isTerrainPartitionBackendEnabled(): boolean {
  const backendBaseUrl = configuredBackendBaseUrl();
  return typeof backendBaseUrl === 'string' && backendBaseUrl.trim().length > 0;
}

export async function solveTerrainPartitionWithBackend(
  request: TerrainPartitionBackendRequest,
): Promise<TerrainPartitionSolutionPreview[]> {
  const backendBaseUrl = normalizedConfiguredBackendBaseUrl();
  if (!isTerrainPartitionBackendEnabled()) {
    throw new Error('Terrain partition backend is not configured.');
  }
  const effectiveRequest: TerrainPartitionBackendRequest = {
    ...request,
    debug: request.debug ?? configuredBackendDebugEnabled(),
  };
  const endpoint = `${backendBaseUrl!}/v1/partition/solve`;
  console.log('[terrain-split][backend-request] POST /v1/partition/solve', {
    polygonId: effectiveRequest.polygonId ?? null,
    payloadKind: effectiveRequest.payloadKind,
    terrainMode: effectiveRequest.terrainSource.mode,
    datasetId: effectiveRequest.terrainSource.datasetId ?? null,
    ringPoints: effectiveRequest.ring.length,
    debug: effectiveRequest.debug ?? false,
    endpoint,
  });
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(effectiveRequest),
  });
  if (!response.ok) {
    const text = await response.text();
    console.warn('[terrain-split][backend-request] POST /v1/partition/solve failed', {
      polygonId: effectiveRequest.polygonId ?? null,
      status: response.status,
      detail: text || null,
    });
    throw new Error(text || `Backend request failed with status ${response.status}`);
  }
  const payload = await response.json() as TerrainPartitionBackendResponse;
  const solutions = Array.isArray(payload.solutions) ? payload.solutions : [];
  console.log('[terrain-split][backend-request] POST /v1/partition/solve succeeded', {
    polygonId: effectiveRequest.polygonId ?? null,
    requestId: payload.requestId,
    solutionCount: solutions.length,
    regionCounts: solutions.map((solution) => solution.regionCount),
    rankingSources: solutions.map((solution) => solution.rankingSource ?? 'surrogate'),
  });
  return solutions;
}
