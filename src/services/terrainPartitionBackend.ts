import type { FlightParams } from '@/domain/types';
import type { TerrainPartitionSolutionPreview } from '@/terrain-partition/types';
import type { TerrainSourceSelection } from '@/terrain/types';

export interface TerrainPartitionBackendRequest {
  polygonId?: string;
  ring: [number, number][];
  payloadKind: 'camera' | 'lidar';
  params: FlightParams;
  terrainSource: TerrainSourceSelection;
  altitudeMode: 'legacy' | 'min-clearance';
  minClearanceM: number;
  turnExtendM: number;
  tradeoff?: number;
  debug?: boolean;
}

interface TerrainPartitionBackendResponse {
  requestId: string;
  solutions: TerrainPartitionSolutionPreview[];
}

const backendBaseUrl = import.meta.env.VITE_TERRAIN_PARTITION_BACKEND_URL as string | undefined;

export function isTerrainPartitionBackendEnabled(): boolean {
  return typeof backendBaseUrl === 'string' && backendBaseUrl.trim().length > 0;
}

export async function solveTerrainPartitionWithBackend(
  request: TerrainPartitionBackendRequest,
): Promise<TerrainPartitionSolutionPreview[]> {
  if (!isTerrainPartitionBackendEnabled()) {
    throw new Error('Terrain partition backend is not configured.');
  }
  const endpoint = `${backendBaseUrl!.replace(/\/$/, '')}/v1/partition/solve`;
  console.log('[terrain-split][backend-request] POST /v1/partition/solve', {
    polygonId: request.polygonId ?? null,
    payloadKind: request.payloadKind,
    terrainMode: request.terrainSource.mode,
    datasetId: request.terrainSource.datasetId ?? null,
    ringPoints: request.ring.length,
    endpoint,
  });
  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const text = await response.text();
    console.warn('[terrain-split][backend-request] POST /v1/partition/solve failed', {
      polygonId: request.polygonId ?? null,
      status: response.status,
      detail: text || null,
    });
    throw new Error(text || `Backend request failed with status ${response.status}`);
  }
  const payload = await response.json() as TerrainPartitionBackendResponse;
  const solutions = Array.isArray(payload.solutions) ? payload.solutions : [];
  console.log('[terrain-split][backend-request] POST /v1/partition/solve succeeded', {
    polygonId: request.polygonId ?? null,
    requestId: payload.requestId,
    solutionCount: solutions.length,
    regionCounts: solutions.map((solution) => solution.regionCount),
    rankingSources: solutions.map((solution) => solution.rankingSource ?? 'surrogate'),
  });
  return solutions;
}
