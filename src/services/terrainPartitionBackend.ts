import type { TerrainPartitionSolutionPreview } from '@/components/MapFlightDirection/api';
import type { FlightParams } from '@/domain/types';
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
  const response = await fetch(`${backendBaseUrl!.replace(/\/$/, '')}/v1/partition/solve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Backend request failed with status ${response.status}`);
  }
  const payload = await response.json() as TerrainPartitionBackendResponse;
  return Array.isArray(payload.solutions) ? payload.solutions : [];
}
