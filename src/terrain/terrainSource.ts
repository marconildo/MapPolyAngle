import { clearActiveDsm, loadDsmFromFile, validateDsmGeoTiffFile } from './dsmSource';
import type { DsmSourceDescriptor, TerrainSourceMode, TerrainSourceSelection, TerrainSourceState } from './types';
import {
  findDsmDatasetFromTerrainBackend,
  getDsmDatasetFromTerrainBackend,
  getTerrainTileUrlForSource,
  getTerrainTileUrlTemplateForSource,
  isDsmTerrainBackendEnabled,
  uploadDsmToTerrainBackend,
} from '@/services/dsmTerrainBackend';

const listeners = new Set<() => void>();
const STORAGE_KEY = 'terrain-source-selection-v1';

interface PersistedTerrainSelection {
  mode: TerrainSourceMode;
  selectedDatasetId: string | null;
}

let terrainSourceState: TerrainSourceState = initialTerrainSourceState();

function readPersistedTerrainSelection(): PersistedTerrainSelection | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PersistedTerrainSelection>;
    return {
      mode: parsed.mode === 'blended' ? 'blended' : 'mapbox',
      selectedDatasetId:
        typeof parsed.selectedDatasetId === 'string' && parsed.selectedDatasetId.trim().length > 0
          ? parsed.selectedDatasetId
          : null,
    };
  } catch {
    return null;
  }
}

function persistTerrainSelection() {
  if (typeof window === 'undefined') return;
  try {
    const payload: PersistedTerrainSelection = {
      mode: terrainSourceState.source.mode,
      selectedDatasetId: terrainSourceState.descriptor?.id ?? null,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Ignore localStorage failures.
  }
}

function emit(options?: { persist?: boolean }) {
  terrainSourceState = {
    ...terrainSourceState,
    backendEnabled: isDsmTerrainBackendEnabled(),
  };
  if (options?.persist !== false) {
    persistTerrainSelection();
  }
  for (const listener of listeners) listener();
}

function initialTerrainSourceState(): TerrainSourceState {
  return {
    source: { mode: 'mapbox', datasetId: null },
    descriptor: null,
    isLoading: false,
    error: null,
    backendEnabled: isDsmTerrainBackendEnabled(),
  };
}

export function subscribeTerrainSource(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getTerrainSourceState(): TerrainSourceState {
  return terrainSourceState;
}

export function getCurrentTerrainSource(): TerrainSourceSelection {
  return terrainSourceState.source;
}

export function getCurrentTerrainDescriptor(): DsmSourceDescriptor | null {
  return terrainSourceState.descriptor;
}

export function getTerrainDemUrlTemplateForCurrentSource(): string | null {
  return getTerrainTileUrlTemplateForSource(terrainSourceState.source);
}

export function getTerrainTileUrlForCurrentSource(z: number, x: number, y: number): string | null {
  return getTerrainTileUrlForSource(terrainSourceState.source, z, x, y);
}

export function setTerrainSourceMode(mode: TerrainSourceMode) {
  const nextMode: TerrainSourceMode = mode === 'blended' && terrainSourceState.descriptor ? 'blended' : 'mapbox';
  terrainSourceState = {
    ...terrainSourceState,
    source: {
      mode: nextMode,
      datasetId: nextMode === 'blended' ? (terrainSourceState.descriptor?.id ?? null) : null,
    },
    error: nextMode === 'blended' || terrainSourceState.error == null ? terrainSourceState.error : null,
  };
  emit();
}

export function clearTerrainSourceSelection() {
  clearActiveDsm();
  terrainSourceState = {
    ...terrainSourceState,
    source: { mode: 'mapbox', datasetId: null },
    descriptor: null,
    isLoading: false,
    error: null,
  };
  emit();
}

export function __resetTerrainSourceForTests(options?: { clearStorage?: boolean }) {
  clearActiveDsm();
  listeners.clear();
  terrainSourceState = initialTerrainSourceState();
  if (options?.clearStorage !== false && typeof window !== 'undefined') {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage cleanup failures in tests.
    }
  }
}

export async function initializeTerrainSourceState(): Promise<void> {
  if (!isDsmTerrainBackendEnabled()) return;
  const persisted = readPersistedTerrainSelection();
  if (persisted?.mode !== 'blended' || !persisted.selectedDatasetId) {
    terrainSourceState = {
      ...terrainSourceState,
      source: { mode: 'mapbox', datasetId: null },
      descriptor: null,
      isLoading: false,
      error: null,
    };
    emit({ persist: false });
    return;
  }

  terrainSourceState = {
    ...terrainSourceState,
    isLoading: true,
    error: null,
  };
  emit({ persist: false });

  try {
    const fetched = await findDsmDatasetFromTerrainBackend(persisted.selectedDatasetId);
    if (!fetched?.descriptor) {
      clearTerrainSourceSelection();
      return;
    }
    terrainSourceState = {
      ...terrainSourceState,
      descriptor: fetched.descriptor,
      source: {
        mode: 'blended',
        datasetId: fetched.descriptor.id,
      },
      isLoading: false,
      error: null,
    };
    emit();
  } catch (error) {
    terrainSourceState = {
      ...terrainSourceState,
      isLoading: false,
      error: error instanceof Error ? error.message : String(error),
    };
    emit({ persist: false });
    throw error;
  }
}

export async function selectTerrainSourceDataset(datasetId: string, mode: TerrainSourceMode = 'blended'): Promise<DsmSourceDescriptor> {
  const normalizedDatasetId = datasetId.trim();
  if (!normalizedDatasetId) {
    throw new Error('A DSM dataset id is required.');
  }
  if (!isDsmTerrainBackendEnabled()) {
    throw new Error('Selecting an existing DSM dataset requires a configured terrain backend.');
  }

  terrainSourceState = {
    ...terrainSourceState,
    isLoading: true,
    error: null,
  };
  emit();

  try {
    const fetched = (await getDsmDatasetFromTerrainBackend(normalizedDatasetId)).descriptor;
    if (!fetched) {
      throw new Error(`DSM dataset ${normalizedDatasetId} is not available.`);
    }
    terrainSourceState = {
      ...terrainSourceState,
      descriptor: fetched,
      source: {
        mode: mode === 'blended' ? 'blended' : 'mapbox',
        datasetId: mode === 'blended' ? fetched.id : null,
      },
      isLoading: false,
      error: null,
    };
    emit();
    return fetched;
  } catch (error) {
    terrainSourceState = {
      ...terrainSourceState,
      isLoading: false,
      error: error instanceof Error ? error.message : String(error),
    };
    emit();
    throw error;
  }
}

export async function loadTerrainSourceFromFile(
  file: File,
  options?: { onProgressPhase?: (phase: 'validating' | 'uploading') => void },
): Promise<DsmSourceDescriptor> {
  terrainSourceState = {
    ...terrainSourceState,
    isLoading: true,
    error: null,
  };
  emit();

  try {
    if (isDsmTerrainBackendEnabled()) {
      try {
        options?.onProgressPhase?.('validating');
        await validateDsmGeoTiffFile(file);
        options?.onProgressPhase?.('uploading');
        const uploaded = await uploadDsmToTerrainBackend(file);
        clearActiveDsm();
        if (!uploaded.descriptor || !uploaded.datasetId) {
          throw new Error('Terrain backend did not return a usable DSM descriptor.');
        }
        terrainSourceState = {
          ...terrainSourceState,
          source: { mode: 'blended', datasetId: uploaded.datasetId },
          descriptor: uploaded.descriptor,
          isLoading: false,
          error: null,
        };
        emit();
        return uploaded.descriptor;
      } catch (error) {
        clearActiveDsm();
        throw error;
      }
    }

    const localDescriptor = await loadDsmFromFile(file);
    terrainSourceState = {
      ...terrainSourceState,
      source: { mode: 'blended', datasetId: localDescriptor.id },
      descriptor: localDescriptor,
      isLoading: false,
      error: null,
    };
    emit();
    return localDescriptor;
  } catch (error) {
    clearActiveDsm();
    terrainSourceState = {
      ...terrainSourceState,
      source: { mode: 'mapbox', datasetId: null },
      descriptor: null,
      isLoading: false,
      error: error instanceof Error ? error.message : String(error),
    };
    emit();
    throw error;
  }
}
