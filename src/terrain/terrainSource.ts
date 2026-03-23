import { clearActiveDsm, loadDsmFromFile } from './dsmSource';
import type { DsmSourceDescriptor, TerrainSourceMode, TerrainSourceSelection, TerrainSourceState } from './types';
import {
  getDsmDatasetFromTerrainBackend,
  getTerrainTileUrlForSource,
  getTerrainTileUrlTemplateForSource,
  isDsmTerrainBackendEnabled,
  listDsmDatasetsFromTerrainBackend,
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

function sortDescriptors(descriptors: DsmSourceDescriptor[]): DsmSourceDescriptor[] {
  return [...descriptors].sort((left, right) => right.loadedAtIso.localeCompare(left.loadedAtIso));
}

function upsertDescriptor(descriptors: DsmSourceDescriptor[], descriptor: DsmSourceDescriptor): DsmSourceDescriptor[] {
  return sortDescriptors([...descriptors.filter((candidate) => candidate.id !== descriptor.id), descriptor]);
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
    datasets: [],
    isLoading: false,
    isDatasetListLoading: false,
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
  await refreshTerrainSourceDatasets();
}

export async function refreshTerrainSourceDatasets(): Promise<DsmSourceDescriptor[]> {
  if (!isDsmTerrainBackendEnabled()) {
    terrainSourceState = {
      ...terrainSourceState,
      datasets: [],
      isDatasetListLoading: false,
    };
    emit();
    return [];
  }

  terrainSourceState = {
    ...terrainSourceState,
    isDatasetListLoading: true,
  };
  emit({ persist: false });

  try {
    const datasets = sortDescriptors(
      (await listDsmDatasetsFromTerrainBackend())
        .map((dataset) => dataset.descriptor)
        .filter((descriptor): descriptor is DsmSourceDescriptor => descriptor != null)
    );
    const persisted = readPersistedTerrainSelection();
    const preferredId = terrainSourceState.descriptor?.id ?? persisted?.selectedDatasetId ?? null;
    const selectedDescriptor = preferredId ? datasets.find((dataset) => dataset.id === preferredId) ?? null : null;
    const nextMode =
      selectedDescriptor && (terrainSourceState.source.mode === 'blended' || persisted?.mode === 'blended')
        ? 'blended'
        : 'mapbox';
    terrainSourceState = {
      ...terrainSourceState,
      datasets,
      descriptor: selectedDescriptor,
      source: {
        mode: nextMode,
        datasetId: nextMode === 'blended' ? selectedDescriptor?.id ?? null : null,
      },
      isDatasetListLoading: false,
      error: null,
    };
    emit();
    return datasets;
  } catch (error) {
    terrainSourceState = {
      ...terrainSourceState,
      isDatasetListLoading: false,
      error: error instanceof Error ? error.message : String(error),
    };
    emit();
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
    const existing = terrainSourceState.datasets.find((dataset) => dataset.id === normalizedDatasetId) ?? null;
    const fetched = existing ?? (await getDsmDatasetFromTerrainBackend(normalizedDatasetId)).descriptor;
    if (!fetched) {
      throw new Error(`DSM dataset ${normalizedDatasetId} is not available.`);
    }
    terrainSourceState = {
      ...terrainSourceState,
      datasets: upsertDescriptor(terrainSourceState.datasets, fetched),
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

export async function loadTerrainSourceFromFile(file: File): Promise<DsmSourceDescriptor> {
  terrainSourceState = {
    ...terrainSourceState,
    isLoading: true,
    error: null,
  };
  emit();

  try {
    if (isDsmTerrainBackendEnabled()) {
      try {
        const uploaded = await uploadDsmToTerrainBackend(file);
        clearActiveDsm();
        if (!uploaded.descriptor || !uploaded.datasetId) {
          throw new Error('Terrain backend did not return a usable DSM descriptor.');
        }
        terrainSourceState = {
          ...terrainSourceState,
          datasets: upsertDescriptor(terrainSourceState.datasets, uploaded.descriptor),
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
      datasets: upsertDescriptor(terrainSourceState.datasets, localDescriptor),
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
