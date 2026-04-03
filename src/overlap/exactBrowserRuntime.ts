import { fetchTerrainRGBA, LidarDensityWorker, OverlapWorker } from "@/overlap/controller";
import type { PaddedDemTileRGBA, TileRGBA } from "@/overlap/types";
import { getActiveDsmDescriptor } from "@/terrain/dsmSource";
import { getCurrentTerrainSource } from "@/terrain/terrainSource";

import type { ExactRegionRuntime, ExactTerrainProvider, ExactTileEvaluator, ExactTileRef, ExactTileWithHalo } from "./exact-region";

function normalizeTileRef(tileRef: ExactTileRef): ExactTileRef {
  const tilesPerAxis = 1 << tileRef.z;
  return {
    z: tileRef.z,
    x: ((tileRef.x % tilesPerAxis) + tilesPerAxis) % tilesPerAxis,
    y: Math.max(0, Math.min(tilesPerAxis - 1, tileRef.y)),
  };
}

function tileKey(tileRef: ExactTileRef) {
  return `${tileRef.z}/${tileRef.x}/${tileRef.y}`;
}

function yieldToFrame() {
  return new Promise<void>((resolve) => {
    if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
      window.requestAnimationFrame(() => resolve());
      return;
    }
    setTimeout(resolve, 0);
  });
}

function resolveBrowserWorkerPoolSize() {
  const hardwareConcurrency = typeof navigator !== "undefined" && Number.isFinite(navigator.hardwareConcurrency)
    ? navigator.hardwareConcurrency
    : 4;
  return Math.min(4, Math.max(2, Math.floor(hardwareConcurrency / 2)));
}

class WorkerPool<TInput, TOutput, TWorker extends { runTile(args: TInput): Promise<TOutput>; terminate(): void }> {
  private readonly workers: TWorker[];
  private readonly idleWorkers: TWorker[];
  private readonly queue: Array<{
    input: TInput;
    resolve: (value: TOutput) => void;
    reject: (error: unknown) => void;
  }> = [];
  private terminating = false;

  constructor(
    private readonly createWorker: () => TWorker,
    readonly size: number,
  ) {
    this.workers = Array.from({ length: size }, () => createWorker());
    this.idleWorkers = [...this.workers];
  }

  run(input: TInput) {
    return new Promise<TOutput>((resolve, reject) => {
      if (this.terminating) {
        reject(new Error("Worker pool has been terminated."));
        return;
      }
      this.queue.push({ input, resolve, reject });
      this.drain();
    });
  }

  terminate() {
    this.terminating = true;
    for (const worker of this.workers) worker.terminate();
    while (this.queue.length > 0) {
      this.queue.shift()?.reject(new Error("Worker pool has been terminated."));
    }
  }

  private drain() {
    while (!this.terminating && this.idleWorkers.length > 0 && this.queue.length > 0) {
      const worker = this.idleWorkers.pop()!;
      const job = this.queue.shift()!;
      void worker.runTile(job.input)
        .then(job.resolve, job.reject)
        .finally(() => {
          if (!this.terminating) {
            this.idleWorkers.push(worker);
            this.drain();
          }
        });
    }
  }
}

class BrowserExactTerrainProvider implements ExactTerrainProvider {
  private readonly tileCache = new Map<string, { width: number; data: Uint8ClampedArray }>();
  private readonly tileWithHaloCache = new Map<string, ExactTileWithHalo>();

  constructor(private readonly mapboxToken: string) {}

  private async getTile(tileRef: ExactTileRef): Promise<TileRGBA> {
    const normalizedRef = normalizeTileRef(tileRef);
    const key = tileKey(normalizedRef);
    let cached = this.tileCache.get(key);
    if (!cached) {
      const imageData = await fetchTerrainRGBA(normalizedRef.z, normalizedRef.x, normalizedRef.y, this.mapboxToken);
      cached = {
        width: imageData.width,
        data: new Uint8ClampedArray(imageData.data),
      };
      this.tileCache.set(key, cached);
    }
    return {
      z: normalizedRef.z,
      x: normalizedRef.x,
      y: normalizedRef.y,
      size: cached.width,
      data: new Uint8ClampedArray(cached.data),
    };
  }

  async getTerrainTiles(tileRefs: ExactTileRef[]) {
    const entries = await Promise.all(tileRefs.map(async (tileRef) => {
      const normalizedRef = normalizeTileRef(tileRef);
      return [tileKey(normalizedRef), await this.getTile(normalizedRef)] as const;
    }));
    return new Map(entries);
  }

  async getTerrainTilesWithHalo(tileRefs: ExactTileRef[], padTiles: number) {
    const entries = await Promise.all(tileRefs.map(async (tileRef) => {
      const normalizedRef = normalizeTileRef(tileRef);
      const key = `${tileKey(normalizedRef)}@${padTiles}`;
      let cached = this.tileWithHaloCache.get(key);
      if (!cached) {
        const centerTile = await this.getTile(normalizedRef);
        if (padTiles <= 0) {
          const demTile: PaddedDemTileRGBA = {
            size: centerTile.size,
            padTiles: 0,
            data: new Uint8ClampedArray(centerTile.data),
          };
          cached = { tile: centerTile, demTile };
        } else {
          const offsets: Array<{ dx: number; dy: number; tileRef: ExactTileRef }> = [];
          for (let dy = -padTiles; dy <= padTiles; dy++) {
            for (let dx = -padTiles; dx <= padTiles; dx++) {
              offsets.push({
                dx,
                dy,
                tileRef: normalizeTileRef({ z: normalizedRef.z, x: normalizedRef.x + dx, y: normalizedRef.y + dy }),
              });
            }
          }
          const neighbors = await Promise.all(offsets.map((entry) => this.getTile(entry.tileRef)));
          const tileSize = centerTile.size;
          const span = padTiles * 2 + 1;
          const demSize = tileSize * span;
          const demData = new Uint8ClampedArray(demSize * demSize * 4);
          for (let index = 0; index < offsets.length; index++) {
            const { dx, dy } = offsets[index];
            const sourceTile = neighbors[index];
            const offsetX = (dx + padTiles) * tileSize;
            const offsetY = (dy + padTiles) * tileSize;
            for (let row = 0; row < tileSize; row++) {
              const srcStart = row * tileSize * 4;
              const dstStart = ((offsetY + row) * demSize + offsetX) * 4;
              demData.set(sourceTile.data.subarray(srcStart, srcStart + tileSize * 4), dstStart);
            }
          }
          cached = {
            tile: centerTile,
            demTile: {
              size: demSize,
              padTiles,
              data: demData,
            },
          };
        }
        this.tileWithHaloCache.set(key, cached);
      }
      return [tileKey(normalizedRef), {
        tile: {
          ...cached.tile,
          data: new Uint8ClampedArray(cached.tile.data),
        },
        demTile: {
          ...cached.demTile,
          data: new Uint8ClampedArray(cached.demTile.data),
        },
      }] as const;
    }));
    return new Map(entries);
  }
}

class BrowserExactTileEvaluator implements ExactTileEvaluator {
  private readonly cameraWorkers: WorkerPool<
    Parameters<OverlapWorker["runTile"]>[0],
    Awaited<ReturnType<OverlapWorker["runTile"]>>,
    OverlapWorker
  >;
  private readonly lidarWorkers: WorkerPool<
    Parameters<LidarDensityWorker["runTile"]>[0],
    Awaited<ReturnType<LidarDensityWorker["runTile"]>>,
    LidarDensityWorker
  >;

  readonly concurrency: number;

  constructor(poolSize: number) {
    this.concurrency = poolSize;
    this.cameraWorkers = new WorkerPool(() => new OverlapWorker(), poolSize);
    this.lidarWorkers = new WorkerPool(() => new LidarDensityWorker(), poolSize);
  }

  evaluateCameraTile(input: Parameters<OverlapWorker["runTile"]>[0]) {
    return this.cameraWorkers.run(input);
  }

  evaluateLidarTile(input: Parameters<LidarDensityWorker["runTile"]>[0]) {
    return this.lidarWorkers.run(input);
  }

  dispose() {
    this.cameraWorkers.terminate();
    this.lidarWorkers.terminate();
  }
}

export function createBrowserExactRegionRuntime(mapboxToken: string): ExactRegionRuntime {
  const tileEvaluator = new BrowserExactTileEvaluator(resolveBrowserWorkerPoolSize());
  return {
    terrainProvider: new BrowserExactTerrainProvider(mapboxToken),
    tileEvaluator,
    yieldToEventLoop: yieldToFrame,
    candidateConcurrency: tileEvaluator.concurrency,
  };
}

type SharedBrowserRuntimeState = {
  key: string;
  runtime: ExactRegionRuntime;
  inFlight: number;
  stale: boolean;
};

let currentBrowserRuntimeState: SharedBrowserRuntimeState | null = null;
const staleBrowserRuntimeStates: SharedBrowserRuntimeState[] = [];

function getBrowserRuntimeCacheKey(mapboxToken: string) {
  const terrainSource = getCurrentTerrainSource();
  const activeDsm = getActiveDsmDescriptor();
  return JSON.stringify({
    mapboxToken,
    terrainMode: terrainSource.mode,
    terrainDatasetId: terrainSource.datasetId ?? null,
    activeDsmId: activeDsm?.id ?? null,
  });
}

function disposeBrowserRuntimeState(state: SharedBrowserRuntimeState) {
  void state.runtime.tileEvaluator.dispose?.();
}

function cleanupStaleBrowserRuntimeStates() {
  for (let index = staleBrowserRuntimeStates.length - 1; index >= 0; index -= 1) {
    const state = staleBrowserRuntimeStates[index];
    if (state.inFlight === 0) {
      staleBrowserRuntimeStates.splice(index, 1);
      disposeBrowserRuntimeState(state);
    }
  }
}

export async function withBrowserExactRegionRuntime<T>(
  mapboxToken: string,
  callback: (runtime: ExactRegionRuntime) => Promise<T>,
): Promise<T> {
  cleanupStaleBrowserRuntimeStates();
  const key = getBrowserRuntimeCacheKey(mapboxToken);
  if (!currentBrowserRuntimeState || currentBrowserRuntimeState.key !== key) {
    if (currentBrowserRuntimeState) {
      currentBrowserRuntimeState.stale = true;
      staleBrowserRuntimeStates.push(currentBrowserRuntimeState);
    }
    currentBrowserRuntimeState = {
      key,
      runtime: createBrowserExactRegionRuntime(mapboxToken),
      inFlight: 0,
      stale: false,
    };
  }
  const runtimeState = currentBrowserRuntimeState;
  runtimeState.inFlight += 1;
  try {
    return await callback(runtimeState.runtime);
  } finally {
    runtimeState.inFlight = Math.max(0, runtimeState.inFlight - 1);
    cleanupStaleBrowserRuntimeStates();
  }
}
