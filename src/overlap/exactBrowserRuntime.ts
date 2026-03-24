import { fetchTerrainRGBA, LidarDensityWorker, OverlapWorker } from "@/overlap/controller";
import type { PaddedDemTileRGBA, TileRGBA } from "@/overlap/types";

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
  private readonly cameraWorker = new OverlapWorker();
  private readonly lidarWorker = new LidarDensityWorker();

  evaluateCameraTile(input: Parameters<OverlapWorker["runTile"]>[0]) {
    return this.cameraWorker.runTile(input);
  }

  evaluateLidarTile(input: Parameters<LidarDensityWorker["runTile"]>[0]) {
    return this.lidarWorker.runTile(input);
  }

  dispose() {
    this.cameraWorker.terminate();
    this.lidarWorker.terminate();
  }
}

export function createBrowserExactRegionRuntime(mapboxToken: string): ExactRegionRuntime {
  return {
    terrainProvider: new BrowserExactTerrainProvider(mapboxToken),
    tileEvaluator: new BrowserExactTileEvaluator(),
    yieldToEventLoop: yieldToFrame,
  };
}

export async function withBrowserExactRegionRuntime<T>(
  mapboxToken: string,
  callback: (runtime: ExactRegionRuntime) => Promise<T>,
): Promise<T> {
  const runtime = createBrowserExactRegionRuntime(mapboxToken);
  try {
    return await callback(runtime);
  } finally {
    await runtime.tileEvaluator.dispose?.();
  }
}
