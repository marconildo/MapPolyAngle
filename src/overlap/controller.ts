import type { PolygonLngLat, WorkerOut, LidarWorkerOut } from "./types";
import { tileCornersLngLat } from "./mercator";
import { tilesCoveringPolygon } from "./tileCoverage";
import { applyActiveDsmToTerrainRgbTile } from "@/terrain/dsmSource";
import { getTerrainTileUrlForCurrentSource } from "@/terrain/terrainSource";

export { tilesCoveringPolygon } from "./tileCoverage";

export async function fetchTerrainRGBA(
  z: number, x: number, y: number, token: string, _size = 512, signal?: AbortSignal
): Promise<ImageData> {
  const terrainUrl = getTerrainTileUrlForCurrentSource(z, x, y);
  const url = terrainUrl
    ?? `https://api.mapbox.com/v4/mapbox.terrain-rgb/${z}/${x}/${y}.pngraw?access_token=${token}`;
  const img = await loadImage(url, signal);
  const canvas = document.createElement("canvas");
  canvas.width = img.width; canvas.height = img.height;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, img.width, img.height);
  if (!terrainUrl) {
    await applyActiveDsmToTerrainRgbTile(z, x, y, img.width, imageData.data);
  }
  return imageData;
}

function loadImage(url: string, signal?: AbortSignal): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    const cleanup = () => {
      img.onload = null; img.onerror = null;
      if (signal) signal.removeEventListener("abort", onAbort);
    };
    const onAbort = () => { cleanup(); rej(new DOMException("aborted", "AbortError")); };
    img.onload = () => { cleanup(); res(img); };
    img.onerror = () => { cleanup(); rej(new Error("image load failed")); };
    if (signal) {
      if (signal.aborted) return onAbort();
      signal.addEventListener("abort", onAbort);
    }
    img.src = url;
  });
}

export function tileCornersForImageSource(z:number,x:number,y:number) {
  return tileCornersLngLat(z,x,y);
}

export class OverlapWorker {
  private worker: Worker;
  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  }
  runTile(args: any) {
    return new Promise<WorkerOut>((resolve) => {
      const onMsg = (e: MessageEvent<WorkerOut>) => {
        this.worker.removeEventListener("message", onMsg as any);
        resolve(e.data);
      };
      this.worker.addEventListener("message", onMsg as any, { once: true });
      this.worker.postMessage(args, [args.tile.data.buffer]); // transfer tile RGBA buffer
    });
  }
  terminate() { this.worker.terminate(); }
}

export class LidarDensityWorker {
  private worker: Worker;
  constructor() {
    this.worker = new Worker(new URL("./lidar-worker.ts", import.meta.url), { type: "module" });
  }
  runTile(args: any) {
    return new Promise<LidarWorkerOut>((resolve, reject) => {
      const onMsg = (e: MessageEvent<LidarWorkerOut>) => {
        this.worker.removeEventListener("error", onErr as any);
        this.worker.removeEventListener("message", onMsg as any);
        resolve(e.data);
      };
      const onErr = (e: ErrorEvent) => {
        this.worker.removeEventListener("error", onErr as any);
        this.worker.removeEventListener("message", onMsg as any);
        reject(e.error ?? new Error(e.message || "Lidar worker failed"));
      };
      this.worker.addEventListener("message", onMsg as any, { once: true });
      this.worker.addEventListener("error", onErr as any, { once: true });
      const transfers: Transferable[] = [args.tile.data.buffer];
      if (args.demTile?.data?.buffer && args.demTile.data.buffer !== args.tile.data.buffer) {
        transfers.push(args.demTile.data.buffer);
      }
      this.worker.postMessage(args, transfers);
    });
  }
  terminate() { this.worker.terminate(); }
}
