import path from "node:path";
import { cpus } from "node:os";
import { Worker } from "node:worker_threads";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { existsSync } from "node:fs";
import { PNG } from "pngjs";

import { evaluateCameraTileExact, evaluateLidarTileExact } from "@/overlap/exact-core";
import {
  evaluateRegionBearingExact,
  evaluatePartitionSolutionCandidateExact,
  optimizeBearingExact,
  rerankPartitionSolutionsExact,
  type ExactRegionRuntime,
  type ExactTerrainProvider,
  type ExactTileEvaluator,
  type ExactTileRef,
  type ExactTileWithHalo,
} from "@/overlap/exact-region";
import type { PaddedDemTileRGBA, TileRGBA } from "@/overlap/types";

import type {
  ExactRuntimeRequest,
  ExactRuntimeResponse,
  ExactRuntimePartitionPreviewPayload,
  ExactRuntimeTerrainBatchRequest,
  ExactRuntimeTerrainBatchResponse,
  ExactRuntimeTilePayload,
} from "./protocol";
import type { ExactCameraTileInput, ExactCameraTileOutput, ExactLidarTileInput, ExactLidarTileOutput } from "@/overlap/exact-core";

type TerrainBatchClient = {
  fetchTerrainBatch(request: ExactRuntimeTerrainBatchRequest): Promise<ExactRuntimeTerrainBatchResponse>;
};

type ExactRuntimeTileWorkerRequest =
  | { id: number; kind: "camera"; input: ExactCameraTileInput }
  | { id: number; kind: "lidar"; input: ExactLidarTileInput };

type ExactRuntimeTileWorkerResponse =
  | { id: number; ok: true; result: ExactCameraTileOutput | ExactLidarTileOutput }
  | { id: number; ok: false; error: string };

function describeError(error: unknown): string {
  if (error instanceof Error) {
    const cause = "cause" in error ? (error as Error & { cause?: unknown }).cause : undefined;
    const suffix = cause === undefined ? "" : ` cause=${describeError(cause)}`;
    return `${error.name}: ${error.message}${suffix}`;
  }
  return String(error);
}

function summarizeTerrainBatchRequest(request: ExactRuntimeTerrainBatchRequest) {
  const haloTileCount = request.tiles.filter((tile) => (tile.padTiles ?? 0) > 0).length;
  const maxPadTiles = request.tiles.reduce((best, tile) => Math.max(best, tile.padTiles ?? 0), 0);
  const firstTile = request.tiles[0];
  const lastTile = request.tiles[request.tiles.length - 1];
  const describeTile = (tile: typeof firstTile | undefined) =>
    tile ? `${tile.z}/${tile.x}/${tile.y}@pad${tile.padTiles ?? 0}` : "none";
  return [
    `terrainSource=${request.terrainSource.mode}${request.terrainSource.datasetId ? `:${request.terrainSource.datasetId}` : ""}`,
    `tiles=${request.tiles.length}`,
    `haloTiles=${haloTileCount}`,
    `maxPadTiles=${maxPadTiles}`,
    `first=${describeTile(firstTile)}`,
    `last=${describeTile(lastTile)}`,
  ].join(" ");
}

function isRetryableLocalTerrainBatchError(error: unknown): boolean {
  const message = describeError(error);
  return message.includes("ECONNRESET") || message.includes("socket hang up") || message.includes("UND_ERR_SOCKET");
}

function delay(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

function resolveExactRuntimeTilePoolSize() {
  const raw = process.env.EXACT_RUNTIME_TILE_POOL_SIZE;
  if (raw !== undefined && raw.trim() !== "") {
    const parsed = Number.parseInt(raw, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  const cpuCount = cpus().length;
  return Math.min(4, Math.max(2, Math.floor(cpuCount / 2)));
}

function resolveExactRuntimeTileWorkerExecArgv() {
  const out: string[] = [];
  for (let index = 0; index < process.execArgv.length; index += 1) {
    const arg = process.execArgv[index];
    if (arg === "--eval" || arg === "-e" || arg === "--print" || arg === "-p") {
      index += 1;
      continue;
    }
    out.push(arg);
  }
  return out;
}

function resolveExactRuntimeTileWorkerPath() {
  const explicitPath = process.env.EXACT_RUNTIME_TILE_WORKER_PATH;
  if (explicitPath && existsSync(explicitPath)) {
    return { path: explicitPath, usesTsSource: explicitPath.endsWith(".ts") };
  }
  const bundledPath = path.resolve(process.cwd(), "backend/terrain_splitter/exact-runtime/tileWorker.node.mjs");
  if (existsSync(bundledPath)) {
    return { path: bundledPath, usesTsSource: false };
  }
  const sourcePath = path.resolve(process.cwd(), "src/overlap/exact-runtime/tileWorker.node.ts");
  return { path: sourcePath, usesTsSource: true };
}

async function postJsonWithoutKeepAlive(urlText: string, body: string) {
  const url = new URL(urlText);
  const requestImpl = url.protocol === "https:" ? httpsRequest : httpRequest;
  return await new Promise<{ statusCode: number; body: string }>((resolve, reject) => {
    const req = requestImpl(
      url,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
          Connection: "close",
        },
        agent: false,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
        res.on("end", () => {
          resolve({
            statusCode: res.statusCode ?? 0,
            body: Buffer.concat(chunks).toString("utf8"),
          });
        });
      },
    );
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

function tileKey(tileRef: ExactTileRef) {
  return `${tileRef.z}/${tileRef.x}/${tileRef.y}`;
}

function decodeTilePng(payload: ExactRuntimeTilePayload): TileRGBA {
  const decoded = PNG.sync.read(Buffer.from(payload.pngBase64, "base64"));
  return {
    z: payload.z,
    x: payload.x,
    y: payload.y,
    size: decoded.width,
    data: new Uint8ClampedArray(decoded.data),
  };
}

function decodeDemTile(payload: ExactRuntimeTilePayload, fallbackTile: TileRGBA): PaddedDemTileRGBA {
  if (!payload.demPngBase64) {
    return {
      size: fallbackTile.size,
      padTiles: 0,
      data: new Uint8ClampedArray(fallbackTile.data),
    };
  }
  const decoded = PNG.sync.read(Buffer.from(payload.demPngBase64, "base64"));
  return {
    size: payload.demSize ?? decoded.width,
    padTiles: payload.demPadTiles ?? 0,
    data: new Uint8ClampedArray(decoded.data),
  };
}

function toPreviewPayload(preview: {
  metricKind: ExactRuntimePartitionPreviewPayload["metricKind"];
  stats: ExactRuntimePartitionPreviewPayload["stats"];
  regionStats: ExactRuntimePartitionPreviewPayload["regionStats"];
  regionCount: number;
  sampleCount: number;
  sampleLabel: string;
}): ExactRuntimePartitionPreviewPayload {
  return {
    metricKind: preview.metricKind,
    stats: preview.stats,
    regionStats: preview.regionStats,
    regionCount: preview.regionCount,
    sampleCount: preview.sampleCount,
    sampleLabel: preview.sampleLabel,
  };
}

class TerrainBatchBackedProvider implements ExactTerrainProvider {
  private readonly tileCache = new Map<string, TileRGBA>();
  private readonly haloCache = new Map<string, ExactTileWithHalo>();

  constructor(
    private readonly client: TerrainBatchClient,
    private readonly terrainSource: ExactRuntimeTerrainBatchRequest["terrainSource"],
  ) {}

  async getTerrainTiles(tileRefs: ExactTileRef[]) {
    const missing = tileRefs.filter((tileRef) => !this.tileCache.has(tileKey(tileRef)));
    if (missing.length > 0) {
      const batch = await this.client.fetchTerrainBatch({
        operation: "terrain-batch",
        terrainSource: this.terrainSource,
        tiles: missing.map((tileRef) => ({ ...tileRef })),
      });
      batch.tiles.forEach((payload) => {
        this.tileCache.set(tileKey(payload), decodeTilePng(payload));
      });
    }
    return new Map(tileRefs.map((tileRef) => {
      const cached = this.tileCache.get(tileKey(tileRef));
      if (!cached) {
        throw new Error(`Missing terrain tile ${tileKey(tileRef)} from terrain batch response.`);
      }
      return [tileKey(tileRef), {
        ...cached,
        data: new Uint8ClampedArray(cached.data),
      }] as const;
    }));
  }

  async getTerrainTilesWithHalo(tileRefs: ExactTileRef[], padTiles: number) {
    const missing = tileRefs.filter((tileRef) => !this.haloCache.has(`${tileKey(tileRef)}@${padTiles}`));
    if (missing.length > 0) {
      const batch = await this.client.fetchTerrainBatch({
        operation: "terrain-batch",
        terrainSource: this.terrainSource,
        tiles: missing.map((tileRef) => ({ ...tileRef, padTiles })),
      });
      batch.tiles.forEach((payload) => {
        const tile = decodeTilePng(payload);
        this.haloCache.set(`${tileKey(payload)}@${padTiles}`, {
          tile,
          demTile: decodeDemTile(payload, tile),
        });
      });
    }
    return new Map(tileRefs.map((tileRef) => {
      const cached = this.haloCache.get(`${tileKey(tileRef)}@${padTiles}`);
      if (!cached) {
        throw new Error(`Missing terrain halo tile ${tileKey(tileRef)} from terrain batch response.`);
      }
      return [tileKey(tileRef), {
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
  }
}

class DirectExactTileEvaluator implements ExactTileEvaluator {
  async evaluateCameraTile(input: Parameters<typeof evaluateCameraTileExact>[0]) {
    return evaluateCameraTileExact(input);
  }

  async evaluateLidarTile(input: Parameters<typeof evaluateLidarTileExact>[0]) {
    return evaluateLidarTileExact(input);
  }
}

class RuntimeExactTileWorker {
  private readonly worker: Worker;
  private readonly pending = new Map<number, {
    resolve: (value: ExactCameraTileOutput | ExactLidarTileOutput) => void;
    reject: (reason?: unknown) => void;
  }>();
  private nextRequestId = 1;
  private alive = true;
  private terminateRequested = false;

  constructor(workerPath: string, execArgv: string[]) {
    this.worker = new Worker(workerPath, { execArgv });
    this.worker.unref();
    this.worker.on("message", this.handleMessage);
    this.worker.on("error", this.handleError);
    this.worker.on("exit", this.handleExit);
  }

  run(input: Omit<ExactRuntimeTileWorkerRequest, "id">) {
    return new Promise<ExactCameraTileOutput | ExactLidarTileOutput>((resolve, reject) => {
      if (!this.alive) {
        reject(new Error("Exact runtime tile worker is not available."));
        return;
      }
      const id = this.nextRequestId++;
      this.pending.set(id, { resolve, reject });
      let request: ExactRuntimeTileWorkerRequest;
      if (input.kind === "camera") {
        request = { id, kind: "camera", input: input.input as ExactCameraTileInput };
      } else {
        request = { id, kind: "lidar", input: input.input as ExactLidarTileInput };
      }
      this.worker.postMessage(request);
    });
  }

  terminate() {
    this.alive = false;
    this.terminateRequested = true;
    this.worker.off("message", this.handleMessage);
    this.worker.off("error", this.handleError);
    this.worker.off("exit", this.handleExit);
    const error = new Error("Exact runtime tile worker terminated.");
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
    this.worker.terminate().catch(() => undefined);
  }

  isUsable() {
    return this.alive;
  }

  private readonly handleMessage = (message: ExactRuntimeTileWorkerResponse) => {
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.ok) {
      pending.resolve(message.result);
      return;
    }
    pending.reject(new Error(message.error));
  };

  private readonly handleError = (error: Error) => {
    this.alive = false;
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  };

  private readonly handleExit = (code: number) => {
    this.alive = false;
    if (this.terminateRequested) return;
    if (this.pending.size === 0) return;
    const error = new Error(`Exact runtime tile worker exited unexpectedly (code=${code}).`);
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  };
}

class WorkerPool<TInput, TOutput, TWorker extends { run(args: TInput): Promise<TOutput>; terminate(): void; isUsable?(): boolean }> {
  private readonly workers: TWorker[];
  private readonly idleWorkers: TWorker[];
  private readonly queue: Array<{
    input: TInput;
    resolve: (value: TOutput) => void;
    reject: (reason?: unknown) => void;
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
      this.flush();
    });
  }

  terminate() {
    this.terminating = true;
    while (this.queue.length > 0) {
      this.queue.shift()?.reject(new Error("Worker pool has been terminated."));
    }
    for (const worker of this.workers) {
      worker.terminate();
    }
    this.idleWorkers.length = 0;
  }

  private flush() {
    while (!this.terminating && this.idleWorkers.length > 0 && this.queue.length > 0) {
      let worker = this.idleWorkers.pop()!;
      if (worker.isUsable && !worker.isUsable()) {
        worker = this.replaceWorker(worker);
      }
      const job = this.queue.shift()!;
      worker.run(job.input)
        .then(job.resolve, job.reject)
        .finally(() => {
          if (!this.terminating) {
            if (worker.isUsable && !worker.isUsable()) {
              this.idleWorkers.push(this.replaceWorker(worker));
            } else {
              this.idleWorkers.push(worker);
            }
            this.flush();
          }
        });
    }
  }

  private replaceWorker(deadWorker: TWorker) {
    const replacement = this.createWorker();
    const index = this.workers.indexOf(deadWorker);
    if (index >= 0) {
      this.workers[index] = replacement;
    }
    return replacement;
  }
}

class NodeWorkerBackedExactTileEvaluator implements ExactTileEvaluator {
  readonly concurrency: number;
  private readonly pool: WorkerPool<
    Omit<ExactRuntimeTileWorkerRequest, "id">,
    ExactCameraTileOutput | ExactLidarTileOutput,
    RuntimeExactTileWorker
  >;

  constructor(poolSize: number) {
    const workerSpec = resolveExactRuntimeTileWorkerPath();
    const execArgv = workerSpec.usesTsSource ? resolveExactRuntimeTileWorkerExecArgv() : [];
    this.concurrency = poolSize;
    this.pool = new WorkerPool(
      () => new RuntimeExactTileWorker(workerSpec.path, execArgv),
      poolSize,
    );
  }

  evaluateCameraTile(input: Parameters<typeof evaluateCameraTileExact>[0]) {
    return this.pool.run({ kind: "camera", input }) as Promise<ExactCameraTileOutput>;
  }

  evaluateLidarTile(input: Parameters<typeof evaluateLidarTileExact>[0]) {
    return this.pool.run({ kind: "lidar", input }) as Promise<ExactLidarTileOutput>;
  }

  dispose() {
    this.pool.terminate();
  }
}

class LocalHttpTerrainBatchClient implements TerrainBatchClient {
  constructor(private readonly baseUrl: string) {}

  async fetchTerrainBatch(request: ExactRuntimeTerrainBatchRequest) {
    const url = `${this.baseUrl.replace(/\/$/, "")}/v1/internal/terrain-batch`;
    const requestSummary = summarizeTerrainBatchRequest(request);
    const body = JSON.stringify(request);
    let lastError: Error | null = null;
    for (let attempt = 1; attempt <= 2; attempt++) {
      const startedAt = Date.now();
      try {
        const response = await postJsonWithoutKeepAlive(url, body);
        if (response.statusCode < 200 || response.statusCode >= 300) {
          throw new Error(
            `Local terrain batch request failed attempt=${attempt} url=${url} status=${response.statusCode} elapsedMs=${Date.now() - startedAt} bytes=${response.body.length} ${requestSummary} body=${response.body.slice(0, 240)}`,
          );
        }
        try {
          return JSON.parse(response.body) as ExactRuntimeTerrainBatchResponse;
        } catch (error) {
          throw new Error(
            `Local terrain batch parse failed attempt=${attempt} url=${url} status=${response.statusCode} elapsedMs=${Date.now() - startedAt} bytes=${response.body.length} ${requestSummary} ${describeError(error)}`,
          );
        }
      } catch (error) {
        const wrapped = new Error(
          `Local terrain batch fetch failed attempt=${attempt} url=${url} elapsedMs=${Date.now() - startedAt} ${requestSummary} ${describeError(error)}`,
        );
        if (attempt < 2 && isRetryableLocalTerrainBatchError(error)) {
          lastError = wrapped;
          await delay(25);
          continue;
        }
        throw wrapped;
      }
    }
    throw lastError ?? new Error(`Local terrain batch failed unexpectedly ${requestSummary}`);
  }
}

class LambdaTerrainBatchClient implements TerrainBatchClient {
  constructor(private readonly functionName: string) {}

  async fetchTerrainBatch(request: ExactRuntimeTerrainBatchRequest) {
    const { InvokeCommand, LambdaClient } = await import("@aws-sdk/client-lambda");
    const lambdaClient = new LambdaClient({});
    const command = new InvokeCommand({
      FunctionName: this.functionName,
      InvocationType: "RequestResponse",
      Payload: Buffer.from(JSON.stringify({
        terrainSplitterInternal: "terrain-batch",
        payload: request,
      })),
    });
    const response = await lambdaClient.send(command);
    const text = response.Payload ? Buffer.from(response.Payload).toString("utf8") : "";
    if (!text) {
      throw new Error("Lambda terrain batch returned an empty payload.");
    }
    const payload = JSON.parse(text) as ExactRuntimeTerrainBatchResponse | { errorMessage?: string };
    if (!("operation" in payload) || payload.operation !== "terrain-batch") {
      const lambdaError = "errorMessage" in payload ? payload.errorMessage : undefined;
      throw new Error(lambdaError || "Lambda terrain batch failed.");
    }
    return payload;
  }
}

function createTerrainBatchClientFromEnv() {
  const providerMode = process.env.EXACT_RUNTIME_PROVIDER_MODE ?? "local-http";
  if (providerMode === "lambda") {
    const functionName = process.env.EXACT_RUNTIME_TERRAIN_BATCH_FUNCTION_NAME;
    if (!functionName) {
      throw new Error("EXACT_RUNTIME_TERRAIN_BATCH_FUNCTION_NAME is required when EXACT_RUNTIME_PROVIDER_MODE=lambda.");
    }
    return new LambdaTerrainBatchClient(functionName);
  }
  const baseUrl = process.env.EXACT_RUNTIME_INTERNAL_BASE_URL ?? "http://127.0.0.1:8090";
  return new LocalHttpTerrainBatchClient(baseUrl);
}

let sharedNodeExactTileEvaluator: NodeWorkerBackedExactTileEvaluator | null = null;

export function disposeExactRuntimeSharedResources() {
  sharedNodeExactTileEvaluator?.dispose?.();
  sharedNodeExactTileEvaluator = null;
}

function getSharedExactTileEvaluator() {
  const poolSize = resolveExactRuntimeTilePoolSize();
  if (poolSize <= 1) {
    return { tileEvaluator: new DirectExactTileEvaluator(), candidateConcurrency: 1 };
  }
  if (!sharedNodeExactTileEvaluator || sharedNodeExactTileEvaluator.concurrency !== poolSize) {
    disposeExactRuntimeSharedResources();
    sharedNodeExactTileEvaluator = new NodeWorkerBackedExactTileEvaluator(poolSize);
  }
  return {
    tileEvaluator: sharedNodeExactTileEvaluator,
    candidateConcurrency: sharedNodeExactTileEvaluator.concurrency,
  };
}

function createRuntimeForRequest(request: Extract<ExactRuntimeRequest, { terrainSource: unknown }>): ExactRegionRuntime {
  const terrainBatchClient = createTerrainBatchClientFromEnv();
  const { tileEvaluator, candidateConcurrency } = getSharedExactTileEvaluator();
  return {
    terrainProvider: new TerrainBatchBackedProvider(terrainBatchClient, request.terrainSource),
    tileEvaluator,
    yieldToEventLoop: async () => undefined,
    candidateConcurrency,
  };
}

export async function handleExactRuntimeRequest(request: ExactRuntimeRequest): Promise<ExactRuntimeResponse> {
  if (request.operation === "terrain-batch") {
    const terrainBatchClient = createTerrainBatchClientFromEnv();
    return terrainBatchClient.fetchTerrainBatch(request);
  }

  const runtime = createRuntimeForRequest(request);
  if (request.operation === "evaluate-region") {
    return {
      operation: "evaluate-region",
      candidate: await evaluateRegionBearingExact(runtime, {
        scopeId: request.scopeId ?? "exact-region",
        ring: request.ring,
        params: request.params,
        altitudeMode: request.altitudeMode,
        minClearanceM: request.minClearanceM,
        turnExtendM: request.turnExtendM,
        exactOptimizeZoom: request.exactOptimizeZoom,
        timeWeight: request.timeWeight,
        clipInnerBufferM: request.clipInnerBufferM,
        minOverlapForGsd: request.minOverlapForGsd,
        bearingDeg: request.bearingDeg,
      }),
    };
  }

  if (request.operation === "evaluate-solution") {
    const result = await evaluatePartitionSolutionCandidateExact(runtime, {
      scopeId: request.polygonId,
      polygonId: request.polygonId,
      ring: request.ring,
      params: request.params,
      altitudeMode: request.altitudeMode,
      minClearanceM: request.minClearanceM,
      turnExtendM: request.turnExtendM,
      exactOptimizeZoom: request.exactOptimizeZoom,
      timeWeight: request.timeWeight,
      clipInnerBufferM: request.clipInnerBufferM,
      minOverlapForGsd: request.minOverlapForGsd,
      solution: request.solution,
      fastestMissionTimeSec: request.fastestMissionTimeSec,
      rankingSource: request.rankingSource ?? "backend-exact",
      debugTrace: request.debugTrace,
    });
    return {
      operation: "evaluate-solution",
      solution: result.solution,
      preview: toPreviewPayload(result.preview),
      debugTrace: request.debugTrace ? result.debugTrace : undefined,
    };
  }

  if (request.operation === "optimize-bearing") {
    const result = await optimizeBearingExact(runtime, {
      scopeId: request.scopeId ?? request.polygonId ?? "optimize-bearing",
      ring: request.ring,
      params: request.params,
      altitudeMode: request.altitudeMode,
      minClearanceM: request.minClearanceM,
      turnExtendM: request.turnExtendM,
      exactOptimizeZoom: request.exactOptimizeZoom,
      timeWeight: request.timeWeight,
      clipInnerBufferM: request.clipInnerBufferM,
      minOverlapForGsd: request.minOverlapForGsd,
      seedBearingDeg: request.seedBearingDeg,
      mode: request.mode,
      halfWindowDeg: request.halfWindowDeg,
    });
    return {
      operation: "optimize-bearing",
      best: result.best,
      evaluated: result.evaluated,
      seedBearingDeg: result.seedBearingDeg,
      lineSpacingM: result.lineSpacingM,
      metricKind: result.best?.metricKind ?? null,
    };
  }

  const reranked = await rerankPartitionSolutionsExact(runtime, {
    scopeId: request.polygonId,
    polygonId: request.polygonId,
    ring: request.ring,
    params: request.params,
    altitudeMode: request.altitudeMode,
    minClearanceM: request.minClearanceM,
    turnExtendM: request.turnExtendM,
    exactOptimizeZoom: request.exactOptimizeZoom,
    timeWeight: request.timeWeight,
    clipInnerBufferM: request.clipInnerBufferM,
    minOverlapForGsd: request.minOverlapForGsd,
    solutions: request.solutions,
    rankingSource: request.rankingSource ?? "backend-exact",
    debugTrace: request.debugTrace,
  });
  const sortedSolutions = [...reranked.solutions].sort((left, right) => {
    const leftScore = Number.isFinite(left.exactScore ?? Number.NaN) ? left.exactScore! : Number.POSITIVE_INFINITY;
    const rightScore = Number.isFinite(right.exactScore ?? Number.NaN) ? right.exactScore! : Number.POSITIVE_INFINITY;
    return leftScore - rightScore;
  });
  return {
    operation: "rerank-solutions",
    bestIndex: sortedSolutions.length > 0 ? 0 : reranked.bestIndex,
    solutions: sortedSolutions,
    previewsBySignature: Object.fromEntries(
      Object.entries(reranked.previewsBySignature).map(([signature, preview]) => [
        signature,
        toPreviewPayload(preview),
      ]),
    ),
    debugBySignature: request.debugTrace ? reranked.debugBySignature : undefined,
  };
}
