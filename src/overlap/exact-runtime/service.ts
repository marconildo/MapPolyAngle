import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
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

type TerrainBatchClient = {
  fetchTerrainBatch(request: ExactRuntimeTerrainBatchRequest): Promise<ExactRuntimeTerrainBatchResponse>;
};

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

function createRuntimeForRequest(request: Extract<ExactRuntimeRequest, { terrainSource: unknown }>): ExactRegionRuntime {
  const terrainBatchClient = createTerrainBatchClientFromEnv();
  return {
    terrainProvider: new TerrainBatchBackedProvider(terrainBatchClient, request.terrainSource),
    tileEvaluator: new DirectExactTileEvaluator(),
    yieldToEventLoop: async () => undefined,
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
    });
    return {
      operation: "evaluate-solution",
      solution: result.solution,
      preview: toPreviewPayload(result.preview),
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
  };
}
