import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { PNG } from "pngjs";

import { disposeExactRuntimeSharedResources, handleExactRuntimeRequest } from "../overlap/exact-runtime/service.ts";
import { handler } from "../overlap/exact-runtime/lambda.ts";
import type {
  ExactRuntimeEnvelopeResponse,
  ExactRuntimeRequest,
  ExactRuntimeTerrainBatchResponse,
} from "../overlap/exact-runtime/protocol.ts";

function encodeTerrainRgb(size: number, elevationForPixel: (row: number, col: number) => number) {
  const out = new Uint8ClampedArray(size * size * 4);
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const elevationM = elevationForPixel(row, col);
      const encoded = Math.max(0, Math.min(16777215, Math.round((elevationM + 10000) * 10)));
      const offset = (row * size + col) * 4;
      out[offset] = (encoded >> 16) & 255;
      out[offset + 1] = (encoded >> 8) & 255;
      out[offset + 2] = encoded & 255;
      out[offset + 3] = 255;
    }
  }
  return out;
}

function tilePngBase64(size: number) {
  const png = new PNG({ width: size, height: size });
  png.data = Buffer.from(encodeTerrainRgb(size, (row, col) => 220 + row * 0.4 + col * 0.3));
  return PNG.sync.write(png).toString("base64");
}

function demPngBase64(size: number, padTiles: number) {
  const span = padTiles * 2 + 1;
  const demSize = size * span;
  const png = new PNG({ width: demSize, height: demSize });
  const tile = Buffer.from(encodeTerrainRgb(size, (row, col) => 220 + row * 0.4 + col * 0.3));
  for (let dy = 0; dy < span; dy++) {
    for (let dx = 0; dx < span; dx++) {
      for (let row = 0; row < size; row++) {
        const srcStart = row * size * 4;
        const dstStart = ((dy * size + row) * demSize + dx * size) * 4;
        tile.copy(png.data, dstStart, srcStart, srcStart + size * 4);
      }
    }
  }
  return {
    demSize,
    demPngBase64: PNG.sync.write(png).toString("base64"),
  };
}

async function startTerrainBatchServer() {
  const requests: ExactRuntimeRequest[] = [];
  const server = http.createServer((req, res) => {
    if (req.method !== "POST" || req.url !== "/v1/internal/terrain-batch") {
      res.statusCode = 404;
      res.end("not found");
      return;
    }
    const chunks: Buffer[] = [];
    req.on("data", (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
    req.on("end", () => {
      const request = JSON.parse(Buffer.concat(chunks).toString("utf8")) as ExactRuntimeRequest;
      requests.push(request);
      const size = 64;
      const payload: ExactRuntimeTerrainBatchResponse = {
        operation: "terrain-batch",
        tiles: (request.operation === "terrain-batch" ? request.tiles : []).map((tile) => ({
          z: tile.z,
          x: tile.x,
          y: tile.y,
          size,
          pngBase64: tilePngBase64(size),
          ...(tile.padTiles && tile.padTiles > 0 ? demPngBase64(size, tile.padTiles) : {}),
          demPadTiles: tile.padTiles ?? null,
        })),
      };
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify(payload));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Failed to bind terrain batch test server.");
  return {
    url: `http://127.0.0.1:${address.port}`,
    requests,
    async close() {
      await new Promise<void>((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
    },
  };
}

function withEnv<T>(vars: Record<string, string>, callback: () => Promise<T>) {
  const previous = new Map<string, string | undefined>();
  Object.entries(vars).forEach(([key, value]) => {
    previous.set(key, process.env[key]);
    process.env[key] = value;
  });
  return callback().finally(() => {
    previous.forEach((value, key) => {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    });
  });
}

async function readSingleJsonLine(proc: ReturnType<typeof spawn>) {
  return await new Promise<ExactRuntimeEnvelopeResponse>((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error("Timed out waiting for sidecar response.")), 15000);
    let stdout = "";
    let stderr = "";
    proc.stdout?.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
      const newline = stdout.indexOf("\n");
      if (newline >= 0) {
        clearTimeout(timeout);
        try {
          resolve(JSON.parse(stdout.slice(0, newline)) as ExactRuntimeEnvelopeResponse);
        } catch (error) {
          reject(error);
        }
      }
    });
    proc.stderr?.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    proc.on("exit", (code) => {
      clearTimeout(timeout);
      reject(new Error(`Sidecar exited before responding (code=${code}). ${stderr}`.trim()));
    });
  });
}

async function terminateChildProcess(proc: ReturnType<typeof spawn>) {
  if (proc.exitCode !== null || proc.signalCode !== null) {
    return;
  }
  await new Promise<void>((resolve) => {
    const timeout = setTimeout(() => {
      proc.off("exit", onExit);
      resolve();
    }, 1000);
    timeout.unref();
    const onExit = () => {
      clearTimeout(timeout);
      resolve();
    };
    proc.once("exit", onExit);
    proc.kill("SIGKILL");
  });
}

async function main() {
  const server = await startTerrainBatchServer();
  try {
    const commonRequest = {
      payloadKind: "camera" as const,
      terrainSource: { mode: "mapbox" as const, datasetId: undefined },
      params: {
        payloadKind: "camera" as const,
        altitudeAGL: 100,
        frontOverlap: 75,
        sideOverlap: 70,
      },
      ring: [
        [0, 0],
        [0.018, 0],
        [0.018, 0.0045],
        [0, 0.0045],
        [0, 0],
      ] as [number, number][],
      altitudeMode: "legacy" as const,
      minClearanceM: 0,
      turnExtendM: 0,
      exactOptimizeZoom: 14,
      minOverlapForGsd: 1,
    };

    await withEnv(
      {
        EXACT_RUNTIME_PROVIDER_MODE: "local-http",
        EXACT_RUNTIME_INTERNAL_BASE_URL: server.url,
      },
      async () => {
        const evaluateResponse = await handleExactRuntimeRequest({
          operation: "evaluate-region",
          ...commonRequest,
          bearingDeg: 90,
        });
        assert.equal(evaluateResponse.operation, "evaluate-region");
        assert.ok(evaluateResponse.candidate);
        assert.equal(evaluateResponse.candidate?.metricKind, "gsd");
        assert.ok(server.requests.some((request) => request.operation === "terrain-batch"));

        const optimizeResponse = await handler({
          operation: "optimize-bearing",
          ...commonRequest,
          polygonId: "poly-1",
          seedBearingDeg: 30,
          mode: "global",
        });
        assert.equal(optimizeResponse.operation, "optimize-bearing");
        assert.ok(optimizeResponse.best);
        assert.equal(optimizeResponse.best?.metricKind, "gsd");

        const evaluateSolutionResponse = await handleExactRuntimeRequest({
          operation: "evaluate-solution",
          ...commonRequest,
          polygonId: "poly-1",
          fastestMissionTimeSec: 120,
          rankingSource: "backend-exact",
          debugTrace: true,
          solution: {
            signature: "candidate-a",
            tradeoff: 0.5,
            regionCount: 1,
            totalMissionTimeSec: 120,
            normalizedQualityCost: 0.2,
            weightedMeanMismatchDeg: 0,
            hierarchyLevel: 0,
            largestRegionFraction: 1,
            meanConvexity: 1,
            boundaryBreakAlignment: 1,
            isFirstPracticalSplit: true,
            regions: [
              {
                areaM2: 10,
                bearingDeg: 90,
                atomCount: 2,
                ring: commonRequest.ring,
                convexity: 1,
                compactness: 1,
              },
            ],
          },
        });
        assert.equal(evaluateSolutionResponse.operation, "evaluate-solution");
        assert.equal(evaluateSolutionResponse.solution.signature, "candidate-a");
        assert.equal(evaluateSolutionResponse.solution.rankingSource, "backend-exact");
        assert.ok(Number.isFinite(evaluateSolutionResponse.solution.exactScore ?? Number.NaN));
        assert.equal(evaluateSolutionResponse.preview.metricKind, "gsd");
        assert.ok(evaluateSolutionResponse.preview.sampleCount > 0);
        assert.ok(evaluateSolutionResponse.debugTrace);
        assert.equal(evaluateSolutionResponse.debugTrace?.signature, "candidate-a");
        assert.equal(evaluateSolutionResponse.debugTrace?.partitionScoreBreakdown.modelVersion, "camera-partition-v1");

        const evaluateSolutionWithoutTrace = await handleExactRuntimeRequest({
          operation: "evaluate-solution",
          ...commonRequest,
          polygonId: "poly-1",
          fastestMissionTimeSec: 120,
          rankingSource: "backend-exact",
          solution: evaluateSolutionResponse.solution,
        });
        assert.equal(evaluateSolutionWithoutTrace.operation, "evaluate-solution");
        assert.equal(evaluateSolutionWithoutTrace.debugTrace, undefined);

        const rerankResponse = await handleExactRuntimeRequest({
          operation: "rerank-solutions",
          ...commonRequest,
          polygonId: "poly-1",
          rankingSource: "backend-exact",
          debugTrace: true,
          solutions: [
            evaluateSolutionResponse.solution,
            {
              ...evaluateSolutionResponse.solution,
              signature: "candidate-b",
              isFirstPracticalSplit: false,
              regions: evaluateSolutionResponse.solution.regions.map((region) => ({
                ...region,
                bearingDeg: 0,
              })),
            },
          ],
        });
        assert.equal(rerankResponse.operation, "rerank-solutions");
        assert.ok(rerankResponse.debugBySignature);
        assert.ok(rerankResponse.debugBySignature?.["candidate-a"]);
        assert.ok(rerankResponse.debugBySignature?.["candidate-b"]);
      },
    );

    const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
    const tsxBin = path.join(repoRoot, "node_modules", ".bin", "tsx");
    const proc = spawn(tsxBin, ["src/overlap/exact-runtime/sidecar.ts"], {
      cwd: repoRoot,
      env: {
        ...process.env,
        EXACT_RUNTIME_PROVIDER_MODE: "local-http",
        EXACT_RUNTIME_INTERNAL_BASE_URL: server.url,
        NODE_NO_WARNINGS: "1",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });
    try {
      proc.stdin.write(`${JSON.stringify({
        id: "sidecar-1",
        request: {
          operation: "terrain-batch",
          terrainSource: { mode: "mapbox" },
          tiles: [{ z: 0, x: 0, y: 0, padTiles: 1 }],
        },
      })}\n`);
      const envelope = await readSingleJsonLine(proc);
      assert.equal(envelope.id, "sidecar-1");
      assert.equal(envelope.ok, true);
      if (envelope.ok) {
        assert.equal(envelope.response.operation, "terrain-batch");
        assert.equal(envelope.response.tiles.length, 1);
        assert.ok(envelope.response.tiles[0].pngBase64);
        assert.ok(envelope.response.tiles[0].demPngBase64);
      }
    } finally {
      await terminateChildProcess(proc);
    }

    console.log("exact_runtime tests passed");
  } finally {
    disposeExactRuntimeSharedResources();
    await server.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
