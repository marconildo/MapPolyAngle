import { createInterface } from "node:readline";
import { performance } from "node:perf_hooks";

import { handleExactRuntimeRequest } from "./service";
import type { ExactRuntimeEnvelope, ExactRuntimeEnvelopeResponse } from "./protocol";

function resolvedExactZoom(request: ExactRuntimeEnvelope["request"]) {
  return "exactOptimizeZoom" in request && typeof request.exactOptimizeZoom === "number"
    ? request.exactOptimizeZoom
    : 14;
}

function summarizeRequest(envelope: ExactRuntimeEnvelope) {
  const { request } = envelope;
  switch (request.operation) {
    case "terrain-batch":
      return `operation=terrain-batch tiles=${request.tiles.length}`;
    case "evaluate-region":
      return `operation=evaluate-region scopeId=${request.scopeId ?? "exact-region"} bearingDeg=${request.bearingDeg} exactZoom=${resolvedExactZoom(request)}`;
    case "evaluate-solution":
      return `operation=evaluate-solution polygonId=${request.polygonId} signature=${request.solution.signature} regions=${request.solution.regions.length} exactZoom=${resolvedExactZoom(request)}`;
    case "optimize-bearing":
      return `operation=optimize-bearing scopeId=${request.scopeId ?? request.polygonId ?? "optimize-bearing"} mode=${request.mode ?? "global"} seedBearingDeg=${request.seedBearingDeg} exactZoom=${resolvedExactZoom(request)}`;
    case "rerank-solutions":
      return `operation=rerank-solutions polygonId=${request.polygonId} solutions=${request.solutions.length} exactZoom=${resolvedExactZoom(request)}`;
  }
}

function describeError(error: unknown): string {
  if (error instanceof Error) {
    const cause = "cause" in error ? (error as Error & { cause?: unknown }).cause : undefined;
    const suffix = cause === undefined ? "" : ` cause=${describeError(cause)}`;
    return `${error.name}: ${error.message}${suffix}`;
  }
  return String(error);
}

async function main() {
  const readline = createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  for await (const line of readline) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let requestId = "unknown";
    let requestSummary = "operation=unknown";
    let response: ExactRuntimeEnvelopeResponse;
    try {
      const envelope = JSON.parse(trimmed) as ExactRuntimeEnvelope;
      requestId = envelope.id;
      requestSummary = summarizeRequest(envelope);
      const startedAt = performance.now();
      process.stderr.write(`[exact-runtime-sidecar] start ${requestSummary}\n`);
      const exactResponse = await handleExactRuntimeRequest(envelope.request);
      process.stderr.write(
        `[exact-runtime-sidecar] finish ${requestSummary} elapsedMs=${(performance.now() - startedAt).toFixed(1)}\n`,
      );
      response = {
        id: envelope.id,
        ok: true,
        response: exactResponse,
      };
    } catch (error) {
      const errorSummary = `${requestSummary} ${describeError(error)} request=${trimmed.slice(0, 4000)}`;
      process.stderr.write(`[exact-runtime-sidecar] ${errorSummary}\n`);
      response = {
        id: requestId,
        ok: false,
        error: `${requestSummary} ${describeError(error)}`,
      };
    }
    process.stdout.write(`${JSON.stringify(response)}\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack || error.message : String(error)}\n`);
  process.exitCode = 1;
});
