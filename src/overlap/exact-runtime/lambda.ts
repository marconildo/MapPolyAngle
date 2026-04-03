import { existsSync } from "node:fs";
import path from "node:path";

import { handleExactRuntimeRequest } from "./service";
import type { ExactRuntimeRequest } from "./protocol";

const bundledWorkerPath = typeof __dirname === "string"
  ? path.join(__dirname, "tileWorker.node.mjs")
  : path.resolve(process.cwd(), "backend/terrain_splitter/exact-runtime/tileWorker.node.mjs");

if (!process.env.EXACT_RUNTIME_TILE_WORKER_PATH && existsSync(bundledWorkerPath)) {
  process.env.EXACT_RUNTIME_TILE_WORKER_PATH = bundledWorkerPath;
}

export async function handler(event: ExactRuntimeRequest) {
  return handleExactRuntimeRequest(event);
}
