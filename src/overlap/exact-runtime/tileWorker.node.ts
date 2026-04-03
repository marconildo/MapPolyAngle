import type { TransferListItem } from "node:worker_threads";
import { parentPort } from "node:worker_threads";

import type { ExactCameraTileInput, ExactCameraTileOutput, ExactLidarTileInput, ExactLidarTileOutput } from "@/overlap/exact-core";
import { runLidarWorkerMessage } from "@/overlap/lidar-worker";
import { runCameraWorkerMessage } from "@/overlap/worker";

type ExactTileWorkerRequest =
  | { id: number; kind: "camera"; input: ExactCameraTileInput }
  | { id: number; kind: "lidar"; input: ExactLidarTileInput };

type ExactTileWorkerResponse =
  | { id: number; ok: true; result: ExactCameraTileOutput | ExactLidarTileOutput }
  | { id: number; ok: false; error: string };

function describeError(error: unknown) {
  if (error instanceof Error) {
    return error.stack || `${error.name}: ${error.message}`;
  }
  return String(error);
}

function getTransferList(result: ExactCameraTileOutput | ExactLidarTileOutput) {
  const transfers: TransferListItem[] = [result.overlap.buffer as ArrayBuffer, result.gsdMin.buffer as ArrayBuffer];
  if ("density" in result && result.density?.buffer) {
    const densityBuffer = result.density.buffer as ArrayBuffer;
    if (!transfers.includes(densityBuffer)) {
      transfers.push(densityBuffer);
    }
  }
  return transfers;
}

if (!parentPort) {
  throw new Error("Exact runtime tile worker requires a worker_threads parentPort.");
}

parentPort.on("message", (message: ExactTileWorkerRequest) => {
  try {
    const result = message.kind === "camera"
      ? runCameraWorkerMessage(message.input)
      : runLidarWorkerMessage(message.input);
    const response: ExactTileWorkerResponse = { id: message.id, ok: true, result };
    parentPort!.postMessage(response, getTransferList(result));
  } catch (error) {
    const response: ExactTileWorkerResponse = {
      id: message.id,
      ok: false,
      error: describeError(error),
    };
    parentPort!.postMessage(response);
  }
});
