import type { LidarWorkerIn, LidarWorkerOut } from "./types";
import { evaluateLidarTileExact } from "./exact-core/lidar";

type Msg = LidarWorkerIn;
type Ret = LidarWorkerOut;
type WorkerScope = {
  onmessage: ((event: MessageEvent<Msg>) => void) | null;
  postMessage: (message: Ret, transfer?: Transferable[]) => void;
};

export function runLidarWorkerMessage(message: Msg): Ret {
  return evaluateLidarTileExact(message);
}

if (typeof self !== "undefined") {
  const workerSelf = self as unknown as WorkerScope;
  workerSelf.onmessage = (ev: MessageEvent<Msg>) => {
    const ret = runLidarWorkerMessage(ev.data);
    const transfers: Transferable[] = [ret.overlap.buffer, ret.gsdMin.buffer];
    if (ret.density?.buffer && ret.density.buffer !== ret.gsdMin.buffer && ret.density.buffer !== ret.overlap.buffer) {
      transfers.push(ret.density.buffer);
    }
    workerSelf.postMessage(ret, transfers);
  };
}
