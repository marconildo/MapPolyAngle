import type { WorkerIn, WorkerOut } from "./types";
import { evaluateCameraTileExact } from "./exact-core/camera";

type Msg = WorkerIn;
type Ret = WorkerOut;
type WorkerScope = {
  onmessage: ((event: MessageEvent<Msg>) => void) | null;
  postMessage: (message: Ret, transfer?: Transferable[]) => void;
};

export function runCameraWorkerMessage(message: Msg): Ret {
  return evaluateCameraTileExact(message);
}

if (typeof self !== "undefined") {
  const workerSelf = self as unknown as WorkerScope;
  workerSelf.onmessage = (ev: MessageEvent<Msg>) => {
    const ret = runCameraWorkerMessage(ev.data);
    workerSelf.postMessage(ret, [ret.overlap.buffer, ret.gsdMin.buffer]);
  };
}
