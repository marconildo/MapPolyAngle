import type { PlannedFlightGeometry } from "@/domain/types";
import type { PlannedGeometryWorkerRequest, PlannedGeometryWorkerResponse } from "./plannedGeometryWorker.types";

type PendingRequest = {
  resolve: (geometry: PlannedFlightGeometry) => void;
  reject: (error: Error) => void;
};

export class PlannedGeometryWorkerController {
  private worker: Worker | null = null;
  private nextRequestId = 1;
  private pending = new Map<number, PendingRequest>();

  constructor() {
    this.ensureWorker();
  }

  run(args: Omit<PlannedGeometryWorkerRequest, "id">): Promise<PlannedFlightGeometry> {
    const worker = this.ensureWorker();
    const id = this.nextRequestId;
    this.nextRequestId += 1;

    return new Promise<PlannedFlightGeometry>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ id, ...args } satisfies PlannedGeometryWorkerRequest);
    });
  }

  terminate() {
    if (this.worker) {
      this.worker.removeEventListener("message", this.handleMessage as EventListener);
      this.worker.removeEventListener("error", this.handleError as EventListener);
      this.worker.terminate();
      this.worker = null;
    }

    const terminationError = new Error("Planned geometry worker terminated");
    for (const pendingRequest of this.pending.values()) {
      pendingRequest.reject(terminationError);
    }
    this.pending.clear();
  }

  private ensureWorker() {
    if (this.worker) return this.worker;

    this.worker = new Worker(new URL("./plannedGeometry.worker.ts", import.meta.url), { type: "module" });
    this.worker.addEventListener("message", this.handleMessage as EventListener);
    this.worker.addEventListener("error", this.handleError as EventListener);
    return this.worker;
  }

  private readonly handleMessage = (event: MessageEvent<PlannedGeometryWorkerResponse>) => {
    const pendingRequest = this.pending.get(event.data.id);
    if (!pendingRequest) return;

    this.pending.delete(event.data.id);

    if ("error" in event.data) {
      pendingRequest.reject(new Error(event.data.error));
      return;
    }

    pendingRequest.resolve(event.data.geometry);
  };

  private readonly handleError = (event: ErrorEvent) => {
    const workerError = event.error instanceof Error ? event.error : new Error(event.message || "Planned geometry worker failed");
    for (const pendingRequest of this.pending.values()) {
      pendingRequest.reject(workerError);
    }
    this.pending.clear();

    if (this.worker) {
      this.worker.removeEventListener("message", this.handleMessage as EventListener);
      this.worker.removeEventListener("error", this.handleError as EventListener);
      this.worker.terminate();
      this.worker = null;
    }
  };
}
