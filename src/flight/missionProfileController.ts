import type {
  MissionProfileData,
  MissionProfileSnapshot,
  MissionProfileWorkerRequest,
  MissionProfileWorkerResponse,
} from "./missionProfileWorker.types";

type PendingRequest =
  | {
      type: "ack";
      resolve: () => void;
      reject: (error: Error) => void;
    }
  | {
      type: "overview";
      resolve: (profile: MissionProfileData | null) => void;
      reject: (error: Error) => void;
    }
  | {
      type: "detail";
      resolve: (result: { requestKey: string; profile: MissionProfileData | null }) => void;
      reject: (error: Error) => void;
    };

export class MissionProfileWorkerController {
  private worker: Worker | null = null;
  private nextRequestId = 1;
  private pending = new Map<number, PendingRequest>();

  constructor() {
    this.ensureWorker();
  }

  setMissionSnapshot(snapshot: MissionProfileSnapshot | null): Promise<void> {
    const worker = this.ensureWorker();
    const id = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { type: "ack", resolve, reject });
      worker.postMessage({ id, type: "setMissionSnapshot", snapshot } satisfies MissionProfileWorkerRequest);
    });
  }

  clearSnapshot(): Promise<void> {
    const worker = this.ensureWorker();
    const id = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { type: "ack", resolve, reject });
      worker.postMessage({ id, type: "clearSnapshot" } satisfies MissionProfileWorkerRequest);
    });
  }

  computeOverview(): Promise<MissionProfileData | null> {
    const worker = this.ensureWorker();
    const id = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { type: "overview", resolve, reject });
      const request: MissionProfileWorkerRequest = { id, type: "computeOverview" };
      worker.postMessage(request);
    });
  }

  computeDetail(args: {
    requestKey: string;
    rangeStartM: number;
    rangeEndM: number;
    spacingBucketM: number;
    maxSamples: number;
  }): Promise<{ requestKey: string; profile: MissionProfileData | null }> {
    const worker = this.ensureWorker();
    const id = this.nextRequestId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { type: "detail", resolve, reject });
      const request: MissionProfileWorkerRequest = { id, type: "computeDetail", ...args };
      worker.postMessage(request);
    });
  }

  terminate() {
    if (this.worker) {
      this.worker.removeEventListener("message", this.handleMessage as EventListener);
      this.worker.removeEventListener("error", this.handleError as EventListener);
      this.worker.terminate();
      this.worker = null;
    }

    const terminationError = new Error("Mission profile worker terminated");
    for (const pendingRequest of this.pending.values()) {
      pendingRequest.reject(terminationError);
    }
    this.pending.clear();
  }

  private ensureWorker() {
    if (this.worker) return this.worker;
    this.worker = new Worker(new URL("./missionProfile.worker.ts", import.meta.url), { type: "module" });
    this.worker.addEventListener("message", this.handleMessage as EventListener);
    this.worker.addEventListener("error", this.handleError as EventListener);
    return this.worker;
  }

  private readonly handleMessage = (event: MessageEvent<MissionProfileWorkerResponse>) => {
    const pendingRequest = this.pending.get(event.data.id);
    if (!pendingRequest) return;
    this.pending.delete(event.data.id);

    if ("error" in event.data) {
      pendingRequest.reject(new Error(event.data.error));
      return;
    }

    if (pendingRequest.type === "ack") {
      pendingRequest.resolve();
      return;
    }
    if (pendingRequest.type === "overview" && event.data.type === "overview") {
      pendingRequest.resolve(event.data.profile);
      return;
    }
    if (pendingRequest.type === "detail" && event.data.type === "detail") {
      pendingRequest.resolve({ requestKey: event.data.requestKey, profile: event.data.profile });
      return;
    }

    pendingRequest.reject(new Error("Mission profile worker returned an unexpected response"));
  };

  private readonly handleError = (event: ErrorEvent) => {
    const workerError = event.error instanceof Error
      ? event.error
      : new Error(event.message || "Mission profile worker failed");
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
