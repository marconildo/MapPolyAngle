import {
  buildMissionProfileDetail,
  buildMissionProfileOverview,
} from "./missionProfile";
import type {
  MissionProfileData,
  MissionProfileSnapshot,
  MissionProfileWorkerRequest,
  MissionProfileWorkerResponse,
} from "./missionProfileWorker.types";

let activeSnapshot: MissionProfileSnapshot | null = null;
let overviewCache: MissionProfileData | null = null;
const detailCache = new Map<string, MissionProfileData | null>();
const terrainQueryCache = new Map<string, number | null>();

function clearCaches() {
  overviewCache = null;
  detailCache.clear();
  terrainQueryCache.clear();
}

self.onmessage = (event: MessageEvent<MissionProfileWorkerRequest>) => {
  const request = event.data;

  try {
    if (request.type === "setMissionSnapshot") {
      activeSnapshot = request.snapshot;
      clearCaches();
      const response: MissionProfileWorkerResponse = { id: request.id, type: "ack" };
      self.postMessage(response);
      return;
    }

    if (request.type === "clearSnapshot") {
      activeSnapshot = null;
      clearCaches();
      const response: MissionProfileWorkerResponse = { id: request.id, type: "ack" };
      self.postMessage(response);
      return;
    }

    if (!activeSnapshot) {
      const response: MissionProfileWorkerResponse =
        request.type === "computeDetail"
          ? { id: request.id, type: "detail", requestKey: request.requestKey, profile: null }
          : { id: request.id, type: "overview", profile: null };
      self.postMessage(response);
      return;
    }

    if (request.type === "computeOverview") {
      if (overviewCache === null) {
        overviewCache = buildMissionProfileOverview(activeSnapshot, undefined, terrainQueryCache);
      }
      const response: MissionProfileWorkerResponse = {
        id: request.id,
        type: "overview",
        profile: overviewCache,
      };
      self.postMessage(response);
      return;
    }

    if (request.type === "computeDetail") {
      if (!detailCache.has(request.requestKey)) {
        detailCache.set(
          request.requestKey,
          buildMissionProfileDetail(
            activeSnapshot,
            request.rangeStartM,
            request.rangeEndM,
            request.spacingBucketM,
            request.maxSamples,
            undefined,
            terrainQueryCache,
          ),
        );
      }
      const response: MissionProfileWorkerResponse = {
        id: request.id,
        type: "detail",
        requestKey: request.requestKey,
        profile: detailCache.get(request.requestKey) ?? null,
      };
      self.postMessage(response);
    }
  } catch (error) {
    const response: MissionProfileWorkerResponse = {
      id: request.id,
      error: error instanceof Error ? error.message : String(error),
    };
    self.postMessage(response);
  }
};

export {};
