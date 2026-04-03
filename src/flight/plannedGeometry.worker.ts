import { generatePlannedFlightGeometryForPolygon } from "./plannedGeometry";
import type { PlannedGeometryWorkerRequest, PlannedGeometryWorkerResponse } from "./plannedGeometryWorker.types";

self.onmessage = (event: MessageEvent<PlannedGeometryWorkerRequest>) => {
  const { id, ring, bearingDeg, lineSpacingM, params } = event.data;

  try {
    const geometry = generatePlannedFlightGeometryForPolygon(ring, bearingDeg, lineSpacingM, params);
    const response: PlannedGeometryWorkerResponse = { id, geometry };
    self.postMessage(response);
  } catch (error) {
    const response: PlannedGeometryWorkerResponse = {
      id,
      error: error instanceof Error ? error.message : String(error),
    };
    self.postMessage(response);
  }
};

export {};
