import type { FlightParams, PlannedFlightGeometry } from "@/domain/types";

export type PlannedGeometryWorkerRequest = {
  id: number;
  ring: [number, number][];
  bearingDeg: number;
  lineSpacingM: number;
  params: FlightParams;
};

export type PlannedGeometryWorkerResponse =
  | {
      id: number;
      geometry: PlannedFlightGeometry;
    }
  | {
      id: number;
      error: string;
    };
