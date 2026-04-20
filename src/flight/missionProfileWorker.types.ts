import type { TerrainTile } from "@/domain/types";

export type MissionProfileSegmentKind = "area" | "connector";

export type MissionProfileCoreSample = {
  distanceM: number;
  lng: number;
  lat: number;
  droneAltitudeM: number;
  terrainAltitudeM: number | null;
  clearanceM: number | null;
  segmentLabel: string;
  segmentKind: MissionProfileSegmentKind;
};

export type MissionProfileSummary = {
  totalDistanceM: number;
  minClearanceM: number | null;
  meanClearanceM: number | null;
  maxClearanceM: number | null;
  sampleCount: number;
};

export type MissionProfileData = {
  samples: MissionProfileCoreSample[];
  summary: MissionProfileSummary;
};

export type MissionProfileSegmentSnapshot = {
  key: string;
  segmentLabel: string;
  segmentKind: MissionProfileSegmentKind;
  path3D: [number, number, number][][];
  terrainTiles: TerrainTile[];
};

export type MissionProfileSnapshot = {
  missionId: string;
  totalDistanceM: number;
  segments: MissionProfileSegmentSnapshot[];
};

export type MissionProfileViewport = {
  mode: "full" | "zoomed";
  startDistanceM: number;
  endDistanceM: number;
};

export type MissionProfileSamplingOptions = {
  rangeStartM?: number;
  rangeEndM?: number;
  targetSpacingM: number;
  maxSamples: number;
  terrainToleranceM: number;
  clearanceToleranceM: number;
  maxDepth: number;
};

export type MissionProfileWorkerRequest =
  | {
      id: number;
      type: "setMissionSnapshot";
      snapshot: MissionProfileSnapshot | null;
    }
  | {
      id: number;
      type: "clearSnapshot";
    }
  | {
      id: number;
      type: "computeOverview";
    }
  | {
      id: number;
      type: "computeDetail";
      requestKey: string;
      rangeStartM: number;
      rangeEndM: number;
      spacingBucketM: number;
      maxSamples: number;
    };

export type MissionProfileWorkerResponse =
  | {
      id: number;
      type: "ack";
    }
  | {
      id: number;
      type: "overview";
      profile: MissionProfileData | null;
    }
  | {
      id: number;
      type: "detail";
      requestKey: string;
      profile: MissionProfileData | null;
    }
  | {
      id: number;
      error: string;
    };
