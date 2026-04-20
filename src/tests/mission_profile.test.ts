import assert from "node:assert/strict";

import type { TerrainTile } from "../domain/types.ts";
import {
  buildMissionProfileDetail,
  buildMissionProfileDetailRange,
  buildMissionProfileOverview,
  clipMissionProfileToRange,
  quantizeMissionProfileSpacingBucket,
} from "../flight/missionProfile.ts";
import type { MissionProfileSnapshot } from "../flight/missionProfileWorker.types.ts";

const ORIGIN: [number, number] = [8.54, 47.37];
const METERS_PER_DEG_LAT = 111_320;

function fromMeters(xM: number, yM: number): [number, number] {
  const latScale = Math.cos((ORIGIN[1] * Math.PI) / 180);
  return [
    ORIGIN[0] + xM / (METERS_PER_DEG_LAT * latScale),
    ORIGIN[1] + yM / METERS_PER_DEG_LAT,
  ];
}

function makeDummyTile(): TerrainTile {
  return {
    z: 14,
    x: 0,
    y: 0,
    width: 1,
    height: 1,
    data: new Uint8ClampedArray(4),
  };
}

function makeSnapshot(lengthM = 1_200): MissionProfileSnapshot {
  return {
    missionId: "fixture",
    totalDistanceM: lengthM,
    segments: [
      {
        key: "A",
        segmentLabel: "Area 1",
        segmentKind: "area",
        terrainTiles: [makeDummyTile()],
        path3D: [[fromMeters(0, 0), fromMeters(lengthM, 0)].map(([lng, lat]) => [lng, lat, 620] as [number, number, number])],
      },
    ],
  };
}

function ridgeTerrainQuery(lng: number, lat: number) {
  const latScale = Math.cos((ORIGIN[1] * Math.PI) / 180);
  const xM = (lng - ORIGIN[0]) * METERS_PER_DEG_LAT * latScale;
  const yM = (lat - ORIGIN[1]) * METERS_PER_DEG_LAT;
  const ridge = 420 + 90 * Math.exp(-((xM - 600) ** 2) / (2 * 110 ** 2));
  return ridge + yM * 0.01;
}

function testOverviewBuildsReasonableProfile() {
  const snapshot = makeSnapshot();
  const overview = buildMissionProfileOverview(snapshot, ridgeTerrainQuery);
  assert.ok(overview, "overview profile should exist");
  assert.ok(overview!.samples.length >= 10, "overview should sample the mission");
  assert.ok(overview!.samples.length <= 800, "overview should respect the sample budget");
  assert.ok(overview!.summary.maxClearanceM !== null, "overview should summarize clearance");
}

function testDetailRefinesRidgeRegion() {
  const snapshot = makeSnapshot();
  const detail = buildMissionProfileDetail(snapshot, 450, 750, 5, 2000, ridgeTerrainQuery);
  assert.ok(detail, "detail profile should exist");
  assert.ok(detail!.samples.length > 40, "detail should add finer samples in the visible range");
  const highestTerrain = Math.max(...detail!.samples.map((sample) => sample.terrainAltitudeM ?? Number.NEGATIVE_INFINITY));
  assert.ok(highestTerrain > 470, "detail should resolve the ridge peak");
}

function testClipRetainsContextForNarrowWindow() {
  const snapshot = makeSnapshot();
  const overview = buildMissionProfileOverview(snapshot, ridgeTerrainQuery);
  const clipped = clipMissionProfileToRange(overview, 598, 602);
  assert.ok(clipped, "narrow range should still clip to a usable local profile");
  assert.ok(clipped!.samples.length >= 2, "clipped range should keep enough points for a chart");
}

function testClipSummaryUsesOnlyVisibleSamples() {
  const clipped = clipMissionProfileToRange({
    samples: [
      {
        distanceM: 0,
        lng: ORIGIN[0],
        lat: ORIGIN[1],
        droneAltitudeM: 620,
        terrainAltitudeM: 500,
        clearanceM: 120,
        segmentLabel: "A",
        segmentKind: "area",
      },
      {
        distanceM: 100,
        lng: ORIGIN[0],
        lat: ORIGIN[1],
        droneAltitudeM: 620,
        terrainAltitudeM: 560,
        clearanceM: 60,
        segmentLabel: "A",
        segmentKind: "area",
      },
      {
        distanceM: 200,
        lng: ORIGIN[0],
        lat: ORIGIN[1],
        droneAltitudeM: 620,
        terrainAltitudeM: 520,
        clearanceM: 100,
        segmentLabel: "A",
        segmentKind: "area",
      },
    ],
    summary: {
      totalDistanceM: 200,
      minClearanceM: 60,
      meanClearanceM: (120 + 60 + 100) / 3,
      maxClearanceM: 120,
      sampleCount: 3,
    },
  }, 90, 110);
  assert.ok(clipped, "single-point visible window should still render with borrowed context");
  assert.equal(clipped!.samples.length, 3, "display samples should keep borrowed context");
  assert.equal(clipped!.summary.sampleCount, 1, "summary should only count in-range samples");
  assert.equal(clipped!.summary.minClearanceM, 60, "summary should reflect only the visible point");
  assert.equal(clipped!.summary.maxClearanceM, 60, "summary should ignore borrowed extrema");
}

function testBucketsAndRangesAreStable() {
  assert.equal(quantizeMissionProfileSpacingBucket(3.7), 5, "spacing bucket should use 1/2/5 quantization");
  assert.equal(quantizeMissionProfileSpacingBucket(17), 20, "spacing bucket should round to the next 1/2/5 decade");
  const detailRange = buildMissionProfileDetailRange(100, 300, 1_200, 10);
  assert.ok(detailRange.requestStartM <= 100, "detail range should pad backward");
  assert.ok(detailRange.requestEndM >= 300, "detail range should pad forward");

  const nearbyDetailRange = buildMissionProfileDetailRange(102, 298, 1_200, 10);
  assert.equal(
    nearbyDetailRange.requestStartM,
    detailRange.requestStartM,
    "tiny zoom deltas should reuse the same padded detail-window start",
  );
  assert.equal(
    nearbyDetailRange.requestEndM,
    detailRange.requestEndM,
    "tiny zoom deltas should reuse the same padded detail-window end",
  );
}

testOverviewBuildsReasonableProfile();
testDetailRefinesRidgeRegion();
testClipRetainsContextForNarrowWindow();
testClipSummaryUsesOnlyVisibleSamples();
testBucketsAndRangesAreStable();

console.log("mission_profile.test.ts: all assertions passed");
