import assert from "node:assert/strict";

import type { FlightParams, PlannedLeadManeuver } from "../domain/types.ts";
import type { TerrainTile } from "../utils/terrainAspectHybrid.ts";
import {
  buildInterAreaConnectionGeometry,
  buildMissionFlightGeometry,
  buildMissionTravelSummary,
} from "../flight/missionGeometry.ts";
import {
  generatePlannedFlightGeometryForPolygon,
  getFlightTurnModel,
} from "../flight/plannedGeometry.ts";
import { queryMinMaxElevationAlongPolylineWGS84 } from "../flight/geometry.ts";

const ORIGIN: [number, number] = [8.54, 47.37];
const METERS_PER_DEG_LAT = 111_320;

function fromMeters(xM: number, yM: number): [number, number] {
  const latScale = Math.cos((ORIGIN[1] * Math.PI) / 180);
  return [
    ORIGIN[0] + xM / (METERS_PER_DEG_LAT * latScale),
    ORIGIN[1] + yM / METERS_PER_DEG_LAT,
  ];
}

function toMeters(point: [number, number]) {
  const latScale = Math.cos((ORIGIN[1] * Math.PI) / 180);
  return {
    x: (point[0] - ORIGIN[0]) * METERS_PER_DEG_LAT * latScale,
    y: (point[1] - ORIGIN[1]) * METERS_PER_DEG_LAT,
  };
}

function normalize2D(point: { x: number; y: number }) {
  const length = Math.hypot(point.x, point.y);
  return length > 1e-9 ? { x: point.x / length, y: point.y / length } : { x: 0, y: 0 };
}

function dot2D(left: { x: number; y: number }, right: { x: number; y: number }) {
  return left.x * right.x + left.y * right.y;
}

function tangentDirectionAtCirclePoint(
  maneuver: PlannedLeadManeuver,
  point: [number, number],
) {
  const sample = toMeters(point);
  const center = toMeters(maneuver.loiterCenter);
  const radial = normalize2D({ x: sample.x - center.x, y: sample.y - center.y });
  return maneuver.loiterDirection > 0
    ? normalize2D({ x: radial.y, y: -radial.x })
    : normalize2D({ x: -radial.y, y: radial.x });
}

function rectangleRing(widthM: number, heightM: number): [number, number][] {
  const halfWidthM = widthM / 2;
  const halfHeightM = heightM / 2;
  return [
    fromMeters(-halfWidthM, -halfHeightM),
    fromMeters(halfWidthM, -halfHeightM),
    fromMeters(halfWidthM, halfHeightM),
    fromMeters(-halfWidthM, halfHeightM),
    fromMeters(-halfWidthM, -halfHeightM),
  ];
}

function makeLead(
  centerXM: number,
  centerYM: number,
  anchorXM: number,
  anchorYM: number,
  direction: 1 | -1,
  radiusM = 60,
): PlannedLeadManeuver {
  return {
    anchorPoint: fromMeters(anchorXM, anchorYM),
    loiterCenter: fromMeters(centerXM, centerYM),
    loiterRadiusM: radiusM,
    loiterDirection: direction,
  };
}

function tileCoordsForLngLat(point: [number, number], z: number) {
  const z2 = 2 ** z;
  const normX = (point[0] + 180) / 360;
  const normY = (1 - Math.log(Math.tan(Math.PI / 4 + (point[1] * Math.PI / 180) / 2)) / Math.PI) / 2;
  return {
    x: Math.floor(normX * z2),
    y: Math.floor(normY * z2),
  };
}

function makeSteppedDemTile(
  originPoint: [number, number],
  z: number,
  width: number,
  boundaryPx: number,
  lowElevationM: number,
  highElevationM: number,
): TerrainTile {
  const { x, y } = tileCoordsForLngLat(originPoint, z);
  const data = new Float32Array(width * width);
  for (let py = 0; py < width; py += 1) {
    for (let px = 0; px < width; px += 1) {
      data[py * width + px] = px >= boundaryPx ? highElevationM : lowElevationM;
    }
  }
  return {
    x,
    y,
    z,
    width,
    height: width,
    data,
    format: "dem",
  };
}

function testPlannedGeometryExposesLeadManeuvers() {
  const params: FlightParams = {
    altitudeAGL: 120,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "SONY_RX1R2",
  };
  const geometry = generatePlannedFlightGeometryForPolygon(rectangleRing(360, 240), 90, 48, params);
  const firstConnectedPoint = geometry.connectedLines[0]?.[0];
  const lastSegment = geometry.connectedLines.at(-1);
  const lastConnectedPoint = lastSegment?.[lastSegment.length - 1];

  assert.ok(geometry.leadIn, "planned geometry should expose a structured lead-in maneuver");
  assert.ok(geometry.leadOut, "planned geometry should expose a structured lead-out maneuver");
  assert.deepEqual(geometry.leadIn?.anchorPoint, firstConnectedPoint, "lead-in anchor should match the first connected path point");
  assert.deepEqual(geometry.leadOut?.anchorPoint, lastConnectedPoint, "lead-out anchor should match the last connected path point");
  assert.ok((geometry.leadIn?.loiterRadiusM ?? 0) > 0, "lead-in radius should be populated");
  assert.ok((geometry.leadOut?.loiterRadiusM ?? 0) > 0, "lead-out radius should be populated");
  assert.ok(
    geometry.leadIn?.loiterDirection === 1 || geometry.leadIn?.loiterDirection === -1,
    "lead-in direction should use the same 1/-1 convention as turn blocks",
  );
  assert.ok(
    geometry.leadOut?.loiterDirection === 1 || geometry.leadOut?.loiterDirection === -1,
    "lead-out direction should use the same 1/-1 convention as turn blocks",
  );

  const flightTurnModel = getFlightTurnModel(params);
  assert.deepEqual(
    geometry.leadIn?.loiterCenter,
    geometry.leadInPoints[0],
    "lead-in center should match the survey-grid lead-in center used by the planner preview",
  );
  assert.deepEqual(
    geometry.leadOut?.loiterCenter,
    geometry.leadOutPoints.at(-1),
    "lead-out center should match the survey-grid lead-out center used by the planner preview",
  );
  assert.equal(
    geometry.leadIn?.loiterRadiusM,
    flightTurnModel.defaultTurnaroundRadiusM,
    "lead-in radius should use the configured turnaround radius like WingtraCloud",
  );
  assert.equal(
    geometry.leadOut?.loiterRadiusM,
    flightTurnModel.defaultTurnaroundRadiusM,
    "lead-out radius should use the configured turnaround radius like WingtraCloud",
  );
  assert.ok(geometry.leadIn?.pathJoinPoint, "lead-in should expose the sweep-to-loiter tangent join point");
  assert.ok(geometry.leadOut?.pathJoinPoint, "lead-out should expose the sweep-to-loiter tangent join point");
}

function testConnectorTangentSelectionReflectsLoiterDirections() {
  const source = makeLead(0, 0, 0, -140, 1);
  const sameDirectionTarget = makeLead(420, 0, 420, -140, 1);
  const oppositeDirectionTarget = makeLead(420, 0, 420, -140, -1);

  const direct = buildInterAreaConnectionGeometry({
    key: "same",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: source,
    targetManeuver: sameDirectionTarget,
    sourceAnchorAltitudeM: 120,
    targetAnchorAltitudeM: 120,
    cruiseSpeedMps: 12,
    terrainZoom: 14,
    terrainTileCount: 0,
  });
  const transverse = buildInterAreaConnectionGeometry({
    key: "opposite",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: source,
    targetManeuver: oppositeDirectionTarget,
    sourceAnchorAltitudeM: 120,
    targetAnchorAltitudeM: 120,
    cruiseSpeedMps: 12,
    terrainZoom: 14,
    terrainTileCount: 0,
  });

  const directStartOffsetY = toMeters(direct.transfer[0]).y - toMeters(source.loiterCenter).y;
  const directEndOffsetY = toMeters(direct.transfer[1]).y - toMeters(sameDirectionTarget.loiterCenter).y;
  const transverseStartOffsetY = toMeters(transverse.transfer[0]).y - toMeters(source.loiterCenter).y;
  const transverseEndOffsetY = toMeters(transverse.transfer[1]).y - toMeters(oppositeDirectionTarget.loiterCenter).y;

  assert.ok(
    Math.sign(directStartOffsetY) === Math.sign(directEndOffsetY),
    "equal loiter directions should choose a direct tangent on the same side of both circles",
  );
  assert.ok(
    Math.sign(transverseStartOffsetY) === -Math.sign(transverseEndOffsetY),
    "opposite loiter directions should choose a transverse tangent that switches circle sides",
  );
}

function testConnectorTransferDirectionMatchesArcTravelDirection() {
  const source = makeLead(0, 0, 0, -140, 1);
  const target = makeLead(420, 0, 420, -140, -1);
  const connection = buildInterAreaConnectionGeometry({
    key: "continuity",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: source,
    targetManeuver: target,
    sourceAnchorAltitudeM: 120,
    targetAnchorAltitudeM: 120,
    cruiseSpeedMps: 12,
    terrainZoom: 14,
    terrainTileCount: 0,
  });

  const transferStart = connection.transfer[0]!;
  const transferEnd = connection.transfer[1]!;
  const transferDirection = normalize2D({
    x: toMeters(transferEnd).x - toMeters(transferStart).x,
    y: toMeters(transferEnd).y - toMeters(transferStart).y,
  });
  const sourceArcDirection = tangentDirectionAtCirclePoint(source, transferStart);
  const targetArcDirection = tangentDirectionAtCirclePoint(target, transferEnd);

  assert.ok(
    dot2D(sourceArcDirection, transferDirection) > 0.99,
    "connector should leave the source loiter in the same direction as the transfer line",
  );
  assert.ok(
    dot2D(targetArcDirection, transferDirection) > 0.99,
    "connector should enter the target loiter in the same direction as the transfer line",
  );
}

function testConnector3DLiftAddsCoilsOnlyOnTheLowerSide() {
  const source = makeLead(0, 0, 0, -140, 1);
  const target = makeLead(420, 0, 420, -140, 1);

  const flat = buildInterAreaConnectionGeometry({
    key: "flat",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: source,
    targetManeuver: target,
    sourceAnchorAltitudeM: 180,
    targetAnchorAltitudeM: 180,
    cruiseSpeedMps: 12,
    terrainZoom: 14,
    terrainTileCount: 0,
  });
  const climbing = buildInterAreaConnectionGeometry({
    key: "climb",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: source,
    targetManeuver: target,
    sourceAnchorAltitudeM: 120,
    targetAnchorAltitudeM: 260,
    cruiseSpeedMps: 12,
    terrainZoom: 14,
    terrainTileCount: 0,
  });
  const descending = buildInterAreaConnectionGeometry({
    key: "descend",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: source,
    targetManeuver: target,
    sourceAnchorAltitudeM: 260,
    targetAnchorAltitudeM: 120,
    cruiseSpeedMps: 12,
    terrainZoom: 14,
    terrainTileCount: 0,
  });

  assert.equal(flat.transferAltitudeM, 180, "equal endpoint altitudes should keep the transfer at that altitude");
  assert.equal(climbing.transferAltitudeM, 260, "transfer altitude should be the higher endpoint altitude");
  assert.equal(descending.transferAltitudeM, 260, "descending case should also keep the transfer at the higher endpoint altitude");
  assert.ok(
    climbing.leadOut.length > flat.leadOut.length && climbing.leadIn.length === flat.leadIn.length,
    "climbing should add coil geometry only on the lower source side",
  );
  assert.ok(
    descending.leadIn.length > flat.leadIn.length && descending.leadOut.length === flat.leadOut.length,
    "descending should add coil geometry only on the lower target side",
  );
}

function testTerrainAwareConnectorSamplingRaisesConnectorAltitude() {
  const source = makeLead(0, 0, 0, -140, 1);
  const target = makeLead(420, 0, 420, -140, 1);
  const sourceTile = tileCoordsForLngLat(source.anchorPoint, 13);
  const sourceNormX = (source.anchorPoint[0] + 180) / 360;
  const sourcePixelX = Math.floor(sourceNormX * (2 ** 13) * 256) - sourceTile.x * 256;
  const terrainTiles = [makeSteppedDemTile(source.anchorPoint, 13, 256, sourcePixelX + 8, 0, 420)];

  const baseline = buildInterAreaConnectionGeometry({
    key: "terrain-baseline",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: source,
    targetManeuver: target,
    sourceAnchorAltitudeM: 120,
    targetAnchorAltitudeM: 120,
    cruiseSpeedMps: 12,
    terrainZoom: 0,
    terrainTileCount: terrainTiles.length,
  });
  const terrainAware = buildInterAreaConnectionGeometry({
    key: "terrain-aware",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: source,
    targetManeuver: target,
    sourceAnchorAltitudeM: 120,
    targetAnchorAltitudeM: 120,
    cruiseSpeedMps: 12,
    terrainZoom: 0,
    terrainTileCount: terrainTiles.length,
    terrainTiles,
    altitudeMode: "min-clearance",
    minClearanceM: 60,
  });

  assert.ok(
    terrainAware.transferAltitudeM > baseline.transferAltitudeM + 100,
    "terrain-aware connector sampling should materially raise the transfer altitude when terrain under the connector path is high",
  );

  const transferTerrainMax = queryMinMaxElevationAlongPolylineWGS84(terrainAware.transfer, terrainTiles, 12).max;
  assert.ok(
    terrainAware.transferAltitudeM >= transferTerrainMax + 60 - 1e-6,
    "terrain-aware transfer altitude should satisfy the sampled min-clearance constraint",
  );

  const minClearance = terrainAware.path3D
    .flatMap((segment) => segment)
    .reduce((minimumClearance, [lng, lat, altitudeM]) => {
      const terrainMax = queryMinMaxElevationAlongPolylineWGS84([[lng, lat]], terrainTiles, 1).max;
      return Number.isFinite(terrainMax)
        ? Math.min(minimumClearance, altitudeM - terrainMax)
        : minimumClearance;
    }, Number.POSITIVE_INFINITY);
  assert.ok(
    minClearance >= 60 - 1e-6,
    "terrain-aware connector lift should preserve the requested clearance margin at sampled connector points",
  );
}

function testMissionSummaryIncludesAreaAndConnectorTravel() {
  const params: FlightParams = {
    altitudeAGL: 120,
    frontOverlap: 75,
    sideOverlap: 70,
    cameraKey: "SONY_RX1R2",
  };
  const geometry = generatePlannedFlightGeometryForPolygon(rectangleRing(280, 220), 90, 45, params);
  const connection = buildInterAreaConnectionGeometry({
    key: "summary",
    fromPolygonId: "A",
    toPolygonId: "B",
    sourceManeuver: makeLead(0, 0, 0, -140, 1),
    targetManeuver: makeLead(420, 0, 420, -140, 1),
    sourceAnchorAltitudeM: 180,
    targetAnchorAltitudeM: 180,
    cruiseSpeedMps: 12,
    terrainZoom: 14,
    terrainTileCount: 0,
  });

  const summary = buildMissionTravelSummary([
    { polygonId: "A", params, geometry },
    { polygonId: "B", params, geometry },
  ], [connection]);
  const mission = buildMissionFlightGeometry(["A", "B"], [connection], [
    { polygonId: "A", params, geometry },
    { polygonId: "B", params, geometry },
  ]);

  assert.ok(summary.areaDistanceM > 0, "mission summary should include area-local path distance");
  assert.ok(summary.connectorDistanceM > 0, "mission summary should include connector distance");
  assert.equal(summary.totalDistanceM, summary.areaDistanceM + summary.connectorDistanceM);
  assert.equal(summary.totalTimeSec, summary.areaTimeSec + summary.connectorTimeSec);
  assert.deepEqual(mission.orderedPolygonIds, ["A", "B"]);
  assert.equal(mission.summary.totalDistanceM, summary.totalDistanceM);
}

testPlannedGeometryExposesLeadManeuvers();
testConnectorTangentSelectionReflectsLoiterDirections();
testConnectorTransferDirectionMatchesArcTravelDirection();
testConnector3DLiftAddsCoilsOnlyOnTheLowerSide();
testTerrainAwareConnectorSamplingRaisesConnectorAltitude();
testMissionSummaryIncludesAreaAndConnectorTravel();

console.log("mission_connectors.test.ts passed");
