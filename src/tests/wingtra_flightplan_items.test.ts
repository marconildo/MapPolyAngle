import assert from "node:assert/strict";

import {
  exportToWingtraFlightPlan,
  importWingtraFlightPlan,
  isWingtraFlightPlanTemplateExportReady,
  replaceAreaItemsInWingtraFlightPlan,
  resolveFreshWingtraExportPayloadOptionFromAreas,
} from "../interop/wingtra/convert.ts";
import type { ExportedArea, WingtraAreaItem, WingtraFlightPlan } from "../interop/wingtra/types.ts";

function makeAreaItem(latStart: number, label: string): WingtraAreaItem & { label: string } {
  return {
    type: "ComplexItem",
    complexItemType: "area",
    version: 3,
    terrainFollowing: true,
    grid: {
      angle: 0,
      spacing: 40,
      altitude: 80,
      multithreading: false,
      turnAroundDistance: 80,
      turnAroundSideOffset: 70,
      safeRTHMaxSurveyAltitude: null,
    },
    camera: {
      imageSideOverlap: 70,
      imageFrontalOverlap: 75,
      cameraTriggerDistance: 20,
    },
    polygon: [
      [latStart, 8.0],
      [latStart, 8.001],
      [latStart + 0.001, 8.001],
      [latStart + 0.001, 8.0],
      [latStart, 8.0],
    ],
    wasFlown: false,
    label,
  };
}

function makeArea(latStart: number, angleDeg: number): ExportedArea {
  return {
    ring: [
      [8.0, latStart],
      [8.001, latStart],
      [8.001, latStart + 0.001],
      [8.0, latStart + 0.001],
      [8.0, latStart],
    ],
    payloadKind: "camera",
    altitudeAGL: 80,
    frontOverlap: 75,
    sideOverlap: 70,
    angleDeg,
    lineSpacingM: 40,
    triggerDistanceM: 20,
    terrainFollowing: true,
  };
}

function isAreaItem(item: unknown): item is WingtraAreaItem {
  return !!item && typeof item === "object" && (item as { type?: unknown }).type === "ComplexItem";
}

function runPreservesNonAreaItemsWhenAreaCountGrows() {
  const takeoff = { type: "SimpleItem", simpleItemType: "takeoff", label: "takeoff" };
  const loiter = { type: "SimpleItem", simpleItemType: "loiter", label: "loiter" };
  const landing = { type: "SimpleItem", simpleItemType: "landing", label: "landing" };
  const template: WingtraFlightPlan = {
    version: 1,
    fileType: "Plan",
    flightPlan: {
      version: 6,
      items: [
        takeoff,
        makeAreaItem(47.0, "area-a"),
        loiter,
        makeAreaItem(47.01, "area-b"),
        landing,
      ],
    },
  };

  const merged = replaceAreaItemsInWingtraFlightPlan(template, [
    makeArea(47.0, 10),
    makeArea(47.01, 20),
    makeArea(47.02, 30),
  ]);

  assert.equal(merged.flightPlan.items.length, 6);
  assert.deepEqual(merged.flightPlan.items[0], takeoff);
  assert.deepEqual(merged.flightPlan.items[2], loiter);
  assert.deepEqual(merged.flightPlan.items[5], landing);
  assert.equal(merged.flightPlan.items.filter(isAreaItem).length, 3);
}

function runPreservesNonAreaItemsWhenAreaCountShrinks() {
  const takeoff = { type: "SimpleItem", simpleItemType: "takeoff", label: "takeoff" };
  const waypoint = { type: "SimpleItem", simpleItemType: "waypoint", label: "waypoint" };
  const landing = { type: "SimpleItem", simpleItemType: "landing", label: "landing" };
  const template: WingtraFlightPlan = {
    version: 1,
    fileType: "Plan",
    flightPlan: {
      version: 6,
      items: [
        takeoff,
        makeAreaItem(47.0, "area-a"),
        waypoint,
        makeAreaItem(47.01, "area-b"),
        makeAreaItem(47.02, "area-c"),
        landing,
      ],
    },
  };

  const merged = replaceAreaItemsInWingtraFlightPlan(template, [makeArea(47.0, 12)]);

  assert.equal(merged.flightPlan.items.length, 4);
  assert.deepEqual(merged.flightPlan.items[0], takeoff);
  assert.deepEqual(merged.flightPlan.items[2], waypoint);
  assert.deepEqual(merged.flightPlan.items[3], landing);
  assert.equal(merged.flightPlan.items.filter(isAreaItem).length, 1);
}

runPreservesNonAreaItemsWhenAreaCountGrows();
runPreservesNonAreaItemsWhenAreaCountShrinks();

function runFreshExportIncludesMinimumWicDraftFields() {
  const exported = exportToWingtraFlightPlan([makeArea(47.0, 12)], {
    payloadKind: "camera",
    payloadUniqueString: "MAPSTAROblique_v5",
    payloadName: "MAPSTAROblique",
  });

  assert.equal(exported.fileType, "Plan");
  assert.equal(exported.version, 1);
  assert.equal(exported.groundStation, "WingtraPilot");
  assert.equal(exported.flightPlan.payloadUniqueString, "MAPSTAROblique_v5");
  assert.equal((exported.flightPlan as { planeHardware?: { hwVersion?: string } }).planeHardware?.hwVersion, "5");
  assert.equal(typeof (exported.flightPlan as { creationTime?: unknown }).creationTime, "number");
  assert.equal(typeof (exported.flightPlan as { lastModifiedTime?: unknown }).lastModifiedTime, "number");
  assert.ok((exported.flightPlan as { elevationData?: unknown }).elevationData);
  assert.ok(exported.geofence);
  assert.ok(exported.safety);
  assert.equal((exported.safety as { maxGroundClearance?: unknown })?.maxGroundClearance, 200);
  assert.equal((exported.flightPlan as { cruiseSpeed?: unknown }).cruiseSpeed, 15.375008);
  assert.ok(isWingtraFlightPlanTemplateExportReady(exported));
}

function runTemplateReadinessRejectsMissingMinimumFields() {
  const invalidTemplate: WingtraFlightPlan = {
    version: 1,
    fileType: "Plan",
    flightPlan: {
      version: 6,
      items: [makeAreaItem(47.0, "area-a")],
      payloadUniqueString: "MAPSTAROblique_v5",
      planeHardware: {
        hwVersion: "5",
      },
    } as any,
  };

  assert.equal(isWingtraFlightPlanTemplateExportReady(invalidTemplate), false);
}

function runHardwareVersionSelectsMatchingWingtraPayload() {
  const v4Option = resolveFreshWingtraExportPayloadOptionFromAreas([
    {
      ...makeArea(47.0, 12),
      cameraKey: "MAP61_17MM",
      planeHardwareVersion: "4",
    },
  ]);
  const v5Option = resolveFreshWingtraExportPayloadOptionFromAreas([
    {
      ...makeArea(47.0, 12),
      cameraKey: "MAP61_17MM",
      planeHardwareVersion: "5",
    },
  ]);

  assert.equal(v4Option?.payloadUniqueString, "MAPSTAROblique_v4");
  assert.equal(v5Option?.payloadUniqueString, "MAPSTAROblique_v5");
}

function runCameraImportPreservesCruiseSpeed() {
  const template = exportToWingtraFlightPlan([makeArea(47.0, 12)], {
    payloadKind: "camera",
    payloadUniqueString: "MAPSTAROblique_v5",
    payloadName: "MAPSTAROblique",
    defaults: {
      cruiseSpeed: 17.25,
    },
  });

  const imported = importWingtraFlightPlan(template, { angleConvention: "northCW" });
  assert.equal(imported.items.length, 1);
  assert.equal(imported.items[0]?.payloadKind, "camera");
  assert.equal(imported.items[0]?.speedMps, 17.25);
}

runFreshExportIncludesMinimumWicDraftFields();
runTemplateReadinessRejectsMissingMinimumFields();
runHardwareVersionSelectsMatchingWingtraPayload();
runCameraImportPreservesCruiseSpeed();

console.log("wingtra_flightplan_items.test.ts: all assertions passed");
