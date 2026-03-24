import assert from "node:assert/strict";

import {
  createCoveragePanelResetState,
  createHomeClearAllState,
  shouldConsumeClearAllEpoch,
} from "../state/clearAllState.ts";

function testCoveragePanelResetStateClearsTransientAnalysisState() {
  const cleared = createCoveragePanelResetState(12_345);

  assert.equal(cleared.runId, "12345");
  assert.equal(cleared.running, false);
  assert.equal(cleared.autoRetryCount, 0);
  assert.deepEqual(cleared.importedPoses, []);
  assert.deepEqual(cleared.poseAreaRing, []);
  assert.deepEqual(cleared.overallStats, { gsd: null, density: null });
  assert.equal(cleared.perPolygonStats.size, 0);
  assert.deepEqual(cleared.partitionOptionsByPolygon, {});
  assert.deepEqual(cleared.partitionSelectionByPolygon, {});
  assert.deepEqual(cleared.loadingPartitionOptionsIds, {});
  assert.deepEqual(cleared.applyingPartitionIds, {});
  assert.deepEqual(cleared.exactPartitionPreviewByKey, {});
  assert.deepEqual(cleared.splittingPolygonIds, {});
  assert.equal(cleared.selectedPolygonId, null);
}

function testCoveragePanelResetStateReturnsFreshContainers() {
  const first = createCoveragePanelResetState(1);
  const second = createCoveragePanelResetState(2);

  first.importedPoses.push("pose");
  first.poseAreaRing.push([8.5, 47.3]);
  first.perPolygonStats.set("poly-1", { count: 1 });
  first.partitionOptionsByPolygon["poly-1"] = ["option"];
  first.partitionSelectionByPolygon["poly-1"] = 2;
  first.loadingPartitionOptionsIds["poly-1"] = true;
  first.applyingPartitionIds["poly-1"] = true;
  first.exactPartitionPreviewByKey["poly-1"] = { score: 1 };
  first.splittingPolygonIds["poly-1"] = true;

  assert.deepEqual(second.importedPoses, []);
  assert.deepEqual(second.poseAreaRing, []);
  assert.equal(second.perPolygonStats.size, 0);
  assert.deepEqual(second.partitionOptionsByPolygon, {});
  assert.deepEqual(second.partitionSelectionByPolygon, {});
  assert.deepEqual(second.loadingPartitionOptionsIds, {});
  assert.deepEqual(second.applyingPartitionIds, {});
  assert.deepEqual(second.exactPartitionPreviewByKey, {});
  assert.deepEqual(second.splittingPolygonIds, {});
}

function testHomeClearAllStateResetsSelectionAndDialogState() {
  const cleared = createHomeClearAllState();

  assert.deepEqual(cleared.polygonResults, []);
  assert.equal(cleared.analyzingPolygons.size, 0);
  assert.deepEqual(cleared.paramsByPolygon, {});
  assert.deepEqual(cleared.paramsDialog, { open: false, polygonId: null });
  assert.deepEqual(cleared.importedOriginals, {});
  assert.deepEqual(cleared.overrides, {});
  assert.equal(cleared.importedPoseCount, 0);
  assert.equal(cleared.selectedPolygonId, null);
}

function testClearAllEpochIsConsumedOnlyOncePerValue() {
  assert.equal(shouldConsumeClearAllEpoch(0, 0), false);
  assert.equal(shouldConsumeClearAllEpoch(0, 1), true);
  assert.equal(shouldConsumeClearAllEpoch(1, 1), false);
  assert.equal(shouldConsumeClearAllEpoch(1, 2), true);
}

testCoveragePanelResetStateClearsTransientAnalysisState();
testCoveragePanelResetStateReturnsFreshContainers();
testHomeClearAllStateResetsSelectionAndDialogState();
testClearAllEpochIsConsumedOnlyOncePerValue();

console.log("clear_all_reset.test.ts passed");
