import assert from "node:assert/strict";

import { planCoverageAutoRun } from "../overlap/coverageAutoRun.ts";

function testLidarRefreshSchedulesComputeWithoutCameraPoses() {
  const plan = planCoverageAutoRun({
    request: { reason: "spacing" },
    nowMs: 1_000,
    suppressAutoRunUntilMs: 0,
    autoGenerate: true,
    importedPosesCount: 0,
    ready: true,
    havePolys: true,
    haveLines: true,
    haveLidarPolys: true,
    posesCount: 0,
    retryCount: 0,
  });

  assert.equal(plan.kind, "compute");
  assert.deepEqual(plan.computeRequest, {
    polygonId: undefined,
    suppressMapNotReadyToast: true,
  });
  assert.equal(plan.nextRetryCount, 0);
}

function testReadyRefreshStillSchedulesComputeAfterRetries() {
  const plan = planCoverageAutoRun({
    request: { polygonId: "poly-1", reason: "lines" },
    nowMs: 2_000,
    suppressAutoRunUntilMs: 0,
    autoGenerate: true,
    importedPosesCount: 0,
    ready: true,
    havePolys: true,
    haveLines: true,
    haveLidarPolys: true,
    posesCount: 0,
    retryCount: 6,
  });

  assert.equal(
    plan.kind,
    "compute",
    "a ready follow-up refresh must still schedule compute instead of being dropped while another run is active",
  );
  assert.deepEqual(plan.computeRequest, {
    polygonId: undefined,
    suppressMapNotReadyToast: true,
  });
  assert.equal(plan.nextRetryCount, 0);
}

function testMapNotReadySchedulesRetry() {
  const plan = planCoverageAutoRun({
    request: { polygonId: "poly-1", reason: "lines" },
    nowMs: 3_000,
    suppressAutoRunUntilMs: 0,
    autoGenerate: true,
    importedPosesCount: 0,
    ready: false,
    havePolys: true,
    haveLines: true,
    haveLidarPolys: true,
    posesCount: 0,
    retryCount: 2,
  });

  assert.deepEqual(plan, {
    kind: "retry",
    nextRetryCount: 3,
    delayMs: 250,
  });
}

testLidarRefreshSchedulesComputeWithoutCameraPoses();
testReadyRefreshStillSchedulesComputeAfterRetries();
testMapNotReadySchedulesRetry();

console.log("coverage_autorun.test.ts passed");
