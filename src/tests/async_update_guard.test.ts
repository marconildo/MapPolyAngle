import assert from "node:assert/strict";

import { shouldApplyAsyncPolygonUpdate, shouldRunAsyncGeneration } from "../state/asyncUpdateGuard.ts";

function testAllowsMatchingGenerationForExistingPolygon() {
  assert.equal(
    shouldApplyAsyncPolygonUpdate({
      startedGeneration: 4,
      currentGeneration: 4,
      polygonStillExists: true,
    }),
    true,
  );
}

function testRejectsLateAsyncResultAfterReset() {
  assert.equal(
    shouldApplyAsyncPolygonUpdate({
      startedGeneration: 4,
      currentGeneration: 5,
      polygonStillExists: true,
    }),
    false,
  );
}

function testRejectsLateGenerationForNonPolygonAsyncWork() {
  assert.equal(shouldRunAsyncGeneration(4, 5), false);
  assert.equal(shouldRunAsyncGeneration(5, 5), true);
}

function testRejectsMissingPolygonEvenWithoutReset() {
  assert.equal(
    shouldApplyAsyncPolygonUpdate({
      startedGeneration: 4,
      currentGeneration: 4,
      polygonStillExists: false,
    }),
    false,
  );
}

testAllowsMatchingGenerationForExistingPolygon();
testRejectsLateAsyncResultAfterReset();
testRejectsLateGenerationForNonPolygonAsyncWork();
testRejectsMissingPolygonEvenWithoutReset();

console.log("async_update_guard.test.ts passed");
