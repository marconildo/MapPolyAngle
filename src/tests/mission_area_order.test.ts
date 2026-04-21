import assert from "node:assert/strict";
import {
  appendMissionAreaOrderIds,
  removeMissionAreaOrderIds,
  replaceMissionAreaOrderIds,
} from "../state/missionAreaOrder.ts";

assert.deepEqual(
  appendMissionAreaOrderIds(["a"], ["b", "a", "c"]),
  ["a", "b", "c"],
  "append should preserve order and skip duplicates",
);

assert.deepEqual(
  appendMissionAreaOrderIds(["a"], ["b", "b", "c", "c"]),
  ["a", "b", "c"],
  "append should dedupe ids introduced by the same update",
);

assert.deepEqual(
  removeMissionAreaOrderIds(["a", "b", "c"], ["b"]),
  ["a", "c"],
  "remove should drop requested ids only",
);

assert.deepEqual(
  replaceMissionAreaOrderIds(["before", "parent", "after"], ["parent", "child-a", "child-b"], ["child-a", "child-b"]),
  ["before", "child-a", "child-b", "after"],
  "split should replace parent with child ids even though child ids are transaction-affected ids",
);

assert.deepEqual(
  replaceMissionAreaOrderIds(["before", "child-a", "child-b", "after"], ["parent", "child-a", "child-b"], ["parent"]),
  ["before", "parent", "after"],
  "undo split should replace child ids with parent even though parent is a transaction-affected id",
);

assert.deepEqual(
  replaceMissionAreaOrderIds(["a", "b", "c", "d"], ["b", "c", "merged"], ["merged"]),
  ["a", "merged", "d"],
  "merge should insert merged id at the first affected source position",
);

assert.deepEqual(
  replaceMissionAreaOrderIds(["a", "merged", "d"], ["b", "c", "merged"], ["b", "c"]),
  ["a", "b", "c", "d"],
  "undo merge should restore the source ids in-place",
);

console.log("mission_area_order.test.ts passed");
