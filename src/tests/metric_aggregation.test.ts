import assert from "node:assert/strict";

import type { GSDStats } from "../overlap/types.ts";
import { aggregateMetricStats, aggregateOverallMetricStats } from "../overlap/metricAggregation.ts";

function makeStats(values: Array<{ bin: number; count: number; areaM2: number }>): GSDStats {
  const totalAreaM2 = values.reduce((sum, value) => sum + value.areaM2, 0);
  const count = values.reduce((sum, value) => sum + value.count, 0);
  const mean = values.reduce((sum, value) => sum + value.bin * value.areaM2, 0) / totalAreaM2;
  return {
    min: Math.min(...values.map((value) => value.bin)),
    max: Math.max(...values.map((value) => value.bin)),
    mean,
    count,
    totalAreaM2,
    histogram: values,
  };
}

function testSinglePolygonOverallMatchesPerPolygonStats() {
  const tileStats = [
    makeStats([
      { bin: 80, count: 10, areaM2: 50 },
      { bin: 120, count: 10, areaM2: 50 },
    ]),
    makeStats([
      { bin: 140, count: 10, areaM2: 50 },
      { bin: 220, count: 10, areaM2: 50 },
    ]),
  ];

  const perPolygon = aggregateMetricStats(tileStats);
  const overall = aggregateOverallMetricStats([{ metricKind: "density", tileStats }]).density;

  assert.ok(overall, "single-polygon overall density stats should exist");
  assert.deepEqual(overall, perPolygon);
}

function testOverallFlattensRawTileStatsAcrossPolygons() {
  const polygonA = [
    makeStats([
      { bin: 100, count: 5, areaM2: 20 },
      { bin: 150, count: 5, areaM2: 20 },
    ]),
  ];
  const polygonB = [
    makeStats([
      { bin: 200, count: 5, areaM2: 40 },
      { bin: 260, count: 5, areaM2: 40 },
    ]),
  ];

  const overall = aggregateOverallMetricStats([
    { metricKind: "density", tileStats: polygonA },
    { metricKind: "density", tileStats: polygonB },
  ]).density;

  const expected = aggregateMetricStats([...polygonA, ...polygonB]);
  assert.ok(overall, "overall density stats should exist");
  assert.deepEqual(overall, expected);
}

function testDensityAggregationPreservesExactZeroHoleBucket() {
  const holeHeavyStats = [
    makeStats([
      { bin: 0, count: 10, areaM2: 100 },
      { bin: 20, count: 10, areaM2: 20 },
      { bin: 60, count: 10, areaM2: 20 },
    ]),
  ];

  const aggregated = aggregateMetricStats(holeHeavyStats);
  assert.equal(aggregated.histogram[0]?.bin, 0);
  assert.equal(aggregated.histogram[0]?.areaM2, 100);
}

testSinglePolygonOverallMatchesPerPolygonStats();
testOverallFlattensRawTileStatsAcrossPolygons();
testDensityAggregationPreservesExactZeroHoleBucket();

console.log("metric_aggregation.test.ts passed");
