import type { GSDStats } from "./types";

export type AggregatedMetricKind = "gsd" | "density";

function extractExactZeroBucket(stats: GSDStats[]) {
  let zeroCount = 0;
  let zeroAreaM2 = 0;
  for (const stat of stats) {
    for (const bin of stat.histogram || []) {
      if (bin.bin === 0 && (bin.areaM2 || 0) > 0) {
        zeroCount += bin.count || 0;
        zeroAreaM2 += bin.areaM2 || 0;
      }
    }
  }
  return { zeroCount, zeroAreaM2 };
}

export function aggregateMetricStats(tileStats: GSDStats[], tailAreaAcres = 1): GSDStats {
  const valid = tileStats.filter((s) => s && s.count > 0 && isFinite(s.min) && isFinite(s.max) && s.max > 0);
  if (valid.length === 0) return { min: 0, max: 0, mean: 0, count: 0, totalAreaM2: 0, histogram: [] } as any;

  let totalCount = 0;
  let totalArea = 0;
  let weightedSum = 0;
  let globalMin = Number.POSITIVE_INFINITY;
  let globalMax = Number.NEGATIVE_INFINITY;

  for (const s of valid) {
    totalCount += s.count;
    const areaWeight = s.totalAreaM2 && s.totalAreaM2 > 0 ? s.totalAreaM2 : s.count;
    totalArea += areaWeight;
    weightedSum += s.mean * areaWeight;
    if (s.min < globalMin) globalMin = s.min;
    if (s.max > globalMax) globalMax = s.max;
  }

  const accurateMean = totalArea > 0 ? weightedSum / totalArea : 0;
  const span = globalMax - globalMin;
  const { zeroCount, zeroAreaM2 } = extractExactZeroBucket(valid);
  if (!(span > 0)) {
    return {
      min: globalMin,
      max: globalMax,
      mean: accurateMean,
      count: totalCount,
      totalAreaM2: totalArea,
      histogram: [{ bin: globalMin, count: totalCount, areaM2: totalArea }],
    } as any;
  }

  const targetBins = 20;
  const hasExactZeroBucket = zeroAreaM2 > 0 && globalMin === 0;
  const positiveMin = hasExactZeroBucket
    ? Math.min(...valid.flatMap((s) => (s.histogram || []).map((bin) => bin.bin)).filter((bin) => bin > 0))
    : globalMin;
  const positiveSpan = globalMax - positiveMin;
  const positiveBinCount = hasExactZeroBucket ? targetBins - 1 : targetBins;
  const positiveBinSize = positiveBinCount > 0 && positiveSpan > 0 ? positiveSpan / positiveBinCount : 0;
  const bins = new Array<{ bin: number; count: number; areaM2: number }>();
  if (hasExactZeroBucket) bins.push({ bin: 0, count: zeroCount, areaM2: zeroAreaM2 });
  for (let i = 0; i < positiveBinCount; i++) {
    bins.push({ bin: positiveMin + (i + 0.5) * positiveBinSize, count: 0, areaM2: 0 });
  }

  for (const s of valid) {
    if (!s.histogram || s.histogram.length === 0) continue;
    for (const hb of s.histogram) {
      if (!hb || hb.count === 0) continue;
      if (hasExactZeroBucket && hb.bin === 0) continue;
      let bi = hasExactZeroBucket
        ? 1 + Math.floor((hb.bin - positiveMin) / positiveBinSize)
        : Math.floor((hb.bin - globalMin) / positiveBinSize);
      const lowerIndex = hasExactZeroBucket ? 1 : 0;
      if (bi < lowerIndex) bi = lowerIndex;
      if (bi >= bins.length) bi = bins.length - 1;
      bins[bi].count += hb.count;
      bins[bi].areaM2 += hb.areaM2 || 0;
    }
  }

  const acreToM2 = 4046.8564224;
  const tailAreaM2 = Math.max(0, tailAreaAcres * acreToM2);
  const areaSum = bins.reduce((sum, bin) => sum + (bin.areaM2 || 0), 0);
  let minTrim = globalMin;
  let maxTrim = globalMax;
  if (areaSum > 0 && tailAreaM2 > 0 && tailAreaM2 * 2 < areaSum) {
    let cum = 0;
    let i = 0;
    for (; i < bins.length; i++) {
      const area = bins[i].areaM2 || 0;
      if (cum + area >= tailAreaM2) break;
      cum += area;
    }
    if (i < bins.length) minTrim = bins[i].bin;

    cum = 0;
    let j = bins.length - 1;
    for (; j >= 0; j--) {
      const area = bins[j].areaM2 || 0;
      if (cum + area >= tailAreaM2) break;
      cum += area;
    }
    if (j >= 0) maxTrim = bins[j].bin;
    if (!(maxTrim > minTrim)) {
      minTrim = globalMin;
      maxTrim = globalMax;
    }
  }

  return {
    min: minTrim,
    max: maxTrim,
    mean: accurateMean,
    count: totalCount,
    totalAreaM2: totalArea,
    histogram: bins.filter((bin) => bin.count > 0 || (bin.areaM2 || 0) > 0),
  } as any;
}

export function aggregateOverallMetricStats(
  groups: Array<{ metricKind: AggregatedMetricKind; tileStats: GSDStats[] }>,
  tailAreaAcres = 1,
): { gsd: GSDStats | null; density: GSDStats | null } {
  const gsdTileStats: GSDStats[] = [];
  const densityTileStats: GSDStats[] = [];

  for (const group of groups) {
    if (!group.tileStats.length) continue;
    if (group.metricKind === "density") densityTileStats.push(...group.tileStats);
    else gsdTileStats.push(...group.tileStats);
  }

  return {
    gsd: gsdTileStats.length > 0 ? aggregateMetricStats(gsdTileStats, tailAreaAcres) : null,
    density: densityTileStats.length > 0 ? aggregateMetricStats(densityTileStats, tailAreaAcres) : null,
  };
}
