import type { FlightParams, TerrainTile } from "@/domain/types";
import {
  buildTerrainGuidanceField,
  combinePartitionObjectives,
  evaluateSensorNodeCostForCells,
  findBestRegionOrientation,
  type PartitionObjective,
  type RegionOrientationObjective,
  type SensorNodeCost,
  type TerrainGuidanceCell,
  type TerrainGuidanceField,
  type TerrainPartitionTradeoffOptions,
} from "@/utils/terrainPartitionObjective";
import { partitionPolygonByTerrainFaces } from "@/utils/terrainFacePartition";

// @ts-ignore Turf typings are inconsistent in this repo.
import * as turf from "@turf/turf";

type Ring = [number, number][];

export type TerrainAtom = {
  id: string;
  cellIndices: number[];
  ring: Ring;
  areaM2: number;
  dominantBearingDeg: number | null;
  meanBreakStrength: number;
  meanConfidence: number;
  internalDispersionDeg: number;
  centroidLngLat: [number, number];
};

export type TerrainAtomAdjacency = {
  atomA: string;
  atomB: string;
  sharedBoundaryM: number;
  meanBearingDeltaDeg: number;
  meanBreakBarrier: number;
};

export type TerrainAtomGraph = {
  guidance: TerrainGuidanceField;
  atoms: TerrainAtom[];
  adjacency: TerrainAtomAdjacency[];
};

export type TerrainAtomizationOptions = TerrainPartitionTradeoffOptions & {
  atomDirectionMergeDeg?: number;
  atomBreakThreshold?: number;
  minAtomCells?: number;
  maxInitialAtoms?: number;
  candidateBearingStepDeg?: number;
  mergeImprovementSlack?: number;
  splitImprovementSlack?: number;
  tradeoffSamples?: number[];
  majorSplitMinChildAreaFraction?: number;
  refinementMinChildAreaFraction?: number;
  childAreaFractionDecay?: number;
  minModeSeparationDeg?: number;
  maxHierarchyRegions?: number;
  maxModeSeedPairs?: number;
};

export type FrontierRegion = {
  atomIds: string[];
  ring: Ring;
  objective: RegionOrientationObjective;
  convexity: number;
  compactness: number;
};

export type FrontierSolution = {
  tradeoff: number;
  signature: string;
  partition: PartitionObjective;
  regions: FrontierRegion[];
  hierarchyLevel: number;
  largestRegionFraction: number;
  meanConvexity: number;
  boundaryBreakAlignment: number;
  isFirstPracticalSplit: boolean;
  totalScore: number;
};

type ModeSeedPair = {
  modeA: number;
  modeB: number;
  separationDeg: number;
  score: number;
};

const DEFAULT_OPTIONS: Required<TerrainAtomizationOptions> = {
  tradeoff: 0.5,
  gridStepM: 0,
  searchSampleStepM: 0,
  minAreaM2: 4000,
  maxAspectRatio: 10,
  minConvexity: 0.38,
  cameraCruiseSpeedMps: 12,
  avgTurnSeconds: 8,
  perRegionOverheadSec: 25,
  interRegionTransitionSec: 35,
  shortLineThresholdFactor: 5,
  minWidthLineSpacingFactor: 2.5,
  atomDirectionMergeDeg: 14,
  atomBreakThreshold: 9,
  minAtomCells: 2,
  maxInitialAtoms: 36,
  candidateBearingStepDeg: 15,
  mergeImprovementSlack: 0.02,
  splitImprovementSlack: 0.02,
  tradeoffSamples: [0.1, 0.3, 0.5, 0.7, 0.9],
  majorSplitMinChildAreaFraction: 0.24,
  refinementMinChildAreaFraction: 0.12,
  childAreaFractionDecay: 0.05,
  minModeSeparationDeg: 18,
  maxHierarchyRegions: 8,
  maxModeSeedPairs: 18,
};

function degToRad(value: number) {
  return (value * Math.PI) / 180;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function mean(values: number[]) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function normalizedAxialBearing(value: number) {
  return ((value % 180) + 180) % 180;
}

function axialAngleDeltaDeg(a: number, b: number) {
  const aa = normalizedAxialBearing(a);
  const bb = normalizedAxialBearing(b);
  const delta = Math.abs(aa - bb);
  return Math.min(delta, 180 - delta);
}

function weightedAxialMeanDeg(values: Array<{ angleDeg: number; weight: number }>): number | null {
  let sumSin = 0;
  let sumCos = 0;
  let totalWeight = 0;
  for (const { angleDeg, weight } of values) {
    if (!(weight > 0) || !Number.isFinite(angleDeg)) continue;
    const doubled = degToRad(normalizedAxialBearing(angleDeg) * 2);
    sumSin += Math.sin(doubled) * weight;
    sumCos += Math.cos(doubled) * weight;
    totalWeight += weight;
  }
  if (!(totalWeight > 0)) return null;
  const meanRad = 0.5 * Math.atan2(sumSin, sumCos);
  return normalizedAxialBearing((meanRad * 180) / Math.PI);
}

function normalizeRing(ring: Ring): Ring | null {
  const cleaned = ring.filter(
    (coord): coord is [number, number] =>
      Array.isArray(coord) &&
      coord.length >= 2 &&
      Number.isFinite(coord[0]) &&
      Number.isFinite(coord[1]),
  );
  if (cleaned.length < 3) return null;
  const [firstLng, firstLat] = cleaned[0];
  const [lastLng, lastLat] = cleaned[cleaned.length - 1];
  if (firstLng === lastLng && firstLat === lastLat) return cleaned;
  return [...cleaned, [firstLng, firstLat]];
}

function lngLatToMercatorMeters(lng: number, lat: number): [number, number] {
  const R = 6378137;
  const lambda = degToRad(lng);
  const phi = Math.max(-85.05112878, Math.min(85.05112878, lat)) * Math.PI / 180;
  return [R * lambda, R * Math.log(Math.tan(Math.PI / 4 + phi / 2))];
}

function mercatorMetersToLngLat(x: number, y: number): [number, number] {
  const R = 6378137;
  const lng = (x / R) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * (180 / Math.PI);
  return [lng, lat];
}

function ringAreaM2(ring: Ring) {
  return turf.area(turf.polygon([ring]));
}

function buildCandidateBearings(graph: TerrainAtomGraph, stepDeg: number) {
  const sampled: number[] = [];
  for (let angle = 0; angle < 180; angle += stepDeg) sampled.push(angle);
  for (const atom of graph.atoms) {
    if (Number.isFinite(atom.dominantBearingDeg)) sampled.push(atom.dominantBearingDeg!);
  }
  const seen = new Set<number>();
  const out: number[] = [];
  for (const value of sampled) {
    const normalized = normalizedAxialBearing(value);
    const rounded = Math.round(normalized * 1000) / 1000;
    if (seen.has(rounded)) continue;
    seen.add(rounded);
    out.push(normalized);
  }
  return out;
}

function featureToSingleRing(feature: any): Ring | null {
  if (!feature?.geometry) return null;
  const cleaned = turf.cleanCoords(feature as any);
  const geom = cleaned?.geometry;
  if (!geom) return null;
  if (geom.type === "Polygon") {
    if (!Array.isArray(geom.coordinates) || geom.coordinates.length === 0) return null;
    return normalizeRing(geom.coordinates[0] as unknown as Ring);
  }
  if (geom.type === "MultiPolygon") {
    const polys = geom.coordinates as Ring[];
    if (!Array.isArray(polys) || polys.length === 0) return null;
    let best: Ring | null = null;
    let bestArea = -Infinity;
    for (const coords of polys) {
      const ring = normalizeRing(coords[0] as unknown as Ring);
      if (!ring) continue;
      const area = ringAreaM2(ring);
      if (area > bestArea) {
        bestArea = area;
        best = ring;
      }
    }
    return best;
  }
  return null;
}

function unionFeatures(a: any, b: any) {
  const unionFn = (turf as any).union;
  if (typeof unionFn !== "function") return null;
  const attempt = (left: any, right: any) => (
    unionFn.length >= 2
      ? unionFn(left, right)
      : unionFn(turf.featureCollection([left, right]))
  );
  try {
    return attempt(a, b);
  } catch {
    try {
      return attempt(normalizePolygonFeature(a), normalizePolygonFeature(b));
    } catch {
      return null;
    }
  }
}

function intersectFeatures(a: any, b: any) {
  const intersectFn = (turf as any).intersect;
  if (typeof intersectFn !== "function") return null;
  const attempt = (left: any, right: any) => (
    intersectFn.length >= 2
      ? intersectFn(left, right)
      : intersectFn(turf.featureCollection([left, right]))
  );
  try {
    return attempt(a, b);
  } catch {
    try {
      return attempt(normalizePolygonFeature(a), normalizePolygonFeature(b));
    } catch {
      return null;
    }
  }
}

function differenceFeatures(a: any, b: any) {
  const differenceFn = (turf as any).difference;
  if (typeof differenceFn !== "function") return null;
  const attempt = (left: any, right: any) => (
    differenceFn.length >= 2
      ? differenceFn(left, right)
      : differenceFn(turf.featureCollection([left, right]))
  );
  try {
    return attempt(a, b);
  } catch {
    try {
      return attempt(normalizePolygonFeature(a), normalizePolygonFeature(b));
    } catch {
      return null;
    }
  }
}

function normalizePolygonFeature(feature: any) {
  if (!feature?.geometry) return feature;
  let next = feature;
  try {
    next = turf.cleanCoords(next as any);
  } catch {}
  try {
    const truncateFn = (turf as any).truncate;
    if (typeof truncateFn === "function") {
      next = truncateFn(next, { precision: 10, coordinates: 2, mutate: false });
    }
  } catch {}
  try {
    const bufferFn = (turf as any).buffer;
    if (typeof bufferFn === "function" && (next.geometry?.type === "Polygon" || next.geometry?.type === "MultiPolygon")) {
      const repaired = bufferFn(next, 0, { units: "meters" });
      if (repaired?.geometry) next = repaired;
    }
  } catch {}
  try {
    next = turf.cleanCoords(next as any);
  } catch {}
  return next;
}

function featureToRings(feature: any): Ring[] {
  if (!feature?.geometry) return [];
  const cleaned = turf.cleanCoords(feature as any);
  const geom = cleaned?.geometry;
  if (!geom) return [];
  if (geom.type === "Polygon") {
    const ring = normalizeRing(geom.coordinates?.[0] as Ring);
    return ring ? [ring] : [];
  }
  if (geom.type === "MultiPolygon") {
    return (geom.coordinates as Ring[][])
      .map((coords) => normalizeRing(coords[0] as Ring))
      .filter((ring): ring is Ring => ring !== null);
  }
  return [];
}

function componentCellsToRing(cells: TerrainGuidanceCell[], gridStepM: number): Ring | null {
  if (cells.length === 0) return null;
  let merged: any = null;
  const half = gridStepM * 0.5;
  for (const cell of cells) {
    const square = turf.polygon([[
      mercatorMetersToLngLat(cell.x - half, cell.y - half),
      mercatorMetersToLngLat(cell.x + half, cell.y - half),
      mercatorMetersToLngLat(cell.x + half, cell.y + half),
      mercatorMetersToLngLat(cell.x - half, cell.y + half),
      mercatorMetersToLngLat(cell.x - half, cell.y - half),
    ]]);
    merged = merged ? unionFeatures(merged, square) : square;
    if (!merged) return null;
  }
  return featureToSingleRing(merged);
}

class UnionFind {
  parent: number[];
  rank: number[];
  constructor(size: number) {
    this.parent = Array.from({ length: size }, (_, index) => index);
    this.rank = new Array(size).fill(0);
  }
  find(x: number): number {
    if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]);
    return this.parent[x];
  }
  union(a: number, b: number) {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return false;
    if (this.rank[ra] < this.rank[rb]) this.parent[ra] = rb;
    else if (this.rank[ra] > this.rank[rb]) this.parent[rb] = ra;
    else {
      this.parent[rb] = ra;
      this.rank[ra] += 1;
    }
    return true;
  }
}

type CellEdge = {
  a: number;
  b: number;
  similarityCost: number;
  breakBarrier: number;
  bearingDeltaDeg: number;
  sharedBoundaryM: number;
};

function buildGridIndex(cells: TerrainGuidanceCell[], gridStepM: number) {
  const xs = cells.map((cell) => cell.x);
  const ys = cells.map((cell) => cell.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const keyToIndex = new Map<string, number>();
  const rows: number[] = [];
  const cols: number[] = [];
  cells.forEach((cell, index) => {
    const col = Math.round((cell.x - minX) / gridStepM);
    const row = Math.round((cell.y - minY) / gridStepM);
    rows[index] = row;
    cols[index] = col;
    keyToIndex.set(`${row}:${col}`, index);
  });
  return { keyToIndex, rows, cols };
}

function buildCellEdges(cells: TerrainGuidanceCell[], gridStepM: number) {
  const { keyToIndex, rows, cols } = buildGridIndex(cells, gridStepM);
  const edges: CellEdge[] = [];
  for (let i = 0; i < cells.length; i++) {
    const neighbors: Array<[number, number]> = [
      [rows[i], cols[i] + 1],
      [rows[i] + 1, cols[i]],
    ];
    for (const [row, col] of neighbors) {
      const neighborIndex = keyToIndex.get(`${row}:${col}`);
      if (neighborIndex == null) continue;
      const a = cells[i];
      const b = cells[neighborIndex];
      const bearingDeltaDeg = axialAngleDeltaDeg(a.preferredBearingDeg, b.preferredBearingDeg);
      const breakBarrier = 0.5 * (a.breakStrength + b.breakStrength);
      const confidenceGap = Math.abs(a.confidence - b.confidence);
      const similarityCost =
        bearingDeltaDeg / 18 +
        breakBarrier / 10 +
        confidenceGap * 0.8;
      edges.push({
        a: i,
        b: neighborIndex,
        similarityCost,
        breakBarrier,
        bearingDeltaDeg,
        sharedBoundaryM: gridStepM,
      });
    }
  }
  return edges;
}

function reassignTinyComponents(
  components: number[][],
  cellToComponent: number[],
  edges: CellEdge[],
  minAtomCells: number,
) {
  const tinyComponents = new Set<number>(
    components
      .map((cells, index) => ({ index, count: cells.length }))
      .filter((item) => item.count < minAtomCells)
      .map((item) => item.index),
  );
  if (tinyComponents.size === 0) return cellToComponent;

  const updated = [...cellToComponent];
  let changed = true;
  while (changed) {
    changed = false;
    for (const componentIndex of [...tinyComponents]) {
      const componentCells = updated
        .map((value, cellIndex) => ({ value, cellIndex }))
        .filter((item) => item.value === componentIndex)
        .map((item) => item.cellIndex);
      if (componentCells.length >= minAtomCells || componentCells.length === 0) {
        tinyComponents.delete(componentIndex);
        continue;
      }
      let bestNeighbor: { component: number; cost: number } | null = null;
      for (const edge of edges) {
        const ca = updated[edge.a];
        const cb = updated[edge.b];
        if (ca === cb) continue;
        const touchesTiny = ca === componentIndex || cb === componentIndex;
        if (!touchesTiny) continue;
        const otherComponent = ca === componentIndex ? cb : ca;
        const cost = edge.similarityCost;
        if (!bestNeighbor || cost < bestNeighbor.cost) {
          bestNeighbor = { component: otherComponent, cost };
        }
      }
      if (!bestNeighbor) continue;
      for (const cellIndex of componentCells) updated[cellIndex] = bestNeighbor.component;
      tinyComponents.delete(componentIndex);
      changed = true;
    }
  }
  return updated;
}

function cellsToComponents(uf: UnionFind, cellCount: number) {
  const groups = new Map<number, number[]>();
  for (let i = 0; i < cellCount; i++) {
    const root = uf.find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root)!.push(i);
  }
  const components = [...groups.values()];
  const cellToComponent = new Array(cellCount).fill(-1);
  components.forEach((cellIndices, componentIndex) => {
    cellIndices.forEach((cellIndex) => { cellToComponent[cellIndex] = componentIndex; });
  });
  return { components, cellToComponent };
}

function remapComponents(cellToComponent: number[]) {
  const componentMap = new Map<number, number>();
  const components: number[][] = [];
  cellToComponent.forEach((componentId, cellIndex) => {
    if (!componentMap.has(componentId)) {
      componentMap.set(componentId, components.length);
      components.push([]);
    }
    components[componentMap.get(componentId)!].push(cellIndex);
  });
  const normalizedCellMap = new Array(cellToComponent.length).fill(-1);
  components.forEach((component, componentIndex) => {
    component.forEach((cellIndex) => { normalizedCellMap[cellIndex] = componentIndex; });
  });
  return { components, cellToComponent: normalizedCellMap };
}

function buildCellAdjacencyLookup(cellCount: number, edges: CellEdge[]) {
  const lookup: number[][] = Array.from({ length: cellCount }, () => []);
  for (const edge of edges) {
    lookup[edge.a].push(edge.b);
    lookup[edge.b].push(edge.a);
  }
  return lookup;
}

function connectedCellComponents(cellIndices: number[], adjacencyLookup: number[][]) {
  const allowed = new Set(cellIndices);
  const visited = new Set<number>();
  const components: number[][] = [];
  for (const cellIndex of cellIndices) {
    if (visited.has(cellIndex)) continue;
    const stack = [cellIndex];
    visited.add(cellIndex);
    const component: number[] = [];
    while (stack.length > 0) {
      const current = stack.pop()!;
      component.push(current);
      for (const neighbor of adjacencyLookup[current] ?? []) {
        if (!allowed.has(neighbor) || visited.has(neighbor)) continue;
        visited.add(neighbor);
        stack.push(neighbor);
      }
    }
    components.push(component);
  }
  return components;
}

function componentAreaWeight(cellIndices: number[], cells: TerrainGuidanceCell[]) {
  return cellIndices.reduce((sum, cellIndex) => sum + Math.max(1e-6, cells[cellIndex].areaWeightM2), 0);
}

function weightedMeanAxialDeltaDeg(
  cellIndices: number[],
  cells: TerrainGuidanceCell[],
  modeDeg: number | null,
) {
  if (!Number.isFinite(modeDeg)) return Infinity;
  let weightedDelta = 0;
  let totalWeight = 0;
  for (const cellIndex of cellIndices) {
    const cell = cells[cellIndex];
    const weight = Math.max(1e-6, cell.areaWeightM2 * (0.25 + 0.75 * cell.confidence));
    weightedDelta += axialAngleDeltaDeg(cell.preferredBearingDeg, modeDeg!) * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? weightedDelta / totalWeight : Infinity;
}

function buildCellModeSeedPairs(cellIndices: number[], cells: TerrainGuidanceCell[], opts: Required<TerrainAtomizationOptions>) {
  const binStepDeg = 15;
  const binCount = Math.round(180 / binStepDeg);
  const bins = Array.from({ length: binCount }, (_, index) => ({
    angleDeg: index * binStepDeg + binStepDeg * 0.5,
    weight: 0,
  }));
  for (const cellIndex of cellIndices) {
    const cell = cells[cellIndex];
    const angle = normalizedAxialBearing(cell.preferredBearingDeg);
    const binIndex = Math.max(0, Math.min(binCount - 1, Math.floor(angle / binStepDeg)));
    bins[binIndex].weight += Math.max(1e-6, cell.areaWeightM2 * (0.25 + 0.75 * cell.confidence));
  }
  const activeBins = bins
    .filter((bin) => bin.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, Math.max(4, opts.maxModeSeedPairs));
  const pairs: ModeSeedPair[] = [];
  for (let i = 0; i < activeBins.length; i++) {
    for (let j = i + 1; j < activeBins.length; j++) {
      const modeA = activeBins[i].angleDeg;
      const modeB = activeBins[j].angleDeg;
      const separationDeg = axialAngleDeltaDeg(modeA, modeB);
      if (separationDeg < opts.minModeSeparationDeg) continue;
      pairs.push({
        modeA,
        modeB,
        separationDeg,
        score: separationDeg * Math.min(activeBins[i].weight, activeBins[j].weight),
      });
    }
  }
  return pairs
    .sort((a, b) => b.score - a.score)
    .slice(0, opts.maxModeSeedPairs);
}

function buildCellFeatureVector(
  cell: TerrainGuidanceCell,
  centerX: number,
  centerY: number,
  spatialScaleM: number,
) {
  const doubled = degToRad(normalizedAxialBearing(cell.preferredBearingDeg) * 2);
  return [
    Math.cos(doubled),
    Math.sin(doubled),
    (cell.x - centerX) / Math.max(1, spatialScaleM),
    (cell.y - centerY) / Math.max(1, spatialScaleM),
  ];
}

function squaredFeatureDistance(a: number[], b: number[]) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return sum;
}

function fallbackKMeansCellSplit(
  cellIndices: number[],
  cells: TerrainGuidanceCell[],
  adjacencyLookup: number[][],
  minChildArea: number,
) {
  const componentCells = cellIndices.map((cellIndex) => cells[cellIndex]);
  const centerX = mean(componentCells.map((cell) => cell.x));
  const centerY = mean(componentCells.map((cell) => cell.y));
  const xs = componentCells.map((cell) => cell.x);
  const ys = componentCells.map((cell) => cell.y);
  const spatialScaleM = Math.max(1, Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)));
  const featureVectors = new Map<number, number[]>();
  for (const cellIndex of cellIndices) {
    featureVectors.set(cellIndex, buildCellFeatureVector(cells[cellIndex], centerX, centerY, spatialScaleM));
  }

  const sortedByBearing = [...cellIndices].sort((a, b) => cells[a].preferredBearingDeg - cells[b].preferredBearingDeg);
  let centerA = featureVectors.get(sortedByBearing[0])!;
  let centerB = featureVectors.get(sortedByBearing[sortedByBearing.length - 1])!;
  const assignments = new Map<number, 0 | 1>();

  for (let iteration = 0; iteration < 6; iteration++) {
    assignments.clear();
    for (const cellIndex of cellIndices) {
      const feature = featureVectors.get(cellIndex)!;
      const distA = squaredFeatureDistance(feature, centerA);
      const distB = squaredFeatureDistance(feature, centerB);
      assignments.set(cellIndex, distA <= distB ? 0 : 1);
    }
    const repaired = repairCellAssignmentsForConnectivity(assignments, cellIndices, adjacencyLookup, cells);
    assignments.clear();
    repaired.forEach((value, index) => {
      assignments.set(index, value as 0 | 1);
    });

    const left = cellIndices.filter((cellIndex) => assignments.get(cellIndex) === 0);
    const right = cellIndices.filter((cellIndex) => assignments.get(cellIndex) === 1);
    if (left.length === 0 || right.length === 0) return null;

    const leftArea = componentAreaWeight(left, cells);
    const rightArea = componentAreaWeight(right, cells);
    if (leftArea < minChildArea || rightArea < minChildArea) return null;

    const nextCenterA = new Array(4).fill(0);
    const nextCenterB = new Array(4).fill(0);
    let weightA = 0;
    let weightB = 0;
    for (const cellIndex of left) {
      const feature = featureVectors.get(cellIndex)!;
      const weight = Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence));
      for (let i = 0; i < feature.length; i++) nextCenterA[i] += feature[i] * weight;
      weightA += weight;
    }
    for (const cellIndex of right) {
      const feature = featureVectors.get(cellIndex)!;
      const weight = Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence));
      for (let i = 0; i < feature.length; i++) nextCenterB[i] += feature[i] * weight;
      weightB += weight;
    }
    centerA = nextCenterA.map((value) => value / Math.max(1e-6, weightA));
    centerB = nextCenterB.map((value) => value / Math.max(1e-6, weightB));
  }

  const left = cellIndices.filter((cellIndex) => assignments.get(cellIndex) === 0);
  const right = cellIndices.filter((cellIndex) => assignments.get(cellIndex) === 1);
  if (left.length === 0 || right.length === 0) return null;
  const leftMode = weightedAxialMeanDeg(
    left.map((cellIndex) => ({
      angleDeg: cells[cellIndex].preferredBearingDeg,
      weight: Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence)),
    })),
  );
  const rightMode = weightedAxialMeanDeg(
    right.map((cellIndex) => ({
      angleDeg: cells[cellIndex].preferredBearingDeg,
      weight: Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence)),
    })),
  );
  const modeSeparation = axialAngleDeltaDeg(leftMode ?? 0, rightMode ?? 0);
  return { left, right, modeSeparation };
}

function splitComponentByBearingBands(
  cellIndices: number[],
  cells: TerrainGuidanceCell[],
  adjacencyLookup: number[][],
  minChildArea: number,
  bandStepDeg = 24,
) {
  const byBand = new Map<number, number[]>();
  for (const cellIndex of cellIndices) {
    const band = Math.floor(normalizedAxialBearing(cells[cellIndex].preferredBearingDeg) / bandStepDeg);
    if (!byBand.has(band)) byBand.set(band, []);
    byBand.get(band)!.push(cellIndex);
  }
  const splitComponents: number[][] = [];
  for (const bandCells of byBand.values()) {
    const components = connectedCellComponents(bandCells, adjacencyLookup);
    for (const component of components) {
      if (componentAreaWeight(component, cells) >= minChildArea) {
        splitComponents.push(component);
      }
    }
  }
  return splitComponents.length > 1 ? splitComponents : null;
}

function repairCellAssignmentsForConnectivity(
  assignments: Map<number, 0 | 1>,
  componentCellIndices: number[],
  adjacencyLookup: number[][],
  cells: TerrainGuidanceCell[],
) {
  const next = new Map(assignments);
  for (let iteration = 0; iteration < 4; iteration++) {
    let changed = false;
    for (const side of [0, 1] as const) {
      const sideCells = componentCellIndices.filter((cellIndex) => next.get(cellIndex) === side);
      if (sideCells.length <= 1) continue;
      const components = connectedCellComponents(sideCells, adjacencyLookup);
      if (components.length <= 1) continue;
      components.sort((a, b) => componentAreaWeight(b, cells) - componentAreaWeight(a, cells));
      for (const fragment of components.slice(1)) {
        for (const cellIndex of fragment) next.set(cellIndex, side === 0 ? 1 : 0);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return next;
}

function splitLargeMultimodalComponents(
  components: number[][],
  cells: TerrainGuidanceCell[],
  adjacencyLookup: number[][],
  totalGuidanceAreaM2: number,
  opts: Required<TerrainAtomizationOptions>,
) {
  const refined: number[][] = [];
  let changed = false;
  const queue = [...components];
  while (queue.length > 0) {
    const cellIndices = queue.shift()!;
    if (cellIndices.length < Math.max(opts.minAtomCells * 6, 18)) {
      refined.push(cellIndices);
      continue;
    }

    const singleMode = weightedAxialMeanDeg(
      cellIndices.map((cellIndex) => ({
        angleDeg: cells[cellIndex].preferredBearingDeg,
        weight: Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence)),
      })),
    );
    const singleMeanDelta = weightedMeanAxialDeltaDeg(cellIndices, cells, singleMode);
    const componentArea = componentAreaWeight(cellIndices, cells);
    const areaFraction = componentArea / Math.max(1, totalGuidanceAreaM2);
    const needsSplitForSize = areaFraction > 0.2 && singleMeanDelta >= 10;
    const needsSplitForModes = singleMeanDelta >= opts.minModeSeparationDeg * 0.45;
    if (!(needsSplitForSize || needsSplitForModes)) {
      refined.push(cellIndices);
      continue;
    }

    const totalArea = componentArea;
    const minChildArea = totalArea * Math.min(0.35, Math.max(opts.majorSplitMinChildAreaFraction, 0.14));
    const seedPairs = buildCellModeSeedPairs(cellIndices, cells, opts);
    let bestSplit: { left: number[]; right: number[]; score: number } | null = null;

    for (const seed of seedPairs) {
      let assignments = new Map<number, 0 | 1>();
      let modeA = seed.modeA;
      let modeB = seed.modeB;
      for (let iteration = 0; iteration < 3; iteration++) {
        assignments = new Map<number, 0 | 1>();
        for (const cellIndex of cellIndices) {
          const cell = cells[cellIndex];
          const deltaA = axialAngleDeltaDeg(cell.preferredBearingDeg, modeA);
          const deltaB = axialAngleDeltaDeg(cell.preferredBearingDeg, modeB);
          assignments.set(cellIndex, deltaA <= deltaB ? 0 : 1);
        }
        assignments = repairCellAssignmentsForConnectivity(assignments, cellIndices, adjacencyLookup, cells);
        const left = cellIndices.filter((cellIndex) => assignments.get(cellIndex) === 0);
        const right = cellIndices.filter((cellIndex) => assignments.get(cellIndex) === 1);
        if (left.length === 0 || right.length === 0) break;
        const nextModeA = weightedAxialMeanDeg(
          left.map((cellIndex) => ({
            angleDeg: cells[cellIndex].preferredBearingDeg,
            weight: Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence)),
          })),
        );
        const nextModeB = weightedAxialMeanDeg(
          right.map((cellIndex) => ({
            angleDeg: cells[cellIndex].preferredBearingDeg,
            weight: Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence)),
          })),
        );
        if (nextModeA == null || nextModeB == null) break;
        if (axialAngleDeltaDeg(nextModeA, modeA) < 1e-3 && axialAngleDeltaDeg(nextModeB, modeB) < 1e-3) break;
        modeA = nextModeA;
        modeB = nextModeB;
      }

      const left = cellIndices.filter((cellIndex) => assignments.get(cellIndex) === 0);
      const right = cellIndices.filter((cellIndex) => assignments.get(cellIndex) === 1);
      if (left.length === 0 || right.length === 0) continue;
      const leftArea = componentAreaWeight(left, cells);
      const rightArea = componentAreaWeight(right, cells);
      if (leftArea < minChildArea || rightArea < minChildArea) continue;

      const leftMode = weightedAxialMeanDeg(
        left.map((cellIndex) => ({
          angleDeg: cells[cellIndex].preferredBearingDeg,
          weight: Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence)),
        })),
      );
      const rightMode = weightedAxialMeanDeg(
        right.map((cellIndex) => ({
          angleDeg: cells[cellIndex].preferredBearingDeg,
          weight: Math.max(1e-6, cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * cells[cellIndex].confidence)),
        })),
      );
      const modeSeparation = axialAngleDeltaDeg(leftMode ?? modeA, rightMode ?? modeB);
      if (modeSeparation < opts.minModeSeparationDeg) continue;

      const splitMeanDelta =
        (weightedMeanAxialDeltaDeg(left, cells, leftMode) * leftArea +
          weightedMeanAxialDeltaDeg(right, cells, rightMode) * rightArea) /
        Math.max(1e-6, leftArea + rightArea);
      const improvement = singleMeanDelta - splitMeanDelta;
      if (improvement < Math.max(6, opts.minModeSeparationDeg * 0.3)) continue;

      const balance = Math.min(leftArea, rightArea) / Math.max(leftArea, rightArea);
      const score = improvement + 0.2 * modeSeparation + 8 * balance;
      if (!bestSplit || score > bestSplit.score) {
        bestSplit = { left, right, score };
      }
    }

    if (!bestSplit) {
      const fallbackSplit = fallbackKMeansCellSplit(
        cellIndices,
        cells,
        adjacencyLookup,
        minChildArea,
      );
      if (fallbackSplit) {
        const leftArea = componentAreaWeight(fallbackSplit.left, cells);
        const rightArea = componentAreaWeight(fallbackSplit.right, cells);
        const balance = Math.min(leftArea, rightArea) / Math.max(leftArea, rightArea);
        bestSplit = {
          left: fallbackSplit.left,
          right: fallbackSplit.right,
          score: fallbackSplit.modeSeparation + 6 * balance,
        };
      }
    }

    if (!bestSplit) {
      refined.push(cellIndices);
      continue;
    }

    changed = true;
    queue.push(bestSplit.left, bestSplit.right);
  }
  return { components: refined, changed };
}

export function buildTerrainAtomGraph(
  ring: Ring,
  tiles: TerrainTile[],
  options: TerrainAtomizationOptions = {},
): TerrainAtomGraph {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const guidance = buildTerrainGuidanceField(ring, tiles, opts);
  if (guidance.cells.length === 0) return { guidance, atoms: [], adjacency: [] };

  const edges = buildCellEdges(guidance.cells, guidance.gridStepM);
  const uf = new UnionFind(guidance.cells.length);
  const sortedEdges = [...edges].sort((a, b) => a.similarityCost - b.similarityCost);
  let componentCount = guidance.cells.length;

  for (const edge of sortedEdges) {
    const canMerge =
      edge.bearingDeltaDeg <= opts.atomDirectionMergeDeg &&
      edge.breakBarrier <= opts.atomBreakThreshold * 1.25 &&
      edge.similarityCost <= 1.75;
    if (!canMerge) continue;
    if (uf.union(edge.a, edge.b)) componentCount -= 1;
  }

  if (componentCount > opts.maxInitialAtoms) {
    for (const edge of sortedEdges) {
      if (componentCount <= opts.maxInitialAtoms) break;
      if (uf.union(edge.a, edge.b)) componentCount -= 1;
    }
  }

  const grouped = cellsToComponents(uf, guidance.cells.length);
  const reassigned = reassignTinyComponents(grouped.components, grouped.cellToComponent, edges, opts.minAtomCells);
  let { components, cellToComponent } = remapComponents(reassigned);
  const adjacencyLookup = buildCellAdjacencyLookup(guidance.cells.length, edges);
  const multimodalSplit = splitLargeMultimodalComponents(
    components,
    guidance.cells,
    adjacencyLookup,
    guidance.areaM2,
    opts,
  );
  if (multimodalSplit.changed) {
    const nextCellToComponent = new Array(guidance.cells.length).fill(-1);
    multimodalSplit.components.forEach((component, componentIndex) => {
      component.forEach((cellIndex) => { nextCellToComponent[cellIndex] = componentIndex; });
    });
    const normalized = remapComponents(nextCellToComponent);
    components = normalized.components;
    cellToComponent = normalized.cellToComponent;
  }
  for (let oversizedIteration = 0; oversizedIteration < 4; oversizedIteration++) {
    const oversizedRefined: number[][] = [];
    let oversizedChanged = false;
    for (const component of components) {
      const componentArea = componentAreaWeight(component, guidance.cells);
      const componentDispersion = weightedMeanAxialDeltaDeg(
        component,
        guidance.cells,
        weightedAxialMeanDeg(
          component.map((cellIndex) => ({
            angleDeg: guidance.cells[cellIndex].preferredBearingDeg,
            weight: Math.max(1e-6, guidance.cells[cellIndex].areaWeightM2 * (0.25 + 0.75 * guidance.cells[cellIndex].confidence)),
          })),
        ),
      );
      const areaFraction = componentArea / Math.max(1, guidance.areaM2);
      if (areaFraction > 0.2 && componentDispersion >= 10) {
        const fallbackSplit = fallbackKMeansCellSplit(
          component,
          guidance.cells,
          adjacencyLookup,
          componentArea * 0.14,
        );
        if (fallbackSplit) {
          oversizedRefined.push(fallbackSplit.left, fallbackSplit.right);
          oversizedChanged = true;
          continue;
        }
        const bandSplit = splitComponentByBearingBands(
          component,
          guidance.cells,
          adjacencyLookup,
          componentArea * 0.1,
        );
        if (bandSplit) {
          oversizedRefined.push(...bandSplit);
          oversizedChanged = true;
          continue;
        }
      }
      oversizedRefined.push(component);
    }
    if (!oversizedChanged) break;
    const nextCellToComponent = new Array(guidance.cells.length).fill(-1);
    oversizedRefined.forEach((component, componentIndex) => {
      component.forEach((cellIndex) => { nextCellToComponent[cellIndex] = componentIndex; });
    });
    const normalized = remapComponents(nextCellToComponent);
    components = normalized.components;
    cellToComponent = normalized.cellToComponent;
  }
  const finalCellToComponent = new Array(guidance.cells.length).fill(-1);
  components.forEach((component, componentIndex) => {
    component.forEach((cellIndex) => { finalCellToComponent[cellIndex] = componentIndex; });
  });
  const finalReassigned = reassignTinyComponents(components, finalCellToComponent, edges, opts.minAtomCells);
  const finalNormalized = remapComponents(finalReassigned);
  components = finalNormalized.components;
  cellToComponent = finalNormalized.cellToComponent;

  const componentAtoms = components.map((cellIndices, componentIndex) => {
      const cells = cellIndices.map((cellIndex) => guidance.cells[cellIndex]);
      const ring = componentCellsToRing(cells, guidance.gridStepM);
      if (!ring) return null;
      const areaM2 = ringAreaM2(ring);
      const dominantBearingDeg = weightedAxialMeanDeg(
        cells.map((cell) => ({ angleDeg: cell.preferredBearingDeg, weight: Math.max(1e-6, cell.confidence * cell.areaWeightM2) })),
      );
      const internalDispersionDeg = weightedMeanAxialDeltaDeg(
        cellIndices,
        guidance.cells,
        dominantBearingDeg,
      );
      const centroidLngLat: [number, number] = [
        mean(cells.map((cell) => cell.lng)),
        mean(cells.map((cell) => cell.lat)),
      ];
      return {
        id: `atom-${componentIndex + 1}`,
        cellIndices,
        ring,
        areaM2,
        dominantBearingDeg,
        meanBreakStrength: mean(cells.map((cell) => cell.breakStrength)),
        meanConfidence: mean(cells.map((cell) => cell.confidence)),
        internalDispersionDeg,
        centroidLngLat,
      } satisfies TerrainAtom;
    })
  const atoms: TerrainAtom[] = componentAtoms.filter((atom): atom is TerrainAtom => atom !== null);

  const adjacencyMap = new Map<string, { atomA: string; atomB: string; sharedBoundaryM: number; bearingDeltaSum: number; breakBarrierSum: number; count: number }>();
  for (const edge of edges) {
    const componentA = cellToComponent[edge.a];
    const componentB = cellToComponent[edge.b];
    if (componentA === componentB) continue;
    const atomA = componentAtoms[componentA]?.id;
    const atomB = componentAtoms[componentB]?.id;
    if (!atomA || !atomB) continue;
    const [left, right] = atomA < atomB ? [atomA, atomB] : [atomB, atomA];
    const key = `${left}|${right}`;
    const existing = adjacencyMap.get(key) ?? {
      atomA: left,
      atomB: right,
      sharedBoundaryM: 0,
      bearingDeltaSum: 0,
      breakBarrierSum: 0,
      count: 0,
    };
    existing.sharedBoundaryM += edge.sharedBoundaryM;
    existing.bearingDeltaSum += edge.bearingDeltaDeg;
    existing.breakBarrierSum += edge.breakBarrier;
    existing.count += 1;
    adjacencyMap.set(key, existing);
  }

  const adjacency: TerrainAtomAdjacency[] = [...adjacencyMap.values()].map((item) => ({
    atomA: item.atomA,
    atomB: item.atomB,
    sharedBoundaryM: item.sharedBoundaryM,
    meanBearingDeltaDeg: item.count > 0 ? item.bearingDeltaSum / item.count : 0,
    meanBreakBarrier: item.count > 0 ? item.breakBarrierSum / item.count : 0,
  }));

  return { guidance, atoms, adjacency };
}

function unionRings(a: Ring, b: Ring): Ring | null {
  try {
    const merged = unionFeatures(turf.polygon([a]), turf.polygon([b]));
    return featureToSingleRing(merged);
  } catch {
    return null;
  }
}

type InternalRegion = FrontierRegion & {
  regionId: string;
  depth: number;
  centroidLngLat: [number, number];
};

type RegionSeedMode = {
  angleDeg: number;
  weight: number;
  centroidLngLat: [number, number];
};

type RegionSeed = {
  regionId: string;
  atomId: string;
  angleDeg: number;
  centroidLngLat: [number, number];
};

type RegionState = {
  regionId: string;
  atomIds: string[];
  areaM2: number;
  bearingDeg: number;
  centroidLngLat: [number, number];
};

type RegionAdjacencyStats = {
  regionA: string;
  regionB: string;
  sharedBoundaryM: number;
  meanBreakBarrier: number;
  meanBearingDeltaDeg: number;
};

type AtomBearingCostTable = Map<string, Map<number, SensorNodeCost>>;

type PartitionEvaluation = {
  regions: InternalRegion[];
  assignments: Map<string, string>;
  partition: PartitionObjective;
  totalScore: number;
  largestRegionFraction: number;
  meanConvexity: number;
  boundaryBreakAlignment: number;
};

type FineSegmentationResult = {
  selected: PartitionEvaluation;
  bestByRegionCount: Map<number, PartitionEvaluation>;
};

const QUALITY_HEAVY_TRADEOFF = 0.92;
const MAX_SEGMENTATION_ITERATIONS = 10;
const MAX_SEED_VARIANTS = 3;

function atomRegionSignature(atomIds: string[]) {
  return [...atomIds].sort().join(",");
}

function bestObjectiveForRing(
  ring: Ring,
  tiles: TerrainTile[],
  params: FlightParams,
  candidateBearings: number[],
  options: TerrainAtomizationOptions,
) {
  return findBestRegionOrientation(ring, tiles, params, candidateBearings, options);
}

function distanceMetersBetweenLngLat(a: [number, number], b: [number, number]) {
  const [ax, ay] = lngLatToMercatorMeters(a[0], a[1]);
  const [bx, by] = lngLatToMercatorMeters(b[0], b[1]);
  return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
}

function buildSolutionSignature(regions: FrontierRegion[]) {
  return regions
    .map((region) => `${atomRegionSignature(region.atomIds)}@${Math.round(region.objective.bearingDeg * 10) / 10}`)
    .sort()
    .join("|");
}

function cloneSortedRegions(regions: FrontierRegion[]) {
  return [...regions]
    .map((region) => ({
      atomIds: [...region.atomIds].sort(),
      ring: region.ring,
      objective: region.objective,
      convexity: region.convexity,
      compactness: region.compactness,
    }))
    .sort((a, b) => a.atomIds.join(",").localeCompare(b.atomIds.join(",")));
}

function buildFrontierSolution(
  evaluation: PartitionEvaluation,
  hierarchyLevel: number,
  isFirstPracticalSplit: boolean,
  tradeoff: number,
): FrontierSolution {
  const sortedRegions = cloneSortedRegions(evaluation.regions);
  return {
    tradeoff,
    signature: buildSolutionSignature(sortedRegions),
    partition: evaluation.partition,
    regions: sortedRegions,
    hierarchyLevel,
    largestRegionFraction: evaluation.largestRegionFraction,
    meanConvexity: evaluation.meanConvexity,
    boundaryBreakAlignment: evaluation.boundaryBreakAlignment,
    isFirstPracticalSplit,
    totalScore: evaluation.totalScore,
  };
}

function computeSolutionCoverageMetrics(parentRing: Ring, regions: FrontierRegion[]) {
  const parentFeature = turf.polygon([parentRing]);
  const parentArea = Math.max(1e-6, turf.area(parentFeature));
  let unionFeature: any = null;
  let summedArea = 0;
  for (const region of regions) {
    const feature = turf.polygon([region.ring]);
    summedArea += turf.area(feature);
    unionFeature = unionFeature ? unionFeatures(unionFeature, feature) : feature;
    if (!unionFeature) {
      return { coverageRatio: 0, overlapRatio: 1 };
    }
  }
  const coveredFeature = intersectFeatures(parentFeature, unionFeature);
  const coveredArea = coveredFeature ? turf.area(coveredFeature) : 0;
  const unionArea = turf.area(unionFeature);
  return {
    coverageRatio: coveredArea / parentArea,
    overlapRatio: Math.max(0, summedArea - unionArea) / parentArea,
  };
}

function filterViableCoverageSolutions(parentRing: Ring, solutions: FrontierSolution[]) {
  return solutions.filter((solution) => {
    const { coverageRatio, overlapRatio } = computeSolutionCoverageMetrics(parentRing, solution.regions);
    return coverageRatio >= 0.5 && overlapRatio <= 0.35;
  });
}

function buildAtomMap(graph: TerrainAtomGraph) {
  return new Map(graph.atoms.map((atom) => [atom.id, atom]));
}

function buildAdjacencyLookup(adjacency: TerrainAtomAdjacency[]) {
  const byAtom = new Map<string, TerrainAtomAdjacency[]>();
  for (const edge of adjacency) {
    if (!byAtom.has(edge.atomA)) byAtom.set(edge.atomA, []);
    if (!byAtom.has(edge.atomB)) byAtom.set(edge.atomB, []);
    byAtom.get(edge.atomA)!.push(edge);
    byAtom.get(edge.atomB)!.push(edge);
  }
  return byAtom;
}

function completeRegionCoverage(
  parentRing: Ring,
  regions: InternalRegion[],
  tiles: TerrainTile[],
  params: FlightParams,
  candidateBearings: number[],
  options: Required<TerrainAtomizationOptions>,
) {
  const parentFeature = turf.polygon([parentRing]);
  const parentArea = Math.max(1e-6, turf.area(parentFeature));
  const regionFeatures = new Map<string, any>();
  for (const region of regions) {
    regionFeatures.set(region.regionId, turf.polygon([region.ring]));
  }

  let unionFeature: any = null;
  for (const feature of regionFeatures.values()) {
    unionFeature = unionFeature ? unionFeatures(unionFeature, feature) : feature;
    if (!unionFeature) return null;
  }

  const uncovered = unionFeature ? differenceFeatures(parentFeature, unionFeature) : parentFeature;
  const uncoveredRings = featureToRings(uncovered);
  const uncoveredArea = uncoveredRings.reduce((sum, ring) => sum + ringAreaM2(ring), 0);
  if (!(uncoveredArea > parentArea * 0.002)) return regions;

  for (const gapRing of uncoveredRings) {
    const gapFeature = turf.polygon([gapRing]);
    const gapCentroidPoint = turf.centroid(gapFeature);
    let bestRegionId: string | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const region of regions) {
      const currentFeature = regionFeatures.get(region.regionId);
      if (!currentFeature) continue;
      const merged = unionFeatures(currentFeature, gapFeature);
      const mergedRing = featureToSingleRing(merged);
      if (!mergedRing) continue;
      const distance = turf.pointToLineDistance(
        gapCentroidPoint,
        turf.lineString(region.ring),
        { units: "meters" },
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        bestRegionId = region.regionId;
      }
    }

    if (!bestRegionId) {
      for (const region of regions) {
        const distance = turf.pointToLineDistance(
          gapCentroidPoint,
          turf.lineString(region.ring),
          { units: "meters" },
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          bestRegionId = region.regionId;
        }
      }
    }

    if (!bestRegionId) continue;
    const currentFeature = regionFeatures.get(bestRegionId);
    const merged = currentFeature ? unionFeatures(currentFeature, gapFeature) : gapFeature;
    if (!merged) continue;
    regionFeatures.set(bestRegionId, merged);
  }

  const rebuilt: InternalRegion[] = [];
  for (const region of regions) {
    const feature = regionFeatures.get(region.regionId);
    const clipped = feature ? intersectFeatures(parentFeature, feature) : null;
    const ring = featureToSingleRing(clipped ?? feature);
    if (!ring) return null;
    const objective = bestObjectiveForRing(ring, tiles, params, candidateBearings, options);
    if (!objective) return null;
    rebuilt.push({
      ...region,
      ring,
      objective,
      convexity: objective.regularization.convexity,
      compactness: objective.regularization.compactness,
      centroidLngLat: turf.centroid(turf.polygon([ring])).geometry.coordinates as [number, number],
    });
  }

  return rebuilt;
}

function mergeAtomRings(atomIds: string[], atomMap: Map<string, TerrainAtom>) {
  const sorted = [...new Set(atomIds)].sort();
  if (sorted.length === 0) return null;
  let merged: Ring | null = null;
  for (const atomId of sorted) {
    const ring = atomMap.get(atomId)?.ring;
    if (!ring) return null;
    merged = merged ? unionRings(merged, ring) : ring;
    if (!merged) return null;
  }
  return merged;
}

function buildRegionRingFromCells(
  atomIds: string[],
  atomMap: Map<string, TerrainAtom>,
  guidance: TerrainGuidanceField,
) {
  const seen = new Set<number>();
  const cells: TerrainGuidanceCell[] = [];
  for (const atomId of [...new Set(atomIds)].sort()) {
    const atom = atomMap.get(atomId);
    if (!atom) return null;
    for (const cellIndex of atom.cellIndices) {
      if (seen.has(cellIndex)) continue;
      seen.add(cellIndex);
      const cell = guidance.cells[cellIndex];
      if (cell) cells.push(cell);
    }
  }
  if (cells.length === 0) return null;
  return componentCellsToRing(cells, guidance.gridStepM);
}

function regionAreaFromAtoms(atomIds: string[], atomMap: Map<string, TerrainAtom>) {
  return atomIds.reduce((sum, atomId) => sum + (atomMap.get(atomId)?.areaM2 ?? 0), 0);
}

function axialModeCost(bearingDeg: number | null, modeDeg: number) {
  if (!Number.isFinite(bearingDeg)) return 45;
  return axialAngleDeltaDeg(bearingDeg!, modeDeg);
}

function connectedComponentsForAtomIds(
  atomIds: string[],
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
) {
  const allowed = new Set(atomIds);
  const visited = new Set<string>();
  const components: string[][] = [];
  for (const atomId of atomIds) {
    if (visited.has(atomId)) continue;
    const queue = [atomId];
    visited.add(atomId);
    const component: string[] = [];
    while (queue.length > 0) {
      const current = queue.pop()!;
      component.push(current);
      for (const edge of adjacencyLookup.get(current) ?? []) {
        const neighbor = edge.atomA === current ? edge.atomB : edge.atomA;
        if (!allowed.has(neighbor) || visited.has(neighbor)) continue;
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
    components.push(component);
  }
  return components;
}

function canonicalizeAssignmentsByConnectivity(
  assignments: Map<string, string>,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  atomMap: Map<string, TerrainAtom>,
) {
  const grouped = new Map<string, string[]>();
  assignments.forEach((regionId, atomId) => {
    if (!grouped.has(regionId)) grouped.set(regionId, []);
    grouped.get(regionId)!.push(atomId);
  });
  const next = new Map<string, string>();
  for (const [regionId, atomIds] of grouped.entries()) {
    const components = connectedComponentsForAtomIds(atomIds, adjacencyLookup)
      .sort((a, b) => regionAreaFromAtoms(b, atomMap) - regionAreaFromAtoms(a, atomMap));
    components.forEach((component, index) => {
      const nextRegionId = index === 0 ? regionId : `${regionId}:${index + 1}`;
      for (const atomId of component) next.set(atomId, nextRegionId);
    });
  }
  return next;
}

function weightedCentroidForAtoms(atomIds: string[], atomMap: Map<string, TerrainAtom>): [number, number] {
  let sumLng = 0;
  let sumLat = 0;
  let total = 0;
  for (const atomId of atomIds) {
    const atom = atomMap.get(atomId);
    if (!atom) continue;
    const weight = Math.max(1e-6, atom.areaM2 * (0.35 + 0.65 * atom.meanConfidence));
    sumLng += atom.centroidLngLat[0] * weight;
    sumLat += atom.centroidLngLat[1] * weight;
    total += weight;
  }
  if (!(total > 0)) return [0, 0];
  return [sumLng / total, sumLat / total];
}

function atomPriorityScore(atom: TerrainAtom) {
  return atom.areaM2 *
    (0.3 + 0.7 * atom.meanConfidence) *
    (1 + atom.internalDispersionDeg / 18) *
    (0.7 + Math.min(1.2, atom.meanBreakStrength / 18));
}

function weightedBearingForAtoms(
  atomIds: string[],
  atomMap: Map<string, TerrainAtom>,
  fallbackBearingDeg: number,
) {
  return weightedAxialMeanDeg(
    atomIds.map((atomId) => {
      const atom = atomMap.get(atomId)!;
      return {
        angleDeg: atom.dominantBearingDeg ?? fallbackBearingDeg,
        weight: Math.max(1e-6, atom.areaM2 * (0.3 + 0.7 * atom.meanConfidence)),
      };
    }),
  ) ?? fallbackBearingDeg;
}

function buildGuidanceModes(
  graph: TerrainAtomGraph,
  maxCount: number,
  minSeparationDeg: number,
) {
  const binStepDeg = 15;
  const binCount = Math.round(180 / binStepDeg);
  const bins = Array.from({ length: binCount }, (_, index) => ({
    angleDeg: index * binStepDeg + binStepDeg * 0.5,
    weight: 0,
    sumLng: 0,
    sumLat: 0,
  }));
  for (const cell of graph.guidance.cells) {
    const angle = normalizedAxialBearing(cell.preferredBearingDeg);
    const binIndex = Math.max(0, Math.min(binCount - 1, Math.floor(angle / binStepDeg)));
    const weight = Math.max(1e-6, cell.areaWeightM2 * (0.25 + 0.75 * cell.confidence));
    bins[binIndex].weight += weight;
    bins[binIndex].sumLng += cell.lng * weight;
    bins[binIndex].sumLat += cell.lat * weight;
  }
  const sorted = bins.filter((bin) => bin.weight > 0).sort((a, b) => b.weight - a.weight);
  const selected: RegionSeedMode[] = [];
  for (const bin of sorted) {
    const tooClose = selected.some((mode) => axialAngleDeltaDeg(mode.angleDeg, bin.angleDeg) < minSeparationDeg);
    if (tooClose) continue;
    selected.push({
      angleDeg: bin.angleDeg,
      weight: bin.weight,
      centroidLngLat: [bin.sumLng / bin.weight, bin.sumLat / bin.weight],
    });
    if (selected.length >= maxCount) break;
  }
  return selected;
}

function buildPriorityAtoms(graph: TerrainAtomGraph) {
  return [...graph.atoms]
    .filter((atom) => Number.isFinite(atom.dominantBearingDeg))
    .sort((a, b) => atomPriorityScore(b) - atomPriorityScore(a));
}

function buildSeedSequence(
  graph: TerrainAtomGraph,
  atomMap: Map<string, TerrainAtom>,
  options: Required<TerrainAtomizationOptions>,
) {
  const totalAreaM2 = Math.max(1, graph.guidance.areaM2);
  const maxSeedCount = Math.min(8, options.maxHierarchyRegions, graph.atoms.length);
  const minSeedDistanceM = Math.max(graph.guidance.gridStepM * 1.5, Math.sqrt(totalAreaM2) * 0.12);
  const modes = buildGuidanceModes(graph, maxSeedCount, options.minModeSeparationDeg);
  const byPriority = buildPriorityAtoms(graph);
  const chosen: RegionSeed[] = [];
  const chosenAtoms = new Set<string>();

  for (const mode of modes) {
    const candidate = byPriority
      .filter((atom) => !chosenAtoms.has(atom.id))
      .filter((atom) => axialAngleDeltaDeg(atom.dominantBearingDeg!, mode.angleDeg) <= Math.max(24, options.minModeSeparationDeg + 6))
      .find((atom) => (
        chosen.every((seed) => distanceMetersBetweenLngLat(seed.centroidLngLat, atom.centroidLngLat) >= minSeedDistanceM)
      ))
      ?? byPriority
        .filter((atom) => !chosenAtoms.has(atom.id))
        .sort((a, b) => axialModeCost(a.dominantBearingDeg, mode.angleDeg) - axialModeCost(b.dominantBearingDeg, mode.angleDeg))[0];
    if (!candidate) continue;
    chosen.push({
      regionId: `seed-${chosen.length + 1}`,
      atomId: candidate.id,
      angleDeg: candidate.dominantBearingDeg ?? mode.angleDeg,
      centroidLngLat: candidate.centroidLngLat,
    });
    chosenAtoms.add(candidate.id);
    if (chosen.length >= maxSeedCount) break;
  }

  for (const atom of byPriority) {
    if (chosen.length >= maxSeedCount) break;
    if (chosenAtoms.has(atom.id)) continue;
    const farEnough = chosen.every((seed) => (
      distanceMetersBetweenLngLat(seed.centroidLngLat, atom.centroidLngLat) >= minSeedDistanceM ||
      axialAngleDeltaDeg(seed.angleDeg, atom.dominantBearingDeg!) >= options.minModeSeparationDeg
    ));
    if (!farEnough) continue;
    chosen.push({
      regionId: `seed-${chosen.length + 1}`,
      atomId: atom.id,
      angleDeg: atom.dominantBearingDeg ?? 0,
      centroidLngLat: atom.centroidLngLat,
    });
    chosenAtoms.add(atom.id);
  }

  return chosen;
}

function buildSpatialSeedSequence(
  graph: TerrainAtomGraph,
  options: Required<TerrainAtomizationOptions>,
) {
  const byPriority = buildPriorityAtoms(graph);
  const totalAreaM2 = Math.max(1, graph.guidance.areaM2);
  const maxSeedCount = Math.min(8, options.maxHierarchyRegions, graph.atoms.length);
  const polygonScaleM = Math.max(1, Math.sqrt(totalAreaM2));
  const chosen: RegionSeed[] = [];

  if (byPriority.length === 0) return chosen;

  const first = byPriority[0];
  chosen.push({
    regionId: "seed-1",
    atomId: first.id,
    angleDeg: first.dominantBearingDeg ?? 0,
    centroidLngLat: first.centroidLngLat,
  });

  while (chosen.length < maxSeedCount) {
    let best: TerrainAtom | null = null;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const atom of byPriority) {
      if (chosen.some((seed) => seed.atomId === atom.id)) continue;
      const minDistance = Math.min(
        ...chosen.map((seed) => distanceMetersBetweenLngLat(seed.centroidLngLat, atom.centroidLngLat)),
      );
      const minBearingDelta = Math.min(
        ...chosen.map((seed) => axialAngleDeltaDeg(seed.angleDeg, atom.dominantBearingDeg ?? seed.angleDeg)),
      );
      const score =
        Math.min(2.2, minDistance / Math.max(graph.guidance.gridStepM * 2, polygonScaleM * 0.18)) +
        0.9 * Math.min(1.8, minBearingDelta / Math.max(12, options.minModeSeparationDeg)) +
        0.45 * Math.min(1.5, atom.internalDispersionDeg / 20) +
        0.35 * Math.min(1.5, atom.meanBreakStrength / 18) +
        0.25 * Math.min(1.5, atomPriorityScore(atom) / Math.max(1, atomPriorityScore(byPriority[0])));
      if (score > bestScore) {
        bestScore = score;
        best = atom;
      }
    }
    if (!best) break;
    chosen.push({
      regionId: `seed-${chosen.length + 1}`,
      atomId: best.id,
      angleDeg: best.dominantBearingDeg ?? 0,
      centroidLngLat: best.centroidLngLat,
    });
  }

  return chosen;
}

function buildBreakHotspotSeedSequence(
  graph: TerrainAtomGraph,
  options: Required<TerrainAtomizationOptions>,
) {
  const maxSeedCount = Math.min(8, options.maxHierarchyRegions, graph.atoms.length);
  const totalAreaM2 = Math.max(1, graph.guidance.areaM2);
  const polygonScaleM = Math.max(1, Math.sqrt(totalAreaM2));
  const minSeedDistanceM = Math.max(graph.guidance.gridStepM * 1.25, polygonScaleM * 0.1);
  const ranked = [...graph.atoms]
    .filter((atom) => Number.isFinite(atom.dominantBearingDeg))
    .sort((a, b) => (
      (b.meanBreakStrength * (0.35 + 0.65 * b.meanConfidence) * (1 + b.internalDispersionDeg / 18) * Math.sqrt(Math.max(1, b.areaM2))) -
      (a.meanBreakStrength * (0.35 + 0.65 * a.meanConfidence) * (1 + a.internalDispersionDeg / 18) * Math.sqrt(Math.max(1, a.areaM2)))
    ));
  const chosen: RegionSeed[] = [];
  for (const atom of ranked) {
    if (chosen.length >= maxSeedCount) break;
    const farEnough = chosen.every((seed) => (
      distanceMetersBetweenLngLat(seed.centroidLngLat, atom.centroidLngLat) >= minSeedDistanceM ||
      axialAngleDeltaDeg(seed.angleDeg, atom.dominantBearingDeg ?? seed.angleDeg) >= options.minModeSeparationDeg * 0.75
    ));
    if (!farEnough) continue;
    chosen.push({
      regionId: `seed-${chosen.length + 1}`,
      atomId: atom.id,
      angleDeg: atom.dominantBearingDeg ?? 0,
      centroidLngLat: atom.centroidLngLat,
    });
  }
  return chosen;
}

function bestBearingForAtom(
  atomId: string,
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable,
) {
  const byBearing = bearingCostTable.get(atomId);
  let bestBearing = candidateBearings[0] ?? 0;
  let bestCost = Number.POSITIVE_INFINITY;
  if (!byBearing) return { bearingDeg: bestBearing, qualityCost: bestCost, holeRisk: 0, lowCoverageRisk: 0 };
  for (const bearingDeg of candidateBearings) {
    const cost = byBearing.get(bearingDeg);
    if (!cost) continue;
    if (cost.qualityCost < bestCost) {
      bestCost = cost.qualityCost;
      bestBearing = bearingDeg;
    }
  }
  const bestNode = byBearing.get(bestBearing);
  return {
    bearingDeg: bestBearing,
    qualityCost: bestNode?.qualityCost ?? bestCost,
    holeRisk: bestNode?.holeRisk ?? 0,
    lowCoverageRisk: bestNode?.lowCoverageRisk ?? 0,
  };
}

function buildLidarHotspotSeedSequence(
  graph: TerrainAtomGraph,
  atomMap: Map<string, TerrainAtom>,
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable,
  params: FlightParams,
  parentBearingDeg: number,
  options: Required<TerrainAtomizationOptions>,
) {
  if ((params.payloadKind ?? "camera") !== "lidar") return [] as RegionSeed[];

  const maxSeedCount = Math.min(8, options.maxHierarchyRegions, graph.atoms.length);
  const totalAreaM2 = Math.max(1, graph.guidance.areaM2);
  const polygonScaleM = Math.max(1, Math.sqrt(totalAreaM2));
  const minSeedDistanceM = Math.max(graph.guidance.gridStepM * 1.25, polygonScaleM * 0.11);
  const parentBearingBucket = nearestBearingBucket(parentBearingDeg, candidateBearings);

  const ranked = graph.atoms
    .map((atom) => {
      const current = bearingCostTable.get(atom.id)?.get(parentBearingBucket);
      const best = bestBearingForAtom(atom.id, candidateBearings, bearingCostTable);
      const improvement = current ? Math.max(0, current.qualityCost - best.qualityCost) : 0;
      const hotspotSeverity =
        (current?.holeRisk ?? 0) * 4.6 +
        (current?.lowCoverageRisk ?? 0) * 2.1 +
        improvement * 1.4 +
        Math.max(0, axialAngleDeltaDeg(parentBearingDeg, best.bearingDeg) - 12) / 26;
      const weight = Math.sqrt(Math.max(1, atom.areaM2)) * (0.35 + 0.65 * atom.meanConfidence);
      return {
        atom,
        bestBearingDeg: best.bearingDeg,
        hotspotSeverity,
        score: hotspotSeverity * weight,
      };
    })
    .filter((item) => item.hotspotSeverity > 0.2)
    .sort((a, b) => b.score - a.score);

  const chosen: RegionSeed[] = [];
  for (const item of ranked) {
    if (chosen.length >= maxSeedCount) break;
    const farEnough = chosen.every((seed) => (
      distanceMetersBetweenLngLat(seed.centroidLngLat, item.atom.centroidLngLat) >= minSeedDistanceM ||
      axialAngleDeltaDeg(seed.angleDeg, item.bestBearingDeg) >= Math.max(14, options.minModeSeparationDeg * 0.8)
    ));
    if (!farEnough) continue;
    chosen.push({
      regionId: `seed-${chosen.length + 1}`,
      atomId: item.atom.id,
      angleDeg: item.bestBearingDeg,
      centroidLngLat: item.atom.centroidLngLat,
    });
  }

  if (chosen.length < 2) return [];

  const baseSequence = buildSeedSequence(graph, atomMap, options);
  for (const seed of baseSequence) {
    if (chosen.length >= maxSeedCount) break;
    if (chosen.some((existing) => existing.atomId === seed.atomId)) continue;
    const farEnough = chosen.every((existing) => (
      distanceMetersBetweenLngLat(existing.centroidLngLat, seed.centroidLngLat) >= minSeedDistanceM ||
      axialAngleDeltaDeg(existing.angleDeg, seed.angleDeg) >= Math.max(12, options.minModeSeparationDeg * 0.7)
    ));
    if (!farEnough) continue;
    chosen.push({
      regionId: `seed-${chosen.length + 1}`,
      atomId: seed.atomId,
      angleDeg: seed.angleDeg,
      centroidLngLat: seed.centroidLngLat,
    });
  }

  return chosen;
}

function buildSeedVariants(
  graph: TerrainAtomGraph,
  atomMap: Map<string, TerrainAtom>,
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable,
  params: FlightParams,
  parentBearingDeg: number,
  options: Required<TerrainAtomizationOptions>,
) {
  const candidates = [
    buildSeedSequence(graph, atomMap, options),
    buildSpatialSeedSequence(graph, options),
    buildBreakHotspotSeedSequence(graph, options),
    buildLidarHotspotSeedSequence(graph, atomMap, candidateBearings, bearingCostTable, params, parentBearingDeg, options),
  ]
    .filter((variant) => variant.length >= 2);
  const unique = new Map<string, RegionSeed[]>();
  for (const variant of candidates) {
    const signature = variant.map((seed) => seed.atomId).join("|");
    if (!unique.has(signature)) unique.set(signature, variant);
  }
  return [...unique.values()].slice(0, MAX_SEED_VARIANTS);
}

function buildRegionStates(
  assignments: Map<string, string>,
  regionIds: string[],
  atomMap: Map<string, TerrainAtom>,
  seedAngles: Map<string, number>,
) {
  const grouped = new Map<string, string[]>();
  for (const regionId of regionIds) grouped.set(regionId, []);
  assignments.forEach((regionId, atomId) => {
    if (!grouped.has(regionId)) grouped.set(regionId, []);
    grouped.get(regionId)!.push(atomId);
  });
  const states = new Map<string, RegionState>();
  for (const regionId of regionIds) {
    const atomIds = grouped.get(regionId) ?? [];
    if (atomIds.length === 0) continue;
    const areaM2 = regionAreaFromAtoms(atomIds, atomMap);
    const centroidLngLat = weightedCentroidForAtoms(atomIds, atomMap);
    const fallback = seedAngles.get(regionId) ?? 0;
    const bearingDeg = weightedBearingForAtoms(atomIds, atomMap, fallback);
    states.set(regionId, { regionId, atomIds, areaM2, centroidLngLat, bearingDeg });
  }
  return states;
}

function nearestBearingBucket(bearingDeg: number, candidateBearings: number[]) {
  let best = candidateBearings[0] ?? normalizedAxialBearing(bearingDeg);
  let bestDelta = axialAngleDeltaDeg(best, bearingDeg);
  for (const candidate of candidateBearings) {
    const delta = axialAngleDeltaDeg(candidate, bearingDeg);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = candidate;
    }
  }
  return best;
}

function buildAtomBearingCostTable(
  graph: TerrainAtomGraph,
  atomMap: Map<string, TerrainAtom>,
  candidateBearings: number[],
  params: FlightParams,
) {
  const table: AtomBearingCostTable = new Map();
  for (const atom of graph.atoms) {
    const cells = atom.cellIndices
      .map((cellIndex) => graph.guidance.cells[cellIndex])
      .filter((cell): cell is TerrainGuidanceCell => cell != null);
    const byBearing = new Map<number, SensorNodeCost>();
    for (const bearingDeg of candidateBearings) {
      byBearing.set(bearingDeg, evaluateSensorNodeCostForCells(cells, bearingDeg, params));
    }
    table.set(atom.id, byBearing);
  }
  return table;
}

function computeEdgeSmoothnessCost(
  edge: Pick<TerrainAtomAdjacency, "meanBearingDeltaDeg" | "meanBreakBarrier">,
  leftBearingDeg: number,
  rightBearingDeg: number,
) {
  const atomContinuity = 1 - clamp(edge.meanBearingDeltaDeg / 55, 0, 1);
  const breakContinuity = 1 - clamp(edge.meanBreakBarrier / 18, 0, 1);
  const regionBearingDeltaDeg = axialAngleDeltaDeg(leftBearingDeg, rightBearingDeg);
  const regionCutDiscount = 1 - 0.55 * clamp(regionBearingDeltaDeg / 60, 0, 1);
  return clamp(
    (0.18 + 0.82 * (0.68 * atomContinuity + 0.32 * breakContinuity)) * regionCutDiscount,
    0.04,
    1.05,
  );
}

function computeCutEdgeCost(
  edge: Pick<TerrainAtomAdjacency, "sharedBoundaryM" | "meanBearingDeltaDeg" | "meanBreakBarrier">,
  leftBearingDeg: number,
  rightBearingDeg: number,
  polygonScaleM: number,
) {
  const normalizedBoundary = edge.sharedBoundaryM / Math.max(1, polygonScaleM);
  return normalizedBoundary * computeEdgeSmoothnessCost(edge, leftBearingDeg, rightBearingDeg);
}

function computeAtomNodeCost(
  atom: TerrainAtom,
  region: RegionState,
  polygonScaleM: number,
  totalAreaM2: number,
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable | null,
) {
  const bearingBucket = nearestBearingBucket(region.bearingDeg, candidateBearings);
  const sensorCost = bearingCostTable?.get(atom.id)?.get(bearingBucket) ?? null;
  const fitWeight = 0.42 + 0.38 * atom.meanConfidence + Math.min(0.35, atom.internalDispersionDeg / 70);
  const fitCost = sensorCost
    ? sensorCost.qualityCost + 0.15 * sensorCost.bearingPriorLoss
    : fitWeight * axialModeCost(atom.dominantBearingDeg, region.bearingDeg) / 24;
  const distanceCost = distanceMetersBetweenLngLat(atom.centroidLngLat, region.centroidLngLat) / Math.max(1, polygonScaleM) * 0.22;
  const balanceCost = Math.max(0, region.areaM2 / Math.max(1, totalAreaM2) - 0.45) * 0.16;
  return fitCost + distanceCost + balanceCost;
}

function computeLocalCutEnergy(
  atomId: string,
  targetRegionId: string,
  assignments: Map<string, string>,
  regionStates: Map<string, RegionState>,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  atomMap: Map<string, TerrainAtom>,
  polygonScaleM: number,
) {
  let penalty = 0;
  const targetBearing = regionStates.get(targetRegionId)?.bearingDeg
    ?? atomMap.get(atomId)?.dominantBearingDeg
    ?? 0;
  for (const edge of adjacencyLookup.get(atomId) ?? []) {
    const neighborId = edge.atomA === atomId ? edge.atomB : edge.atomA;
    const neighborRegionId = assignments.get(neighborId);
    if (!neighborRegionId) continue;
    if (neighborRegionId === targetRegionId) continue;
    const neighborBearing = regionStates.get(neighborRegionId)?.bearingDeg
      ?? atomMap.get(neighborId)?.dominantBearingDeg
      ?? targetBearing;
    penalty += computeCutEdgeCost(
      edge,
      targetBearing,
      neighborBearing,
      polygonScaleM,
    );
  }
  return penalty;
}

function atomAssignmentCost(
  atomId: string,
  targetRegionId: string,
  assignments: Map<string, string>,
  regionStates: Map<string, RegionState>,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  atomMap: Map<string, TerrainAtom>,
  polygonScaleM: number,
  totalAreaM2: number,
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable | null,
) {
  const atom = atomMap.get(atomId)!;
  const region = regionStates.get(targetRegionId)!;
  const nodeCost = computeAtomNodeCost(atom, region, polygonScaleM, totalAreaM2, candidateBearings, bearingCostTable);
  const cutEnergy = computeLocalCutEnergy(
    atomId,
    targetRegionId,
    assignments,
    regionStates,
    adjacencyLookup,
    atomMap,
    polygonScaleM,
  );
  return nodeCost + cutEnergy;
}

function pickBestNeighborRegion(
  fragmentAtomIds: string[],
  currentRegionId: string,
  assignments: Map<string, string>,
  regionStates: Map<string, RegionState>,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  atomMap: Map<string, TerrainAtom>,
  polygonScaleM: number,
  totalAreaM2: number,
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable | null,
) {
  const neighborCandidates = new Set<string>();
  for (const atomId of fragmentAtomIds) {
    for (const edge of adjacencyLookup.get(atomId) ?? []) {
      const neighborId = edge.atomA === atomId ? edge.atomB : edge.atomA;
      const neighborRegionId = assignments.get(neighborId);
      if (neighborRegionId && neighborRegionId !== currentRegionId) neighborCandidates.add(neighborRegionId);
    }
  }
  let best: { regionId: string; cost: number } | null = null;
  for (const regionId of neighborCandidates) {
    const cost = mean(fragmentAtomIds.map((atomId) => (
      atomAssignmentCost(
        atomId,
        regionId,
        assignments,
        regionStates,
        adjacencyLookup,
        atomMap,
        polygonScaleM,
        totalAreaM2,
        candidateBearings,
        bearingCostTable,
      )
    )));
    if (!best || cost < best.cost) best = { regionId, cost };
  }
  return best?.regionId ?? null;
}

function repairDisconnectedAssignments(
  assignments: Map<string, string>,
  regionIds: string[],
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  atomMap: Map<string, TerrainAtom>,
  seedAngles: Map<string, number>,
  polygonScaleM: number,
  totalAreaM2: number,
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable | null,
) {
  const next = new Map(assignments);
  for (let iteration = 0; iteration < 4; iteration++) {
    let changed = false;
    const regionStates = buildRegionStates(next, regionIds, atomMap, seedAngles);
    for (const regionId of regionIds) {
      const atomIds = regionStates.get(regionId)?.atomIds ?? [];
      if (atomIds.length <= 1) continue;
      const components = connectedComponentsForAtomIds(atomIds, adjacencyLookup);
      if (components.length <= 1) continue;
      components.sort((a, b) => regionAreaFromAtoms(b, atomMap) - regionAreaFromAtoms(a, atomMap));
      for (const fragment of components.slice(1)) {
        const targetRegionId = pickBestNeighborRegion(
          fragment,
          regionId,
          next,
          regionStates,
          adjacencyLookup,
          atomMap,
          polygonScaleM,
          totalAreaM2,
          candidateBearings,
          bearingCostTable,
        );
        if (!targetRegionId) continue;
        for (const atomId of fragment) next.set(atomId, targetRegionId);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return next;
}

function mergeTinyRegions(
  assignments: Map<string, string>,
  regionIds: string[],
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  atomMap: Map<string, TerrainAtom>,
  seedAngles: Map<string, number>,
  polygonScaleM: number,
  totalAreaM2: number,
  minAreaM2: number,
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable | null,
) {
  const next = new Map(assignments);
  for (let iteration = 0; iteration < 4; iteration++) {
    let changed = false;
    const regionStates = buildRegionStates(next, regionIds, atomMap, seedAngles);
    const currentIds = [...new Set(regionIds.filter((regionId) => (regionStates.get(regionId)?.atomIds.length ?? 0) > 0))];
    const totalArea = Math.max(1, totalAreaM2);
    for (const regionId of currentIds) {
      const state = regionStates.get(regionId);
      if (!state) continue;
      const areaFraction = state.areaM2 / totalArea;
      if (state.areaM2 >= minAreaM2 && areaFraction >= 0.06) continue;
      if (currentIds.length <= 2) continue;
      const targetRegionId = pickBestNeighborRegion(
        state.atomIds,
        regionId,
        next,
        regionStates,
        adjacencyLookup,
        atomMap,
        polygonScaleM,
        totalAreaM2,
        candidateBearings,
        bearingCostTable,
      );
      if (!targetRegionId) continue;
      for (const atomId of state.atomIds) next.set(atomId, targetRegionId);
      changed = true;
    }
    if (!changed) break;
  }
  return next;
}

function buildRegionFromAtomIds(
  regionId: string,
  atomIds: string[],
  atomMap: Map<string, TerrainAtom>,
  guidance: TerrainGuidanceField,
  tiles: TerrainTile[],
  params: FlightParams,
  candidateBearings: number[],
  options: TerrainAtomizationOptions,
  depth: number,
): InternalRegion | null {
  const sortedIds = [...new Set(atomIds)].sort();
  const ring = buildRegionRingFromCells(sortedIds, atomMap, guidance) ?? mergeAtomRings(sortedIds, atomMap);
  if (!ring) return null;
  const objective = bestObjectiveForRing(ring, tiles, params, candidateBearings, options);
  if (!objective) return null;
  return {
    regionId,
    atomIds: sortedIds,
    ring,
    objective,
    convexity: objective.regularization.convexity,
    compactness: objective.regularization.compactness,
    centroidLngLat: weightedCentroidForAtoms(sortedIds, atomMap),
    depth,
  };
}

function buildRegionAdjacency(
  assignments: Map<string, string>,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
) {
  const stats = new Map<string, RegionAdjacencyStats>();
  for (const [atomId, regionId] of assignments.entries()) {
    for (const edge of adjacencyLookup.get(atomId) ?? []) {
      const neighborId = edge.atomA === atomId ? edge.atomB : edge.atomA;
      const neighborRegionId = assignments.get(neighborId);
      if (!neighborRegionId || neighborRegionId === regionId) continue;
      const [left, right] = regionId < neighborRegionId ? [regionId, neighborRegionId] : [neighborRegionId, regionId];
      const key = `${left}|${right}|${edge.atomA < edge.atomB ? edge.atomA : edge.atomB}|${edge.atomA < edge.atomB ? edge.atomB : edge.atomA}`;
      if (stats.has(key)) continue;
      stats.set(key, {
        regionA: left,
        regionB: right,
        sharedBoundaryM: edge.sharedBoundaryM,
        meanBreakBarrier: edge.meanBreakBarrier,
        meanBearingDeltaDeg: edge.meanBearingDeltaDeg,
      });
    }
  }
  const aggregated = new Map<string, RegionAdjacencyStats>();
  for (const edge of stats.values()) {
    const key = `${edge.regionA}|${edge.regionB}`;
    const existing = aggregated.get(key) ?? {
      regionA: edge.regionA,
      regionB: edge.regionB,
      sharedBoundaryM: 0,
      meanBreakBarrier: 0,
      meanBearingDeltaDeg: 0,
    };
    const nextBoundary = existing.sharedBoundaryM + edge.sharedBoundaryM;
    existing.meanBreakBarrier =
      nextBoundary > 0
        ? ((existing.meanBreakBarrier * existing.sharedBoundaryM) + edge.meanBreakBarrier * edge.sharedBoundaryM) / nextBoundary
        : 0;
    existing.meanBearingDeltaDeg =
      nextBoundary > 0
        ? ((existing.meanBearingDeltaDeg * existing.sharedBoundaryM) + edge.meanBearingDeltaDeg * edge.sharedBoundaryM) / nextBoundary
        : 0;
    existing.sharedBoundaryM = nextBoundary;
    aggregated.set(key, existing);
  }
  return [...aggregated.values()];
}

function evaluatePartitionRegions(
  regions: InternalRegion[],
  assignments: Map<string, string>,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  options: Required<TerrainAtomizationOptions>,
) {
  const partition = combinePartitionObjectives(regions.map((region) => region.objective), options);
  const totalAreaM2 = Math.max(1, regions.reduce((sum, region) => sum + region.objective.regularization.areaM2, 0));
  const regionAdjacency = buildRegionAdjacency(assignments, adjacencyLookup);
  const regionBearingMap = new Map(regions.map((region) => [region.regionId, region.objective.bearingDeg]));
  const internalBoundaryM = regionAdjacency.reduce((sum, edge) => sum + edge.sharedBoundaryM, 0);
  const weightedBreakBarrier = regionAdjacency.reduce((sum, edge) => sum + edge.meanBreakBarrier * edge.sharedBoundaryM, 0);
  const boundaryBreakAlignment = internalBoundaryM > 0
    ? clamp((weightedBreakBarrier / internalBoundaryM) / 24, 0, 1)
    : 0;
  const largestRegionFraction = Math.max(
    0,
    ...regions.map((region) => region.objective.regularization.areaM2 / totalAreaM2),
  );
  const meanConvexity = regions.reduce(
    (sum, region) => sum + region.convexity * (region.objective.regularization.areaM2 / totalAreaM2),
    0,
  );
  const shapePenalty = regions.reduce((sum, region) => {
    const areaWeight = region.objective.regularization.areaM2 / totalAreaM2;
    const convexityPenalty = Math.max(0, 0.76 - region.convexity) * 8;
    const compactnessPenalty = Math.max(0, region.compactness - 2.8) * 0.65;
    const neckPenalty = region.convexity < 0.78 && region.compactness > 4.2
      ? (4.2 - Math.max(0, 0.92 - region.convexity) * 2) * Math.min(2.2, (region.compactness - 4.2) * 0.8)
      : 0;
    const widthPenalty = region.objective.regularization.widthPenalty * 1.6;
    const fragmentationPenalty = region.objective.regularization.fragmentedLinePenalty * 1.2;
    const interSegmentGapPenalty = region.objective.regularization.interSegmentGapPenalty * 1.1;
    const overflightPenalty = region.objective.regularization.overflightTransitPenalty * 1.15;
    const shortLinePenalty = region.objective.flightTime.shortLineFraction * 0.35;
    return sum + areaWeight * (
      convexityPenalty +
      compactnessPenalty +
      neckPenalty +
      widthPenalty +
      fragmentationPenalty +
      interSegmentGapPenalty +
      overflightPenalty +
      shortLinePenalty
    );
  }, 0);
  const polygonScaleM = Math.max(1, Math.sqrt(totalAreaM2));
  const boundaryPenalty = regionAdjacency.reduce((sum, edge) => {
    const leftBearing = regionBearingMap.get(edge.regionA);
    const rightBearing = regionBearingMap.get(edge.regionB);
    if (!Number.isFinite(leftBearing) || !Number.isFinite(rightBearing)) return sum;
    return sum + 0.42 * computeCutEdgeCost(edge, leftBearing!, rightBearing!, polygonScaleM);
  }, 0);
  const regionCountPenalty = Math.max(0, regions.length - 1) * 0.03;
  const dominancePenalty = Math.max(0, largestRegionFraction - 0.7) * 2.4;
  const totalScore = partition.totalCost + boundaryPenalty + shapePenalty + regionCountPenalty + dominancePenalty;
  return {
    partition,
    totalScore,
    largestRegionFraction,
    meanConvexity,
    boundaryBreakAlignment,
  };
}

function isPracticalPartitionEvaluation(
  evaluation: PartitionEvaluation,
  minAreaM2: number,
) {
  if (evaluation.regions.length <= 1) return false;
  return evaluation.regions.every((region) => (
    region.objective.regularization.areaM2 >= minAreaM2 &&
    region.convexity >= 0.7 &&
    region.compactness <= 5.25 &&
    !(region.convexity < 0.74 && region.compactness > 4.25) &&
    !region.objective.regularization.isHardInvalid
  ));
}

function buildPartitionEvaluationFromAssignments(
  parentRing: Ring,
  assignments: Map<string, string>,
  atomMap: Map<string, TerrainAtom>,
  guidance: TerrainGuidanceField,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  tiles: TerrainTile[],
  params: FlightParams,
  candidateBearings: number[],
  options: Required<TerrainAtomizationOptions>,
  depth = 0,
) {
  const grouped = new Map<string, string[]>();
  assignments.forEach((regionId, atomId) => {
    if (!grouped.has(regionId)) grouped.set(regionId, []);
    grouped.get(regionId)!.push(atomId);
  });
  const regions = [...grouped.entries()]
    .map(([regionId, atomIds]) => buildRegionFromAtomIds(regionId, atomIds, atomMap, guidance, tiles, params, candidateBearings, options, depth))
    .filter((region): region is InternalRegion => region !== null);
  if (regions.length !== grouped.size || regions.length === 0) return null;
  const completedRegions = completeRegionCoverage(parentRing, regions, tiles, params, candidateBearings, options);
  if (!completedRegions || completedRegions.length !== regions.length) return null;
  const metrics = evaluatePartitionRegions(completedRegions, assignments, adjacencyLookup, options);
  return {
    regions: completedRegions,
    assignments,
    partition: metrics.partition,
    totalScore: metrics.totalScore,
    largestRegionFraction: metrics.largestRegionFraction,
    meanConvexity: metrics.meanConvexity,
    boundaryBreakAlignment: metrics.boundaryBreakAlignment,
  } satisfies PartitionEvaluation;
}

function initializeAssignments(
  seedAtoms: RegionSeed[],
  graph: TerrainAtomGraph,
  _atomMap: Map<string, TerrainAtom>,
) {
  const polygonScaleM = Math.max(1, Math.sqrt(Math.max(1, graph.guidance.areaM2)));
  const assignments = new Map<string, string>();
  for (const seed of seedAtoms) assignments.set(seed.atomId, seed.regionId);
  for (const atom of graph.atoms) {
    if (assignments.has(atom.id)) continue;
    let bestRegionId = seedAtoms[0]?.regionId;
    let bestCost = Number.POSITIVE_INFINITY;
    for (const seed of seedAtoms) {
      const fitCost = axialModeCost(atom.dominantBearingDeg, seed.angleDeg) / 22;
      const distanceCost = distanceMetersBetweenLngLat(atom.centroidLngLat, seed.centroidLngLat) / polygonScaleM * 0.28;
      const score = fitCost + distanceCost;
      if (score < bestCost) {
        bestCost = score;
        bestRegionId = seed.regionId;
      }
    }
    if (bestRegionId) assignments.set(atom.id, bestRegionId);
  }
  return assignments;
}

function segmentAtomsIntoRegions(
  graph: TerrainAtomGraph,
  atomMap: Map<string, TerrainAtom>,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  seedAtoms: RegionSeed[],
  candidateBearings: number[],
  bearingCostTable: AtomBearingCostTable | null,
  options: Required<TerrainAtomizationOptions>,
) {
  const regionIds = seedAtoms.map((seed) => seed.regionId);
  const seedAngles = new Map(seedAtoms.map((seed) => [seed.regionId, seed.angleDeg]));
  const lockedSeedAtoms = new Map(seedAtoms.map((seed) => [seed.atomId, seed.regionId]));
  const totalAreaM2 = Math.max(1, graph.guidance.areaM2);
  const polygonScaleM = Math.max(1, Math.sqrt(totalAreaM2));
  let assignments = initializeAssignments(seedAtoms, graph, atomMap);

  for (let iteration = 0; iteration < MAX_SEGMENTATION_ITERATIONS; iteration++) {
    let changed = false;
    let regionStates = buildRegionStates(assignments, regionIds, atomMap, seedAngles);
    for (const atom of graph.atoms) {
      const lockedRegionId = lockedSeedAtoms.get(atom.id);
      if (lockedRegionId) {
        assignments.set(atom.id, lockedRegionId);
        continue;
      }
      const currentRegionId = assignments.get(atom.id);
      if (!currentRegionId) continue;
      if ((regionStates.get(currentRegionId)?.atomIds.length ?? 0) <= 1) continue;
      let bestRegionId = currentRegionId;
      let bestCost = atomAssignmentCost(
        atom.id,
        currentRegionId,
        assignments,
        regionStates,
        adjacencyLookup,
        atomMap,
        polygonScaleM,
        totalAreaM2,
        candidateBearings,
        bearingCostTable,
      );
      for (const regionId of regionIds) {
        if (regionId === currentRegionId || !regionStates.has(regionId)) continue;
        const score = atomAssignmentCost(
          atom.id,
          regionId,
          assignments,
          regionStates,
          adjacencyLookup,
          atomMap,
          polygonScaleM,
          totalAreaM2,
          candidateBearings,
          bearingCostTable,
        );
        if (score + 1e-6 < bestCost) {
          bestCost = score;
          bestRegionId = regionId;
        }
      }
      if (bestRegionId !== currentRegionId) {
        assignments.set(atom.id, bestRegionId);
        changed = true;
      }
    }

    assignments = repairDisconnectedAssignments(
      assignments,
      regionIds,
      adjacencyLookup,
      atomMap,
      seedAngles,
      polygonScaleM,
      totalAreaM2,
      candidateBearings,
      bearingCostTable,
    );
    assignments = mergeTinyRegions(
      assignments,
      regionIds,
      adjacencyLookup,
      atomMap,
      seedAngles,
      polygonScaleM,
      totalAreaM2,
      options.minAreaM2,
      candidateBearings,
      bearingCostTable,
    );
    regionStates = buildRegionStates(assignments, regionIds, atomMap, seedAngles);
    if (regionStates.size < 2) return null;
    if (!changed) break;
  }

  const canonical = canonicalizeAssignmentsByConnectivity(assignments, adjacencyLookup, atomMap);
  return canonical;
}

function buildFineSegmentation(
  ring: Ring,
  graph: TerrainAtomGraph,
  tiles: TerrainTile[],
  params: FlightParams,
  candidateBearings: number[],
  options: Required<TerrainAtomizationOptions>,
): FineSegmentationResult | null {
  const atomMap = buildAtomMap(graph);
  const adjacencyLookup = buildAdjacencyLookup(graph.adjacency);
  const bearingCostTable = buildAtomBearingCostTable(graph, atomMap, candidateBearings, params);
  const parentObjective = bestObjectiveForRing(ring, tiles, params, candidateBearings, options);
  const parentBearingDeg =
    parentObjective?.bearingDeg ??
    graph.guidance.dominantPreferredBearingDeg ??
    candidateBearings[0] ??
    0;
  const seedVariants = buildSeedVariants(
    graph,
    atomMap,
    candidateBearings,
    bearingCostTable,
    params,
    parentBearingDeg,
    options,
  );
  if (seedVariants.length === 0) return null;

  const candidates: PartitionEvaluation[] = [];
  const scoringOptions = { ...options, tradeoff: Math.max(options.tradeoff, QUALITY_HEAVY_TRADEOFF) };
  const maxK = Math.min(
    Math.max(...seedVariants.map((variant) => variant.length)),
    8,
    graph.atoms.length,
  );

  for (let k = 2; k <= maxK; k++) {
    for (const seedVariant of seedVariants) {
      if (seedVariant.length < k) continue;
      const seeds = seedVariant.slice(0, k);
      const assignments = segmentAtomsIntoRegions(
        graph,
        atomMap,
        adjacencyLookup,
        seeds,
        candidateBearings,
        bearingCostTable,
        scoringOptions,
      );
      if (!assignments) continue;
      const evaluation = buildPartitionEvaluationFromAssignments(
        ring,
        assignments,
        atomMap,
        graph.guidance,
        adjacencyLookup,
        tiles,
        params,
        candidateBearings,
        scoringOptions,
        0,
      );
      if (!evaluation) continue;
      if (evaluation.largestRegionFraction > 0.965) continue;
      candidates.push(evaluation);
    }
  }

  if (candidates.length === 0) return null;
  const deduped = new Map<string, PartitionEvaluation>();
  for (const candidate of candidates) {
    const signature = buildSolutionSignature(candidate.regions);
    const existing = deduped.get(signature);
    if (!existing || candidate.totalScore < existing.totalScore) deduped.set(signature, candidate);
  }
  const candidatePool = [...deduped.values()];
  const bestByRegionCount = new Map<number, PartitionEvaluation>();
  for (const candidate of candidatePool) {
    if (!isPracticalPartitionEvaluation(candidate, options.minAreaM2)) continue;
    if (candidate.largestRegionFraction >= 0.92) continue;
    const count = candidate.regions.length;
    const existing = bestByRegionCount.get(count);
    if (!existing) {
      bestByRegionCount.set(count, candidate);
      continue;
    }
    const candidateBetter =
      candidate.largestRegionFraction < existing.largestRegionFraction - 1e-9 ||
      (
        Math.abs(candidate.largestRegionFraction - existing.largestRegionFraction) <= 1e-9 &&
        (
          candidate.partition.normalizedQualityCost < existing.partition.normalizedQualityCost - 1e-9 ||
          (
            Math.abs(candidate.partition.normalizedQualityCost - existing.partition.normalizedQualityCost) <= 1e-9 &&
            candidate.totalScore < existing.totalScore
          )
        )
      );
    if (candidateBetter) bestByRegionCount.set(count, candidate);
  }
  const practical = candidatePool.filter((candidate) => (
    isPracticalPartitionEvaluation(candidate, options.minAreaM2) &&
    candidate.largestRegionFraction < 0.88 &&
    (tokenSplitSatisfied(candidate) || candidate.regions.length >= 3)
  ));
  const ranked = (practical.length > 0 ? practical : candidatePool).sort((a, b) => {
    if (a.partition.normalizedQualityCost !== b.partition.normalizedQualityCost) {
      return a.partition.normalizedQualityCost - b.partition.normalizedQualityCost;
    }
    if (a.largestRegionFraction !== b.largestRegionFraction) {
      return a.largestRegionFraction - b.largestRegionFraction;
    }
    if (b.regions.length !== a.regions.length) return b.regions.length - a.regions.length;
    return a.totalScore - b.totalScore;
  });
  const selected = ranked[0] ?? null;
  return selected ? { selected, bestByRegionCount } : null;
}

function tokenSplitSatisfied(solution: PartitionEvaluation) {
  const totalArea = Math.max(1, solution.regions.reduce((sum, region) => sum + region.objective.regularization.areaM2, 0));
  const largest = Math.max(...solution.regions.map((region) => region.objective.regularization.areaM2));
  return (totalArea - largest) / totalArea >= 0.25;
}

function buildHierarchyFromFineSegmentation(
  parentRing: Ring,
  fineResult: FineSegmentationResult,
  guidance: TerrainGuidanceField,
  atomMap: Map<string, TerrainAtom>,
  adjacencyLookup: Map<string, TerrainAtomAdjacency[]>,
  tiles: TerrainTile[],
  params: FlightParams,
  candidateBearings: number[],
  options: Required<TerrainAtomizationOptions>,
) {
  const hierarchy: PartitionEvaluation[] = [fineResult.selected];
  let current = fineResult.selected;

  while (current.regions.length > 1) {
    const regionAdjacency = buildRegionAdjacency(current.assignments, adjacencyLookup);
    if (regionAdjacency.length === 0) break;
    let bestMerge: PartitionEvaluation | null = null;
    let bestDelta = Number.POSITIVE_INFINITY;

    for (const edge of regionAdjacency) {
      const nextAssignments = new Map(current.assignments);
      const mergedRegionId = `${edge.regionA}+${edge.regionB}`;
      nextAssignments.forEach((regionId, atomId) => {
        if (regionId === edge.regionA || regionId === edge.regionB) nextAssignments.set(atomId, mergedRegionId);
      });
      const evaluation = buildPartitionEvaluationFromAssignments(
        parentRing,
        nextAssignments,
        atomMap,
        guidance,
        adjacencyLookup,
        tiles,
        params,
        candidateBearings,
        options,
        0,
      );
      if (!evaluation) continue;
      const delta = evaluation.totalScore - current.totalScore;
      if (delta < bestDelta - 1e-9) {
        bestDelta = delta;
        bestMerge = evaluation;
      }
    }

    if (!bestMerge) break;
    hierarchy.push(bestMerge);
    current = bestMerge;
  }

  const ordered = [...hierarchy].reverse();
  let firstPracticalIndex = ordered.findIndex((solution) => (
    solution.regions.length > 1 &&
    solution.largestRegionFraction < 0.7 &&
    tokenSplitSatisfied(solution) &&
    isPracticalPartitionEvaluation(solution, options.minAreaM2)
  ));
  if (firstPracticalIndex < 0) {
    firstPracticalIndex = ordered.findIndex((solution) => (
      solution.regions.length > 1 &&
      solution.largestRegionFraction < 0.82
    ));
  }
  const visibleHierarchy = firstPracticalIndex < 0
    ? [] as PartitionEvaluation[]
    : ordered.slice(firstPracticalIndex).filter((solution) => solution.regions.length > 1);
  const firstPracticalSignature = visibleHierarchy.length > 0
    ? buildSolutionSignature(visibleHierarchy[0].regions)
    : null;

  const byRegionCount = new Map<number, PartitionEvaluation>();
  const consider = (solution: PartitionEvaluation) => {
    if (solution.regions.length <= 1) return;
    const count = solution.regions.length;
    const existing = byRegionCount.get(count);
    if (!existing) {
      byRegionCount.set(count, solution);
      return;
    }
    const solutionBetter =
      solution.largestRegionFraction < existing.largestRegionFraction - 1e-9 ||
      (
        Math.abs(solution.largestRegionFraction - existing.largestRegionFraction) <= 1e-9 &&
        (
          solution.partition.normalizedQualityCost < existing.partition.normalizedQualityCost - 1e-9 ||
          (
            Math.abs(solution.partition.normalizedQualityCost - existing.partition.normalizedQualityCost) <= 1e-9 &&
            solution.totalScore < existing.totalScore
          )
        )
      );
    if (solutionBetter) byRegionCount.set(count, solution);
  };

  visibleHierarchy.forEach(consider);
  fineResult.bestByRegionCount.forEach((solution, count) => {
    if (count === 2 || !byRegionCount.has(count)) consider(solution);
  });

  const visible = [...byRegionCount.values()].sort((a, b) => {
    if (a.regions.length !== b.regions.length) return a.regions.length - b.regions.length;
    if (a.largestRegionFraction !== b.largestRegionFraction) return a.largestRegionFraction - b.largestRegionFraction;
    if (a.partition.normalizedQualityCost !== b.partition.normalizedQualityCost) {
      return a.partition.normalizedQualityCost - b.partition.normalizedQualityCost;
    }
    return a.totalScore - b.totalScore;
  });
  if (visible.length === 0) return [];
  return visible.map((solution, index) => buildFrontierSolution(
    solution,
    index,
    firstPracticalSignature != null && buildSolutionSignature(solution.regions) === firstPracticalSignature,
    visible.length <= 1 ? 0.5 : index / (visible.length - 1),
  ));
}

function buildHeuristicFallbackSolutions(
  ring: Ring,
  graph: TerrainAtomGraph,
  tiles: TerrainTile[],
  params: FlightParams,
  candidateBearings: number[],
  options: Required<TerrainAtomizationOptions>,
) {
  const atomMap = buildAtomMap(graph);
  const adjacencyLookup = buildAdjacencyLookup(graph.adjacency);
  const fallbackCandidates: PartitionEvaluation[] = [];
  for (let maxPolygons = 2; maxPolygons <= Math.min(options.maxHierarchyRegions, 6); maxPolygons++) {
    const heuristic = partitionPolygonByTerrainFaces(ring, tiles, params, {
      forceAtLeastOneSplit: true,
      maxPolygons,
      candidateAngleStepDeg: 10,
      candidateOffsetFractions: [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45],
    });
    if (heuristic.polygons.length <= 1) continue;

    const assignments = new Map<string, string>();
    const regionCenters = heuristic.polygons.map((poly) => {
      const feature = turf.polygon([poly]);
      const centroid = turf.centroid(feature);
      return {
        feature,
        centroid: centroid.geometry.coordinates as [number, number],
      };
    });

    for (const atom of graph.atoms) {
      let regionIndex = regionCenters.findIndex((region) => turf.booleanPointInPolygon(turf.point(atom.centroidLngLat), region.feature));
      if (regionIndex < 0) {
        let best = Number.POSITIVE_INFINITY;
        for (let index = 0; index < regionCenters.length; index++) {
          const distance = distanceMetersBetweenLngLat(atom.centroidLngLat, regionCenters[index].centroid);
          if (distance < best) {
            best = distance;
            regionIndex = index;
          }
        }
      }
      assignments.set(atom.id, `fallback-${regionIndex + 1}`);
    }

    const evaluation = buildPartitionEvaluationFromAssignments(
      ring,
      assignments,
      atomMap,
      graph.guidance,
      adjacencyLookup,
      tiles,
      params,
      candidateBearings,
      options,
      0,
    );
    if (!evaluation || evaluation.regions.length <= 1) continue;
    const practicalEnough = (
      evaluation.largestRegionFraction < 0.86 &&
      (tokenSplitSatisfied(evaluation) || evaluation.regions.length >= 3) &&
      isPracticalPartitionEvaluation(evaluation, options.minAreaM2)
    );
    if (!practicalEnough) continue;
    fallbackCandidates.push(evaluation);
  }
  const deduped = new Map<string, PartitionEvaluation>();
  for (const candidate of fallbackCandidates) {
    const signature = buildSolutionSignature(candidate.regions);
    const existing = deduped.get(signature);
    if (!existing || candidate.totalScore < existing.totalScore) deduped.set(signature, candidate);
  }
  const ranked = [...deduped.values()].sort((a, b) => {
    if (a.partition.normalizedQualityCost !== b.partition.normalizedQualityCost) {
      return a.partition.normalizedQualityCost - b.partition.normalizedQualityCost;
    }
    if (a.largestRegionFraction !== b.largestRegionFraction) {
      return a.largestRegionFraction - b.largestRegionFraction;
    }
    if (b.regions.length !== a.regions.length) return b.regions.length - a.regions.length;
    return a.totalScore - b.totalScore;
  });
  return ranked.map((evaluation, index, visible) => buildFrontierSolution(
    evaluation,
    index,
    index === 0,
    visible.length <= 1 ? 0.5 : index / (visible.length - 1),
  ));
}

function buildRefinedOptionsForRetry(
  ring: Ring,
  options: Required<TerrainAtomizationOptions>,
): Required<TerrainAtomizationOptions> {
  const areaM2 = ringAreaM2(ring);
  const refinedGridStepM = options.gridStepM > 0
    ? Math.max(40, options.gridStepM * 0.72)
    : clamp(Math.sqrt(Math.max(areaM2, 1)) / 24, 40, 120);
  const refinedSearchSampleStepM = options.searchSampleStepM > 0
    ? Math.max(25, options.searchSampleStepM * 0.8)
    : clamp(refinedGridStepM * 0.72, 25, 90);
  return {
    ...options,
    gridStepM: refinedGridStepM,
    searchSampleStepM: refinedSearchSampleStepM,
    maxInitialAtoms: Math.max(options.maxInitialAtoms, Math.min(72, options.maxInitialAtoms * 2)),
    atomDirectionMergeDeg: Math.max(10, options.atomDirectionMergeDeg - 2),
    atomBreakThreshold: Math.max(5, options.atomBreakThreshold - 1),
    minAtomCells: Math.max(1, options.minAtomCells - 1),
  };
}

function runPartitionFrontierAttempt(
  ring: Ring,
  tiles: TerrainTile[],
  params: FlightParams,
  options: Required<TerrainAtomizationOptions>,
) {
  const graph = buildTerrainAtomGraph(ring, tiles, options);
  if (graph.atoms.length < 2) return { graph, solutions: [] as FrontierSolution[] };

  const candidateBearings = buildCandidateBearings(graph, options.candidateBearingStepDeg);
  const fine = buildFineSegmentation(ring, graph, tiles, params, candidateBearings, options);
  if (!fine) {
    return {
      graph,
      solutions: buildHeuristicFallbackSolutions(ring, graph, tiles, params, candidateBearings, options),
    };
  }

  const atomMap = buildAtomMap(graph);
  const adjacencyLookup = buildAdjacencyLookup(graph.adjacency);
  const solutions = buildHierarchyFromFineSegmentation(
    ring,
    fine,
    graph.guidance,
    atomMap,
    adjacencyLookup,
    tiles,
    params,
    candidateBearings,
    options,
  );
  if (solutions.length > 0) return { graph, solutions };
  return {
    graph,
    solutions: buildHeuristicFallbackSolutions(ring, graph, tiles, params, candidateBearings, options),
  };
}

export function buildPartitionFrontier(
  ring: Ring,
  tiles: TerrainTile[],
  params: FlightParams,
  options: TerrainAtomizationOptions = {},
): { graph: TerrainAtomGraph; solutions: FrontierSolution[] } {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const initialAttempt = runPartitionFrontierAttempt(ring, tiles, params, opts);
  const initialSolutions = filterViableCoverageSolutions(ring, initialAttempt.solutions);
  const needsRetry =
    initialSolutions.length === 0 ||
    initialSolutions.every((solution) => solution.largestRegionFraction > 0.9 || solution.partition.regionCount <= 1);
  if (!needsRetry) {
    return { graph: initialAttempt.graph, solutions: initialSolutions };
  }

  const refinedOpts = buildRefinedOptionsForRetry(ring, opts);
  const refinedAttempt = runPartitionFrontierAttempt(ring, tiles, params, refinedOpts);
  const refinedSolutions = filterViableCoverageSolutions(ring, refinedAttempt.solutions);
  if (refinedSolutions.length > 0) {
    return { graph: refinedAttempt.graph, solutions: refinedSolutions };
  }
  return { graph: initialAttempt.graph, solutions: initialSolutions };
}
