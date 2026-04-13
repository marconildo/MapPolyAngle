import { getPolygonBounds, haversineDistance } from "./geometry";

const EARTH_RADIUS_M = 6378137;
const INTERSECTION_EPSILON_M = 1e-6;

type LocalPoint = {
  north: number;
  east: number;
};

type SweepPoint = {
  along: number;
  cross: number;
};

function ensureClosedRing(ring: number[][]) {
  if (ring.length < 2) return ring;
  const first = ring[0];
  const last = ring[ring.length - 1];
  if (first[0] === last[0] && first[1] === last[1]) return ring;
  return [...ring, first];
}

function toLocalPoint(
  point: [number, number],
  reference: [number, number],
): LocalPoint {
  const dLat = ((point[1] - reference[1]) * Math.PI) / 180;
  const dLng = ((point[0] - reference[0]) * Math.PI) / 180;
  const cosLat = Math.cos((reference[1] * Math.PI) / 180);
  return {
    north: dLat * EARTH_RADIUS_M,
    east: dLng * EARTH_RADIUS_M * cosLat,
  };
}

function fromLocalPoint(
  point: LocalPoint,
  reference: [number, number],
): [number, number] {
  const lat = reference[1] + (point.north / EARTH_RADIUS_M) * (180 / Math.PI);
  const lng =
    reference[0] +
    (point.east / (EARTH_RADIUS_M * Math.cos((reference[1] * Math.PI) / 180))) * (180 / Math.PI);
  return [lng, lat];
}

function toSweepPoint(
  point: LocalPoint,
  sinBearing: number,
  cosBearing: number,
): SweepPoint {
  return {
    along: point.north * cosBearing + point.east * sinBearing,
    cross: -point.north * sinBearing + point.east * cosBearing,
  };
}

function fromSweepPoint(
  point: SweepPoint,
  sinBearing: number,
  cosBearing: number,
): LocalPoint {
  return {
    north: point.along * cosBearing - point.cross * sinBearing,
    east: point.along * sinBearing + point.cross * cosBearing,
  };
}

function dedupeSortedIntersections(intersections: number[]) {
  if (intersections.length <= 1) return intersections;

  const deduped: number[] = [intersections[0]];
  for (let index = 1; index < intersections.length; index += 1) {
    const value = intersections[index];
    if (Math.abs(value - deduped[deduped.length - 1]) <= INTERSECTION_EPSILON_M) {
      deduped[deduped.length - 1] = (deduped[deduped.length - 1] + value) * 0.5;
      continue;
    }
    deduped.push(value);
  }
  return deduped;
}

export function generateFlightLinesForPolygon(
  ring: number[][],
  bearingDeg: number,
  lineSpacingM: number,
): { flightLines: number[][][]; sweepIndices: number[]; lineSpacing: number; bounds: ReturnType<typeof getPolygonBounds> } {
  const closedRing = ensureClosedRing(ring);
  const bounds = getPolygonBounds(closedRing);
  const lineSpacing = lineSpacingM;
  const flightLines: number[][][] = [];
  const sweepIndices: number[] = [];

  const centerLat = (bounds.minLat + bounds.maxLat) / 2;
  const centerLng = (bounds.minLng + bounds.maxLng) / 2;
  const diagonal = haversineDistance([bounds.minLng, bounds.minLat], [bounds.maxLng, bounds.maxLat]);
  const referenceCoordinate: [number, number] = [centerLng, centerLat];
  const bearingRad = (bearingDeg * Math.PI) / 180;
  const sinBearing = Math.sin(bearingRad);
  const cosBearing = Math.cos(bearingRad);
  const projectedRing = closedRing.map((point) =>
    toSweepPoint(toLocalPoint(point as [number, number], referenceCoordinate), sinBearing, cosBearing),
  );

  let minCrossTrackM = Infinity;
  let maxCrossTrackM = -Infinity;
  for (const point of projectedRing) {
    minCrossTrackM = Math.min(minCrossTrackM, point.cross);
    maxCrossTrackM = Math.max(maxCrossTrackM, point.cross);
  }

  const minLineIndex = Math.ceil((minCrossTrackM - INTERSECTION_EPSILON_M) / lineSpacing);
  const maxLineIndex = Math.floor((maxCrossTrackM + INTERSECTION_EPSILON_M) / lineSpacing);

  for (let lineIndex = minLineIndex; lineIndex <= maxLineIndex; lineIndex += 1) {
    const crossTrackM = lineIndex * lineSpacing;
    const intersections: number[] = [];

    for (let pointIndex = 0; pointIndex < projectedRing.length - 1; pointIndex += 1) {
      const start = projectedRing[pointIndex];
      const end = projectedRing[pointIndex + 1];
      const crossDelta = end.cross - start.cross;
      if (Math.abs(crossDelta) <= INTERSECTION_EPSILON_M) continue;

      const minCross = Math.min(start.cross, end.cross);
      const maxCross = Math.max(start.cross, end.cross);
      if (crossTrackM < minCross || crossTrackM >= maxCross) continue;

      const t = (crossTrackM - start.cross) / crossDelta;
      intersections.push(start.along + t * (end.along - start.along));
    }

    if (intersections.length < 2) continue;
    intersections.sort((left, right) => left - right);
    const deduped = dedupeSortedIntersections(intersections);
    const pairCount = deduped.length - (deduped.length % 2);

    for (let pairIndex = 0; pairIndex < pairCount; pairIndex += 2) {
      const startAlongM = deduped[pairIndex];
      const endAlongM = deduped[pairIndex + 1];
      if (endAlongM - startAlongM <= INTERSECTION_EPSILON_M) continue;

      const startPoint = fromLocalPoint(
        fromSweepPoint({ along: startAlongM, cross: crossTrackM }, sinBearing, cosBearing),
        referenceCoordinate,
      );
      const endPoint = fromLocalPoint(
        fromSweepPoint({ along: endAlongM, cross: crossTrackM }, sinBearing, cosBearing),
        referenceCoordinate,
      );
      flightLines.push([startPoint, endPoint]);
      sweepIndices.push(lineIndex);
    }
  }

  return { flightLines, sweepIndices, lineSpacing, bounds };
}
