import { destination as geoDestination } from "@/utils/terrainAspectHybrid";

import { getPolygonBounds, haversineDistance } from "./geometry";

export function generateFlightLinesForPolygon(
  ring: number[][],
  bearingDeg: number,
  lineSpacingM: number,
): { flightLines: number[][][]; sweepIndices: number[]; lineSpacing: number; bounds: ReturnType<typeof getPolygonBounds> } {
  const bounds = getPolygonBounds(ring);
  const lineSpacing = lineSpacingM;
  const flightLines: number[][][] = [];
  const sweepIndices: number[] = [];

  const centerLat = (bounds.minLat + bounds.maxLat) / 2;
  const centerLng = (bounds.minLng + bounds.maxLng) / 2;
  const diagonal = haversineDistance([bounds.minLng, bounds.minLat], [bounds.maxLng, bounds.maxLat]);

  const numLines = Math.ceil(diagonal / lineSpacing);
  const perpBearing = (bearingDeg + 90) % 360;

  const pointInPolygon = (lng: number, lat: number): boolean => {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      const intersect = ((yi > lat) !== (yj > lat)) &&
        (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  };

  const refineBoundaryPoint = (
    outsidePoint: [number, number],
    insidePoint: [number, number],
  ): [number, number] => {
    let lo = outsidePoint;
    let hi = insidePoint;

    for (let iter = 0; iter < 18; iter++) {
      const mid: [number, number] = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
      ];
      if (pointInPolygon(mid[0], mid[1])) {
        hi = mid;
      } else {
        lo = mid;
      }
    }

    return hi;
  };

  for (let i = -numLines; i <= numLines; i++) {
    const distance = i * lineSpacing;
    const [centerLineLng, centerLineLat] = geoDestination([centerLng, centerLat], perpBearing, distance);

    const extendDistance = diagonal * 0.6;
    const p1 = geoDestination([centerLineLng, centerLineLat], bearingDeg, extendDistance);
    const p2 = geoDestination([centerLineLng, centerLineLat], (bearingDeg + 180) % 360, extendDistance);

    const sampleStepM = Math.max(5, Math.min(15, lineSpacing * 0.2));
    const samples = Math.max(120, Math.min(1200, Math.ceil((extendDistance * 2) / sampleStepM)));

    let previousPoint: [number, number] = [p2[0], p2[1]];
    let previousInside = pointInPolygon(previousPoint[0], previousPoint[1]);
    let currentSegmentStart: [number, number] | null = previousInside ? previousPoint : null;

    for (let s = 1; s <= samples; s++) {
      const t = s / samples;
      const currentPoint: [number, number] = [
        p2[0] + t * (p1[0] - p2[0]),
        p2[1] + t * (p1[1] - p2[1]),
      ];
      const currentInside = pointInPolygon(currentPoint[0], currentPoint[1]);

      if (currentInside && !previousInside) {
        currentSegmentStart = refineBoundaryPoint(previousPoint, currentPoint);
      } else if (!currentInside && previousInside && currentSegmentStart) {
        const segmentEnd = refineBoundaryPoint(currentPoint, previousPoint);
        flightLines.push([currentSegmentStart, segmentEnd]);
        sweepIndices.push(i);
        currentSegmentStart = null;
      }

      if (s === samples && currentInside && currentSegmentStart) {
        flightLines.push([currentSegmentStart, currentPoint]);
        sweepIndices.push(i);
      }

      previousPoint = currentPoint;
      previousInside = currentInside;
    }
  }

  return { flightLines, sweepIndices, lineSpacing, bounds };
}
