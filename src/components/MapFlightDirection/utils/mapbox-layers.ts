/***********************************************************************
 * utils/mapbox-layers.ts
 *
 * Functions for adding and removing Mapbox GL JS layers (flight lines).
 *
 * © 2025 <your-name>. MIT License.
 ***********************************************************************/

import type { FlightParams, PlannedFlightGeometry } from '@/domain/types';
import type { Map as MapboxMap } from 'mapbox-gl';
import { haversineDistance } from '@/flight/geometry';
import { generateFlightLinesForPolygon } from '@/flight/flightLines';
import { generatePlannedFlightGeometryForPolygon } from '@/flight/plannedGeometry';
import { destination as geoDestination } from '@/utils/terrainAspectHybrid';

export { generateFlightLinesForPolygon } from '@/flight/flightLines';

function toPlannedFlightGeometry(
  rawGeometry: ReturnType<typeof generateFlightLinesForPolygon>,
): PlannedFlightGeometry {
  return {
    ...rawGeometry,
    flightLines: rawGeometry.flightLines as [number, number][][],
    sweepLines: [],
    gridPoints: [],
    leadInPoints: [],
    leadOutPoints: [],
    connectedLines: [],
    turnaroundRadiusM: 0,
    turnBlocks: [],
  };
}

function getLineColor(quality?: string) {
  switch (quality) {
    case 'excellent':
      return '#22c55e'; // green-500
    case 'good':
      return '#3b82f6'; // blue-500
    case 'fair':
      return '#f97316'; // orange-500
    case 'poor':
      return '#ef4444'; // red-500
    default:
      return '#6b7280'; // gray-500
  }
}

function getDrawLayerAnchor(map: MapboxMap): string | undefined {
  const layers = map.getStyle().layers || [];
  return layers.find((layer) => layer.id.startsWith('gl-draw-'))?.id;
}

const PROCESSING_PERIMETER_SOURCE_ID = 'terrain-processing-perimeter-source';
const PROCESSING_PERIMETER_GLOW_LAYER_ID = 'terrain-processing-perimeter-glow';
const PROCESSING_PERIMETER_CORE_LAYER_ID = 'terrain-processing-perimeter-core';
const PROCESSING_PERIMETER_PULSE_LAYER_ID = 'terrain-processing-perimeter-pulse';
const DSM_FOOTPRINT_SOURCE_ID = 'uploaded-dsm-footprint-source';
const DSM_FOOTPRINT_FILL_LAYER_ID = 'uploaded-dsm-footprint-fill';
const DSM_FOOTPRINT_LINE_LAYER_ID = 'uploaded-dsm-footprint-line';
const IMAGERY_OVERLAY_SOURCE_ID = 'imagery-overlay-source';
const IMAGERY_OVERLAY_LAYER_ID = 'imagery-overlay-layer';
const SELECTED_POLYGON_SOURCE_ID = 'selected-polygon-highlight-source';
const SELECTED_POLYGON_FILL_LAYER_ID = 'selected-polygon-highlight-fill';
const SELECTED_POLYGON_OUTER_LAYER_ID = 'selected-polygon-highlight-outer';
const SELECTED_POLYGON_INNER_LAYER_ID = 'selected-polygon-highlight-inner';
const NON_SELECTED_POLYGONS_SOURCE_ID = 'non-selected-polygons-dim-source';
const NON_SELECTED_POLYGONS_FILL_LAYER_ID = 'non-selected-polygons-dim-fill';
const NON_SELECTED_POLYGONS_LINE_LAYER_ID = 'non-selected-polygons-dim-line';
const MISSION_ENDPOINTS_SOURCE_ID = 'mission-sequence-endpoints-source';
const MISSION_ENDPOINTS_MARKER_LAYER_ID = 'mission-sequence-endpoints-marker';
const MISSION_ENDPOINTS_LABEL_LAYER_ID = 'mission-sequence-endpoints-label';

function emptyProcessingPerimeterData() {
  return {
    type: 'FeatureCollection',
    features: [],
  } as const;
}

function ensureClosedRing(ring: [number, number][]): [number, number][] {
  if (ring.length < 2) return ring;
  const [firstLng, firstLat] = ring[0];
  const [lastLng, lastLat] = ring[ring.length - 1];
  if (firstLng === lastLng && firstLat === lastLat) return ring;
  return [...ring, [firstLng, firstLat]];
}

function pointsAreNear(
  left: [number, number],
  right: [number, number],
  epsilonDeg = 1e-7,
) {
  return Math.abs(left[0] - right[0]) <= epsilonDeg && Math.abs(left[1] - right[1]) <= epsilonDeg;
}

function buildProcessingPulseGradient(phase: number) {
  const base = 'rgba(147, 197, 253, 0)';
  const shoulder = 'rgba(219, 234, 254, 0.72)';
  const peak = 'rgba(255, 255, 255, 1)';
  const tail = 0.16;
  const lead = 0.085;
  const shoulderBefore = 0.07;
  const shoulderAfter = 0.05;
  const resolution = 10000;
  const rawStops: Array<[number, string, number]> = [
    [0, base, 0],
    [1, base, 0],
  ];

  for (const center of [phase - 1, phase, phase + 1]) {
    const stops: Array<[number, string, number]> = [
      [center - tail, base, 0],
      [center - shoulderBefore, shoulder, 1],
      [center, peak, 2],
      [center + shoulderAfter, shoulder, 1],
      [center + lead, base, 0],
    ];
    for (const [position, color, priority] of stops) {
      if (position < 0 || position > 1) continue;
      rawStops.push([position, color, priority]);
    }
  }

  const byPosition = new Map<number, { color: string; priority: number }>();
  for (const [position, color, priority] of rawStops) {
    const key = Math.round(position * resolution);
    const existing = byPosition.get(key);
    if (!existing || priority >= existing.priority) {
      byPosition.set(key, { color, priority });
    }
  }

  const sortedStops = Array.from(byPosition.entries())
    .map(([key, value]) => ({
      position: key / resolution,
      color: value.color,
    }))
    .sort((a, b) => a.position - b.position);

  const normalized: Array<number | string | ['line-progress'] | ['linear'] | ['interpolate']> = [
    'interpolate',
    ['linear'],
    ['line-progress'],
  ];
  for (const stop of sortedStops) {
    normalized.push(stop.position, stop.color);
  }
  return normalized as any;
}

function ensureProcessingPerimeterLayers(map: MapboxMap) {
  if (!map.getSource(PROCESSING_PERIMETER_SOURCE_ID)) {
    map.addSource(PROCESSING_PERIMETER_SOURCE_ID, {
      type: 'geojson',
      lineMetrics: true,
      data: emptyProcessingPerimeterData(),
    } as any);
  }

  if (!map.getLayer(PROCESSING_PERIMETER_GLOW_LAYER_ID)) {
    map.addLayer({
      id: PROCESSING_PERIMETER_GLOW_LAYER_ID,
      type: 'line',
      source: PROCESSING_PERIMETER_SOURCE_ID,
      layout: {
        'line-join': 'round',
        'line-cap': 'round',
      },
      paint: {
        'line-color': '#60a5fa',
        'line-width': 18,
        'line-opacity': 0.32,
        'line-blur': 6.2,
      },
    });
  }

  if (!map.getLayer(PROCESSING_PERIMETER_CORE_LAYER_ID)) {
    map.addLayer({
      id: PROCESSING_PERIMETER_CORE_LAYER_ID,
      type: 'line',
      source: PROCESSING_PERIMETER_SOURCE_ID,
      layout: {
        'line-join': 'round',
        'line-cap': 'round',
      },
      paint: {
        'line-color': '#bfdbfe',
        'line-width': 3.25,
        'line-opacity': 0.86,
      },
    });
  }

  if (!map.getLayer(PROCESSING_PERIMETER_PULSE_LAYER_ID)) {
    map.addLayer({
      id: PROCESSING_PERIMETER_PULSE_LAYER_ID,
      type: 'line',
      source: PROCESSING_PERIMETER_SOURCE_ID,
      layout: {
        'line-join': 'round',
        'line-cap': 'round',
      },
      paint: {
        'line-width': 11.5,
        'line-opacity': 1,
        'line-blur': 0.9,
        'line-gradient': buildProcessingPulseGradient(0),
      },
    });
  }
}

export function setProcessingPerimeterPolygons(
  map: MapboxMap,
  polygons: Array<{ polygonId: string; ring: [number, number][] }>,
) {
  ensureProcessingPerimeterLayers(map);
  const source = map.getSource(PROCESSING_PERIMETER_SOURCE_ID) as any;
  if (!source) return;

  const data = {
    type: 'FeatureCollection',
    features: polygons
      .map(({ polygonId, ring }) => {
        const closedRing = ensureClosedRing(ring);
        if (closedRing.length < 2) return null;
        return {
          type: 'Feature',
          geometry: {
            type: 'LineString',
            coordinates: closedRing,
          },
          properties: {
            polygonId,
          },
        };
      })
      .filter((feature): feature is NonNullable<typeof feature> => feature !== null),
  } as const;

  source.setData(data);
}

export function animateProcessingPerimeter(map: MapboxMap, timestampMs: number) {
  ensureProcessingPerimeterLayers(map);
  const pulse = (Math.sin(timestampMs * 0.004) + 1) / 2;
  const phase = (timestampMs * 0.00018) % 1;

  if (map.getLayer(PROCESSING_PERIMETER_GLOW_LAYER_ID)) {
    map.setPaintProperty(PROCESSING_PERIMETER_GLOW_LAYER_ID, 'line-opacity', 0.26 + pulse * 0.22);
    map.setPaintProperty(PROCESSING_PERIMETER_GLOW_LAYER_ID, 'line-width', 16 + pulse * 7);
  }
  if (map.getLayer(PROCESSING_PERIMETER_CORE_LAYER_ID)) {
    map.setPaintProperty(PROCESSING_PERIMETER_CORE_LAYER_ID, 'line-opacity', 0.68 + pulse * 0.16);
  }
  if (map.getLayer(PROCESSING_PERIMETER_PULSE_LAYER_ID)) {
    map.setPaintProperty(PROCESSING_PERIMETER_PULSE_LAYER_ID, 'line-width', 10.5 + pulse * 3.2);
    map.setPaintProperty(PROCESSING_PERIMETER_PULSE_LAYER_ID, 'line-gradient', buildProcessingPulseGradient(phase));
  }
}

export function setDsmFootprintPolygon(
  map: MapboxMap,
  dsmId: string,
  ring: [number, number][],
) {
  const closedRing = ensureClosedRing(ring);
  if (closedRing.length < 4) return;
  const data = {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [closedRing],
        },
        properties: {
          dsmId,
        },
      },
    ],
  } as const;

  if (map.getSource(DSM_FOOTPRINT_SOURCE_ID)) {
    (map.getSource(DSM_FOOTPRINT_SOURCE_ID) as any).setData(data);
  } else {
    map.addSource(DSM_FOOTPRINT_SOURCE_ID, {
      type: 'geojson',
      data,
    } as any);
  }

  const beforeId = getDrawLayerAnchor(map);
  if (!map.getLayer(DSM_FOOTPRINT_FILL_LAYER_ID)) {
    map.addLayer({
      id: DSM_FOOTPRINT_FILL_LAYER_ID,
      type: 'fill',
      source: DSM_FOOTPRINT_SOURCE_ID,
      paint: {
        'fill-color': '#0ea5e9',
        'fill-opacity': 0.08,
      },
    }, beforeId);
  }
  if (!map.getLayer(DSM_FOOTPRINT_LINE_LAYER_ID)) {
    map.addLayer({
      id: DSM_FOOTPRINT_LINE_LAYER_ID,
      type: 'line',
      source: DSM_FOOTPRINT_SOURCE_ID,
      paint: {
        'line-color': '#0284c7',
        'line-width': 2,
        'line-dasharray': [2, 2],
        'line-opacity': 0.9,
      },
    }, beforeId);
  }
}

export function clearDsmFootprintPolygon(map: MapboxMap) {
  try { if (map.getLayer(DSM_FOOTPRINT_LINE_LAYER_ID)) map.removeLayer(DSM_FOOTPRINT_LINE_LAYER_ID); } catch {}
  try { if (map.getLayer(DSM_FOOTPRINT_FILL_LAYER_ID)) map.removeLayer(DSM_FOOTPRINT_FILL_LAYER_ID); } catch {}
  try { if (map.getSource(DSM_FOOTPRINT_SOURCE_ID)) map.removeSource(DSM_FOOTPRINT_SOURCE_ID); } catch {}
}

export function setMissionSequenceEndpoints(
  map: MapboxMap,
  endpoints: {
    startEndpoint?: { point: [number, number] };
    endEndpoint?: { point: [number, number] };
  } | null,
) {
  if (!endpoints?.startEndpoint && !endpoints?.endEndpoint) {
    clearMissionSequenceEndpoints(map);
    return;
  }

  const uniquePoints: [number, number][] = [];
  for (const point of [endpoints.startEndpoint?.point, endpoints.endEndpoint?.point]) {
    if (!point) continue;
    if (uniquePoints.some((existing) => pointsAreNear(existing, point))) continue;
    uniquePoints.push(point);
  }

  const features = uniquePoints.map((point) => ({
    type: 'Feature',
    geometry: {
      type: 'Point',
      coordinates: point,
    },
    properties: {
      label: 'H',
      radius: 6,
    },
  }));

  const data = {
    type: 'FeatureCollection',
    features,
  } as const;

  if (map.getSource(MISSION_ENDPOINTS_SOURCE_ID)) {
    (map.getSource(MISSION_ENDPOINTS_SOURCE_ID) as any).setData(data);
  } else {
    map.addSource(MISSION_ENDPOINTS_SOURCE_ID, {
      type: 'geojson',
      data,
    } as any);
  }

  const beforeId = getDrawLayerAnchor(map);
  if (!map.getLayer(MISSION_ENDPOINTS_MARKER_LAYER_ID)) {
    map.addLayer({
      id: MISSION_ENDPOINTS_MARKER_LAYER_ID,
      type: 'circle',
      source: MISSION_ENDPOINTS_SOURCE_ID,
      paint: {
        'circle-radius': ['get', 'radius'],
        'circle-color': '#14b8a6',
        'circle-stroke-width': 2,
        'circle-stroke-color': '#ffffff',
        'circle-opacity': 0.96,
      },
    }, beforeId);
  }
  if (!map.getLayer(MISSION_ENDPOINTS_LABEL_LAYER_ID)) {
    map.addLayer({
      id: MISSION_ENDPOINTS_LABEL_LAYER_ID,
      type: 'symbol',
      source: MISSION_ENDPOINTS_SOURCE_ID,
      layout: {
        'text-field': ['get', 'label'],
        'text-size': 10,
        'text-font': ['Open Sans Bold', 'Arial Unicode MS Bold'],
        'text-anchor': 'center',
        'text-allow-overlap': true,
        'text-ignore-placement': true,
      },
      paint: {
        'text-color': '#ffffff',
        'text-halo-color': '#ffffff',
        'text-halo-width': 0,
      }
    }, beforeId);
  }
}

export function clearMissionSequenceEndpoints(map: MapboxMap) {
  try { if (map.getLayer(MISSION_ENDPOINTS_LABEL_LAYER_ID)) map.removeLayer(MISSION_ENDPOINTS_LABEL_LAYER_ID); } catch {}
  try { if (map.getLayer(MISSION_ENDPOINTS_MARKER_LAYER_ID)) map.removeLayer(MISSION_ENDPOINTS_MARKER_LAYER_ID); } catch {}
  try { if (map.getSource(MISSION_ENDPOINTS_SOURCE_ID)) map.removeSource(MISSION_ENDPOINTS_SOURCE_ID); } catch {}
}

export function setNonSelectedPolygonDimMask(
  map: MapboxMap,
  polygons: Array<{ polygonId: string; ring: [number, number][] }>,
) {
  if (polygons.length === 0) {
    try { if (map.getLayer(NON_SELECTED_POLYGONS_LINE_LAYER_ID)) map.removeLayer(NON_SELECTED_POLYGONS_LINE_LAYER_ID); } catch {}
    try { if (map.getLayer(NON_SELECTED_POLYGONS_FILL_LAYER_ID)) map.removeLayer(NON_SELECTED_POLYGONS_FILL_LAYER_ID); } catch {}
    try { if (map.getSource(NON_SELECTED_POLYGONS_SOURCE_ID)) map.removeSource(NON_SELECTED_POLYGONS_SOURCE_ID); } catch {}
    return;
  }

  const data = {
    type: 'FeatureCollection',
    features: polygons
      .map((polygon) => {
        const closedRing = ensureClosedRing(polygon.ring);
        if (closedRing.length < 4) return null;
        return {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [closedRing],
          },
          properties: {
            polygonId: polygon.polygonId,
          },
        };
      })
      .filter((feature): feature is NonNullable<typeof feature> => feature !== null),
  } as const;

  if (data.features.length === 0) {
    setNonSelectedPolygonDimMask(map, []);
    return;
  }

  if (map.getSource(NON_SELECTED_POLYGONS_SOURCE_ID)) {
    (map.getSource(NON_SELECTED_POLYGONS_SOURCE_ID) as any).setData(data);
  } else {
    map.addSource(NON_SELECTED_POLYGONS_SOURCE_ID, {
      type: 'geojson',
      data,
    } as any);
  }

  if (!map.getLayer(NON_SELECTED_POLYGONS_FILL_LAYER_ID)) {
    const beforeId = getDrawLayerAnchor(map);
    map.addLayer({
      id: NON_SELECTED_POLYGONS_FILL_LAYER_ID,
      type: 'fill',
      source: NON_SELECTED_POLYGONS_SOURCE_ID,
      paint: {
        'fill-color': '#ffffff',
        'fill-opacity': 0.08,
      },
    }, beforeId);
  }

  if (!map.getLayer(NON_SELECTED_POLYGONS_LINE_LAYER_ID)) {
    const beforeId = getDrawLayerAnchor(map);
    map.addLayer({
      id: NON_SELECTED_POLYGONS_LINE_LAYER_ID,
      type: 'line',
      source: NON_SELECTED_POLYGONS_SOURCE_ID,
      layout: {
        'line-join': 'round',
        'line-cap': 'round',
      },
      paint: {
        'line-color': '#94a3b8',
        'line-width': 1.5,
        'line-opacity': 0.3,
      },
    }, beforeId);
  }
}

export function setSelectedPolygonHighlight(
  map: MapboxMap,
  polygon: { polygonId: string; ring: [number, number][] } | null,
) {
  if (!polygon) {
    try { if (map.getLayer(SELECTED_POLYGON_INNER_LAYER_ID)) map.removeLayer(SELECTED_POLYGON_INNER_LAYER_ID); } catch {}
    try { if (map.getLayer(SELECTED_POLYGON_OUTER_LAYER_ID)) map.removeLayer(SELECTED_POLYGON_OUTER_LAYER_ID); } catch {}
    try { if (map.getLayer(SELECTED_POLYGON_FILL_LAYER_ID)) map.removeLayer(SELECTED_POLYGON_FILL_LAYER_ID); } catch {}
    try { if (map.getSource(SELECTED_POLYGON_SOURCE_ID)) map.removeSource(SELECTED_POLYGON_SOURCE_ID); } catch {}
    return;
  }

  const closedRing = ensureClosedRing(polygon.ring);
  if (closedRing.length < 4) {
    setSelectedPolygonHighlight(map, null);
    return;
  }

  const data = {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [closedRing],
        },
        properties: {
          polygonId: polygon.polygonId,
        },
      },
    ],
  } as const;

  if (map.getSource(SELECTED_POLYGON_SOURCE_ID)) {
    (map.getSource(SELECTED_POLYGON_SOURCE_ID) as any).setData(data);
  } else {
    map.addSource(SELECTED_POLYGON_SOURCE_ID, {
      type: 'geojson',
      data,
    } as any);
  }

  if (!map.getLayer(SELECTED_POLYGON_FILL_LAYER_ID)) {
    map.addLayer({
      id: SELECTED_POLYGON_FILL_LAYER_ID,
      type: 'fill',
      source: SELECTED_POLYGON_SOURCE_ID,
      paint: {
        'fill-color': '#f59e0b',
        'fill-opacity': 0.06,
      },
    });
  }

  if (!map.getLayer(SELECTED_POLYGON_OUTER_LAYER_ID)) {
    map.addLayer({
      id: SELECTED_POLYGON_OUTER_LAYER_ID,
      type: 'line',
      source: SELECTED_POLYGON_SOURCE_ID,
      layout: {
        'line-join': 'round',
        'line-cap': 'round',
      },
      paint: {
        'line-color': '#ffffff',
        'line-width': 7,
        'line-opacity': 0.96,
        'line-blur': 1.25,
      },
    });
  }

  if (!map.getLayer(SELECTED_POLYGON_INNER_LAYER_ID)) {
    map.addLayer({
      id: SELECTED_POLYGON_INNER_LAYER_ID,
      type: 'line',
      source: SELECTED_POLYGON_SOURCE_ID,
      layout: {
        'line-join': 'round',
        'line-cap': 'round',
      },
      paint: {
        'line-color': '#f59e0b',
        'line-width': 3,
        'line-opacity': 1,
      },
    });
  }
}

export function setFlightLineSelectionEmphasis(
  map: MapboxMap,
  selectedPolygonId: string | null,
  visible: boolean,
) {
  const layers = map.getStyle()?.layers ?? [];
  for (const layer of layers) {
    const layerId = layer.id;
    const isLineLayer = layerId.startsWith('flight-lines-layer-');
    const isTriggerCircleLayer = layerId.startsWith('flight-triggers-layer-');
    const isTriggerLabelLayer = layerId.startsWith('flight-triggers-label-');
    if (!isLineLayer && !isTriggerCircleLayer && !isTriggerLabelLayer) continue;

    const polygonId = isLineLayer
      ? layerId.slice('flight-lines-layer-'.length)
      : isTriggerCircleLayer
        ? layerId.slice('flight-triggers-layer-'.length)
        : layerId.slice('flight-triggers-label-'.length);
    const isSelected = !!selectedPolygonId && polygonId === selectedPolygonId;
    const isDimmed = !!selectedPolygonId && !isSelected;

    try {
      map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
    } catch {}

    if (isLineLayer) {
      try { map.setPaintProperty(layerId, 'line-opacity', isDimmed ? 0.16 : 0.95); } catch {}
      try { map.setPaintProperty(layerId, 'line-width', isSelected ? 1.8 : 0.5); } catch {}
    }

    if (isTriggerCircleLayer) {
      try { map.setPaintProperty(layerId, 'circle-opacity', isDimmed ? 0.18 : 0.95); } catch {}
      try { map.setPaintProperty(layerId, 'circle-stroke-opacity', isDimmed ? 0.2 : 1); } catch {}
      try { map.setPaintProperty(layerId, 'circle-radius', isSelected ? 2.2 : 1.5); } catch {}
    }

    if (isTriggerLabelLayer) {
      try { map.setPaintProperty(layerId, 'text-opacity', isDimmed ? 0.16 : 0.92); } catch {}
      try { map.setPaintProperty(layerId, 'text-halo-color', isSelected ? '#f8fafc' : '#ffffff'); } catch {}
    }
  }
}

export function setImageryOverlayOnMap(
  map: MapboxMap,
  overlay: {
    url: string;
    coordinates: [[number, number], [number, number], [number, number], [number, number]];
  } | null,
) {
  try {
    if (!map.getStyle()) {
      return;
    }
  } catch {
    return;
  }

  try { if (map.getLayer(IMAGERY_OVERLAY_LAYER_ID)) map.removeLayer(IMAGERY_OVERLAY_LAYER_ID); } catch {}
  try { if (map.getSource(IMAGERY_OVERLAY_SOURCE_ID)) map.removeSource(IMAGERY_OVERLAY_SOURCE_ID); } catch {}

  if (!overlay) return;

  map.addSource(IMAGERY_OVERLAY_SOURCE_ID, {
    type: 'image',
    url: overlay.url,
    coordinates: overlay.coordinates,
  } as any);

  map.addLayer({
    id: IMAGERY_OVERLAY_LAYER_ID,
    type: 'raster',
    source: IMAGERY_OVERLAY_SOURCE_ID,
    paint: {
      'raster-opacity': 0.92,
      'raster-resampling': 'linear',
    },
  }, getDrawLayerAnchor(map));
}

export function renderFlightLinesForPolygon(
  map: MapboxMap,
  polygonId: string,
  geometry: PlannedFlightGeometry & {
    bounds?: { minLng: number; minLat: number; maxLng: number; maxLat: number };
  },
  quality?: string,
  debugBearingDeg?: number,
): PlannedFlightGeometry {
  const { flightLines, connectedLines, lineSpacing, bounds } = geometry;
  const displayLines = connectedLines.length > 0 ? connectedLines : flightLines;

  const sourceId = `flight-lines-source-${polygonId}`;
  const layerId = `flight-lines-layer-${polygonId}`;

  const data = {
    type: 'FeatureCollection',
    features: displayLines.length > 0 ? [{
      type: 'Feature',
      geometry: {
        type: 'MultiLineString',
        coordinates: displayLines,
      },
      properties: {},
    }] : [],
  } as const;

  // Source: update or add
  if (map.getSource(sourceId)) {
    (map.getSource(sourceId) as any).setData(data);
  } else {
    map.addSource(sourceId, { type: 'geojson', data } as any);
  }

  // Layer: update or add (idempotent)
  if (map.getLayer(layerId)) {
    // Update paint properties if needed
    map.setPaintProperty(layerId, 'line-color', getLineColor(quality));
    map.setPaintProperty(layerId, 'line-opacity', 0.8);
    map.setPaintProperty(layerId, 'line-width', 0.5);
  } else {
    const beforeId = getDrawLayerAnchor(map);
    map.addLayer({
      id: layerId,
      type: 'line',
      source: sourceId,
      layout: {
        'line-join': 'round',
        'line-cap': 'round',
      },
      paint: {
        'line-color': getLineColor(quality),
        'line-width': 0.5,
        'line-opacity': 0.8,
      },
    }, beforeId);
  }

  if (displayLines.length === 0) {
    try {
      const b = bounds;
      if (!b) {
        throw new Error('bounds unavailable');
      }
      const centerLng = (b.minLng + b.maxLng) / 2;
      const centerLat = (b.minLat + b.maxLat) / 2;
      console.warn(
        `[flight-lines] No segments inside polygon for ${polygonId}. Debug: bearing=${(debugBearingDeg ?? 0).toFixed(2)}, spacing=${lineSpacing.toFixed(2)}m, center=(${centerLng.toFixed(5)},${centerLat.toFixed(5)}), bbox=lng[${b.minLng.toFixed(5)},${b.maxLng.toFixed(5)}], lat[${b.minLat.toFixed(5)},${b.maxLat.toFixed(5)}]`
      );
    } catch {}
  }

  return geometry;
}

export function addFlightLinesForPolygon(
  map: MapboxMap,
  polygonId: string,
  ring: number[][],
  bearingDeg: number,
  lineSpacingM: number,
  params?: FlightParams,
  quality?: string
): PlannedFlightGeometry {
  const rawGeometry = params ? null : generateFlightLinesForPolygon(ring, bearingDeg, lineSpacingM);
  const geometry = params
    ? generatePlannedFlightGeometryForPolygon(ring as [number, number][], bearingDeg, lineSpacingM, params)
    : toPlannedFlightGeometry(rawGeometry!);

  return renderFlightLinesForPolygon(map, polygonId, geometry, quality, bearingDeg);
}

export function removeFlightLinesForPolygon(map: MapboxMap, polygonId: string) {
  const layerId = `flight-lines-layer-${polygonId}`;
  const sourceId = `flight-lines-source-${polygonId}`;

  try { if (map.getLayer(layerId)) map.removeLayer(layerId); } catch {}
  try { if (map.getSource(sourceId)) map.removeSource(sourceId); } catch {}
}

// -------------------------------------------------------------------
// Trigger tick marks (camera trigger positions) along flight lines
// -------------------------------------------------------------------

function sampleTriggerPoints(line: [number, number][], spacingM: number): [number, number][] {
  if (line.length < 2 || spacingM <= 0) return [];
  const [A, B] = line as [[number, number], [number, number]];
  const total = haversineDistance(A, B);
  if (total === 0) return [A];

  // Calculate bearing from A to B
  const dLng = B[0] - A[0];
  const dLat = B[1] - A[1];
  const bearing = Math.atan2(dLng * Math.cos(A[1] * Math.PI / 180), dLat) * 180 / Math.PI;

  const pts: [number, number][] = [];
  // Place a trigger at the line start
  pts.push(A);
  let d = spacingM;
  while (d < total) {
    pts.push(geoDestination(A, bearing, d) as [number, number]);
    d += spacingM;
  }
  // Always place one at the end to avoid uncovered tail
  pts.push(B);
  return pts;
}

export function addTriggerPointsForPolygon(
  map: MapboxMap,
  polygonId: string,
  flightLines: number[][][],
  spacingM: number
) {
  const points: any[] = [];
  for (const ln of flightLines) {
    const pts = sampleTriggerPoints(ln as [number, number][], spacingM);
    for (const p of pts) {
      points.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: p },
        properties: { spacing: spacingM }
      });
    }
  }

  const sourceId = `flight-triggers-source-${polygonId}`;
  const circleLayerId = `flight-triggers-layer-${polygonId}`;
  const labelLayerId = `flight-triggers-label-${polygonId}`;

  if (map.getLayer(circleLayerId)) map.removeLayer(circleLayerId);
  if (map.getLayer(labelLayerId)) map.removeLayer(labelLayerId);
  if (map.getSource(sourceId)) map.removeSource(sourceId);

  map.addSource(sourceId, {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: points },
  } as any);

  map.addLayer({
    id: circleLayerId,
    type: 'circle',
    source: sourceId,
    paint: {
      'circle-radius': 1.5,
      'circle-color': '#111827',
      'circle-stroke-width': 1,
      'circle-stroke-color': '#ffffff'
    }
  }, getDrawLayerAnchor(map));

  map.addLayer({
    id: labelLayerId,
    type: 'symbol',
    source: sourceId,
    layout: {
      'text-field': ['concat', ['to-string', ['round', ['get', 'spacing']]], ' m'],
      'text-size': 10,
      'text-offset': [0, 1.2],
      'text-anchor': 'top',
      'symbol-avoid-edges': true
    },
    paint: {
      'text-color': '#374151',
      'text-halo-color': '#ffffff',
      'text-halo-width': 0.75
    }
  }, getDrawLayerAnchor(map));
}

export function removeTriggerPointsForPolygon(map: MapboxMap, polygonId: string) {
  const sourceId = `flight-triggers-source-${polygonId}`;
  const circleLayerId = `flight-triggers-layer-${polygonId}`;
  const labelLayerId = `flight-triggers-label-${polygonId}`;

  try { if (map.getLayer(circleLayerId)) map.removeLayer(circleLayerId); } catch {}
  try { if (map.getLayer(labelLayerId)) map.removeLayer(labelLayerId); } catch {}
  try { if (map.getSource(sourceId)) map.removeSource(sourceId); } catch {}
}

/**
 * Clear all flight lines from the map (for use when clearing all polygons)
 */
export function clearAllFlightLines(map: MapboxMap) {
  const layers = map.getStyle().layers || [];
  const sources = map.getStyle().sources || {};

  // Remove all flight line layers
  for (const layer of layers) {
    if (layer.id.startsWith('flight-lines-layer-')) {
      try { map.removeLayer(layer.id); } catch {}
    }
  }

  // Remove all flight line sources
  for (const sourceId of Object.keys(sources)) {
    if (sourceId.startsWith('flight-lines-source-')) {
      try { map.removeSource(sourceId); } catch {}
    }
  }
}

/**
 * Clear all trigger points from the map (for use when clearing all polygons)
 */
export function clearAllTriggerPoints(map: MapboxMap) {
  const layers = map.getStyle().layers || [];
  const sources = map.getStyle().sources || {};

  // Remove all trigger point layers
  for (const layer of layers) {
    if (layer.id.startsWith('flight-triggers-')) {
      try { map.removeLayer(layer.id); } catch {}
    }
  }

  // Remove all trigger point sources
  for (const sourceId of Object.keys(sources)) {
    if (sourceId.startsWith('flight-triggers-source-')) {
      try { map.removeSource(sourceId); } catch {}
    }
  }
}
