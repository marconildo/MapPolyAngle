/***********************************************************************
 * utils/deckgl-layers.ts
 *
 * Functions for creating and managing Deck.gl layers for 3D visualization.
 *
 * © 2025 <your-name>. MIT License.
 ***********************************************************************/

import { MapboxOverlay } from '@deck.gl/mapbox';
import { PathLayer, ScatterplotLayer } from '@deck.gl/layers';
import { COORDINATE_SYSTEM } from '@deck.gl/core';
import { mergeContiguous3DPathSegmentsForRender } from './geometry';

export function update3DPathLayer(
  overlay: MapboxOverlay,
  polygonId: string,
  path3d: [number, number, number][][],
  setLayers: React.Dispatch<React.SetStateAction<any[]>>
) {
  const layers: any[] = [];
  const renderPaths = mergeContiguous3DPathSegmentsForRender(path3d);

  // Collapse contiguous sweep/turn segments into continuous render paths to avoid visible seams.
  const pathLayer = new PathLayer({
    id: `drone-path-${polygonId}`,
    data: renderPaths,
    getPath: (d: any) => d,
    getColor: [100, 200, 255, 240], // thin light blue line
    getWidth: 2,
    widthUnits: 'meters',
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    billboard: false,
    parameters: {
      depthTest: true,
      depthWrite: true,
    },
  });

  layers.push(pathLayer);

  setLayers((currentLayers) => {
    const filteredLayers = currentLayers.filter(
      (l) => !l.id.includes(`drone-`) || !l.id.includes(polygonId)
    );
    const newLayers = [...filteredLayers, ...layers];
    overlay.setProps({ layers: newLayers });
    return newLayers;
  });
}

export function remove3DPathLayer(
  overlay: MapboxOverlay,
  polygonId: string,
  setLayers: React.Dispatch<React.SetStateAction<any[]>>
) {
  setLayers((currentLayers) => {
    const filteredLayers = currentLayers.filter(
      (l) => !l.id.includes(`drone-path-${polygonId}`) && !l.id.includes(`drone-centerline-${polygonId}`)
    );
    overlay.setProps({ layers: filteredLayers });
    return filteredLayers;
  });
}

export function update3DMissionConnectorLayer(
  overlay: MapboxOverlay,
  connectorPaths: [number, number, number][][],
  setLayers: React.Dispatch<React.SetStateAction<any[]>>
) {
  const renderPaths = mergeContiguous3DPathSegmentsForRender(connectorPaths);
  const connectorLayer = new PathLayer({
    id: 'drone-mission-connectors',
    data: renderPaths,
    getPath: (d: any) => d,
    getColor: [125, 211, 252, 220],
    getWidth: 2.4,
    widthUnits: 'meters',
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    billboard: false,
    parameters: {
      depthTest: true,
      depthWrite: true,
    },
  });

  setLayers((currentLayers) => {
    const filteredLayers = currentLayers.filter((layer) => String(layer?.id ?? '') !== 'drone-mission-connectors');
    const newLayers = [...filteredLayers, connectorLayer];
    overlay.setProps({ layers: newLayers });
    return newLayers;
  });
}

export function remove3DMissionConnectorLayer(
  overlay: MapboxOverlay,
  setLayers: React.Dispatch<React.SetStateAction<any[]>>
) {
  setLayers((currentLayers) => {
    const filteredLayers = currentLayers.filter((layer) => String(layer?.id ?? '') !== 'drone-mission-connectors');
    overlay.setProps({ layers: filteredLayers });
    return filteredLayers;
  });
}

export function update3DCameraPointsLayer(
  overlay: MapboxOverlay,
  polygonId: string,
  cameraPositions: [number, number, number][],
  setLayers: React.Dispatch<React.SetStateAction<any[]>>
) {
  const isImportedPoses = polygonId === '__POSES__';
  const cameraLayer = new ScatterplotLayer({
    id: `camera-points-${polygonId}`,
    data: cameraPositions,
    getPosition: (d: any) => d,
    // Make imported red circles smaller than polygon camera points
    getRadius: isImportedPoses ? 3 : 8,
    getFillColor: [255, 71, 87, 255], // Red color #ff4757
    getLineColor: [255, 255, 255, 255], // White outline
    lineWidthMinPixels: isImportedPoses ? 1 : 2,
    radiusUnits: 'meters',
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    parameters: {
      depthTest: true,
      depthWrite: true,
    },
  });

  setLayers((currentLayers) => {
    // Remove existing camera points for this polygon
    const filteredLayers = currentLayers.filter(
      (l) => !l.id.includes(`camera-points-${polygonId}`)
    );
    const newLayers = [...filteredLayers, cameraLayer];
    overlay.setProps({ layers: newLayers });
    return newLayers;
  });
}

export function remove3DCameraPointsLayer(
  overlay: MapboxOverlay,
  polygonId: string,
  setLayers: React.Dispatch<React.SetStateAction<any[]>>
) {
  setLayers((currentLayers) => {
    const filteredLayers = currentLayers.filter(
      (l) => !l.id.includes(`camera-points-${polygonId}`)
    );
    overlay.setProps({ layers: filteredLayers });
    return filteredLayers;
  });
}

// Trigger points along flight lines, rendered in the air via Deck.gl
export function update3DTriggerPointsLayer(
  overlay: MapboxOverlay,
  polygonId: string,
  triggerPositions: [number, number, number][],
  setLayers: React.Dispatch<React.SetStateAction<any[]>>
) {
  const triggerLayer = new ScatterplotLayer({
    id: `trigger-points-${polygonId}`,
    data: triggerPositions,
    getPosition: (d: any) => d,
    getRadius: 4, // half the size of camera points
    getFillColor: [12, 36, 97, 230], // Dark blue
    getLineColor: [230, 240, 255, 220],
    lineWidthMinPixels: 1,
    radiusUnits: 'meters',
    coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
    // Draw on top of the path; avoid depth occlusion
    parameters: { depthTest: false, depthWrite: false },
  });

  setLayers((currentLayers) => {
    const filteredLayers = currentLayers.filter(
      (l) => !l.id.includes(`trigger-points-${polygonId}`)
    );
    const newLayers = [...filteredLayers, triggerLayer];
    overlay.setProps({ layers: newLayers });
    return newLayers;
  });
}

export function remove3DTriggerPointsLayer(
  overlay: MapboxOverlay,
  polygonId: string,
  setLayers: React.Dispatch<React.SetStateAction<any[]>>
) {
  setLayers((currentLayers) => {
    const filteredLayers = currentLayers.filter(
      (l) => !l.id.includes(`trigger-points-${polygonId}`)
    );
    overlay.setProps({ layers: filteredLayers });
    return filteredLayers;
  });
}
