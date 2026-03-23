/***********************************************************************
 * hooks/useMapInitialization.ts
 *
 * Custom hook to initialize and manage the Mapbox GL map instance.
 *
 * © 2025 <your-name>. MIT License.
 ***********************************************************************/

import { useRef, useEffect } from 'react';
import mapboxgl, { Map as MapboxMap, LngLatLike } from 'mapbox-gl';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import { MapboxOverlay } from '@deck.gl/mapbox';

const MAPBOX_DEM_SOURCE_ID = 'mapbox-dem';
const BACKEND_DEM_SOURCE_ID = 'backend-dem';

function ensureMapboxDemSource(map: MapboxMap) {
  if (map.getSource(MAPBOX_DEM_SOURCE_ID)) return;
  map.addSource(MAPBOX_DEM_SOURCE_ID, {
    type: 'raster-dem',
    url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
    tileSize: 512,
    maxzoom: 14,
  });
}

export function setTerrainDemSourceOnMap(map: MapboxMap, tileUrlTemplate: string | null) {
  try {
    if (!map.getStyle()) {
      return;
    }
  } catch {
    return;
  }

  ensureMapboxDemSource(map);
  map.setTerrain(null);

  if (tileUrlTemplate) {
    if (map.getSource(BACKEND_DEM_SOURCE_ID)) {
      map.removeSource(BACKEND_DEM_SOURCE_ID);
    }
    map.addSource(BACKEND_DEM_SOURCE_ID, {
      type: 'raster-dem',
      tiles: [tileUrlTemplate],
      tileSize: 512,
      maxzoom: 14,
      encoding: 'mapbox',
    });
    map.setTerrain({ source: BACKEND_DEM_SOURCE_ID, exaggeration: 1 });
    return;
  }

  map.setTerrain({ source: MAPBOX_DEM_SOURCE_ID, exaggeration: 1 });
}

interface UseMapInitializationProps {
  mapboxToken: string;
  center: LngLatLike;
  zoom: number;
  mapContainer: React.RefObject<HTMLDivElement>;
  onLoad: (map: MapboxMap, draw: MapboxDraw, overlay: MapboxOverlay) => void;
  onError: (message: string) => void;
}

export function useMapInitialization({
  mapboxToken,
  center,
  zoom,
  mapContainer,
  onLoad,
  onError,
}: UseMapInitializationProps) {
  const mapRef = useRef<MapboxMap>();
  const drawRef = useRef<MapboxDraw>();
  const deckOverlayRef = useRef<MapboxOverlay>();
  // Keep latest callbacks without retriggering map init
  const onLoadRef = useRef(onLoad);
  const onErrorRef = useRef(onError);

  // Update refs when props change
  useEffect(() => { onLoadRef.current = onLoad; }, [onLoad]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);

  useEffect(() => {
    if (!mapContainer.current) {
      console.warn('Map container not ready yet');
      return;
    }

    if (!mapboxToken) {
      console.error('Mapbox token is missing');
      onErrorRef.current('Mapbox token is missing');
      return;
    }

    const timeoutId = setTimeout(() => {
      if (!mapContainer.current) {
        console.error('Map container became unavailable');
        onErrorRef.current('Map container is not available');
        return;
      }

      try {
        mapboxgl.accessToken = mapboxToken;

        const map = new mapboxgl.Map({
          container: mapContainer.current,
          style: 'mapbox://styles/mapbox/satellite-v9',
          center,
          zoom,
          pitch: 45,
          bearing: 0,
          attributionControl: true,
        });
        mapRef.current = map;

        const draw = new MapboxDraw({
          displayControlsDefault: true,
          controls: {
            polygon: true,
            trash: true,
            line_string: false,
            point: false,
            combine_features: false,
            uncombine_features: false,
          },
        });
        drawRef.current = draw;

        map.on('load', () => {
          ensureMapboxDemSource(map);
          map.setTerrain({ source: MAPBOX_DEM_SOURCE_ID, exaggeration: 1 });
          map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), 'top-right');
          map.addControl(draw, 'top-left');

          const deckOverlay = new MapboxOverlay({
            interleaved: true,
            layers: [],
          });
          deckOverlayRef.current = deckOverlay;
          map.addControl(deckOverlay);

          if (drawRef.current && deckOverlayRef.current) {
            // Use latest callback without reinitializing the map
            onLoadRef.current(map, drawRef.current, deckOverlayRef.current);
          }
        });

        map.on('error', (e) => {
          console.error('Map error:', e);
          onErrorRef.current(`Map loading error: ${e.error?.message || 'Unknown error'}`);
        });
      } catch (error) {
        console.error('Failed to initialize map:', error);
        onErrorRef.current(`Failed to initialize map: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }, 100);

    return () => {
      clearTimeout(timeoutId);
      if (mapRef.current) {
        try {
          // Check if map is still valid before removing
          // remove() is safe even if style isn't fully loaded
          if (mapRef.current.getContainer()) mapRef.current.remove();
        } catch (error) {
          console.warn('Error during map cleanup:', error);
        } finally {
          mapRef.current = undefined;
        }
      }
    };
    // ⚠️ Crucial: do not depend on onLoad/onError here
  }, [mapboxToken, center, zoom, mapContainer]);

  return { map: mapRef.current, draw: drawRef.current };
}
