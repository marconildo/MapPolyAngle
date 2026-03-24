/***********************************************************************
 * hooks/usePolygonAnalysis.ts
 *
 * Custom hook to handle the analysis of a single polygon.
 *
 * © 2025 <your-name>. MIT License.
 ***********************************************************************/

import { useRef, useCallback } from 'react';
import {
  dominantContourDirectionPlaneFit,
  Polygon as AspectPolygon,
  TerrainTile,
} from '../../../utils/terrainAspectHybrid';
import { fetchTilesForPolygon } from '../utils/terrain';
import { calculateOptimalTerrainZoom } from '../utils/geometry';
import { PolygonAnalysisResult } from '../types';

interface UsePolygonAnalysisProps {
  mapboxToken: string;
  sampleStep: number;
  onAnalysisStart?: (polygonId: string) => void;
  onAnalysisComplete?: (result: PolygonAnalysisResult, tiles: TerrainTile[]) => void;
  onError?: (error: string, polygonId?: string) => void;
}

export function usePolygonAnalysis({
  mapboxToken,
  sampleStep,
  onAnalysisStart,
  onAnalysisComplete,
  onError,
}: UsePolygonAnalysisProps) {
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map());

  const analyzePolygon = useCallback(
    async (polygonId: string, feature: any) => {
      const existingController = abortControllersRef.current.get(polygonId);
      if (existingController) {
        existingController.abort();
      }

      const controller = new AbortController();
      abortControllersRef.current.set(polygonId, controller);
      const signal = controller.signal;

      const ring = feature.geometry.coordinates[0];
      const polygon: AspectPolygon = { coordinates: ring as [number, number][] };

      try {
        onAnalysisStart?.(polygonId);
        console.log(`Starting analysis for polygon ${polygonId}`);

        const optimalTerrainZoom = calculateOptimalTerrainZoom(polygon);
        console.log(`Using terrain zoom ${optimalTerrainZoom} for polygon ${polygonId}`);

        const tiles = await fetchTilesForPolygon(polygon, optimalTerrainZoom, mapboxToken, signal);
        console.log(`Fetched ${tiles.length} tiles for polygon ${polygonId}`);

        if (signal.aborted) return;

        if (!tiles.length) {
          console.warn(`No terrain tiles found for polygon ${polygonId}`);
          onError?.('Terrain tiles not found – polygon outside coverage?', polygonId);
          return;
        }

        console.log(`Running terrain analysis for polygon ${polygonId}...`);
        const result = dominantContourDirectionPlaneFit(polygon, tiles, {
          sampleStep,
        });
        console.log(`Analysis result for polygon ${polygonId}:`, result);

        if (signal.aborted) return;

        if (!Number.isFinite(result.contourDirDeg)) {
          const errorMsg =
            result.fitQuality === 'poor'
              ? 'Could not determine reliable direction (insufficient data or flat terrain)'
              : 'Could not determine aspect (flat terrain?)';
          onError?.(errorMsg, polygonId);
          return;
        }

        const polygonResult: PolygonAnalysisResult = {
          polygonId,
          result,
          polygon,
          terrainZoom: optimalTerrainZoom,
        };

        onAnalysisComplete?.(polygonResult, tiles);
      } catch (error) {
        if (error instanceof Error && (error.message.includes('cancelled') || error.message.includes('aborted'))) {
          return;
        }
        const errorMsg = error instanceof Error ? error.message : 'Analysis failed';
        onError?.(errorMsg, polygonId);
      } finally {
        abortControllersRef.current.delete(polygonId);
      }
    },
    [mapboxToken, sampleStep, onAnalysisStart, onAnalysisComplete, onError]
  );

  const cancelAnalysis = useCallback((polygonId: string) => {
    const controller = abortControllersRef.current.get(polygonId);
    if (controller) {
      controller.abort();
    }
  }, []);

  const cancelAllAnalyses = useCallback(() => {
    abortControllersRef.current.forEach((controller) => controller.abort());
  }, []);

  return { analyzePolygon, cancelAnalysis, cancelAllAnalyses };
}
