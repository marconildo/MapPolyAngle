declare module '@mapbox/mapbox-gl-draw' {
  import { IControl, Map } from 'mapbox-gl';

  interface DrawOptions {
    displayControlsDefault?: boolean;
    controls?: {
      polygon?: boolean;
      trash?: boolean;
      line_string?: boolean;
      point?: boolean;
      combine_features?: boolean;
      uncombine_features?: boolean;
    };
    defaultMode?: string;
  }

  class MapboxDraw implements IControl {
    constructor(options?: DrawOptions);

    onAdd(map: Map): HTMLElement;
    onRemove(map: Map): void;

    getAll(): {
      type: 'FeatureCollection';
      features: Array<{
        id: string;
        type: 'Feature';
        properties: any;
        geometry: {
          type: string;
          coordinates: any;
        };
      }>;
    };
    deleteAll(): void;
    delete(featureId: string): void;
    changeMode(mode: string): void;
  }

  export = MapboxDraw;
}
