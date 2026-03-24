import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@shared": path.resolve(__dirname, "shared"),
    },
  },
  // No need to set root since index.html is in project root
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return;

          if (
            id.includes('mapbox-gl') ||
            id.includes('@mapbox/mapbox-gl-draw') ||
            id.includes('@deck.gl')
          ) {
            return 'map-vendor';
          }

          if (id.includes('@turf/turf')) {
            return 'terrain-vendor';
          }

          if (id.includes('recharts')) {
            return 'charts-vendor';
          }

          if (id.includes('jszip')) {
            return 'zip-vendor';
          }

          if (id.includes('egm96-universal')) {
            return 'geodesy-vendor';
          }

          if (id.includes('@radix-ui')) {
            return 'ui-vendor';
          }
        },
      },
    },
  },
  define: {
    // Make sure Mapbox token is available
    'process.env.VITE_MAPBOX_ACCESS_TOKEN': JSON.stringify(process.env.VITE_MAPBOX_ACCESS_TOKEN)
  }
});
