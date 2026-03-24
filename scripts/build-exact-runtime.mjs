import { build } from "esbuild";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const outfile = process.env.EXACT_RUNTIME_OUTFILE
  ? path.resolve(process.env.EXACT_RUNTIME_OUTFILE)
  : path.join(repoRoot, "backend/terrain_splitter/exact-runtime/index.js");

await fs.mkdir(path.dirname(outfile), { recursive: true });

await build({
  entryPoints: [path.join(repoRoot, "src/overlap/exact-runtime/lambda.ts")],
  outfile,
  bundle: true,
  format: "cjs",
  platform: "node",
  target: "node20",
  sourcemap: false,
  tsconfig: path.join(repoRoot, "tsconfig.json"),
  external: [
    "canvas",
  ],
});
