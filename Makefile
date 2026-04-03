.PHONY: build-TerrainSplitterExactFunction

build-TerrainSplitterExactFunction:
	mkdir -p "$(ARTIFACTS_DIR)"
	npm ci
	EXACT_RUNTIME_OUTFILE="$(ARTIFACTS_DIR)/index.js" \
	EXACT_RUNTIME_WORKER_OUTFILE="$(ARTIFACTS_DIR)/tileWorker.node.mjs" \
	node scripts/build-exact-runtime.mjs
