.PHONY: build-TerrainSplitterExactFunction

build-TerrainSplitterExactFunction:
	mkdir -p "$(ARTIFACTS_DIR)"
	npm ci
	EXACT_RUNTIME_OUTFILE="$(ARTIFACTS_DIR)/index.mjs" node scripts/build-exact-runtime.mjs
