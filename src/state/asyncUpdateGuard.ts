export function shouldRunAsyncGeneration(startedGeneration: number, currentGeneration: number) {
  return startedGeneration === currentGeneration;
}

export function shouldApplyAsyncPolygonUpdate(options: {
  startedGeneration: number;
  currentGeneration: number;
  polygonStillExists: boolean;
}) {
  return shouldRunAsyncGeneration(options.startedGeneration, options.currentGeneration) && options.polygonStillExists;
}
