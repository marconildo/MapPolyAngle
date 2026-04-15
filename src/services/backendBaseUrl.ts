declare global {
  // Test-only override used by tsx/node tests outside the Vite runtime.
  var __TERRAIN_BACKEND_URL_FOR_TESTS__: string | undefined;
}

function firstConfigured(values: Array<string | undefined>): string | undefined {
  return values.find(
    (value): value is string => typeof value === 'string' && value.trim().length > 0,
  );
}

export function configuredBackendBaseUrl(): string | undefined {
  const fromImportMeta = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_TERRAIN_PARTITION_BACKEND_URL;
  const fromGlobal = globalThis.__TERRAIN_BACKEND_URL_FOR_TESTS__;
  const fromProcess = typeof process !== 'undefined' ? process.env.VITE_TERRAIN_PARTITION_BACKEND_URL : undefined;
  return firstConfigured([fromImportMeta, fromGlobal, fromProcess]);
}

export function normalizedConfiguredBackendBaseUrl(): string | undefined {
  const configured = configuredBackendBaseUrl();
  return typeof configured === 'string' ? configured.replace(/\/$/, '') : undefined;
}

export function requireConfiguredBackendBaseUrl(errorMessage: string): string {
  const configured = normalizedConfiguredBackendBaseUrl();
  if (!configured) {
    throw new Error(errorMessage);
  }
  return configured;
}
