export type CoverageAutoRunReason = "lines" | "spacing" | "alt" | "manual";

export type CoverageAutoRunRequest = {
  polygonId?: string;
  reason?: CoverageAutoRunReason;
};

export type CoverageComputeRequest = {
  polygonId?: string;
  suppressMapNotReadyToast?: boolean;
};

export type CoverageAutoRunPlan =
  | { kind: "noop"; nextRetryCount: number }
  | { kind: "compute"; nextRetryCount: number; computeRequest: CoverageComputeRequest }
  | { kind: "retry"; nextRetryCount: number; delayMs: number };

export type CoverageAutoRunInputs = {
  request?: CoverageAutoRunRequest;
  nowMs: number;
  suppressAutoRunUntilMs: number;
  autoGenerate: boolean;
  importedPosesCount: number;
  ready: boolean;
  havePolys: boolean;
  haveLines: boolean;
  haveLidarPolys: boolean;
  posesCount: number;
  retryCount: number;
  retryLimit?: number;
};

const DEFAULT_RETRY_DELAY_MS = 250;
const DEFAULT_RETRY_LIMIT = 15;

export function planCoverageAutoRun(inputs: CoverageAutoRunInputs): CoverageAutoRunPlan {
  const retryLimit = inputs.retryLimit ?? DEFAULT_RETRY_LIMIT;
  if (inputs.suppressAutoRunUntilMs > inputs.nowMs) {
    return { kind: "noop", nextRetryCount: inputs.retryCount };
  }

  if (!inputs.autoGenerate && inputs.importedPosesCount > 0) {
    if (inputs.ready) {
      return {
        kind: "compute",
        nextRetryCount: 0,
        computeRequest: { suppressMapNotReadyToast: true },
      };
    }
    if (inputs.retryCount < retryLimit) {
      return { kind: "retry", nextRetryCount: inputs.retryCount + 1, delayMs: DEFAULT_RETRY_DELAY_MS };
    }
    return { kind: "noop", nextRetryCount: 0 };
  }

  if (!inputs.autoGenerate) {
    return { kind: "noop", nextRetryCount: inputs.retryCount };
  }

  if (inputs.ready && !inputs.havePolys && inputs.importedPosesCount === 0) {
    return { kind: "noop", nextRetryCount: 0 };
  }

  if (inputs.ready && inputs.havePolys && inputs.haveLines) {
    if (inputs.posesCount === 0 && !inputs.haveLidarPolys) {
      return { kind: "noop", nextRetryCount: inputs.retryCount };
    }
    const requiresGlobalRasterRefresh = Boolean(
      inputs.request?.polygonId &&
      inputs.request?.reason &&
      ["lines", "spacing", "alt"].includes(inputs.request.reason),
    );
    return {
      kind: "compute",
      nextRetryCount: 0,
      computeRequest: {
        polygonId: requiresGlobalRasterRefresh ? undefined : inputs.request?.polygonId,
        suppressMapNotReadyToast: true,
      },
    };
  }

  if (inputs.retryCount < retryLimit) {
    return { kind: "retry", nextRetryCount: inputs.retryCount + 1, delayMs: DEFAULT_RETRY_DELAY_MS };
  }

  return { kind: "noop", nextRetryCount: 0 };
}
