// Global error handler for unhandled promise rejections
export function setupGlobalErrorHandling() {
  // Handle unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    const error = event.reason;

    // Silently ignore AbortErrors and cancelled requests
    if (error instanceof Error) {
      if (error.name === 'AbortError' ||
          error.message?.includes('cancelled') ||
          error.message?.includes('aborted') ||
          error.message?.includes('Fetch is aborted')) {
        event.preventDefault();
        return;
      }
    }

    // Log other errors for debugging
    console.warn('Unhandled promise rejection:', error);
  });

  // Handle general errors
  window.addEventListener('error', (event) => {
    const error = event.error;

    // Silently ignore AbortErrors
    if (error instanceof Error && error.name === 'AbortError') {
      event.preventDefault();
      return;
    }
  });
}
