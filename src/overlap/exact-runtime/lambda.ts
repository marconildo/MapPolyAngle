import { handleExactRuntimeRequest } from "./service";
import type { ExactRuntimeRequest } from "./protocol";

export async function handler(event: ExactRuntimeRequest) {
  return handleExactRuntimeRequest(event);
}
