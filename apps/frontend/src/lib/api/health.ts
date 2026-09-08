import type { HealthResponse } from "@/types/inference";

export const DEFAULT_AI_API_BASE_URL = "http://localhost:8000";

export interface ApiClientOptions {
  /** Overrides NEXT_PUBLIC_AI_API_BASE_URL, primarily for deployment and tests. */
  baseUrl?: string;
  fetcher?: typeof fetch;
  signal?: AbortSignal;
}

export class ApiRequestError extends Error {
  readonly status: number | undefined;
  readonly body: unknown;

  constructor(message: string, status?: number, body?: unknown) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.body = body;
  }
}

function isHealthResponse(value: unknown): value is HealthResponse {
  if (typeof value !== "object" || value === null) return false;

  const candidate = value as Record<string, unknown>;
  return candidate.status === "ok" && candidate.service === "ai-backend";
}

export function getAiApiBaseUrl(baseUrl?: string): string {
  const configuredBaseUrl = baseUrl ?? process.env.NEXT_PUBLIC_AI_API_BASE_URL ?? DEFAULT_AI_API_BASE_URL;
  return configuredBaseUrl.replace(/\/+$/, "");
}

async function readResponseBody(response: Response): Promise<unknown> {
  const body = await response.text();
  if (!body) return undefined;

  try {
    return JSON.parse(body) as unknown;
  } catch {
    return body;
  }
}

/** Internal shared transport constrained to the two documented AI-backend paths. */
export async function requestAiBackend<T>(
  path: "/health" | "/api/v1/inference",
  init: RequestInit,
  options: ApiClientOptions = {},
): Promise<T> {
  const fetcher = options.fetcher ?? fetch;
  let response: Response;

  try {
    response = await fetcher(`${getAiApiBaseUrl(options.baseUrl)}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...init.headers,
      },
      signal: options.signal,
    });
  } catch (error) {
    throw new ApiRequestError(
      error instanceof Error ? `AI backend request failed: ${error.message}` : "AI backend request failed",
    );
  }

  const body = await readResponseBody(response);
  if (!response.ok) {
    throw new ApiRequestError(`AI backend returned HTTP ${response.status}`, response.status, body);
  }

  return body as T;
}

/** GET /health — verifies that the AI backend is reachable. */
export async function getHealth(options: ApiClientOptions = {}): Promise<HealthResponse> {
  const response = await requestAiBackend<unknown>("/health", { method: "GET", cache: "no-store" }, options);

  if (!isHealthResponse(response)) {
    throw new ApiRequestError("AI backend returned an invalid /health response", undefined, response);
  }

  return response;
}
