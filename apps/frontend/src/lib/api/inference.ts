import { ApiRequestError, type ApiClientOptions, requestAiBackend } from "@/lib/api/health";
import type { InferenceRequest, InferenceResponse } from "@/types/inference";
import type { TelemetrySample } from "@/types/telemetry";

function isInferenceResponse(value: unknown): value is InferenceResponse {
  if (typeof value !== "object" || value === null) return false;

  const candidate = value as Record<string, unknown>;
  if (candidate.success !== true || typeof candidate.data !== "object" || candidate.data === null) return false;

  const data = candidate.data as Record<string, unknown>;
  return (
    typeof data.engine_id === "string" &&
    typeof data.flight_id === "string" &&
    typeof data.timestamp === "string" &&
    Array.isArray(data.anomaly) &&
    Array.isArray(data.fault) &&
    (data.rul === null || (Array.isArray(data.rul) && data.rul.every((value) => typeof value === "number")))
  );
}

/** POST /api/v1/inference — submits one or more telemetry samples for inference. */
export async function getInference(
  telemetry: TelemetrySample[],
  options: ApiClientOptions = {},
): Promise<InferenceResponse> {
  const payload: InferenceRequest = { telemetry };
  const response = await requestAiBackend<unknown>(
    "/api/v1/inference",
    {
      method: "POST",
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    options,
  );

  if (!isInferenceResponse(response)) {
    throw new ApiRequestError("AI backend returned an invalid inference response", undefined, response);
  }

  return response;
}
