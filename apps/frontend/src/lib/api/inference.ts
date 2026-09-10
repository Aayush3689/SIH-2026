import type { EngineDiagnosis, TelemetryPoint, Alert } from "@/lib/types";
import { mockDiagnosis, mockTelemetryHistory } from "@/data/mock";

export interface InferenceResult {
  diagnosis: EngineDiagnosis;
  telemetryHistory: TelemetryPoint[];
}

export async function runInference(uavId: string): Promise<InferenceResult> {
  const diagnosis = mockDiagnosis[uavId] ?? mockDiagnosis["UAV-01"];
  const history = mockTelemetryHistory[uavId] ?? mockTelemetryHistory["UAV-01"];
  return { diagnosis, telemetryHistory: history };
}

export async function getRollupDiagnosis(): Promise<Record<string, EngineDiagnosis>> {
  return mockDiagnosis;
}

export async function getDashboardAlerts(): Promise<Alert[]> {
  const { mockAlerts } = await import("@/data/mock");
  return mockAlerts;
}
