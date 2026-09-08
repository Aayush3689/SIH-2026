import type { TelemetrySample } from "@/types/telemetry";

export type AnomalyState = "NORMAL" | "PRE_FAULT" | "ACTIVE_FAULT";

export type FaultType =
  | "abnormal_vibration"
  | "coking_degradation"
  | "combustion_instability"
  | "injector_abnormality"
  | "lubrication_issue"
  | "misfire"
  | "overheating_trend"
  | "sensor_drift";

export type AnomalyProbabilities = Record<AnomalyState, number>;
export type FaultProbabilities = Record<FaultType, number>;

export interface AnomalyPrediction {
  state: AnomalyState;
  raw_label: number;
  probabilities: AnomalyProbabilities;
  confidence: number;
}

export interface FaultPrediction {
  fault: FaultType;
  probabilities: FaultProbabilities;
  confidence: number;
}

export interface InferenceRequest {
  telemetry: TelemetrySample[];
}

export interface InferenceData {
  engine_id: string;
  flight_id: string;
  timestamp: string;
  anomaly: AnomalyPrediction[];
  fault: FaultPrediction[];
  /** Null means the backend has not collected enough samples to estimate RUL. */
  rul: number[] | null;
}

export interface InferenceResponse {
  success: true;
  data: InferenceData;
}

export interface HealthResponse {
  status: "ok";
  service: "ai-backend";
}
