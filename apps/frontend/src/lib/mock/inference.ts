import type {
  AnomalyPrediction,
  FaultProbabilities,
  FaultType,
  InferenceResponse,
} from "@/types/inference";
import type { TelemetrySample } from "@/types/telemetry";

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

const normalAnomaly: AnomalyPrediction = {
  state: "NORMAL",
  raw_label: 0,
  probabilities: { NORMAL: 0.97, PRE_FAULT: 0.025, ACTIVE_FAULT: 0.005 },
  confidence: 0.97,
};

function probabilities(overrides: Partial<FaultProbabilities>): FaultProbabilities {
  return {
    abnormal_vibration: 0.01,
    coking_degradation: 0.01,
    combustion_instability: 0.01,
    injector_abnormality: 0.01,
    lubrication_issue: 0.01,
    misfire: 0.01,
    overheating_trend: 0.01,
    sensor_drift: 0.01,
    ...overrides,
  };
}

function response(
  engineId: string,
  flightId: string,
  timestamp: string,
  anomaly: AnomalyPrediction,
  fault: FaultType,
  faultProbabilities: Partial<FaultProbabilities>,
  rul: number[] | null,
): InferenceResponse {
  return {
    success: true,
    data: {
      engine_id: engineId,
      flight_id: flightId,
      timestamp,
      anomaly: [anomaly],
      fault: [
        {
          fault,
          probabilities: probabilities(faultProbabilities),
          confidence: faultProbabilities[fault] ?? 0.01,
        },
      ],
      rul,
    },
  };
}

export const mockInferenceByEngine: Record<string, InferenceResponse> = {
  "ENG-F01": response(
    "ENG-F01",
    "FLT-260909-A1",
    "2026-09-09T06:14:25.000Z",
    normalAnomaly,
    "sensor_drift",
    { sensor_drift: 0.04 },
    [38.4],
  ),
  "ENG-F02": response(
    "ENG-F02",
    "FLT-260909-B2",
    "2026-09-09T06:14:25.000Z",
    {
      state: "PRE_FAULT",
      raw_label: 1,
      probabilities: { NORMAL: 0.16, PRE_FAULT: 0.72, ACTIVE_FAULT: 0.12 },
      confidence: 0.72,
    },
    "abnormal_vibration",
    { abnormal_vibration: 0.68, lubrication_issue: 0.12 },
    [13.8],
  ),
  "ENG-R03": response(
    "ENG-R03",
    "FLT-260909-C3",
    "2026-09-09T06:14:25.000Z",
    {
      state: "ACTIVE_FAULT",
      raw_label: 2,
      probabilities: { NORMAL: 0.02, PRE_FAULT: 0.11, ACTIVE_FAULT: 0.87 },
      confidence: 0.87,
    },
    "abnormal_vibration",
    { abnormal_vibration: 0.91, lubrication_issue: 0.58, overheating_trend: 0.44 },
    [6.31],
  ),
  "ENG-K04": response(
    "ENG-K04",
    "FLT-260909-D4",
    "2026-09-09T05:48:12.000Z",
    normalAnomaly,
    "sensor_drift",
    { sensor_drift: 0.02 },
    null,
  ),
};

const fallbackInference = response(
  "ENG-DEMO",
  "FLT-DEMO",
  "2026-09-09T06:14:25.000Z",
  normalAnomaly,
  "sensor_drift",
  { sensor_drift: 0.04 },
  null,
);

export function getMockInference(telemetry: TelemetrySample[]): InferenceResponse {
  const latest = telemetry[telemetry.length - 1];
  const inference = latest ? mockInferenceByEngine[latest.engine_id] : fallbackInference;
  const result = clone(inference ?? fallbackInference);

  if (latest && !inference) {
    result.data.engine_id = latest.engine_id;
    result.data.flight_id = latest.flight_id;
    result.data.timestamp = latest.timestamp;
  }

  // The mock follows the live backend rule: no numerical RUL before 8 samples.
  if (telemetry.length < 8) result.data.rul = null;

  return result;
}
