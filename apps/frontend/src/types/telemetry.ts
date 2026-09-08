/**
 * A single engine-telemetry observation accepted by POST /api/v1/inference.
 * Field names intentionally match the backend contract exactly.
 */
export interface TelemetrySample {
  timestamp: string;
  engine_id: string;
  flight_id: string;
  elapsed_s: number;
  rpm: number;
  throttle_position_pct: number;
  cht_c: number;
  egt_c: number;
  oil_pressure_kpa: number;
  oil_temperature_c: number;
  fuel_flow_lph: number;
  fuel_pressure_kpa: number;
  intake_manifold_pressure_kpa: number;
  engine_vibration_mm_s: number;
  battery_voltage_v: number;
  alternator_output_a: number;
  injection_timing_deg_btdc: number;
  engine_load_pct: number;
  ambient_temp_c: number;
  planned_altitude_ft: number;
  airspeed_kts: number;
}

export type TelemetryField = keyof TelemetrySample;

export interface TelemetrySeries {
  uav_id: string;
  samples: TelemetrySample[];
}

export interface TelemetryMetric {
  key: TelemetryField;
  label: string;
  unit: string;
  precision: number;
}
