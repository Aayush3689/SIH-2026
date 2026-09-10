import type { TelemetryMetric, TelemetrySample, TelemetrySeries } from "@/types/telemetry";

const TELEMETRY_START_MS = Date.parse("2026-09-09T06:13:30.000Z");

interface TelemetryProfile {
  uavId: string;
  engineId: string;
  flightId: string;
  rpm: number;
  cht: number;
  egt: number;
  oilPressure: number;
  oilTemperature: number;
  vibration: number;
  load: number;
  ambient: number;
  altitude: number;
  airspeed: number;
  sampleCount: number;
}

const telemetryProfiles: TelemetryProfile[] = [
  {
    uavId: "UAV-001",
    engineId: "ENG-F01",
    flightId: "FLT-260909-A1",
    rpm: 5_120,
    cht: 171,
    egt: 638,
    oilPressure: 382,
    oilTemperature: 88,
    vibration: 3.2,
    load: 64,
    ambient: 29,
    altitude: 4_850,
    airspeed: 68,
    sampleCount: 12,
  },
  {
    uavId: "UAV-002",
    engineId: "ENG-F02",
    flightId: "FLT-260909-B2",
    rpm: 5_420,
    cht: 184,
    egt: 672,
    oilPressure: 360,
    oilTemperature: 96,
    vibration: 5.8,
    load: 72,
    ambient: 30,
    altitude: 5_120,
    airspeed: 74,
    sampleCount: 12,
  },
  {
    uavId: "UAV-003",
    engineId: "ENG-R03",
    flightId: "FLT-260909-C3",
    rpm: 4_880,
    cht: 207,
    egt: 719,
    oilPressure: 309,
    oilTemperature: 111,
    vibration: 8.7,
    load: 81,
    ambient: 31,
    altitude: 3_760,
    airspeed: 57,
    sampleCount: 12,
  },
  {
    uavId: "UAV-004",
    engineId: "ENG-K04",
    flightId: "FLT-260909-D4",
    rpm: 0,
    cht: 26,
    egt: 27,
    oilPressure: 0,
    oilTemperature: 25,
    vibration: 0.1,
    load: 0,
    ambient: 28,
    altitude: 0,
    airspeed: 0,
    sampleCount: 6,
  },
];

const round = (value: number, digits = 1): number => Number(value.toFixed(digits));

function makeSample(profile: TelemetryProfile, index: number): TelemetrySample {
  const isGrounded = profile.rpm === 0;
  const trend = isGrounded ? 0 : index;
  const oscillation = isGrounded ? 0 : (index % 3) - 1;

  return {
    timestamp: new Date(TELEMETRY_START_MS + index * 5_000).toISOString(),
    engine_id: profile.engineId,
    flight_id: profile.flightId,
    elapsed_s: index * 5,
    rpm: isGrounded ? 0 : profile.rpm + trend * 9 + oscillation * 6,
    throttle_position_pct: isGrounded ? 0 : round(profile.load * 0.78 + (index % 4) * 0.7),
    cht_c: round(profile.cht + trend * (profile.cht > 200 ? 0.55 : 0.18) + oscillation * 0.3),
    egt_c: round(profile.egt + trend * (profile.egt > 700 ? 0.9 : 0.35) + oscillation),
    oil_pressure_kpa: round(profile.oilPressure - trend * (profile.oilPressure < 330 ? 1.15 : 0.25)),
    oil_temperature_c: round(profile.oilTemperature + trend * (profile.oilTemperature > 100 ? 0.45 : 0.12)),
    fuel_flow_lph: isGrounded ? 0 : round(profile.load * 0.21 + trend * 0.04),
    fuel_pressure_kpa: isGrounded ? 0 : round(292 - trend * (profile.vibration > 8 ? 0.75 : 0.15)),
    intake_manifold_pressure_kpa: isGrounded ? 0 : round(82 + profile.load * 0.47 + oscillation * 0.8),
    engine_vibration_mm_s: round(profile.vibration + trend * (profile.vibration > 8 ? 0.22 : profile.vibration > 5 ? 0.1 : 0.02), 2),
    battery_voltage_v: isGrounded ? 12.7 : round(25.2 - trend * 0.015, 2),
    alternator_output_a: isGrounded ? 0 : round(28 + profile.load * 0.18 + oscillation * 0.4),
    injection_timing_deg_btdc: isGrounded ? 0 : round(18.4 + oscillation * 0.15, 2),
    engine_load_pct: isGrounded ? 0 : round(profile.load + (index % 5) * 0.6),
    ambient_temp_c: round(profile.ambient + index * 0.03),
    planned_altitude_ft: profile.altitude,
    airspeed_kts: isGrounded ? 0 : round(profile.airspeed + oscillation * 0.5),
  };
}

export const mockTelemetry: TelemetrySeries[] = telemetryProfiles.map((profile) => ({
  uav_id: profile.uavId,
  samples: Array.from({ length: profile.sampleCount }, (_, index) => makeSample(profile, index)),
}));

export const telemetryMetrics: TelemetryMetric[] = [
  { key: "rpm", label: "RPM", unit: "rpm", precision: 0 },
  { key: "cht_c", label: "Cylinder head temperature", unit: "°C", precision: 1 },
  { key: "egt_c", label: "Exhaust gas temperature", unit: "°C", precision: 1 },
  { key: "oil_pressure_kpa", label: "Oil pressure", unit: "kPa", precision: 1 },
  { key: "oil_temperature_c", label: "Oil temperature", unit: "°C", precision: 1 },
  { key: "engine_vibration_mm_s", label: "Engine vibration", unit: "mm/s", precision: 2 },
  { key: "battery_voltage_v", label: "Battery voltage", unit: "V", precision: 2 },
  { key: "engine_load_pct", label: "Engine load", unit: "%", precision: 1 },
];

export function getMockTelemetry(uavId: string): TelemetrySample[] {
  return mockTelemetry.find((series) => series.uav_id === uavId)?.samples ?? [];
}

export function getMockTelemetryByEngine(engineId: string): TelemetrySample[] {
  return mockTelemetry.find((series) => series.samples[0]?.engine_id === engineId)?.samples ?? [];
}
