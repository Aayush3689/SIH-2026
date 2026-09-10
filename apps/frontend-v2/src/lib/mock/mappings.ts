import type {
  IncomingTelemetryField,
  MappingValidationSummary,
  PlatformTelemetryParameter,
  TelemetryFieldMapping,
  UavGatewayMapping,
} from "@/types/gateway";

export const mockUavGatewayMappings: UavGatewayMapping[] = [
  {
    uav_id: "UAV-001",
    gateway_id: "GW-NORTH-01",
    assigned_at: "2026-09-09T05:45:00.000Z",
    link_quality_pct: 94,
  },
  {
    uav_id: "UAV-002",
    gateway_id: "GW-NORTH-01",
    assigned_at: "2026-09-09T05:47:00.000Z",
    link_quality_pct: 87,
  },
  {
    uav_id: "UAV-003",
    gateway_id: "GW-SOUTH-02",
    assigned_at: "2026-09-09T05:51:00.000Z",
    link_quality_pct: 72,
  },
];

export function getMockGatewayMapping(uavId: string): UavGatewayMapping | undefined {
  return mockUavGatewayMappings.find((mapping) => mapping.uav_id === uavId);
}

/** Raw adapter fields shown on the left side of the data-mapping workspace. */
export const mockIncomingTelemetryFields: IncomingTelemetryField[] = [
  {
    id: "RAW-001",
    source_field: "engine_rpm",
    display_name: "Engine RPM",
    current_value: 5_219,
    unit: "rpm",
    source: "GCS",
    uav_id: "UAV-001",
    status: "valid",
  },
  {
    id: "RAW-002",
    source_field: "oil_temp",
    display_name: "Oil temperature",
    current_value: 89.3,
    unit: "°C",
    source: "GCS",
    uav_id: "UAV-001",
    status: "valid",
  },
  {
    id: "RAW-003",
    source_field: "oil_press",
    display_name: "Oil pressure",
    current_value: 379.3,
    unit: "kPa",
    source: "GCS",
    uav_id: "UAV-001",
    status: "valid",
  },
  {
    id: "RAW-004",
    source_field: "egt_sensor",
    display_name: "Exhaust gas temperature",
    current_value: 641.5,
    unit: "°C",
    source: "GCS",
    uav_id: "UAV-001",
    status: "valid",
  },
  {
    id: "RAW-005",
    source_field: "fuel_rate",
    display_name: "Fuel flow",
    current_value: 13.9,
    unit: "L/h",
    source: "GCS",
    uav_id: "UAV-001",
    status: "valid",
  },
  {
    id: "RAW-006",
    source_field: "vibration_x",
    display_name: "Engine vibration",
    current_value: 3.4,
    unit: "mm/s",
    source: "GCS",
    uav_id: "UAV-001",
    status: "valid",
  },
];

/** Target platform fields and their unit requirements. */
export const mockPlatformTelemetryParameters: PlatformTelemetryParameter[] = [
  { target_field: "rpm", display_name: "Engine RPM", expected_unit: "rpm", required: true },
  { target_field: "oil_temperature_c", display_name: "Oil Temperature", expected_unit: "°C", required: true },
  { target_field: "oil_pressure_kpa", display_name: "Oil Pressure", expected_unit: "kPa", required: true },
  { target_field: "egt_c", display_name: "Exhaust Gas Temperature", expected_unit: "°C", required: true },
  { target_field: "fuel_flow_lph", display_name: "Fuel Flow", expected_unit: "L/h", required: true },
  { target_field: "engine_vibration_mm_s", display_name: "Vibration", expected_unit: "mm/s", required: true },
  { target_field: "ambient_temp_c", display_name: "Ambient Temperature", expected_unit: "°C", required: false },
];

export const mockTelemetryMappings: TelemetryFieldMapping[] = [
  { id: "MAP-001", uav_id: "UAV-001", source_field: "engine_rpm", target_field: "rpm", status: "valid", mapped_at: "2026-09-09T05:45:10.000Z" },
  { id: "MAP-002", uav_id: "UAV-001", source_field: "oil_temp", target_field: "oil_temperature_c", status: "valid", mapped_at: "2026-09-09T05:45:12.000Z" },
  { id: "MAP-003", uav_id: "UAV-001", source_field: "oil_press", target_field: "oil_pressure_kpa", status: "valid", mapped_at: "2026-09-09T05:45:14.000Z" },
  { id: "MAP-004", uav_id: "UAV-001", source_field: "egt_sensor", target_field: "egt_c", status: "valid", mapped_at: "2026-09-09T05:45:16.000Z" },
  { id: "MAP-005", uav_id: "UAV-001", source_field: "fuel_rate", target_field: "fuel_flow_lph", status: "valid", mapped_at: "2026-09-09T05:45:18.000Z" },
  { id: "MAP-006", uav_id: "UAV-001", source_field: "vibration_x", target_field: "engine_vibration_mm_s", status: "valid", mapped_at: "2026-09-09T05:45:20.000Z" },
];

export const mockMappingValidation: MappingValidationSummary = {
  required: 6,
  mapped: 6,
  unmapped: 0,
  invalid: 0,
  is_valid: true,
};
