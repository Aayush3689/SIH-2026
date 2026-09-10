import type { TelemetryField } from "@/types/telemetry";

export type GatewayStatus = "online" | "degraded" | "offline";
export type DataSourceType = "GCS" | "REAL_UAV" | "SIMULATOR";
export type AdapterProtocol = "TCP" | "UDP" | "REST API" | "MAVLink" | "Own Protocol";
export type DataSourceStatus = "connected" | "connecting" | "error" | "disabled";

export interface Gateway {
  id: string;
  name: string;
  status: GatewayStatus;
  location: string;
  connected_uav_count: number;
  signal_strength_pct: number;
  last_heartbeat: string;
  firmware_version: string;
}

export interface UavGatewayMapping {
  uav_id: string;
  gateway_id: string;
  assigned_at: string;
  link_quality_pct: number;
}

export interface GatewayDataSource {
  id: string;
  name: string;
  uav_id: string | null;
  type: DataSourceType;
  protocol: AdapterProtocol | "CAN";
  status: DataSourceStatus;
  latency_ms: number | null;
  packet_rate_hz: number;
  last_packet_at: string | null;
  mapping_count: number;
}

export type MappingStatus = "valid" | "unmapped" | "invalid";

export interface IncomingTelemetryField {
  id: string;
  source_field: string;
  display_name: string;
  current_value: string | number;
  unit: string;
  source: DataSourceType;
  uav_id: string;
  status: MappingStatus;
}

export interface PlatformTelemetryParameter {
  target_field: TelemetryField;
  display_name: string;
  expected_unit: string;
  required: boolean;
}

export interface TelemetryFieldMapping {
  id: string;
  uav_id: string;
  source_field: string;
  target_field: TelemetryField;
  status: MappingStatus;
  mapped_at: string | null;
}

export interface MappingValidationSummary {
  required: number;
  mapped: number;
  unmapped: number;
  invalid: number;
  is_valid: boolean;
}

export type CanInterfaceStatus = "CONNECTED" | "VIRTUAL" | "AVAILABLE" | "DISCONNECTED";

export interface CanInterface {
  id: string;
  interface: string;
  type: "CAN";
  status: CanInterfaceStatus;
  bitrate: number;
  framesPerSecond: number;
  busLoadPercent: number;
  source: "SIMULATOR";
}

export type CanFrameDirection = "rx" | "tx";

export interface CanFrame {
  id: string;
  timestamp: string;
  gateway_id: string;
  uav_id: string;
  direction: CanFrameDirection;
  arbitration_id: string;
  data: string;
  dlc: number;
  label: string;
}
