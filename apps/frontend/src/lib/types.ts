export type EnvMode = "demo" | "live";

export type FlightState = "GROUND" | "TAXI" | "TAKEOFF" | "CRUISE" | "LANDING" | "IDLE";

export type EngineHealthState = "NORMAL" | "PRE_FAULT" | "ACTIVE_FAULT";

export type AlertSeverity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

export type AlertStatus = "ACTIVE" | "ACKNOWLEDGED" | "RESOLVED";

export type SourceType = "GCS" | "REAL_UAV" | "SIMULATOR";

export type Protocol =
  | "TCP"
  | "UDP"
  | "REST_API"
  | "MAVLink"
  | "Own Protocol";

export type ConnectionState =
  | "IDLE"
  | "VALIDATING"
  | "CONNECTING"
  | "RECEIVING_DATA"
  | "CONNECTED"
  | "TIMEOUT"
  | "CONNECTION_REFUSED"
  | "AUTH_FAILED"
  | "NO_DATA"
  | "PARSER_ERROR";

export interface TelemetryPoint {
  timestamp: number;
  engineRpm: number;
  oilTemp: number;
  oilPressure: number;
  egt: number;
  fuelFlow: number;
  vibrationX: number;
  vibrationY: number;
  vibrationZ: number;
}

export interface Uav {
  id: string;
  name: string;
  type: string;
  engineId: string;
  engineModel: string;
  manufacturer: string;
  mission: string;
  flightState: FlightState;
  connected: boolean;
  healthState: EngineHealthState;
  rulHours: number;
  lastTelemetry: TelemetryPoint;
  sourceId?: string;
}

export interface EngineDiagnosis {
  anomalyDetected: boolean;
  faultType: "NONE" | "HIGH_EGT" | "LOW_OIL_PRESSURE" | "HIGH_VIBRATION" | "COMBINED";
  confidence: number;
  rulHours: number;
  recommendedAction: string;
  severity: AlertSeverity;
}

export interface Alert {
  id: string;
  severity: AlertSeverity;
  status: AlertStatus;
  uavId: string;
  engineId: string;
  type: string;
  message: string;
  timestamp: string;
  acknowledgedAt?: string;
  resolvedAt?: string;
}

export interface Adapter {
  id: string;
  name: string;
  protocol: Protocol;
  source: SourceType;
  status: ConnectionState;
  host?: string;
  port?: number;
  packetRate?: number;
  lastPacket?: string;
  latencyMs?: number;
  mappingCount: number;
}

export interface MappingRule {
  rawField: string;
  standardParameter: string;
  unit: string;
  required: boolean;
  valid: boolean;
}

export interface DataSourceConnection {
  id: string;
  uavId: string;
  source: SourceType;
  protocol: Protocol;
  host?: string;
  port?: number;
  status: ConnectionState;
  mappedParams: number;
  totalParams: number;
  packetRate: number;
  lastUpdate: string;
}
