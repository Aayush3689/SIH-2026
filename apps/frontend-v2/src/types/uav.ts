/** Operational state shown for a vehicle in the fleet console. */
export type UavStatus = "active" | "warning" | "critical" | "offline";

export type MissionStatus =
  | "patrolling"
  | "en-route"
  | "returning"
  | "idle"
  | "maintenance";

export interface GeoPosition {
  latitude: number;
  longitude: number;
  altitude_ft: number;
}

export interface Uav {
  id: string;
  callsign: string;
  name: string;
  model: string;
  manufacturer: string;
  engine_model: string;
  mission_profile: string;
  description?: string;
  status: UavStatus;
  mission_status: MissionStatus;
  position: GeoPosition;
  heading_deg: number;
  speed_kts: number;
  battery_pct: number;
  signal_strength_pct: number;
  engine_id: string;
  gateway_id: string | null;
  last_seen: string;
}

export interface FleetSummary {
  total_uavs: number;
  connected_uavs: number;
  in_flight_uavs: number;
  active_uavs: number;
  warning_uavs: number;
  critical_uavs: number;
  offline_uavs: number;
  pre_fault_engines: number;
  active_fault_engines: number;
  low_rul_engines: number;
}
