import type { FaultType } from "@/types/inference";

export type AlertSeverity = "info" | "warning" | "critical";
export type AlertStatus = "open" | "acknowledged" | "resolved";
export type AlertSource = "inference" | "telemetry" | "gateway" | "system";

export interface Alert {
  id: string;
  uav_id: string;
  engine_id: string | null;
  source: AlertSource;
  severity: AlertSeverity;
  status: AlertStatus;
  title: string;
  message: string;
  fault_type?: FaultType;
  created_at: string;
  acknowledged_at: string | null;
}
