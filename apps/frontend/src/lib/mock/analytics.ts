export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
}

export interface FleetAnalytics {
  generated_at: string;
  fleet_health_score: number;
  mission_readiness_pct: number;
  active_alerts: number;
  telemetry_ingestion_pct: number;
  health_score_trend: TimeSeriesPoint[];
  active_airframes_trend: TimeSeriesPoint[];
  fault_distribution: Array<{ fault: string; count: number }>;
  rul_buckets: Array<{ label: ">24h" | "8–24h" | "<8h" | "Unavailable"; count: number }>;
  connection_uptime_pct: number;
}

export const mockAnalytics: FleetAnalytics = {
  generated_at: "2026-09-09T06:15:00.000Z",
  fleet_health_score: 76,
  mission_readiness_pct: 75,
  active_alerts: 3,
  telemetry_ingestion_pct: 98.7,
  health_score_trend: [
    { timestamp: "2026-09-09T05:20:00.000Z", value: 89 },
    { timestamp: "2026-09-09T05:30:00.000Z", value: 87 },
    { timestamp: "2026-09-09T05:40:00.000Z", value: 85 },
    { timestamp: "2026-09-09T05:50:00.000Z", value: 82 },
    { timestamp: "2026-09-09T06:00:00.000Z", value: 79 },
    { timestamp: "2026-09-09T06:10:00.000Z", value: 76 },
  ],
  active_airframes_trend: [
    { timestamp: "2026-09-09T05:20:00.000Z", value: 3 },
    { timestamp: "2026-09-09T05:30:00.000Z", value: 3 },
    { timestamp: "2026-09-09T05:40:00.000Z", value: 3 },
    { timestamp: "2026-09-09T05:50:00.000Z", value: 3 },
    { timestamp: "2026-09-09T06:00:00.000Z", value: 2 },
    { timestamp: "2026-09-09T06:10:00.000Z", value: 2 },
  ],
  fault_distribution: [
    { fault: "Abnormal vibration", count: 2 },
    { fault: "Lubrication issue", count: 1 },
    { fault: "Overheating trend", count: 1 },
  ],
  rul_buckets: [
    { label: ">24h", count: 1 },
    { label: "8–24h", count: 1 },
    { label: "<8h", count: 1 },
    { label: "Unavailable", count: 1 },
  ],
  connection_uptime_pct: 97.6,
};
