import type { Gateway, GatewayDataSource } from "@/types/gateway";

export const mockGateways: Gateway[] = [
  {
    id: "GW-NORTH-01",
    name: "North Operations Gateway",
    status: "online",
    location: "North Sector Control Room",
    connected_uav_count: 2,
    signal_strength_pct: 94,
    last_heartbeat: "2026-09-09T06:14:56.000Z",
    firmware_version: "2.4.1",
  },
  {
    id: "GW-SOUTH-02",
    name: "South Relay Gateway",
    status: "degraded",
    location: "South Sector Relay Tower",
    connected_uav_count: 1,
    signal_strength_pct: 72,
    last_heartbeat: "2026-09-09T06:14:54.000Z",
    firmware_version: "2.4.1",
  },
  {
    id: "GW-HANGAR-03",
    name: "Hangar Gateway",
    status: "online",
    location: "Maintenance Hangar",
    connected_uav_count: 0,
    signal_strength_pct: 99,
    last_heartbeat: "2026-09-09T06:14:57.000Z",
    firmware_version: "2.4.0",
  },
];

export function getMockGateway(gatewayId: string): Gateway | undefined {
  return mockGateways.find((gateway) => gateway.id === gatewayId);
}

/** Demo adapters used by the Gateway data-source management view. */
export const mockGatewayDataSources: GatewayDataSource[] = [
  {
    id: "SRC-GCS-001",
    name: "Falcon GCS MAVLink",
    uav_id: "UAV-001",
    type: "GCS",
    protocol: "MAVLink",
    status: "connected",
    latency_ms: 38,
    packet_rate_hz: 20,
    last_packet_at: "2026-09-09T06:14:56.000Z",
    mapping_count: 21,
  },
  {
    id: "SRC-REAL-002",
    name: "Falcon Two UDP Link",
    uav_id: "UAV-002",
    type: "REAL_UAV",
    protocol: "UDP",
    status: "connected",
    latency_ms: 46,
    packet_rate_hz: 18,
    last_packet_at: "2026-09-09T06:14:54.000Z",
    mapping_count: 21,
  },
  {
    id: "SRC-SIM-003",
    name: "Raptor CAN Simulator",
    uav_id: "UAV-003",
    type: "SIMULATOR",
    protocol: "CAN",
    status: "connecting",
    latency_ms: 71,
    packet_rate_hz: 16,
    last_packet_at: "2026-09-09T06:14:51.000Z",
    mapping_count: 18,
  },
];
