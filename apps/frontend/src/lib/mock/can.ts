import type { CanFrame, CanInterface } from "@/types/gateway";

export const mockCanInterfaces: CanInterface[] = [
  {
    id: "SIM-CAN-0",
    interface: "can0",
    type: "CAN",
    status: "CONNECTED",
    bitrate: 500_000,
    framesPerSecond: 132,
    busLoadPercent: 18.4,
    source: "SIMULATOR",
  },
  {
    id: "SIM-VCAN-0",
    interface: "vcan0",
    type: "CAN",
    status: "VIRTUAL",
    bitrate: 500_000,
    framesPerSecond: 84,
    busLoadPercent: 11.8,
    source: "SIMULATOR",
  },
  {
    id: "SIM-CAN-1",
    interface: "can1",
    type: "CAN",
    status: "AVAILABLE",
    bitrate: 250_000,
    framesPerSecond: 0,
    busLoadPercent: 0,
    source: "SIMULATOR",
  },
];

export const mockCanFrames: CanFrame[] = [
  {
    id: "CAN-0001",
    timestamp: "2026-09-09T06:14:45.000Z",
    gateway_id: "GW-SOUTH-02",
    uav_id: "UAV-003",
    direction: "rx",
    arbitration_id: "0x18FF50E5",
    data: "D0 07 CF 02 35 01 6E 00",
    dlc: 8,
    label: "Engine telemetry",
  },
  {
    id: "CAN-0002",
    timestamp: "2026-09-09T06:14:46.000Z",
    gateway_id: "GW-SOUTH-02",
    uav_id: "UAV-003",
    direction: "rx",
    arbitration_id: "0x18FF51E5",
    data: "36 03 7A 01 6F 00 52 00",
    dlc: 8,
    label: "Oil and fuel telemetry",
  },
  {
    id: "CAN-0003",
    timestamp: "2026-09-09T06:14:48.000Z",
    gateway_id: "GW-NORTH-01",
    uav_id: "UAV-002",
    direction: "rx",
    arbitration_id: "0x18FF50E5",
    data: "2B 15 B0 02 62 01 68 00",
    dlc: 8,
    label: "Engine telemetry",
  },
  {
    id: "CAN-0004",
    timestamp: "2026-09-09T06:14:50.000Z",
    gateway_id: "GW-NORTH-01",
    uav_id: "UAV-001",
    direction: "tx",
    arbitration_id: "0x0CF00400",
    data: "01 00 00 00 FF FF FF FF",
    dlc: 8,
    label: "Gateway acknowledgement",
  },
];

export function getMockCanFrames(uavId?: string): CanFrame[] {
  return uavId ? mockCanFrames.filter((frame) => frame.uav_id === uavId) : mockCanFrames;
}
