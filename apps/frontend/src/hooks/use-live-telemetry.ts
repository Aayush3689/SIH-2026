"use client";

import { useEffect, useState } from "react";
import type { Uav, TelemetryPoint } from "@/lib/types";

function drift(uav: Uav, field: keyof TelemetryPoint, base: number, amplitude: number) {
  const healthy = uav.healthState === "NORMAL";
  const phase = { engineRpm: 0, oilTemp: 0.7, oilPressure: 1.4, egt: 2.1, fuelFlow: 2.8, vibrationX: 3.5, vibrationY: 4.2, vibrationZ: 4.9, timestamp: 0 }[field];
  return base + (Math.sin(Date.now() / 900 + phase) * amplitude) * (healthy ? 1 : 1.4);
}

export function useLiveTelemetry(uavs: Uav[], refreshMs = 2500) {
  const [liveUavs, setLiveUavs] = useState<Uav[]>(uavs);

  useEffect(() => {
    const baseByUav = new Map(uavs.map((u) => [u.id, u]));
    const interval = setInterval(() => {
      setLiveUavs((prev) =>
        prev.map((u) => {
          const base = baseByUav.get(u.id) ?? u;
          const t = base.lastTelemetry;
          return {
            ...u,
            lastTelemetry: {
              timestamp: Date.now(),
              engineRpm: Math.round(drift(u, "engineRpm", t.engineRpm, 40)),
              oilTemp: Math.round(drift(u, "oilTemp", t.oilTemp, 1) * 10) / 10,
              oilPressure: Math.round(drift(u, "oilPressure", t.oilPressure, 0.05) * 100) / 100,
              egt: Math.round(drift(u, "egt", t.egt, 6)),
              fuelFlow: Math.round(drift(u, "fuelFlow", t.fuelFlow, 0.3) * 100) / 100,
              vibrationX: Math.round(drift(u, "vibrationX", t.vibrationX, 0.04) * 100) / 100,
              vibrationY: Math.round(drift(u, "vibrationY", t.vibrationY, 0.04) * 100) / 100,
              vibrationZ: Math.round(drift(u, "vibrationZ", t.vibrationZ, 0.04) * 100) / 100,
            },
          };
        })
      );
    }, refreshMs);
    return () => clearInterval(interval);
  }, [uavs, refreshMs]);

  return liveUavs;
}
