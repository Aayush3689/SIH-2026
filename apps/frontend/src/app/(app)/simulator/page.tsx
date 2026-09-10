"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Play, Pause, Scan, Radio, RefreshCw, Cable, HardDrive,
} from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { SectionCard, StatusBadge, MeterBar } from "@/components/common/Status";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { mockCanBuses, mockUavs } from "@/data/mock";

const SCENARIOS = ["Normal", "High EGT", "Low Oil Pressure", "High Vibration", "Combined Fault"];

export default function SimulatorPage() {
  const [scenario, setScenario] = useState("Normal");
  const [playing, setPlaying] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [buses, setBuses] = useState(mockCanBuses);
  const [detected, setDetected] = useState(false);
  const [selectedBus, setSelectedBus] = useState("CAN-0");
  const [speed, setSpeed] = useState("1x");

  const fault = scenario !== "Normal";

  const startScan = () => {
    setScanning(true);
    setScanProgress(0);
    const interval = setInterval(() => {
      setScanProgress((p) => {
        if (p >= 100) {
          clearInterval(interval);
          setScanning(false);
          setDetected(true);
          setBuses((prev) =>
            prev.map((b) => (b.id === "CAN-1" ? { ...b, detected: true, frameRate: 250, devices: ["ECU-3", "ECU-4"] } : b))
          );
          return 100;
        }
        return p + 10;
      });
    }, 300);
  };

  return (
    <div>
      <PageHeader
        title="Simulator"
        subtitle="Controlled fault simulation and CAN telemetry source"
        actions={
          <>
            <StatusBadge status="CONNECTED" />
            <Button variant={playing ? "outline" : "default"} size="sm" onClick={() => setPlaying(!playing)}>
              {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />} {playing ? "Stop" : "Start"} Simulation
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Scenario control */}
        <div className="lg:col-span-1">
          <SectionCard title="Anomaly Scenarios">
            <div className="space-y-2">
              {SCENARIOS.map((s) => {
                const active = scenario === s;
                const danger = s === "Combined Fault" || s === "Low Oil Pressure";
                return (
                  <button
                    key={s}
                    onClick={() => setScenario(s)}
                    className={`flex w-full items-center justify-between rounded border px-3 py-2 text-sm transition-colors ${
                      active
                        ? danger
                          ? "border-danger/50 bg-danger/10 text-danger"
                          : "border-primary/50 bg-primary/10 text-primary"
                        : "border-border bg-muted/30 text-foreground hover:bg-muted/50"
                    }`}
                  >
                    <span className="font-medium">{s}</span>
                    {active && <span className="size-2 rounded-full bg-current" />}
                  </button>
                );
              })}
            </div>

            <div className="mt-4 rounded border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
              <div className="flex items-center gap-2 font-medium text-foreground">
                <Radio className="size-3.5 text-status-info" /> Active Scenario
              </div>
              <div className="mt-2 font-mono text-lg text-foreground">{scenario}</div>
              <p className="mt-1">
                {fault
                  ? "Signal deviation injected. Mock diagnosis confidence 87%. Maintenance action escalated."
                  : "All observed signals align with the simulated digital-twin baseline."}
              </p>
            </div>
          </SectionCard>
        </div>

        {/* Playback + telemetry */}
        <div className="lg:col-span-2">
          <SectionCard title="Simulator Telemetry Preview" action={<div className="flex items-center gap-2"><label className="text-[10px] text-muted-foreground">Speed <select aria-label="Playback speed" value={speed} onChange={(e) => setSpeed(e.target.value)} className="ml-1 rounded border border-border bg-muted px-1 py-0.5 text-foreground"><option>0.5x</option><option>1x</option><option>2x</option></select></label><Badge variant="outline" className="text-[10px] text-muted-foreground">{playing ? "streaming" : "stopped"}</Badge></div>}>
            <div className="grid grid-cols-3 gap-3">
              {([
                { label: "Engine RPM", value: fault ? 6240 : 8420, pct: fault ? 74 : 100 },
                { label: "EGT", value: fault ? 829 : 648, pct: fault ? 83 : 64 },
                { label: "Oil Pressure", value: fault ? 2.1 : 4.2, pct: fault ? 50 : 100 },
              ] as { label: string; value: number; pct: number }[]).map((m) => (
                <div key={m.label} className="rounded border border-border bg-muted/30 p-3">
                  <div className="text-[10px] uppercase text-muted-foreground">{m.label}</div>
                  <div className={`font-mono text-xl font-bold ${fault && m.label !== "Engine RPM" ? "text-status-danger" : "text-foreground"}`}>
                    {m.value}{m.label === "EGT" ? "°C" : m.label === "Oil Pressure" ? "bar" : ""}
                  </div>
                  <MeterBar value={m.pct} color={fault && m.label !== "Engine RPM" ? "#ef626f" : "#36c98f"} className="mt-2 h-1" />
                </div>
              ))}
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
              <span>Packet rate: {speed === "2x" ? "200" : speed === "0.5x" ? "50" : "100"}/s</span>
              <span>Frame: 09,847</span>
              <span>Δt: 10ms</span>
            </div>
          </SectionCard>

          {/* CAN detection */}
          <div className="mt-4">
            <SectionCard
              title="CAN Bus Detection"
              action={
                <Button variant="outline" size="xs" onClick={startScan} disabled={scanning}>
                  {scanning ? <RefreshCw className="size-3 animate-spin" /> : <Scan className="size-3" />}
                  {scanning ? "Scanning..." : "Scan Bus"}
                </Button>
              }
            >
              {scanning ? (
                <div className="py-2">
                  <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                    <span>Scanning CAN interfaces...</span><span>{scanProgress}%</span>
                  </div>
                  <MeterBar value={scanProgress} className="h-1.5" />
                </div>
              ) : (
                <div className="space-y-2">
                  {buses.map((b) => (
                    <motion.div
                      key={b.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      onClick={() => b.detected && setSelectedBus(b.id)}
                      className={`flex items-center justify-between rounded border px-3 py-2 text-xs ${b.detected ? "cursor-pointer" : ""} ${selectedBus === b.id ? "border-primary bg-primary/10" : "border-border bg-muted/30"}`}
                    >
                      <div className="flex items-center gap-2">
                        <Cable className={`size-3.5 ${b.detected ? "text-status-normal" : "text-muted-foreground"}`} />
                        <span className="font-medium text-foreground">{b.id}</span>
                        <span className="text-muted-foreground">{Math.round(b.bitrate / 1000)} kbit/s</span>
                      </div>
                      <div className="flex items-center gap-3">
                        {b.detected ? (
                          <>
                            <span className="font-mono text-muted-foreground">{b.frameRate} fps · {b.devices.join(", ")}</span>
                            <StatusBadge status="CONNECTED" />
                          </>
                        ) : (
                          <StatusBadge status="NO_DATA" />
                        )}
                      </div>
                    </motion.div>
                  ))}
                  {detected && (
                    <div className="flex items-center justify-between rounded border border-success/40 bg-success/10 px-3 py-2 text-xs text-success">
                      <span className="flex items-center gap-2"><HardDrive className="size-3.5" /> {mockUavs[2].id} simulator profile ready on {selectedBus}</span>
                      <StatusBadge status="READY" />
                    </div>
                  )}
                </div>
              )}
            </SectionCard>
          </div>
        </div>
      </div>
    </div>
  );
}
