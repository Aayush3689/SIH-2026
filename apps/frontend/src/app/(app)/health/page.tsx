"use client";

import { useEffect, useMemo, useState, Suspense } from "react";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer } from "recharts";
import { Gauge, ShieldAlert, BrainCircuit, Activity, Wrench } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { PageHeader } from "@/components/layout/PageHeader";
import { Kpi, SectionCard, StatusBadge, MeterBar } from "@/components/common/Status";
import { Button } from "@/components/ui/button";
import type { EngineDiagnosis } from "@/lib/types";
import { mockUavs } from "@/data/mock";
import { useLiveTelemetry } from "@/hooks/use-live-telemetry";
import { runInference } from "@/lib/api/inference";

const HEALTH_AXIS = [
  { subject: "Thermal", value: 82, color: "#2f80ed" },
  { subject: "Oil", value: 74, color: "#39d0c8" },
  { subject: "Vibration", value: 88, color: "#36c98f" },
  { subject: "Fuel", value: 71, color: "#f1b955" },
  { subject: "RPM", value: 85, color: "#9aa8ba" },
  { subject: "Pressure", value: 64, color: "#ef626f" },
];

export default function HealthPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted-foreground">Loading engine health...</div>}>
      <HealthContent />
    </Suspense>
  );
}

function HealthContent() {
  const searchParams = useSearchParams();
  const selectedId = searchParams.get("uav") ?? "UAV-01";
  const live = useLiveTelemetry(mockUavs, 3000);
  const uav = live.find((u) => u.id === selectedId) ?? live[0];
  const [diagnosis, setDiagnosis] = useState<EngineDiagnosis | null>(null);

  useEffect(() => {
    runInference(uav.id).then((r) => setDiagnosis(r.diagnosis));
  }, [uav.id]);

  const healthPct =
    diagnosis?.rulHours != null
      ? Math.min(100, Math.round((diagnosis.rulHours / 186) * 100))
      : 68;

  const radar = useMemo(() => {
    const scale =
      diagnosis?.faultType === "COMBINED" ? 0.45 : diagnosis?.faultType === "HIGH_EGT" ? 0.62 : diagnosis?.faultType === "HIGH_VIBRATION" ? 0.55 : 1;
    return HEALTH_AXIS.map((h) => ({ ...h, value: Math.round(h.value * scale) }));
  }, [diagnosis]);

  const rulHrs = diagnosis?.rulHours ?? 0;
  const rulColor = rulHrs < 8 ? "#ef626f" : rulHrs < 24 ? "#f1b955" : "#36c98f";

  return (
    <div>
      <PageHeader
        title="Engine Health"
        subtitle={`Deep engine health · ${uav.id} · ${uav.engineModel}`}
        actions={
          <>
            <StatusBadge status={uav.healthState} />
            <StatusBadge status={diagnosis?.anomalyDetected ? "ACTIVE_FAULT" : "NORMAL"} />
            <Button variant="outline" size="sm"><Gauge className="size-3.5" /> Run Diagnosis</Button>
          </>
        }
      />

      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Kpi label="Health Score" value={`${healthPct}%`} tone={healthPct > 70 ? "success" : healthPct > 40 ? "warning" : "danger"} icon={<Activity className="size-4" />} />
        <Kpi label="RUL" value={diagnosis ? `${rulHrs}h` : "—"} tone={rulHrs < 24 ? "warning" : "default"} icon={<Gauge className="size-4" />} />
        <Kpi label="Fault Class" value={diagnosis?.faultType.replace(/_/g, " ") ?? "—"} tone={diagnosis?.faultType === "NONE" ? "success" : "danger"} icon={<ShieldAlert className="size-4" />} />
        <Kpi label="Confidence" value={diagnosis ? `${diagnosis.confidence}%` : "—"} tone="info" icon={<BrainCircuit className="size-4" />} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Radar health */}
        <SectionCard title="Health Signature" className="lg:col-span-1">
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radar} outerRadius={90}>
              <PolarGrid stroke="#2a3a4f" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: "#9aa8ba", fontSize: 11 }} />
              <Radar dataKey="value" stroke="#2f80ed" fill="#2f80ed" fillOpacity={0.25} />
            </RadarChart>
          </ResponsiveContainer>
        </SectionCard>

        {/* System indicators */}
        <SectionCard title="Engine System Indicators" className="lg:col-span-2">
          <div className="space-y-3.5">
            {radar.map((s) => {
              const val = uav.lastTelemetry;
              const metric =
                s.subject === "Thermal" ? `${Math.round(val.egt)}°C`
                : s.subject === "Oil" ? `${val.oilTemp.toFixed(1)}°C / ${val.oilPressure.toFixed(2)}bar`
                : s.subject === "Vibration" ? `${val.vibrationX.toFixed(2)}g`
                : s.subject === "Fuel" ? `${val.fuelFlow.toFixed(1)} L/h`
                : s.subject === "RPM" ? val.engineRpm.toLocaleString()
                : "nominal";
              const barColor = s.value > 70 ? "#36c98f" : s.value > 45 ? "#f1b955" : "#ef626f";
              return (
                <div key={s.subject}>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">{s.subject}</span>
                    <span className="font-mono text-foreground">{metric}</span>
                  </div>
                  <MeterBar value={s.value} color={barColor} className="h-1.5" />
                </div>
              );
            })}
          </div>
        </SectionCard>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
        {/* Diagnosis */}
        <SectionCard
          title="Diagnosis"
          action={<BadgeAI anomaly={diagnosis?.anomalyDetected} />}
          className="lg:col-span-2"
        >
          {diagnosis ? (
            <div className="space-y-2.5 text-sm">
              <div className="flex items-start gap-3">
                <div className="mt-0.5 flex size-8 items-center justify-center rounded bg-primary/15 text-primary">
                  <BrainCircuit className="size-4" />
                </div>
                <div>
                  <div className="font-medium text-foreground">
                    {diagnosis.anomalyDetected ? "Anomaly detected" : "No anomalies detected"}
                  </div>
                  <div className="mt-0.5 text-xs text-muted-foreground">{diagnosis.recommendedAction}</div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 border-t border-border pt-3 sm:grid-cols-4">
                <div className="rounded bg-muted/40 p-2 text-center">
                  <div className="text-[10px] uppercase text-muted-foreground">Severity</div>
                  <div className="font-mono text-sm font-bold" style={{ color: rulColor }}>{diagnosis.severity}</div>
                </div>
                <div className="rounded bg-muted/40 p-2 text-center">
                  <div className="text-[10px] uppercase text-muted-foreground">Confidence</div>
                  <div className="font-mono text-sm font-bold text-status-info">{diagnosis.confidence}%</div>
                </div>
                <div className="rounded bg-muted/40 p-2 text-center">
                  <div className="text-[10px] uppercase text-muted-foreground">RUL</div>
                  <div className="font-mono text-sm font-bold" style={{ color: rulColor }}>{diagnosis.rulHours}h</div>
                </div>
                <div className="rounded bg-muted/40 p-2 text-center">
                  <div className="text-[10px] uppercase text-muted-foreground">Model</div>
                  <div className="font-mono text-sm font-bold text-status-primary">v2.4</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="py-6 text-center text-xs text-muted-foreground">Loading diagnosis...</div>
          )}
        </SectionCard>

        {/* Advisory */}
        <SectionCard title="Maintenance Advisory">
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <Wrench className="size-4 text-status-primary" />
              <span className="text-muted-foreground">Advisory level</span>
              <StatusBadge status={diagnosis?.anomalyDetected ? "HIGH" : "LOW"} />
            </div>
            <div className="rounded border border-border bg-muted/30 p-2.5 text-xs text-muted-foreground">
              {diagnosis?.recommendedAction ?? "Awaiting inference output..."}
            </div>
            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Next inspection</span>
                <span className="font-medium text-foreground">{diagnosis?.rulHours != null && diagnosis.rulHours < 24 ? "Immediate" : "24h"}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Recommended window</span>
                <span className="font-medium text-foreground">{diagnosis?.rulHours} flight h</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Parts required</span>
                <span className="font-medium text-foreground">—</span>
              </div>
            </div>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}

function BadgeAI({ anomaly }: { anomaly?: boolean }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[10px] font-medium ${
        anomaly ? "text-status-danger" : "text-status-normal"
      }`}
      style={{ backgroundColor: anomaly ? "#ef626f1a" : "#36c98f1a" }}
    >
      <BrainCircuit className="size-3" /> AI + Physics
    </span>
  );
}