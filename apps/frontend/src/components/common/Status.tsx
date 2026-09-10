"use client";

import { cn } from "@/lib/utils";
import type { EngineHealthState, FlightState } from "@/lib/types";

export function stateColor(key: string): string {
  switch (key) {
    case "NORMAL":
    case "CONNECTED":
    case "ACTIVE":
    case "RESOLVED":
    case "GROUND":
    case "READY":
    case "OK":
      return "#36c98f";
    case "PRE_FAULT":
    case "ACKNOWLEDGED":
    case "MEDIUM":
    case "TAXI":
    case "IDLE":
    case "RECEIVING_DATA":
    case "VALIDATING":
    case "CONNECTING":
    case "WARNING":
      return "#f1b955";
    case "ACTIVE_FAULT":
    case "CRITICAL":
    case "HIGH":
    case "TIME OUT":
    case "TIMEOUT":
    case "CONNECTION_REFUSED":
    case "AUTH_FAILED":
    case "NO_DATA":
    case "PARSER_ERROR":
      return "#ef626f";
    case "CRUISE":
    case "TAKEOFF":
    case "LANDING":
    case "INFO":
      return "#39d0c8";
    default:
      return "#9aa8ba";
  }
}

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const color = stateColor(status);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[11px] font-medium",
        className
      )}
      style={{ color, backgroundColor: `${color}1a`, border: `1px solid ${color}33` }}
    >
      <span
        className="size-1.5 rounded-full"
        style={{ backgroundColor: color, boxShadow: `0 0 4px ${color}` }}
      />
      {status.replace(/_/g, " ")}
    </span>
  );
}

export function Kpi({
  label,
  value,
  sub,
  tone = "default",
  icon,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  tone?: "default" | "success" | "warning" | "danger" | "info";
  icon?: React.ReactNode;
}) {
  const colors = {
    default: "#e8eef7",
    success: "#36c98f",
    warning: "#f1b955",
    danger: "#ef626f",
    info: "#39d0c8",
  } as const;
  return (
    <div className="ops-kpi rounded-md border border-border bg-card p-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </span>
        {icon && <span className="text-muted-foreground">{icon}</span>}
      </div>
      <div className="mt-1.5 text-2xl font-bold leading-none" style={{ color: colors[tone] }}>
        {value}
      </div>
      {sub && <div className="mt-1 text-[11px] text-muted-foreground">{sub}</div>}
    </div>
  );
}

export function SectionCard({
  title,
  children,
  action,
  className,
}: {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("ops-panel rounded-md border border-border bg-card", className)}>
      <div className="ops-panel-header flex items-center justify-between border-b border-border px-3 py-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-foreground">{title}</h3>
        {action}
      </div>
      <div className="p-3">{children}</div>
    </div>
  );
}

export type HealthStateMap = Record<EngineHealthState, number>;
export type FlightStateMap = Record<FlightState, number>;

export function MeterBar({ value, color = "#2f80ed", className }: { value: number; color?: string; className?: string }) {
  return (
    <div className={cn("flex h-1.5 w-full items-center overflow-hidden rounded-full bg-border", className)}>
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${Math.min(100, Math.max(0, value))}%`, backgroundColor: color }}
      />
    </div>
  );
}
