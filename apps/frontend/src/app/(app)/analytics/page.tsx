"use client";

import { useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Legend,
} from "recharts";
import { Download } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { SectionCard, StatusBadge } from "@/components/common/Status";
import { Button } from "@/components/ui/button";

const healthTrend = [
  { day: "Mon", score: 84, rid: 5 },
  { day: "Tue", score: 86, rid: 4 },
  { day: "Wed", score: 82, rid: 6 },
  { day: "Thu", score: 79, rid: 7 },
  { day: "Fri", score: 74, rid: 9 },
  { day: "Sat", score: 71, rid: 11 },
  { day: "Sun", score: 68, rid: 12 },
];

const flightHours = [
  { month: "Apr", hours: 182 },
  { month: "May", hours: 210 },
  { month: "Jun", hours: 195 },
  { month: "Jul", hours: 236 },
  { month: "Aug", hours: 271 },
  { month: "Sep", hours: 158 },
];

const faultMix = [
  { name: "High EGT", value: 6, color: "#f1b955" },
  { name: "Low Oil Pressure", value: 4, color: "#ef626f" },
  { name: "High Vibration", value: 3, color: "#2f80ed" },
  { name: "Normal", value: 24, color: "#36c98f" },
];

const alerTrend = [
  { week: "W1", critical: 1, high: 3, med: 5 },
  { week: "W2", critical: 0, high: 2, med: 4 },
  { week: "W3", critical: 2, high: 4, med: 6 },
  { week: "W4", critical: 1, high: 3, med: 4 },
];

export default function AnalyticsPage() {
  const [range, setRange] = useState("7 days");
  const [uav, setUav] = useState("All UAVs");
  const multiplier = range === "30 days" ? 1.08 : range === "24 hours" ? 0.94 : 1;
  const trend = useMemo(() => healthTrend.map((row) => ({ ...row, score: Math.round(row.score * multiplier) })), [multiplier]);
  return (
    <div>
      <PageHeader
        title="Analytics"
        subtitle="Fleet and engine long-term insight"
        actions={<><select aria-label="Analytics period" value={range} onChange={(e) => setRange(e.target.value)} className="h-8 rounded border border-border bg-card px-2 text-xs text-foreground"><option>24 hours</option><option>7 days</option><option>30 days</option></select><select aria-label="UAV filter" value={uav} onChange={(e) => setUav(e.target.value)} className="h-8 rounded border border-border bg-card px-2 text-xs text-foreground"><option>All UAVs</option><option>UAV-01</option><option>UAV-02</option><option>UAV-03</option></select><Button variant="outline" size="sm"><Download className="size-3.5" /> Export Report</Button></>}
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <SectionCard
            title={`Fleet Health Trend (${range} · ${uav})`}
            action={<StatusBadge status="INFO" />}
          >
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={trend}>
                <CartesianGrid stroke="#2a3a4f" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="day" stroke="#9aa8ba" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#9aa8ba" fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} />
                <Tooltip contentStyle={{ background: "#1b293b", border: "1px solid #2a3a4f", borderRadius: 6, fontSize: 12 }} labelStyle={{ color: "#9aa8ba" }} />
                <Line type="monotone" dataKey="score" name="Health Score" stroke="#2f80ed" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </SectionCard>
        </div>

        <SectionCard title="Fault Distribution">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={faultMix} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80} paddingAngle={2}>
                {faultMix.map((e) => <Cell key={e.name} fill={e.color} />)}
              </Pie>
              <Legend wrapperStyle={{ fontSize: 11, color: "#9aa8ba" }} />
              <Tooltip contentStyle={{ background: "#1b293b", border: "1px solid #2a3a4f", borderRadius: 6, fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </SectionCard>

        <SectionCard title="Flight Hours (Monthly)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={flightHours}>
              <CartesianGrid stroke="#2a3a4f" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="month" stroke="#9aa8ba" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#9aa8ba" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: "#1b293b", border: "1px solid #2a3a4f", borderRadius: 6, fontSize: 12 }} labelStyle={{ color: "#9aa8ba" }} cursor={{ fill: "rgba(47,128,237,0.08)" }} />
              <Bar dataKey="hours" fill="#39d0c8" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </SectionCard>

        <div className="lg:col-span-2">
          <SectionCard title="Alert Volume by Week & Severity">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={alerTrend}>
                <CartesianGrid stroke="#2a3a4f" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="week" stroke="#9aa8ba" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#9aa8ba" fontSize={11} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "#1b293b", border: "1px solid #2a3a4f", borderRadius: 6, fontSize: 12 }} labelStyle={{ color: "#9aa8ba" }} cursor={{ fill: "rgba(47,128,237,0.08)" }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="critical" name="Critical" stackId="a" fill="#ef626f" />
                <Bar dataKey="high" name="High" stackId="a" fill="#f1b955" />
                <Bar dataKey="med" name="Medium" stackId="a" fill="#2f80ed" />
              </BarChart>
            </ResponsiveContainer>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
