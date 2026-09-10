"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { CheckCheck, BellOff, ShieldAlert, Clock } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Kpi, StatusBadge } from "@/components/common/Status";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { mockAlerts } from "@/data/mock";

export default function AlertsPage() {
  const [alerts, setAlerts] = useState(mockAlerts);
  const [ackIds, setAckIds] = useState<Set<string>>(new Set());

  const active = alerts.filter((a) => a.status === "ACTIVE");
  const acknowledged = alerts.filter((a) => a.status === "ACKNOWLEDGED");
  const resolved = alerts.filter((a) => a.status === "RESOLVED");

  const acknowledge = (id: string) => {
    const next = new Set(ackIds);
    next.add(id);
    setAckIds(next);
    setAlerts((prev) => prev.map((a) => (a.id === id ? { ...a, status: "ACKNOWLEDGED", acknowledgedAt: "09-Sep 08:20" } : a)));
  };

  const resolveAll = () => {
    setAlerts((prev) => prev.map((a) => (a.status === "ACTIVE" ? { ...a, status: "RESOLVED", resolvedAt: "09-Sep 08:21" } : a)));
  };

  const renderRow = (a: (typeof alerts)[number], i: number) => (
    <motion.div
      key={a.id}
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: i * 0.04 }}
      className="flex items-center justify-between border-b border-border/50 px-3 py-2.5 text-xs last:border-0 hover:bg-muted/30"
    >
      <div className="flex min-w-0 items-center gap-3">
        <StatusBadge status={a.severity} />
        <div className="min-w-0">
          <div className="font-medium text-foreground">{a.id} · {a.uavId} · {a.engineId}</div>
          <div className="truncate text-muted-foreground">{a.message}</div>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-3">
        <div className="text-right">
          <div className="text-muted-foreground">{a.type}</div>
          <div className="text-muted-foreground/70">{a.timestamp}</div>
        </div>
        <Badge variant="outline" className="text-[10px] text-muted-foreground">{a.status}</Badge>
        {a.status === "ACTIVE" && (
          <Button variant="outline" size="xs" onClick={() => acknowledge(a.id)}>
            <CheckCheck className="size-3" /> Acknowledge
          </Button>
        )}
      </div>
    </motion.div>
  );

  return (
    <div>
      <PageHeader
        title="Alerts"
        subtitle="Active, acknowledged and resolved engine events"
        actions={
          <Button variant="outline" size="sm" onClick={resolveAll}>
            <BellOff className="size-3.5" /> Resolve All Active
          </Button>
        }
      />

      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Kpi label="Active" value={active.length} tone={active.length > 0 ? "danger" : "success"} icon={<ShieldAlert className="size-4" />} />
        <Kpi label="Acknowledged" value={acknowledged.length} tone="warning" icon={<CheckCheck className="size-4" />} />
        <Kpi label="Resolved" value={resolved.length} tone="success" icon={<BellOff className="size-4" />} />
        <Kpi label="MTTA" value="4m 12s" sub="Mean time to acknowledge" icon={<Clock className="size-4" />} />
      </div>

      <div className="rounded-md border border-border bg-card">
        <Tabs defaultValue="ACTIVE">
          <div className="border-b border-border px-3 pt-2">
            <TabsList>
              <TabsTrigger value="ACTIVE">Active ({active.length})</TabsTrigger>
              <TabsTrigger value="ACKNOWLEDGED">Acknowledged ({acknowledged.length})</TabsTrigger>
              <TabsTrigger value="RESOLVED">Resolved ({resolved.length})</TabsTrigger>
            </TabsList>
          </div>
          <TabsContent value="ACTIVE" className="mt-0">
            <div className="divide-y divide-transparent">
              {active.length === 0
                ? <Empty text="No active alerts." />
                : active.map((a, i) => renderRow(a, i))}
            </div>
          </TabsContent>
          <TabsContent value="ACKNOWLEDGED" className="mt-0">
            <div className="divide-y divide-transparent">
              {acknowledged.length === 0
                ? <Empty text="No acknowledged alerts." />
                : acknowledged.map((a, i) => renderRow(a, i))}
            </div>
          </TabsContent>
          <TabsContent value="RESOLVED" className="mt-0">
            <div className="divide-y divide-transparent">
              {resolved.length === 0
                ? <Empty text="No resolved alerts." />
                : resolved.map((a, i) => renderRow(a, i))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="flex flex-col items-center gap-2 py-10 text-sm text-muted-foreground">
      <CheckCheck className="size-6 text-status-normal" />
      {text}
    </div>
  );
}