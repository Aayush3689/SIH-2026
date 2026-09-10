"use client";

import { useMemo } from "react";
import { Plus, RefreshCw, PlugZap, Pause, Trash2 } from "lucide-react";
import Link from "next/link";
import { PageHeader } from "@/components/layout/PageHeader";
import { Kpi, SectionCard, StatusBadge } from "@/components/common/Status";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { mockAdapters, mockConnections, mockUavs } from "@/data/mock";

export default function GatewayPage() {
  const adapters = mockAdapters;
  const connected = adapters.filter((a) => a.status === "CONNECTED" || a.status === "RECEIVING_DATA").length;

  const sources = useMemo(
    () =>
      mockUavs.map((u) => {
        const conn = mockConnections.find((c) => c.uavId === u.id);
        return { uav: u, conn: conn ?? null };
      }),
    []
  );

  return (
    <div>
      <PageHeader
        title="Gateway"
        subtitle="Data sources, adapters and connection management"
        actions={
          <>
            <Button variant="outline" size="sm"><RefreshCw className="size-3.5" /> Refresh</Button>
            <Link href="/add-uav">
              <Button size="sm"><Plus className="size-3.5" /> Add Data Source</Button>
            </Link>
          </>
        }
      />

      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Kpi label="Adapters" value={adapters.length} sub="Configured adapters" icon={<PlugZap className="size-4" />} />
        <Kpi label="Connected" value={connected} sub="Active data streams" tone="success" icon={<RefreshCw className="size-4" />} />
        <Kpi label="Data Sources" value={sources.length} sub="Linked UAV sources" icon={<Plus className="size-4" />} />
        <Kpi label="Packet Rate" value="165/s" sub="Aggregate throughput" tone="info" icon={<RefreshCw className="size-4" />} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <SectionCard
            title="Adapters"
            action={<Badge variant="outline" className="text-[10px] text-muted-foreground">{adapters.length} total</Badge>}
          >
            <div className="-mx-3 -my-3">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Adapter</TableHead>
                    <TableHead>Source</TableHead>
                    <TableHead>Protocol</TableHead>
                    <TableHead>Endpoint</TableHead>
                    <TableHead>Packet Rate</TableHead>
                    <TableHead>Latency</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {adapters.map((a) => (
                    <TableRow key={a.id}>
                      <TableCell className="font-medium text-foreground">{a.name}</TableCell>
                      <TableCell><Badge variant="outline" className="text-[10px] text-muted-foreground">{a.source.replace(/_/g, " ")}</Badge></TableCell>
                      <TableCell className="text-muted-foreground">{a.protocol}</TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">{a.host}:{a.port}</TableCell>
                      <TableCell className="font-mono text-muted-foreground">{a.packetRate}/s</TableCell>
                      <TableCell className="text-muted-foreground">{a.latencyMs}ms</TableCell>
                      <TableCell><StatusBadge status={a.status} /></TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="icon-sm" className="text-muted-foreground" title="Pause"><Pause className="size-3.5" /></Button>
                        <Button variant="ghost" size="icon-sm" className="text-muted-foreground hover:text-destructive" title="Remove"><Trash2 className="size-3.5" /></Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </SectionCard>
        </div>

        <div className="flex flex-col gap-4">
          <SectionCard title="Linked Sources">
            <div className="space-y-2">
              {sources.map(({ uav, conn }) => (
                <div key={uav.id} className="flex items-center justify-between rounded border border-border bg-muted/30 px-2.5 py-2">
                  <div>
                    <div className="text-xs font-medium text-foreground">{uav.id}</div>
                    <div className="text-[10px] text-muted-foreground">{uav.engineId}</div>
                  </div>
                  <div className="text-right">
                    <StatusBadge status={conn?.status ?? "IDLE"} />
                    <div className="mt-1 text-[10px] text-muted-foreground">
                      {conn ? `${conn.protocol} · ${conn.packetRate}/s` : "no source"}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>

          <SectionCard title="Gateway Health">
            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Last sync</span>
                <span className="font-medium text-foreground">09-Sep 08:15:01</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Uptime</span>
                <span className="font-medium text-status-normal">6d 14h 22m</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Data loss</span>
                <span className="font-medium text-status-normal">0.0%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Auth failures</span>
                <span className="font-medium text-foreground">0</span>
              </div>
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}