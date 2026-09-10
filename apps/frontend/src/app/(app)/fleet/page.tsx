"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { Plus, Plane, ChevronRight, MoreHorizontal, Search } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { StatusBadge } from "@/components/common/Status";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useLiveTelemetry } from "@/hooks/use-live-telemetry";
import { mockUavs, mockConnections } from "@/data/mock";

export default function FleetPage() {
  const uavs = mockUavs;
  const live = useLiveTelemetry(uavs, 3000);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<(typeof uavs)[0] | null>(null);

  const filtered = live.filter(
    (u) =>
      u.id.toLowerCase().includes(query.toLowerCase()) ||
      u.engineId.toLowerCase().includes(query.toLowerCase()) ||
      u.mission.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div>
      <PageHeader
        title="UAV Fleet"
        subtitle="Registered aircraft and engine inventory"
        actions={
          <Link href="/add-uav">
            <Button size="sm"><Plus className="size-3.5" /> Add UAV</Button>
          </Link>
        }
      />

      <div className="mb-4 flex h-11 w-full max-w-xl items-center gap-2 rounded-lg border border-border bg-card px-3 shadow-inner shadow-black/10 focus-within:border-primary">
        <Search className="size-4 shrink-0 text-primary" />
        <Input
          className="h-auto border-0 bg-transparent px-0 text-sm shadow-none focus-visible:ring-0"
          placeholder="Search by UAV ID, engine, mission..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-md border border-border bg-card"
      >
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>UAV</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Engine</TableHead>
              <TableHead>Mission</TableHead>
              <TableHead>Flight State</TableHead>
              <TableHead>Health</TableHead>
              <TableHead>RUL</TableHead>
              <TableHead>Source</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.map((u) => {
              const conn = mockConnections.find((c) => c.uavId === u.id);
              return (
                <TableRow key={u.id} className="cursor-pointer" onClick={() => setSelected(u)}>
                  <TableCell className="font-medium text-foreground">
                    <div className="flex items-center gap-2">
                      <Plane className="size-3.5 text-primary" />
                      {u.id}
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{u.type}</TableCell>
                  <TableCell className="text-muted-foreground">{u.engineModel}</TableCell>
                  <TableCell className="text-muted-foreground">{u.mission}</TableCell>
                  <TableCell><StatusBadge status={u.flightState} /></TableCell>
                  <TableCell><StatusBadge status={u.healthState} /></TableCell>
                  <TableCell>
                    <span className={u.rulHours < 8 ? "text-status-danger" : u.rulHours < 24 ? "text-status-warning" : "text-status-normal"}>
                      {u.rulHours}h
                    </span>
                  </TableCell>
                  <TableCell>
                    {conn ? <Badge variant="outline" className="text-[10px] text-muted-foreground">{conn.source.replace(/_/g, " ")}</Badge> : "—"}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="icon-sm" className="text-muted-foreground">
                      <MoreHorizontal className="size-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </motion.div>

      {/* UAV detail dialog */}
      <Dialog open={!!selected} onOpenChange={(o) => !o && setSelected(null)}>
        <DialogContent className="max-w-lg border-border">
          {selected && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Plane className="size-4 text-primary" /> {selected.id}
                </DialogTitle>
              </DialogHeader>
              <div className="space-y-3 text-sm">
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded bg-muted/50 p-2">
                    <div className="text-[10px] uppercase text-muted-foreground">UAV Name</div>
                    <div className="font-medium text-foreground">{selected.name}</div>
                  </div>
                  <div className="rounded bg-muted/50 p-2">
                    <div className="text-[10px] uppercase text-muted-foreground">Engine</div>
                    <div className="font-medium text-foreground">{selected.engineModel}</div>
                  </div>
                  <div className="rounded bg-muted/50 p-2">
                    <div className="text-[10px] uppercase text-muted-foreground">Health</div>
                    <div><StatusBadge status={selected.healthState} /></div>
                  </div>
                  <div className="rounded bg-muted/50 p-2">
                    <div className="text-[10px] uppercase text-muted-foreground">RUL</div>
                    <div className="font-medium text-foreground">{selected.rulHours}h</div>
                  </div>
                </div>
                <div className="flex justify-end gap-2">
                  <Link href={`/monitoring?uav=${selected.id}`}>
                    <Button variant="outline" size="sm">Live Monitoring</Button>
                  </Link>
                  <Link href={`/health?uav=${selected.id}`}>
                    <Button size="sm">Engine Health <ChevronRight className="size-3.5" /></Button>
                  </Link>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
