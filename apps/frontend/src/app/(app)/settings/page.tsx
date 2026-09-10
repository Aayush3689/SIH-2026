"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Bell, Database, Palette, ShieldCheck, Save } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { SectionCard } from "@/components/common/Status";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

export default function SettingsPage() {
  const [demoMode, setDemoMode] = useState(true);
  const [saved, setSaved] = useState(false);

  return (
    <div>
      <PageHeader
        title="Settings"
        subtitle="User, application and connection defaults"
        actions={
          <Button size="sm" onClick={() => { setSaved(true); setTimeout(() => setSaved(false), 2500); }}>
            <Save className="size-3.5" /> Save Changes
          </Button>
        }
      />

      {saved && (
        <motion.div
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-4 rounded border border-success/40 bg-success/10 px-3 py-2 text-sm text-success"
        >
          Settings saved successfully.
        </motion.div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <SectionCard title="Environment">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <Database className="size-4 text-muted-foreground" /> Simulation mode
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Use deterministic local mock data instead of live API calls.
                </p>
              </div>
              <Switch checked={demoMode} onCheckedChange={setDemoMode} />
            </div>
            <div className="rounded border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
              Current environment: <span className="font-mono text-status-info">{demoMode ? "LOCAL SIMULATION" : "LIVE API"}</span>
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Notifications">
          <div className="space-y-4">
            {([
              ["Critical alerts", true],
              ["High severity alerts", true],
              ["Medium severity alerts", false],
              ["RUL threshold reached", true],
            ] as const).map(([label, def]) => (
              <div key={label} className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-foreground">
                  <Bell className="size-4 text-muted-foreground" /> {label}
                </div>
                <Switch defaultChecked={def} />
              </div>
            ))}
          </div>
        </SectionCard>

        <SectionCard title="Connection Defaults">
          <div className="space-y-3">
            {([
              ["Default host", "192.168.1.100"],
              ["Default port", "14550"],
              ["Connect timeout (ms)", "5000"],
            ] as const).map(([label, val]) => (
              <div key={label} className="space-y-1">
                <Label className="text-xs text-muted-foreground">{label}</Label>
                <Input defaultValue={val} className="h-8" />
              </div>
            ))}
          </div>
        </SectionCard>

        <SectionCard title="Profile">
          <div className="space-y-3">
            {([
              ["Operator", "Mission Control"],
              ["Role", "SIH Operator"],
            ] as const).map(([label, val]) => (
              <div key={label} className="space-y-1">
                <Label className="text-xs text-muted-foreground">{label}</Label>
                <Input defaultValue={val} className="h-8" />
              </div>
            ))}
            <div className="flex items-center gap-2 rounded border border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
              <ShieldCheck className="size-4 text-status-normal" /> Session authenticated · token valid for 8h
            </div>
          </div>
        </SectionCard>

        <SectionCard title="Appearance">
          <div className="space-y-3">
            <div>
              <Label className="text-xs text-muted-foreground">Theme</Label>
              <div className="mt-1 flex gap-2">
                {["Dark Console", "Midnight", "Carbon"].map((t, i) => (
                  <button
                    key={t}
                    className={`flex items-center gap-1.5 rounded border px-3 py-1.5 text-xs ${
                      i === 0 ? "border-primary bg-primary/10 text-primary" : "border-border bg-muted/30 text-foreground hover:bg-muted/50"
                    }`}
                  >
                    <Palette className="size-3" /> {t}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}
