"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import { Check, CircleHelp, Save, WandSparkles } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { mockRawTelemetryFields, mockMappings, mockUavs } from "@/data/mock";

export default function MappingPage() {
  return <Suspense fallback={<div className="text-sm text-muted-foreground">Loading mapping workspace...</div>}><MappingWorkspace /></Suspense>;
}

function MappingWorkspace() {
  const searchParams = useSearchParams();
  const shouldAutoMap = searchParams.get("autoMap") === "1";
  const [mappings, setMappings] = useState(() => shouldAutoMap ? mockMappings : mockMappings.map((rule) => ({ ...rule, rawField: "", valid: false })));
  const [saved, setSaved] = useState(false);
  const [autoMapped, setAutoMapped] = useState(shouldAutoMap);
  const [visibleLines, setVisibleLines] = useState(0);
  const required = mappings.filter((m) => m.required).length;
  const mapped = mappings.filter((m) => m.valid).length;
  const unmapped = mappings.filter((m) => !m.rawField).length;
  const invalid = mappings.filter((m) => m.rawField && !m.valid).length;
  const autoMap = () => { setMappings(mockMappings.map((rule) => ({ ...rule, valid: true }))); setAutoMapped(true); setSaved(false); };
  const updateMapping = (parameter: string, rawField: string) => {
    setMappings((items) => {
      const updated = items.map((item) => item.standardParameter === parameter ? { ...item, rawField } : item);
      return updated.map((item) => ({ ...item, valid: Boolean(item.rawField) && updated.filter((candidate) => candidate.rawField === item.rawField).length === 1 }));
    });
    setAutoMapped(false);
    setSaved(false);
  };
  useEffect(() => {
    const timer = setInterval(() => setVisibleLines((count) => count >= mappings.length ? count : count + 1), 500);
    return () => clearInterval(timer);
  }, [mappings.length]);

  return <div>
    <PageHeader title="Data Mapping" subtitle="Normalize incoming telemetry into platform parameters" actions={<><Button variant="outline" size="sm" onClick={autoMap}><WandSparkles className="size-3.5" /> Auto-map</Button><Button size="sm" disabled={unmapped > 0} onClick={() => setSaved(true)}><Save className="size-3.5" /> Save mapping</Button></>} />
    <div className="mapping-ready-banner"><Check /><b>{autoMapped ? "Gateway data source ready" : "Manual mapping in progress"}</b><span>{autoMapped ? "6 deterministic fields matched automatically" : "Review your changed fields before validation"}</span></div>
    <section className="mapping-board">
      <div className="mapping-column-heading">Incoming telemetry <small>RAW DATA STREAM</small></div>
      <div className="mapping-column-heading mapping-target-heading">Standardized platform parameters <small>PLATFORM SCHEMA</small></div>
      <div className="mapping-grid">
        <div className="mapping-raw-list">{mockRawTelemetryFields.map(({ field, value }) => <div className="mapping-raw-row" key={field}><div><b>{field}</b><small>Latest: {value}</small></div><span className="mapping-node" /><CircleHelp /></div>)}</div>
        <div className="mapping-links" aria-hidden="true"><svg viewBox="0 0 360 420" preserveAspectRatio="none">{mappings.slice(0, visibleLines).map((mapping, targetIndex) => { const sourceIndex = mockRawTelemetryFields.findIndex(({ field }) => field === mapping.rawField); const startY = sourceIndex < 0 ? 34 + targetIndex * 70 : 34 + sourceIndex * 70; const endY = 34 + targetIndex * 70; const path = `M 0 ${startY} C 108 ${startY}, 250 ${endY}, 360 ${endY}`; return <motion.path key={mapping.standardParameter} initial={{ d: path, pathLength: 0, opacity: 0 }} animate={{ d: path, pathLength: 1, opacity: 1 }} transition={{ d: { duration: 0.7, ease: "easeInOut" }, pathLength: { duration: 1, ease: "easeOut" }, opacity: { duration: 0.2 } }} />; })}</svg></div>
        <div className="mapping-target-list">{mappings.map((mapping) => <div className={"mapping-target-row " + (!mapping.valid ? "mapping-invalid" : "")} key={mapping.standardParameter}><label>{mapping.standardParameter}<small>{mapping.unit} · {mapping.required ? "Required" : "Optional"}</small></label><select value={mapping.rawField} onChange={(e) => updateMapping(mapping.standardParameter, e.target.value)}><option value="">Choose field</option>{mockRawTelemetryFields.map(({ field }) => <option key={field} value={field}>{field}</option>)}</select></div>)}</div>
      </div>
      <div className="mapping-validation"><div><span>Required parameters</span><b>{required}</b></div><div><span>Mapped parameters</span><b className="good">{mapped}</b></div><div><span>Unmapped parameters</span><b className={unmapped ? "warn" : "good"}>{unmapped}</b></div><div><span>Invalid parameters</span><b className={invalid ? "warn" : "good"}>{invalid}</b></div></div>
      {invalid > 0 && <p className="mapping-error">Each raw telemetry field can only map to one platform parameter.</p>}
      {saved && <p className="mapping-saved"><Check /> Mapping saved successfully for {mockUavs[0].id}.</p>}
      <Button className="mapping-validate" disabled={unmapped > 0 || invalid > 0} onClick={() => setSaved(true)}>Validate mapping</Button>
    </section>
  </div>;
}
