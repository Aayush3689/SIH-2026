"use client";

import { Activity, ChevronRight } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { Button } from "@/components/ui/button";

const frames = [
  ["engine_rpm", "5,128 rpm", "CAN 0x182"],
  ["oil_temp", "88.4 °C", "CAN 0x191"],
  ["oil_press", "381.8 kPa", "CAN 0x192"],
  ["egt_sensor", "642.1 °C", "CAN 0x1A4"],
  ["fuel_rate", "13.7 L/h", "CAN 0x1B0"],
  ["vibration_x", "3.42 mm/s", "CAN 0x1C5"],
  ["battery_voltage", "25.18 V", "CAN 0x205"],
];

export default function DataPreviewPage() {
  return <Suspense fallback={null}><Preview /></Suspense>;
}

function Preview() {
  const router = useRouter();
  const source = useSearchParams().get("source") ?? "vcan0";
  const protocol = useSearchParams().get("protocol") ?? "Virtual CAN";
  return <div className="telemetry-preview-page">
    <header><div><strong>INCOMING TELEMETRY</strong><h1>Raw data preview</h1><p>Live values are generated from deterministic frontend mock telemetry.</p></div><em><i /> DATA LIVE</em></header>
    <div className="telemetry-preview-meta"><b>{source} source</b><b>{protocol} adapter</b><b>0x182-0x205 raw frames</b><b>FLT-260909-A1 flight</b></div>
    <section className="telemetry-preview-table">{frames.map(([field, value, can]) => <div key={field}><code>{field}</code><b>{value}</b><span>{can}</span><em>VALID</em></div>)}</section>
    <footer><span><Activity /> Frame count <b>18,420</b> · Payload last updated <b>just now</b> · Mapping target <b>platform telemetry v1</b></span><Button onClick={() => router.push("/mapping?autoMap=1")}>Continue to mapping <ChevronRight className="size-4" /></Button></footer>
  </div>;
}
