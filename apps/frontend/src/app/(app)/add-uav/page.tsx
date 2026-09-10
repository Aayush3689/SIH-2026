"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { Braces, Cable, Check, CheckCircle2, Cpu, Globe2, Info, Monitor, Network, Plane, PlugZap, Radio, RefreshCw, X } from "lucide-react";

type View = "source" | "simulator" | "uav" | "protocol" | "connection" | "mapping" | "ready";

const raw = ["engine_rpm", "oil_temp", "oil_press", "egt_sensor", "fuel_rate", "vibration_x"];
const standard = ["Engine RPM", "Oil Temperature", "Oil Pressure", "Exhaust Gas Temperature", "Fuel Flow", "Vibration"];

function Title({ children }: { children: React.ReactNode }) {
  return (
    <div className="wizard-title">
      {children}
      <X />
    </div>
  );
}

function Source({
  source,
  setSource,
  next,
}: {
  source: string;
  setSource: (x: string) => void;
  next: () => void;
}) {
  const cards = [
    ["GCS", "Ground Control Station", "Receive UAV telemetry through a Ground Control Station.", Monitor],
    ["REAL UAV", "Physical UAV", "Connect directly to a physical UAV telemetry source.", Radio],
    ["SIMULATOR", "UAV/Engine Simulator", "Receive simulated telemetry for testing and development.", Plane],
  ] as const;
  return (
    <>
      <Title>Select data source</Title>
      <div className="wizard-sourcecards source-scan-results scan-complete">
        {cards.map(([a, b, c, I]) => (
          <motion.button
            onClick={() => setSource(a)}
            className={"wizard-source " + (source === a ? "wizard-picked" : "")}
            key={a}
            initial={{ opacity: 0, y: 22 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.32, delay: 0.08 + cards.findIndex(([id]) => id === a) * 0.1 }}
            whileHover={{ y: -8, scale: 1.018 }}
            whileTap={{ scale: 0.985 }}
          >
            {source === a && <Check className="wizard-corner" />}
            <I className="wizard-sourceicon" />
            <h2>{a}</h2>
            <h3>{b}</h3>
            <p>{c}</p>
            <span className="source-card-cta">Select source <b>›</b></span>
          </motion.button>
        ))}
      </div>
      <button className="wizard-continue" onClick={next} disabled={!source}>
        Continue
      </button>
    </>
  );
}

function Uav({ next }: { next: () => void }) {
  const aircraft = [["UAV ID", "UAV-017"], ["UAV Name", "MALE-UAV-017"], ["Mission profile", "Long-range surveillance"]];
  return (
    <>
      <Title>Basic UAV Details</Title>
      <div className="uav-console">
        <aside className="uav-console-rail"><span className="uav-console-index">01</span><b>Aircraft identity</b><small>Mission and airframe record</small><span className="uav-console-line" /><span className="uav-console-index muted">02</span><b>Engine profile</b><small>Powerplant specification</small><span className="uav-console-status"><i /> Ready for source setup</span></aside>
        <div className="uav-console-body">
          <div className="uav-console-heading"><span><Plane /></span><div><strong>New aircraft record</strong><p>Build a telemetry-ready identity for this mission asset.</p></div></div>
          <section><h2><i>01</i> Aircraft identity</h2><div className="uav-console-fields">{aircraft.map((x) => <label className={x[0] === "Mission profile" ? "uav-console-field-wide" : ""} key={x[0]}>{x[0]}<input defaultValue={x[1]} /></label>)}</div></section>
          <section><h2><i>02</i> Engine profile</h2><div className="uav-console-fields"><label>Engine ID<input defaultValue="ENG-017" /></label><label>Manufacturer<input defaultValue="Raptor Engines" /></label><label>Engine Type<select defaultValue="R-Type Turbo"><option>R-Type Turbo</option><option>Turbo Prop</option></select></label><label>Model<select><option>Choose model</option><option>R-Type Turbo Mk II</option></select></label></div></section>
          <label className="uav-console-check"><input type="checkbox" /> I want to configure a custom model later.</label>
          <button className="wizard-continue" onClick={next}>Continue to source selection <b>›</b></button>
        </div>
      </div>
    </>
  );
}

function SimulatorDiscovery({ next, back, onTestComplete }: { next: () => void; back: () => void; onTestComplete: () => void }) {
  const [selected, setSelected] = useState("");
  const [scanning, setScanning] = useState(false);
  const [scanPct, setScanPct] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [testState, setTestState] = useState<"idle" | "connecting" | "connected">("idle");
  const [speed, setSpeed] = useState("1x");
  const [profile, setProfile] = useState("Nominal");
  const [interfaces, setInterfaces] = useState([
    { name: "can0", detail: "CAN · SIMULATOR", bitrate: "500 kbps", rate: "132", load: "18.4%", status: "CONNECTED" },
    { name: "vcan0", detail: "CAN · SIMULATOR", bitrate: "500 kbps", rate: "84", load: "11.6%", status: "VIRTUAL" },
    { name: "can1", detail: "CAN · SIMULATOR", bitrate: "250 kbps", rate: "0", load: "0%", status: "AVAILABLE" },
  ]);

  const autoScan = () => {
    if (scanning) return;
    setScanning(true);
    setScanPct(0);
    const iv = setInterval(() => {
      setScanPct((p) => {
        if (p >= 100) {
          clearInterval(iv);
          setScanning(false);
          setInterfaces((prev) =>
            prev.map((it) =>
              it.name === "can1" ? { ...it, bitrate: "500 kbps", rate: "96", load: "12.8%", status: "VIRTUAL" } : it
            )
          );
          return 100;
        }
        return p + 12;
      });
    }, 220);
  };

  const testConnection = () => {
    setTestState("connecting");
    setTimeout(() => {
      setTestState("connected");
      setPlaying(true);
      onTestComplete();
    }, 1400);
  };

  const fault = testState === "connected";

  return (
    <div className="simulator-wizard">
      <div className="simulator-heading">
        <div>
          <h1>Add UAV</h1>
          <p>Configure a frontend-local mock source that can be replaced by a backend adapter later.</p>
        </div>
        <X aria-label="Close" />
      </div>
      <div className="simulator-steps" aria-label="Setup progress">
        {["Details", "Source", "Adapter", "Connection", "Data Preview", "Mapping", "Validate", "Ready"].map((step, index) => (
          <div className={index < 2 ? "complete" : index === 2 ? "active" : ""} key={step}>
            <span>{index < 2 ? <Check /> : index + 1}</span>
            {step}
          </div>
        ))}
      </div>
      <main className="simulator-content">
        <div className="simulator-notice">
          <Cpu /> Simulator discovery uses deterministic local data. The browser is not accessing OS CAN interfaces.
        </div>

        <div className="simulator-title-row">
          <div>
            <strong>SIMULATOR CAN DISCOVERY</strong>
            <h2>Detected simulator CAN interfaces</h2>
            <p>Choose the virtual CAN interface that will provide the simulated telemetry stream.</p>
          </div>
          <button type="button" className="simulator-scan" onClick={autoScan} disabled={scanning}>
            <RefreshCw className={scanning ? "spin" : ""} /> {scanning ? `Scanning ${scanPct}%` : "Auto scan"}
          </button>
        </div>

        {scanning && (
          <div className="simulator-scanbar">
            <div style={{ width: `${scanPct}%` }} />
          </div>
        )}

        {scanPct === 0 && !scanning ? <div className="simulator-empty"><RefreshCw /><span>Run an auto scan to load deterministic simulator interfaces.</span></div> : <div className="simulator-interfaces">
          {interfaces.map((item) => (
            <button type="button" key={item.name} onClick={() => setSelected(item.name)} className={selected === item.name ? "selected" : ""}>
              <span className="simulator-interface-icon"><Cpu /></span>
              <span className="simulator-interface-name"><b>{item.name}</b><small>{item.detail}</small></span>
              <span><b>{item.bitrate}</b><small>bitrate</small></span>
              <span><b>{item.rate}</b><small>frames/s</small></span>
              <span><b>{item.load}</b><small>bus load</small></span>
              <em className={item.status.toLowerCase()}><i />{item.status}</em>
            </button>
          ))}
        </div>}

        {selected && <section className="simulator-selected-config">
          <div className="simulator-selected-source"><span><Cpu /></span><div><b>{selected} telemetry source</b><small>Configure playback for the selected deterministic simulator stream.</small></div><em><i /> READY</em></div>
          <div className="simulator-config-controls">
            <label>Playback speed<select value={speed} onChange={(e) => setSpeed(e.target.value)}><option>0.5x</option><option>1x</option><option>2x</option></select></label>
            <label>Telemetry profile<select value={profile} onChange={(e) => setProfile(e.target.value)}><option>Nominal</option><option>High EGT</option><option>Low oil pressure</option><option>High vibration</option></select></label>
            <label>Simulation<button type="button" onClick={() => setPlaying(!playing)}>{playing ? "Stop simulation" : "Start simulation"}</button></label>
          </div>
          <div className="simulator-connection-test"><div><strong>CONNECTION TEST</strong><h3>Validate telemetry availability</h3></div><em className={testState === "connected" ? "" : "pending"}><i /> {testState === "connected" ? "CONNECTED" : testState === "connecting" ? "TESTING" : "NOT TESTED"}</em><label>Connection test result<div><span>{testState === "connected" ? "Connected" : testState === "connecting" ? "Checking adapter..." : "Run test to check connection"}</span><button type="button" onClick={testConnection} disabled={testState === "connecting"}>{testState === "connecting" ? "Testing..." : testState === "connected" ? "Test again" : "Test connection"}</button></div></label>{testState === "connected" ? <p><CheckCircle2 /> <b>Receiving telemetry</b><small>32 ms latency · {speed === "2x" ? "256" : speed === "0.5x" ? "64" : "128"} packets/s · last packet just now</small></p> : <p className="simulator-test-hint">Run a connection test before submitting this virtual CAN adapter.</p>}</div>
        </section>}

        <div className="simulator-playback">
          <div className="simulator-playback-head">
            <div>
              <strong>TELEMETRY PLAYBACK</strong>
              <h2>Simulated engine stream</h2>
              <p>R-Type Turbo profile · {fault ? "fault scenario injected" : "nominal envelope"}</p>
            </div>
            <button type="button" className={"simulator-play " + (playing ? "on" : "")} onClick={() => setPlaying(!playing)}>
              {playing ? "◼" : "▶"} {playing ? "Playing" : "Play"}
            </button>
          </div>
          <div className="simulator-gauges">
            <div>
              <span className="simulator-gauge-label">Engine RPM</span>
              <b className={fault ? "warn" : ""}>{playing ? (fault ? "6,240" : "8,420") : "0"}</b>
              <small>RPM</small>
            </div>
            <div>
              <span className="simulator-gauge-label">Exhaust Gas Temp</span>
              <b className={fault ? "crit" : ""}>{playing ? (fault ? "829°C" : "648°C") : "—"}</b>
              <small>EGT</small>
            </div>
            <div>
              <span className="simulator-gauge-label">Oil Pressure</span>
              <b className={fault ? "crit" : ""}>{playing ? (fault ? "2.1 bar" : "4.2 bar") : "—"}</b>
              <small>oil_press</small>
            </div>
            <div>
              <span className="simulator-gauge-label">Vibration X</span>
              <b className={fault ? "warn" : ""}>{playing ? (fault ? "3.2 g" : "0.62 g") : "—"}</b>
              <small>mm/s</small>
            </div>
          </div>
          <div className="simulator-streammeta">
            <span>Frame: 09,847</span>
            <span>Packet rate: 100/s</span>
            <span>Δt: 10ms</span>
            <span>Bus: {selected}</span>
            <span className="simulator-linkstate">
              <i className={testState === "connected" ? "up" : ""} />
              {testState === "idle" ? "Not connected" : testState === "connecting" ? "Connecting…" : "Connected · Receiving data"}
            </span>
          </div>
        </div>

        <div className="simulator-testrow">
          <button type="button" className="simulator-test" onClick={testConnection} disabled={testState === "connecting"}>
            {testState === "connected" ? <><CheckCircle2 /> Connected — continue</> : testState === "connecting" ? "Testing connection…" : "Test simulator connection"}
          </button>
          <em>Test states: idle → connecting → connected</em>
        </div>
      </main>
      <footer className="simulator-footer">
        <span>{selected ? `Selected adapter: ${selected}` : "Select a virtual CAN adapter"}</span>
        <button type="button" onClick={back}>‹<small>Back</small></button>
        <button type="button" onClick={next} disabled={testState !== "connected"}>Continue <b>›</b></button>
        <button type="button" className="simulator-submit" onClick={next} disabled={!selected || testState !== "connected"}>Submit <b>›</b></button>
      </footer>
    </div>
  );
}

function Protocol({
  protocol,
  setProtocol,
  next,
}: {
  protocol: string;
  setProtocol: (x: string) => void;
  next: () => void;
}) {
  const adapters = [["MAVLink", "Flight-control telemetry", Radio], ["TCP", "Reliable stream socket", Cable], ["UDP", "Low-latency datagrams", Network], ["REST API", "HTTP telemetry endpoint", Globe2], ["Own Protocol", "Custom parser format", Braces]] as const;
  return (
    <>
      <Title>Choose telemetry adapter</Title>
      <div className="adapter-picker">
        <div className="adapter-picker-intro"><span><PlugZap /></span><div><b>Adapter / protocol</b><p>Select how this source will send its telemetry into the gateway.</p></div><em>5 AVAILABLE</em></div>
        <div className="adapter-picker-grid">{adapters.map(([name, description, Icon], index) => <motion.button key={name} type="button" onClick={() => setProtocol(name)} className={protocol === name ? "selected" : ""} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * .05 }} whileTap={{ scale: .98 }}><span><Icon /></span><div><b>{name}</b><small>{description}</small></div>{protocol === name && <Check />}</motion.button>)}</div>
        <div className="adapter-picker-selected"><i /> <span><b>{protocol}</b> selected</span><small>Connection settings will be tailored for this adapter.</small></div>
        <label className="adapter-picker-check"><input type="checkbox" /> Configure advanced adapter settings after connection.</label>
        <button className="wizard-continue" onClick={next}>Configure {protocol} <b>›</b></button>
      </div>
    </>
  );
}

function Connection({
  connected,
  connect,
  protocol,
  onPreview,
}: {
  connected: boolean;
  connect: () => void;
  protocol: string;
  onPreview: () => void;
}) {
  return (
    <>
      <Title>{protocol} Connection Configuration</Title>
      <div className="connection-console">
        <div className="connection-console-head"><span><Cable /></span><div><b>Gateway connection</b><p>Configure the route used by the selected {protocol} adapter.</p></div><em>{protocol}</em></div>
        <div className="connection-console-grid">
          <div className="connection-console-fields">
            {[["Connection Type", protocol], [protocol === "REST API" ? "Endpoint URL" : "Host / IP Address", protocol === "REST API" ? "https://telemetry.example/api" : "192.168.1.100"], ["Port", protocol === "TCP" ? "9000" : protocol === "UDP" || protocol === "MAVLink" ? "14550" : "8080"], ["Telemetry Stream", "UAV-017"]].map((x) => <label key={x[0]}>{x[0]}<input defaultValue={x[1]} /></label>)}
          </div>
          <div className={"connection-console-status " + (connected ? "connected" : "")}><span><Network /></span><small>Connection status</small><b>{connected ? "Connected" : "Waiting to test"}</b><p>{connected ? "Telemetry gateway responded successfully." : "Run a test to validate this telemetry route."}</p><div><i /> {connected ? "Ready to receive data" : "No active session"}</div></div>
        </div>
        <button className="connection-console-test" onClick={() => (connected ? onPreview() : connect())}>{connected ? "Open live data preview" : "Test connection"} <b>›</b></button>
      </div>
    </>
  );
}

function Mapping({
  ready,
  mapped,
  validate,
}: {
  ready: boolean;
  mapped: boolean;
  validate: () => void;
}) {
  return (
    <>
      {ready && (
        <div className="wizard-ready">
          <Check /> GATEWAY DATA SOURCE READY
        </div>
      )}
      <Title>Data Mapping</Title>
      {mapped && !ready && <div className="wizard-auto-mapped"><Check /> Detected telemetry fields have been mapped automatically. Review the mapping and validate to continue.</div>}
      <div className="wizard-map">
        <div>
          <h2>Incoming GCS Data (Raw Telemetry)</h2>
          {raw.map((x) => (
            <div className="wizard-raw" key={x}>
              {x}
              {ready && <span />}
              <Info />
            </div>
          ))}
        </div>
        <div>
          <h2>Standardized Platform Parameters</h2>
          {standard.map((x) => (
            <label className="wizard-selectrow" key={x}>
              {x}
              <select defaultValue={x}>
                <option>{x}</option>
              </select>
            </label>
          ))}
        </div>
      </div>
      {ready && (
        <div className="wizard-summary">
          Required Parameters:　6
          <br />
          Mapped Parameters:　6
          <br />
          Unmapped Parameters:　0
          <br />
          Invalid Parameters:　0
        </div>
      )}
      <button disabled={ready} className="wizard-validate" onClick={validate}>
        {mapped || ready ? "Mapping Validated" : "Validate Mapping"}
      </button>
    </>
  );
}

export default function AddUavPage() {
  const router = useRouter();
  const [v, setV] = useState<View>("uav");
  const [source, setSource] = useState("");
  const [protocol, setProtocol] = useState("MAVLink");
  const [connected, setConnected] = useState(false);
  const [mapped, setMapped] = useState(false);
  useEffect(() => {
    if (v === "mapping" || v === "ready") router.replace("/mapping");
  }, [router, v]);
  const next = () => {
    if (v === "simulator") {
      setMapped(true);
      router.push("/mapping?autoMap=1");
      return;
    }
    setV(v === "uav" ? "source" : v === "source" ? (source === "SIMULATOR" ? "simulator" : "protocol") : v === "protocol" ? "connection" : v === "connection" ? "mapping" : v === "mapping" ? "ready" : "uav");
  };

  return (
    <div className="wizard-shell" style={{ background: "#162132", minHeight: "calc(100vh - 120px)", borderRadius: 6 }}>
      {v === "source" && <Source source={source} setSource={setSource} next={next} />}
      {v === "simulator" && <SimulatorDiscovery next={next} back={() => setV("source")} onTestComplete={() => router.push("/data-preview?source=vcan0")} />}
      {v === "uav" && <Uav next={next} />}
      {v === "protocol" && <Protocol protocol={protocol} setProtocol={setProtocol} next={next} />}
      {v === "connection" && <Connection connected={connected} connect={() => { setConnected(true); setTimeout(() => router.push(`/data-preview?source=${encodeURIComponent(source)}&protocol=${encodeURIComponent(protocol)}`), 650); }} protocol={protocol} onPreview={() => router.push(`/data-preview?source=${encodeURIComponent(source)}&protocol=${encodeURIComponent(protocol)}`)} />}
      {(v === "mapping" || v === "ready") && (
        <Mapping
          ready={v === "ready"}
          mapped={mapped}
          validate={() => {
            setMapped(true);
            setV("ready");
          }}
        />
      )}
    </div>
  );
}
