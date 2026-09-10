'use client';

import { useMemo, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  BatteryCharging,
  CircleDot,
  Cpu,
  Flame,
  Gauge,
  Radio,
  RefreshCw,
  Thermometer,
  Waves,
} from 'lucide-react';
import { useApp } from '@/components/layout/app-providers';
import { useTelemetryFeed } from '@/hooks/use-telemetry-feed';
import { Badge, Card, CardHeader, ProgressBar, SectionHeading, Select, Tabs } from '@/components/ui';
import { TrendChart } from '@/components/shared/trend-chart';
import { cn } from '@/lib/utils';

const anomalyTone = {
  NORMAL: 'green',
  PRE_FAULT: 'amber',
  ACTIVE_FAULT: 'red',
} as const;

function number(value: number | undefined, digits = 1) {
  return typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
}

function prettyFault(value: string | undefined) {
  return (value || 'Unknown').split('_').map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(' ');
}

function TelemetryCard({
  label,
  value,
  unit,
  icon,
  tone = 'blue',
  detail,
}: {
  label: string;
  value: string;
  unit: string;
  icon: React.ReactNode;
  tone?: 'blue' | 'cyan' | 'amber' | 'red' | 'green';
  detail: string;
}) {
  return (
    <div className="telemetry-card">
      <div className={'telemetry-card-icon telemetry-card-icon--' + tone}>{icon}</div>
      <div>
        <span className="telemetry-card-label">{label}</span>
        <div className="telemetry-value"><strong>{value}</strong><small>{unit}</small></div>
        <span className="telemetry-detail"><i />{detail}</span>
      </div>
    </div>
  );
}

export function LiveMonitoringView() {
  const { alerts, environment, selectedUav, selectedUavId, setSelectedUavId, uavs } = useApp();
  const [payloadView, setPayloadView] = useState('normalized');
  const uav = selectedUav || uavs[0];
  const { history, inference, isRulReady, latest, samplesReceived } = useTelemetryFeed(uav?.id || selectedUavId, uav?.engine_id);
  const anomaly = inference?.data.anomaly[0];
  const fault = inference?.data.fault[0];
  const rul = isRulReady ? inference?.data.rul?.[0] ?? null : null;
  const activeAlerts = alerts.filter((alert) => alert.status === 'open' && alert.uav_id === uav?.id);
  const probabilityEntries = useMemo(
    () => Object.entries(fault?.probabilities || {}).sort((left, right) => right[1] - left[1]).slice(0, 4),
    [fault],
  );

  if (!uav || !latest || !anomaly || !fault) {
    return <div className="page-enter"><SectionHeading description="No telemetry source is selected." title="Live Monitoring" /></div>;
  }

  const metrics = [
    { label: 'RPM', value: number(latest.rpm, 0), unit: 'rpm', icon: <Gauge size={17} />, tone: 'blue' as const, detail: latest.rpm > 5200 ? 'above cruise band' : 'nominal band' },
    { label: 'Engine Load', value: number(latest.engine_load_pct), unit: '%', icon: <Activity size={17} />, tone: 'cyan' as const, detail: 'throttle-correlated' },
    { label: 'EGT', value: number(latest.egt_c), unit: '°C', icon: <Flame size={17} />, tone: latest.egt_c > 700 ? 'red' as const : 'amber' as const, detail: latest.egt_c > 700 ? 'thermal watch' : 'within limit' },
    { label: 'CHT', value: number(latest.cht_c), unit: '°C', icon: <Thermometer size={17} />, tone: latest.cht_c > 200 ? 'red' as const : 'amber' as const, detail: latest.cht_c > 200 ? 'elevated' : 'stable trend' },
    { label: 'Oil Pressure', value: number(latest.oil_pressure_kpa), unit: 'kPa', icon: <Gauge size={17} />, tone: latest.oil_pressure_kpa < 330 ? 'amber' as const : 'green' as const, detail: latest.oil_pressure_kpa < 330 ? 'below expected' : 'healthy pressure' },
    { label: 'Oil Temperature', value: number(latest.oil_temperature_c), unit: '°C', icon: <Thermometer size={17} />, tone: latest.oil_temperature_c > 105 ? 'red' as const : 'amber' as const, detail: 'model input' },
    { label: 'Fuel Flow', value: number(latest.fuel_flow_lph), unit: 'L/h', icon: <Waves size={17} />, tone: 'cyan' as const, detail: 'source normalized' },
    { label: 'Fuel Pressure', value: number(latest.fuel_pressure_kpa), unit: 'kPa', icon: <Gauge size={17} />, tone: 'green' as const, detail: 'within band' },
    { label: 'Vibration', value: number(latest.engine_vibration_mm_s, 2), unit: 'mm/s', icon: <Activity size={17} />, tone: latest.engine_vibration_mm_s > 7 ? 'red' as const : latest.engine_vibration_mm_s > 5 ? 'amber' as const : 'cyan' as const, detail: latest.engine_vibration_mm_s > 7 ? 'active event' : 'rolling RMS' },
    { label: 'Battery', value: number(latest.battery_voltage_v, 2), unit: 'V', icon: <BatteryCharging size={17} />, tone: 'green' as const, detail: 'alternator online' },
  ];

  return (
    <div className="page-enter monitoring-page">
      <SectionHeading
        action={
          <Select aria-label="Selected UAV" className="monitoring-uav-select" onChange={(event) => setSelectedUavId(event.target.value)} value={uav.id}>
            {uavs.map((item) => <option key={item.id} value={item.id}>{item.id} · {item.callsign}</option>)}
          </Select>
        }
        description="Live mock telemetry, normalized data, and current AI inference outputs."
        eyebrow={<><Badge dot tone={environment === 'Demo' ? 'blue' : 'green'}>{environment} mode</Badge><span className="heading-refresh"><CircleDot size={12} /> update cycle 1.4 s</span></>}
        title="Live Monitoring"
      />

      <Card className="monitoring-context-card">
        <div className="monitoring-context-main">
          <span className="monitoring-plane-icon"><Radio size={18} /></span>
          <span><small>Aircraft</small><b>{uav.id} · {uav.callsign}</b></span>
          <span><small>Engine</small><b>{uav.engine_id}</b></span>
          <span><small>Flight</small><b>{latest.flight_id}</b></span>
          <span><small>Phase</small><b>{uav.mission_status.replace('-', ' ')}</b></span>
          <span><small>Source</small><b>{uav.gateway_id || 'Demo simulator'}</b></span>
        </div>
        <div className="monitoring-context-status">
          <Badge dot tone={uav.status === 'critical' ? 'red' : uav.status === 'warning' ? 'amber' : 'green'}>{uav.status === 'active' ? 'Connected' : uav.status}</Badge>
          <span><RefreshCw size={12} /> Updated just now</span>
        </div>
      </Card>

      <div className="telemetry-grid">
        {metrics.map((metric) => <TelemetryCard key={metric.label} {...metric} />)}
      </div>

      <div className="monitoring-analysis-grid">
        <div className="monitoring-chart-grid">
          <TrendChart
            series={[
              { id: 'rpm', label: 'RPM', color: '#39D0C8', values: history.slice(-24).map((item) => item.rpm) },
              { id: 'load', label: 'Load', color: '#2F80ED', values: history.slice(-24).map((item) => item.engine_load_pct * 70) },
            ]}
            subtitle="Rolling 30–60 sample stream"
            title="RPM + engine load"
          />
          <TrendChart
            series={[
              { id: 'egt', label: 'EGT', color: '#F1B955', values: history.slice(-24).map((item) => item.egt_c) },
              { id: 'cht', label: 'CHT', color: '#EF626F', values: history.slice(-24).map((item) => item.cht_c * 3.3) },
            ]}
            subtitle="Thermal response"
            title="EGT + CHT"
          />
          <TrendChart
            series={[
              { id: 'oil-pressure', label: 'Pressure', color: '#39D0C8', values: history.slice(-24).map((item) => item.oil_pressure_kpa) },
              { id: 'oil-temperature', label: 'Temperature', color: '#F1B955', values: history.slice(-24).map((item) => item.oil_temperature_c * 3.8) },
            ]}
            subtitle="Lubrication indicators"
            title="Oil pressure + temperature"
          />
          <TrendChart
            series={[
              { id: 'vibration', label: 'RMS vibration', color: anomaly.state === 'ACTIVE_FAULT' ? '#EF626F' : '#A78BFA', values: history.slice(-24).map((item) => item.engine_vibration_mm_s) },
            ]}
            subtitle="Rolling vibration RMS"
            title="Engine vibration"
          />
        </div>

        <Card className="ai-summary-card">
          <CardHeader
            action={<Badge dot tone="purple">Inference</Badge>}
            subtitle="Current model response from the typed demo service"
            title="AI engine summary"
          />
          <div className="ai-state-block">
            <span className={'ai-state-icon ai-state-icon--' + anomalyTone[anomaly.state]}><Cpu size={20} /></span>
            <div><small>Anomaly state</small><strong>{anomaly.state.replace('_', ' ')}</strong><span>{(anomaly.confidence * 100).toFixed(0)}% confidence</span></div>
            <Badge tone={anomalyTone[anomaly.state]}>{anomaly.state}</Badge>
          </div>
          <div className="ai-probability-group">
            <span className="ai-group-label">Anomaly probabilities</span>
            {Object.entries(anomaly.probabilities).map(([label, value]) => (
              <ProgressBar key={label} label={label.replace('_', ' ')} tone={label === anomaly.state ? anomalyTone[anomaly.state] : 'gray'} value={value * 100} />
            ))}
          </div>
          <div className="ai-fault-block">
            <span className="ai-group-label">Suspected model prediction</span>
            <strong>{prettyFault(fault.fault)}</strong>
            <small>Model prediction — confirmation logic pending · {(fault.confidence * 100).toFixed(1)}% confidence</small>
          </div>
          <div className="ai-probability-group ai-probability-group--compact">
            {probabilityEntries.map(([label, value]) => (
              <ProgressBar key={label} label={prettyFault(label)} tone={label === fault.fault ? 'purple' : 'gray'} value={value * 100} />
            ))}
          </div>
          <div className={cn('ai-rul', rul !== null && 'ai-rul--ready')}>
            <div><small>Remaining useful life</small><strong>{rul !== null ? rul.toFixed(1) + ' h' : 'Collecting history'}</strong></div>
            {rul !== null ? <Badge tone={rul < 8 ? 'red' : rul < 24 ? 'amber' : 'green'}>{rul < 8 ? 'High risk' : rul < 24 ? 'Watch' : 'Healthy'}</Badge> : <Badge tone="gray">{samplesReceived} / 8 samples</Badge>}
            <p>{rul !== null ? 'Numeric estimate is available from the 8-sample inference window.' : 'RUL is not available yet. Collect at least 8 telemetry samples.'}</p>
          </div>
        </Card>
      </div>

      <Card className="monitoring-alert-strip">
        <div className="monitoring-alert-title"><AlertTriangle size={17} /><span><b>Current active alerts</b><small>{activeAlerts.length ? activeAlerts.length + ' require operator review' : 'No active alerts for this UAV'}</small></span></div>
        {activeAlerts.length ? activeAlerts.map((alert) => (
          <div className="monitoring-alert-item" key={alert.id}>
            <Badge tone={alert.severity === 'critical' ? 'red' : alert.severity === 'warning' ? 'amber' : 'blue'}>{alert.severity}</Badge>
            <span><strong>{alert.title}</strong><small>{alert.id} · {alert.message}</small></span>
          </div>
        )) : <Badge dot tone="green">All clear</Badge>}
      </Card>

      <Card className="payload-card">
        <CardHeader
          action={<Badge tone="cyan">{payloadView === 'normalized' ? 'Platform schema' : 'Adapter payload'}</Badge>}
          subtitle="Latest received values for the selected aircraft"
          title="Telemetry payload"
        />
        <div className="payload-tabs"><Tabs active={payloadView} onChange={setPayloadView} tabs={[{ id: 'normalized', label: 'Normalized values' }, { id: 'raw', label: 'Raw adapter fields' }]} /></div>
        <div className="table-scroll">
          <table className="console-table payload-table">
            <thead><tr><th>{payloadView === 'raw' ? 'Raw field' : 'Platform field'}</th><th>Latest value</th><th>Unit</th><th>Validation</th><th>Last received</th></tr></thead>
            <tbody>
              {[
                ['rpm', latest.rpm, 'rpm'],
                ['engine_load_pct', latest.engine_load_pct, '%'],
                ['egt_c', latest.egt_c, '°C'],
                ['cht_c', latest.cht_c, '°C'],
                ['oil_pressure_kpa', latest.oil_pressure_kpa, 'kPa'],
                ['engine_vibration_mm_s', latest.engine_vibration_mm_s, 'mm/s'],
              ].map(([field, value, unit]) => (
                <tr key={String(field)}>
                  <td className="mono-cell">{payloadView === 'raw' ? ({ rpm: 'engine_rpm', engine_load_pct: 'throttle_pct', egt_c: 'egt_sensor', cht_c: 'cht_sensor', oil_pressure_kpa: 'oil_press', engine_vibration_mm_s: 'vibration_x' }[String(field)] || field) : field}</td>
                  <td>{typeof value === 'number' ? number(value, String(field) === 'engine_vibration_mm_s' ? 2 : 1) : value}</td>
                  <td>{unit}</td>
                  <td><Badge tone="green">Valid</Badge></td>
                  <td>just now</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
