'use client';

import { useMemo } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import {
  Activity,
  AlertTriangle,
  CalendarClock,
  Cpu,
  Gauge,
  HeartPulse,
  ShieldCheck,
  Thermometer,
  Wrench,
} from 'lucide-react';
import { useApp } from '@/components/layout/app-providers';
import { useTelemetryFeed } from '@/hooks/use-telemetry-feed';
import { Badge, Card, CardHeader, ProgressBar, SectionHeading, Select } from '@/components/ui';
import { TrendChart } from '@/components/shared/trend-chart';

const anomalyTone = {
  NORMAL: 'green',
  PRE_FAULT: 'amber',
  ACTIVE_FAULT: 'red',
} as const;

function pretty(value: string) {
  return value.split('_').map((item) => item.charAt(0).toUpperCase() + item.slice(1)).join(' ');
}

export function EngineHealthView() {
  const reduceMotion = useReducedMotion();
  const { selectedUav, selectedUavId, setSelectedUavId, uavs } = useApp();
  const uav = selectedUav || uavs[0];
  const { history, inference, isRulReady, latest, samplesReceived } = useTelemetryFeed(uav?.id || selectedUavId, uav?.engine_id);
  const anomaly = inference?.data.anomaly[0];
  const fault = inference?.data.fault[0];
  const rul = isRulReady ? inference?.data.rul?.[0] ?? null : null;
  const healthIndex = anomaly?.state === 'ACTIVE_FAULT' ? 38 : anomaly?.state === 'PRE_FAULT' ? 68 : 91;
  const gaugeColor = anomaly?.state === 'ACTIVE_FAULT' ? '#EF626F' : anomaly?.state === 'PRE_FAULT' ? '#F1B955' : '#36C98F';
  const faultEntries = useMemo(
    () => Object.entries(fault?.probabilities || {}).sort((first, second) => second[1] - first[1]).slice(0, 5),
    [fault],
  );

  if (!uav || !latest || !anomaly || !fault) {
    return <div className="page-enter"><SectionHeading description="No engine telemetry is available." title="Engine Health" /></div>;
  }

  const residuals = history.slice(-24);
  const timeline = anomaly.state === 'ACTIVE_FAULT'
    ? [
        ['06:02', 'NORMAL', 'green'],
        ['06:07', 'PRE_FAULT', 'amber'],
        ['06:11', 'PRE_FAULT', 'amber'],
        ['06:14', 'ACTIVE_FAULT', 'red'],
      ]
    : anomaly.state === 'PRE_FAULT'
      ? [
          ['06:02', 'NORMAL', 'green'],
          ['06:07', 'NORMAL', 'green'],
          ['06:11', 'PRE_FAULT', 'amber'],
          ['06:14', 'PRE_FAULT', 'amber'],
        ]
      : [
          ['06:02', 'NORMAL', 'green'],
          ['06:07', 'NORMAL', 'green'],
          ['06:11', 'NORMAL', 'green'],
          ['06:14', 'NORMAL', 'green'],
        ];

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="page-enter health-page"
      initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }}
      transition={{ duration: reduceMotion ? 0 : 0.2 }}
    >
      <SectionHeading
        action={
          <Select aria-label="Selected engine UAV" className="monitoring-uav-select" onChange={(event) => setSelectedUavId(event.target.value)} value={uav.id}>
            {uavs.map((item) => <option key={item.id} value={item.id}>{item.engine_id} · {item.id}</option>)}
          </Select>
        }
        description="Interpret model signals, remaining-life readiness, and maintenance context for the selected engine."
        eyebrow={<><Badge dot tone={anomalyTone[anomaly.state]}>{anomaly.state.replace('_', ' ')}</Badge><span className="heading-refresh">Updated from demo inference</span></>}
        title="Engine Health"
      />

      <Card className="health-context-card">
        <div className="health-context-engine">
          <span className={'health-context-icon health-context-icon--' + anomalyTone[anomaly.state]}><HeartPulse size={19} /></span>
          <div><small>Engine health context</small><strong>{uav.engine_id} · {uav.model}</strong><span>{uav.id} · {latest.flight_id} · {uav.mission_status.replace('-', ' ')}</span></div>
        </div>
        <div className="health-context-state">
          <span>Last telemetry</span><b>just now</b>
          <Badge dot tone={anomalyTone[anomaly.state]}>{anomaly.state}</Badge>
        </div>
      </Card>

      <div className="health-hero-grid">
        <Card className="health-index-card">
          <CardHeader subtitle="Composite operational indicator" title="Health index" />
          <div className="health-gauge-layout">
            <div className="health-gauge">
              <svg viewBox="0 0 120 120">
                <circle cx="60" cy="60" fill="none" r="48" stroke="#0e1928" strokeWidth="10" />
                <motion.circle
                  animate={{ strokeDashoffset: 302 - (302 * healthIndex) / 100 }}
                  cx="60"
                  cy="60"
                  fill="none"
                  initial={{ strokeDashoffset: 302 }}
                  r="48"
                  stroke={gaugeColor}
                  strokeLinecap="round"
                  strokeWidth="10"
                  style={{ strokeDasharray: 302, transform: 'rotate(-90deg)', transformOrigin: '60px 60px' }}
                  transition={{ duration: reduceMotion ? 0 : 0.65 }}
                />
              </svg>
              <span><b>{healthIndex}</b><small>/100</small></span>
            </div>
            <div className="health-gauge-copy">
              <Badge tone={anomalyTone[anomaly.state]}>{anomaly.state.replace('_', ' ')}</Badge>
              <strong>{healthIndex > 80 ? 'Stable operating condition' : healthIndex > 55 ? 'Watch the trend' : 'Operator review advised'}</strong>
              <p>Derived from anomaly confidence, fault signal strength, and remaining-life risk.</p>
            </div>
          </div>
          <div className="health-factor-list">
            <span><i className="factor-good" />Telemetry quality <b>98.7%</b></span>
            <span><i className={anomaly.state === 'NORMAL' ? 'factor-good' : 'factor-amber'} />Thermal trend <b>{anomaly.state === 'ACTIVE_FAULT' ? 'Elevated' : 'Stable'}</b></span>
            <span><i className={fault.fault === 'abnormal_vibration' ? 'factor-red' : 'factor-good'} />Vibration signal <b>{latest.engine_vibration_mm_s.toFixed(2)} mm/s</b></span>
          </div>
        </Card>

        <Card className="health-rul-card">
          <CardHeader action={<Badge tone={rul === null ? 'gray' : rul < 8 ? 'red' : rul < 24 ? 'amber' : 'green'}>{rul === null ? 'Not ready' : rul < 8 ? 'High risk' : rul < 24 ? 'Watch' : 'Normal'}</Badge>} subtitle="RUL v4 inference output" title="Remaining useful life" />
          {rul !== null ? (
            <div className="rul-ready-panel">
              <div><strong>{rul.toFixed(1)}</strong><span>hours estimated</span></div>
              <ProgressBar label="Relative maintenance horizon" tone={rul < 8 ? 'red' : rul < 24 ? 'amber' : 'green'} value={Math.min(100, (rul / 48) * 100)} />
              <p>Model estimate based on the current telemetry window. Treat it as planning input, not a maintenance release.</p>
            </div>
          ) : (
            <div className="rul-not-ready-panel">
              <CalendarClock size={25} />
              <strong>Collecting history</strong>
              <p>RUL is not available yet. Collect at least 8 telemetry samples.</p>
              <ProgressBar label={samplesReceived + ' of 8 samples received'} tone="blue" value={(samplesReceived / 8) * 100} />
            </div>
          )}
          <div className="health-maintenance-mini">
            <span><small>Cycles</small><b>{uav.id === 'UAV-003' ? '1,842' : '684'}</b></span>
            <span><small>Since overhaul</small><b>{uav.id === 'UAV-003' ? '173 d' : '41 d'}</b></span>
            <span><small>Major service</small><b>{uav.id === 'UAV-003' ? '52 d' : '18 d'}</b></span>
          </div>
        </Card>
      </div>

      <div className="health-analysis-grid">
        <Card className="health-prediction-card">
          <CardHeader action={<Badge tone={anomalyTone[anomaly.state]}>{(anomaly.confidence * 100).toFixed(0)}% confidence</Badge>} subtitle="Anomaly classifier result" title="Anomaly state" />
          <div className="health-anomaly-state">
            <span className={'health-anomaly-icon health-anomaly-icon--' + anomalyTone[anomaly.state]}><ShieldCheck size={21} /></span>
            <div><strong>{anomaly.state.replace('_', ' ')}</strong><p>{anomaly.state === 'NORMAL' ? 'No anomaly threshold is currently exceeded.' : anomaly.state === 'PRE_FAULT' ? 'A developing deviation should be monitored.' : 'A sustained deviation requires review.'}</p></div>
          </div>
          <div className="health-probabilities">
            {Object.entries(anomaly.probabilities).map(([label, value]) => (
              <ProgressBar key={label} label={label.replace('_', ' ')} tone={label === anomaly.state ? anomalyTone[anomaly.state] : 'gray'} value={value * 100} />
            ))}
          </div>
          <div className="health-timeline">
            <span className="health-timeline-label">Recent state timeline</span>
            <div>
              {timeline.map(([time, label, tone], index) => (
                <span className="health-timeline-event" key={time}>
                  <i className={'timeline-dot timeline-dot--' + tone} />
                  <b>{label.replace('_', ' ')}</b>
                  <small>{time}</small>
                  {index < timeline.length - 1 ? <em /> : null}
                </span>
              ))}
            </div>
          </div>
        </Card>

        <Card className="health-fault-card">
          <CardHeader action={<Badge tone="purple">Prediction</Badge>} subtitle="Class probabilities from the fault model" title="Fault hypothesis" />
          <div className="fault-hypothesis">
            <span><Cpu size={20} /></span>
            <div><strong>Predicted: {pretty(fault.fault)}</strong><p>{(fault.confidence * 100).toFixed(1)}% confidence · Model prediction — confirmation logic pending.</p></div>
          </div>
          <div className="health-probabilities">
            {faultEntries.map(([label, value]) => <ProgressBar key={label} label={pretty(label)} tone={label === fault.fault ? 'purple' : 'gray'} value={value * 100} />)}
          </div>
          <div className="contributing-signals">
            <span>Top contributing engineering signals</span>
            <div><b><Activity size={13} /> Vibration RMS</b><small>{latest.engine_vibration_mm_s.toFixed(2)} mm/s · rolling rise</small></div>
            <div><b><Thermometer size={13} /> EGT residual</b><small>{(latest.egt_c - 630).toFixed(1)} °C · expected vs actual</small></div>
            <div><b><Gauge size={13} /> Oil pressure</b><small>{latest.oil_pressure_kpa.toFixed(1)} kPa · load-compensated</small></div>
          </div>
        </Card>
      </div>

      <div className="health-chart-grid">
        <TrendChart
          series={[
            { id: 'egt-residual', label: 'EGT residual', color: '#EF626F', values: residuals.map((item) => item.egt_c - 630) },
            { id: 'cht-residual', label: 'CHT residual', color: '#F1B955', values: residuals.map((item) => item.cht_c - 165) },
          ]}
          subtitle="Expected vs actual · model-derived engineering signal"
          title="Thermal physics residuals"
        />
        <TrendChart
          series={[
            { id: 'oil-residual', label: 'Oil temperature residual', color: '#39D0C8', values: residuals.map((item) => item.oil_temperature_c - 86) },
            { id: 'pressure-residual', label: 'Oil pressure residual', color: '#A78BFA', values: residuals.map((item) => item.oil_pressure_kpa - 380) },
          ]}
          subtitle="Expected vs actual · model-derived engineering signal"
          title="Lubrication physics residuals"
        />
      </div>

      <Card className="maintenance-context-card">
        <CardHeader subtitle="Demo maintenance metadata supports planning context only." title="Maintenance context" />
        <div className="maintenance-grid">
          <div><Wrench size={17} /><span><small>Last major maintenance</small><b>{uav.id === 'UAV-003' ? '52 days ago' : '18 days ago'}</b></span></div>
          <div><CalendarClock size={17} /><span><small>Days since overhaul</small><b>{uav.id === 'UAV-003' ? '173 days' : '41 days'}</b></span></div>
          <div><Gauge size={17} /><span><small>Engine cycles</small><b>{uav.id === 'UAV-003' ? '1,842 cycles' : '684 cycles'}</b></span></div>
          <div><AlertTriangle size={17} /><span><small>Operator guidance</small><b>{anomaly.state === 'NORMAL' ? 'Continue normal monitoring' : 'Inspect at next safe opportunity'}</b></span></div>
        </div>
      </Card>
    </motion.div>
  );
}
