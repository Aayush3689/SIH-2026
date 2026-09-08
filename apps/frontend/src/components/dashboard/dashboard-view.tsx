'use client';

import { motion, useReducedMotion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BellRing,
  CircleDot,
  Clock3,
  Gauge,
  MapPinned,
  Plane,
  Plus,
  Radio,
  ShieldAlert,
  Signal,
} from 'lucide-react';
import { Badge, Button, Card, CardHeader, SectionHeading } from '@/components/ui';
import { useApp } from '@/components/layout/app-providers';
import { cn } from '@/lib/utils';

const statusTone = {
  active: 'green',
  warning: 'amber',
  critical: 'red',
  offline: 'gray',
} as const;

const statusLabel = {
  active: 'Connected',
  warning: 'Degraded',
  critical: 'At risk',
  offline: 'Offline',
} as const;

function MetricCard({
  icon,
  label,
  value,
  detail,
  tone = 'blue',
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  detail: string;
  tone?: 'blue' | 'green' | 'amber' | 'red' | 'cyan';
  onClick?: () => void;
}) {
  return (
    <button className="metric-card" onClick={onClick} type="button">
      <span className={'metric-card-icon metric-card-icon--' + tone}>{icon}</span>
      <span className="metric-card-copy">
        <small>{label}</small>
        <strong>{value}</strong>
        <span>{detail}</span>
      </span>
      <ArrowUpRight className="metric-card-arrow" size={15} />
    </button>
  );
}

export function DashboardView() {
  const router = useRouter();
  const reduceMotion = useReducedMotion();
  const { alerts, environment, openAddUav, setSelectedUavId, uavs } = useApp();
  const connected = uavs.filter((uav) => uav.status !== 'offline').length;
  const inFlight = uavs.filter((uav) => !['idle', 'maintenance'].includes(uav.mission_status)).length;
  const activeAlerts = alerts.filter((alert) => alert.status === 'open');
  const normal = uavs.filter((uav) => uav.status === 'active').length;
  const preFault = uavs.filter((uav) => uav.status === 'warning').length;
  const activeFault = uavs.filter((uav) => uav.status === 'critical').length;
  const lowRul = activeFault + preFault;

  const selectUav = (uavId: string, route = '/uav-fleet') => {
    setSelectedUavId(uavId);
    router.push(route);
  };

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className="page-enter dashboard-page"
      initial={{ opacity: 0, y: reduceMotion ? 0 : 8 }}
      transition={{ duration: reduceMotion ? 0 : 0.2 }}
    >
      <SectionHeading
        action={<Button onClick={openAddUav}><Plus size={16} /> Add UAV</Button>}
        description="Fleet readiness, engine risk, and the latest telemetry activity."
        eyebrow={<><Badge dot tone={environment === 'Demo' ? 'blue' : 'green'}>{environment} data</Badge><span className="heading-refresh"><Clock3 size={12} /> refreshed just now</span></>}
        title="Fleet Operations"
      />

      <div className="metric-grid">
        <MetricCard detail="Registered aircraft" icon={<Plane size={18} />} label="Total UAVs" onClick={() => router.push('/uav-fleet')} value={uavs.length} />
        <MetricCard detail="Telemetry sources available" icon={<Radio size={18} />} label="Connected" onClick={() => router.push('/gateway')} tone="green" value={connected} />
        <MetricCard detail="Active mission profiles" icon={<Activity size={18} />} label="In Flight" onClick={() => router.push('/live-monitoring')} tone="cyan" value={inFlight} />
        <MetricCard detail="Require operator attention" icon={<BellRing size={18} />} label="Active Alerts" onClick={() => router.push('/alerts')} tone="red" value={activeAlerts.length} />
        <MetricCard detail="Pre-fault or active-fault" icon={<ShieldAlert size={18} />} label="Engines at Risk" onClick={() => router.push('/engine-health')} tone="amber" value={preFault + activeFault} />
        <MetricCard detail="Estimated under 24 h" icon={<Gauge size={18} />} label="Low RUL" onClick={() => router.push('/engine-health')} tone="amber" value={lowRul} />
      </div>

      <div className="dashboard-main-grid">
        <Card className="fleet-map-card">
          <CardHeader
            action={<Badge dot tone="cyan">Live activity</Badge>}
            subtitle="Simulated operational positions · Sector 12"
            title="Fleet operations map"
          />
          <div className="fleet-map">
            <div className="map-grid-lines" />
            <div className="map-zone map-zone--one">NORTH CORRIDOR</div>
            <div className="map-zone map-zone--two">OPS SECTOR 12</div>
            <div className="map-route map-route--one" />
            <div className="map-route map-route--two" />
            {uavs.map((uav, index) => (
              <button
                aria-label={'Open ' + uav.id}
                className={cn('uav-marker', 'uav-marker--' + uav.status)}
                key={uav.id}
                onClick={() => selectUav(uav.id)}
                style={{ left: (22 + ((index * 23) % 64)) + '%', top: (32 + ((index * 19) % 42)) + '%' }}
                type="button"
              >
                <span className="uav-marker-core"><Plane size={14} style={{ transform: 'rotate(' + (uav.heading_deg - 45) + 'deg)' }} /></span>
                <span className="uav-marker-label"><b>{uav.callsign}</b><small>{uav.position.altitude_ft ? uav.position.altitude_ft.toLocaleString() + ' ft' : 'grounded'}</small></span>
              </button>
            ))}
            <div className="map-legend">
              <span><i className="legend-dot legend-dot--green" /> Connected</span>
              <span><i className="legend-dot legend-dot--amber" /> Degraded</span>
              <span><i className="legend-dot legend-dot--red" /> Risk</span>
            </div>
          </div>
          <div className="map-selected-strip">
            <MapPinned size={16} />
            <span>Click a UAV marker to inspect its fleet record, source, and engine context.</span>
          </div>
        </Card>

        <div className="dashboard-side-stack">
          <Card className="health-overview-card">
            <CardHeader subtitle="Current engine anomaly state" title="Health overview" />
            <div className="health-distribution">
              <button onClick={() => router.push('/engine-health')} type="button">
                <span className="health-distribution-count health-distribution-count--green">{normal}</span>
                <span><b>Normal</b><small>stable telemetry</small></span>
              </button>
              <button onClick={() => router.push('/engine-health')} type="button">
                <span className="health-distribution-count health-distribution-count--amber">{preFault}</span>
                <span><b>Pre-fault</b><small>watch signal trend</small></span>
              </button>
              <button onClick={() => router.push('/engine-health')} type="button">
                <span className="health-distribution-count health-distribution-count--red">{activeFault}</span>
                <span><b>Active fault</b><small>operator review</small></span>
              </button>
            </div>
          </Card>
          <Card className="rul-overview-card">
            <CardHeader action={<button className="quiet-link" onClick={() => router.push('/engine-health')} type="button">Review risk</button>} subtitle="Model estimate availability" title="Remaining useful life" />
            <div className="rul-buckets">
              <div><span className="rul-dot rul-dot--green" /><b>{normal}</b><small>&gt;24 h</small></div>
              <div><span className="rul-dot rul-dot--amber" /><b>{preFault}</b><small>8–24 h</small></div>
              <div><span className="rul-dot rul-dot--red" /><b>{activeFault}</b><small>&lt;8 h</small></div>
              <div><span className="rul-dot rul-dot--gray" /><b>1</b><small>unavailable</small></div>
            </div>
            <p className="rul-note">RUL appears only after the inference window collects at least 8 telemetry samples.</p>
          </Card>
        </div>
      </div>

      <div className="dashboard-lower-grid">
        <Card className="recent-alerts-card">
          <CardHeader
            action={<button className="quiet-link" onClick={() => router.push('/alerts')} type="button">View alerts</button>}
            subtitle="Latest engine, telemetry, and gateway exceptions"
            title="Recent alerts"
          />
          <div className="recent-alert-list">
            {alerts.slice(0, 6).map((alert) => (
              <button
                className="recent-alert-row"
                key={alert.id}
                onClick={() => {
                  setSelectedUavId(alert.uav_id);
                  router.push('/alerts');
                }}
                type="button"
              >
                <span className={'recent-alert-icon recent-alert-icon--' + alert.severity}>
                  <AlertTriangle size={15} />
                </span>
                <span className="recent-alert-copy">
                  <strong>{alert.title}</strong>
                  <small>{alert.uav_id} · {alert.engine_id || 'Gateway'} · {alert.id}</small>
                </span>
                <Badge tone={alert.status === 'resolved' ? 'green' : alert.severity === 'critical' ? 'red' : alert.severity === 'warning' ? 'amber' : 'blue'}>
                  {alert.status}
                </Badge>
              </button>
            ))}
          </div>
        </Card>

        <Card className="live-activity-card">
          <CardHeader
            action={<Badge dot tone="green">Streaming</Badge>}
            subtitle="Most recent packet per active vehicle"
            title="Live activity"
          />
          <div className="live-activity-list">
            {uavs.filter((uav) => uav.status !== 'offline').map((uav) => (
              <button className="live-activity-row" key={uav.id} onClick={() => selectUav(uav.id, '/live-monitoring')} type="button">
                <span className={'activity-ring activity-ring--' + uav.status}><Signal size={14} /></span>
                <span className="live-activity-name"><strong>{uav.id}</strong><small>{uav.engine_id} · {uav.mission_status}</small></span>
                <span className="live-activity-metric"><b>{uav.status === 'critical' ? '8.7' : uav.status === 'warning' ? '5.8' : '3.2'} mm/s</b><small>vibration</small></span>
                <Badge dot tone={statusTone[uav.status]}>{statusLabel[uav.status]}</Badge>
              </button>
            ))}
          </div>
          <div className="demo-footer"><CircleDot size={13} /> Demo data updates on a deterministic telemetry cycle.</div>
        </Card>
      </div>
    </motion.div>
  );
}
