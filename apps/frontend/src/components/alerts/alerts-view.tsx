'use client';

import { type ReactNode, useMemo, useState } from 'react';
import { BellRing, CheckCheck, CircleAlert, ClipboardCheck, MessageSquareText, ShieldAlert } from 'lucide-react';
import { useApp } from '@/components/layout/app-providers';
import { Badge, Button, Card, EmptyState, Modal, SectionHeading, Tabs } from '@/components/ui';
import { cn, formatTimestamp } from '@/lib/utils';
import type { Alert, AlertSeverity, AlertStatus } from '@/types/alerts';
import styles from './alerts-view.module.css';

type AlertTab = 'active' | 'acknowledged' | 'resolved';

const tabToStatus: Record<AlertTab, AlertStatus> = {
  active: 'open',
  acknowledged: 'acknowledged',
  resolved: 'resolved',
};

const severityTones: Record<AlertSeverity, 'blue' | 'amber' | 'red'> = {
  info: 'blue',
  warning: 'amber',
  critical: 'red',
};

const severityLabels: Record<AlertSeverity, string> = {
  info: 'Info',
  warning: 'Warning',
  critical: 'Critical',
};

const statusTones: Record<AlertStatus, 'red' | 'amber' | 'green'> = {
  open: 'red',
  acknowledged: 'amber',
  resolved: 'green',
};

const statusLabels: Record<AlertStatus, string> = {
  open: 'Active',
  acknowledged: 'Acknowledged',
  resolved: 'Resolved',
};

function formatFault(alert: Alert): string {
  if (alert.fault_type) return alert.fault_type.replace(/_/g, ' ');
  return alert.source === 'gateway' ? 'Gateway link' : alert.source;
}

/** Operations alert workflow: inspect, acknowledge with an optional note, and resolve. */
export function AlertsView() {
  const { alerts, setSelectedUavId, updateAlertStatus, uavs } = useApp();
  const [tab, setTab] = useState<AlertTab>('active');
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null);
  const [operatorNote, setOperatorNote] = useState('');

  const selectedAlert = alerts.find((alert) => alert.id === selectedAlertId) ?? null;
  const visibleAlerts = useMemo(
    () => alerts.filter((alert) => alert.status === tabToStatus[tab]),
    [alerts, tab],
  );

  const counts = useMemo(
    () => ({
      active: alerts.filter((alert) => alert.status === 'open').length,
      acknowledged: alerts.filter((alert) => alert.status === 'acknowledged').length,
      resolved: alerts.filter((alert) => alert.status === 'resolved').length,
    }),
    [alerts],
  );

  const openDetail = (alert: Alert) => {
    setSelectedAlertId(alert.id);
    setOperatorNote('');
  };

  const closeDetail = () => {
    setSelectedAlertId(null);
    setOperatorNote('');
  };

  const changeStatus = (status: AlertStatus) => {
    if (!selectedAlert) return;
    updateAlertStatus(selectedAlert.id, status, operatorNote);
    setOperatorNote('');
  };

  const inspectUav = (uavId: string) => {
    setSelectedUavId(uavId);
    closeDetail();
  };

  return (
    <div className={styles.alertsView}>
      <SectionHeading
        eyebrow="Operations workflow"
        title="Alerts"
        description="Review predicted engine risks, telemetry exceptions, and gateway events."
        action={<Badge dot tone={counts.active ? 'red' : 'green'}>{counts.active ? `${counts.active} active` : 'All clear'}</Badge>}
      />

      <section aria-label="Alert summary" className={styles.summaryGrid}>
        <SummaryCard count={counts.active} icon={<BellRing size={17} />} label="Active" tone="red" />
        <SummaryCard count={counts.acknowledged} icon={<ClipboardCheck size={17} />} label="Acknowledged" tone="amber" />
        <SummaryCard count={counts.resolved} icon={<CheckCheck size={17} />} label="Resolved today" tone="green" />
      </section>

      <Card className={styles.alertCard}>
        <div className={styles.cardHeader}>
          <div>
            <h2>Alert queue</h2>
            <p>Open an alert to record an operator response or close the incident.</p>
          </div>
          <Tabs
            active={tab}
            onChange={(next) => setTab(next as AlertTab)}
            tabs={[
              { id: 'active', label: 'Active', count: counts.active },
              { id: 'acknowledged', label: 'Acknowledged', count: counts.acknowledged },
              { id: 'resolved', label: 'Resolved', count: counts.resolved },
            ]}
          />
        </div>

        {visibleAlerts.length ? (
          <div className={styles.tableScroll}>
            <table className={styles.alertTable}>
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Alert</th>
                  <th>UAV / engine</th>
                  <th>Predicted signal</th>
                  <th>Opened</th>
                  <th>Status</th>
                  <th><span className={styles.srOnly}>Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {visibleAlerts.map((alert) => {
                  const uav = uavs.find((item) => item.id === alert.uav_id);
                  return (
                    <tr key={alert.id}>
                      <td>
                        <Badge dot tone={severityTones[alert.severity]}>{severityLabels[alert.severity]}</Badge>
                      </td>
                      <td>
                        <button className={styles.alertName} onClick={() => openDetail(alert)} type="button">
                          <strong>{alert.title}</strong>
                          <small>{alert.id}</small>
                        </button>
                      </td>
                      <td>
                        <strong className={styles.compactValue}>{uav?.callsign ?? alert.uav_id}</strong>
                        <small>{alert.engine_id ?? 'Gateway / system event'}</small>
                      </td>
                      <td><span className={styles.faultLabel}>{formatFault(alert)}</span></td>
                      <td><time dateTime={alert.created_at}>{formatTimestamp(alert.created_at)}</time></td>
                      <td><Badge tone={statusTones[alert.status]}>{statusLabels[alert.status]}</Badge></td>
                      <td><Button onClick={() => openDetail(alert)} size="sm" variant="ghost">Inspect</Button></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={<ShieldAlert size={21} />}
            title={tab === 'active' ? 'No active alerts' : `No ${tab} alerts`}
            description={tab === 'active' ? 'The fleet is all clear. New operational alerts will appear here.' : 'Alerts moved through this workflow will remain available here.'}
          />
        )}
      </Card>

      <Modal
        description={selectedAlert ? `${selectedAlert.id} · ${selectedAlert.source} event` : undefined}
        onClose={closeDetail}
        open={Boolean(selectedAlert)}
        size="md"
        title={selectedAlert?.title ?? 'Alert detail'}
      >
        {selectedAlert ? (
          <div className={styles.detailContent}>
            <div className={styles.detailStatus}>
              <Badge dot tone={severityTones[selectedAlert.severity]}>{severityLabels[selectedAlert.severity]}</Badge>
              <Badge tone={statusTones[selectedAlert.status]}>{statusLabels[selectedAlert.status]}</Badge>
            </div>

            <section className={styles.messagePanel}>
              <CircleAlert aria-hidden="true" size={18} />
              <p>{selectedAlert.message}</p>
            </section>

            <dl className={styles.detailGrid}>
              <div><dt>UAV</dt><dd>{uavs.find((uav) => uav.id === selectedAlert.uav_id)?.callsign ?? selectedAlert.uav_id}</dd></div>
              <div><dt>Engine</dt><dd>{selectedAlert.engine_id ?? 'Not applicable'}</dd></div>
              <div><dt>Source</dt><dd>{selectedAlert.source}</dd></div>
              <div><dt>Predicted signal</dt><dd>{formatFault(selectedAlert)}</dd></div>
              <div><dt>Opened</dt><dd>{formatTimestamp(selectedAlert.created_at)}</dd></div>
              <div><dt>Acknowledged</dt><dd>{selectedAlert.acknowledged_at ? formatTimestamp(selectedAlert.acknowledged_at) : 'Not yet acknowledged'}</dd></div>
            </dl>

            {selectedAlert.status !== 'resolved' ? (
              <label className={styles.noteField} htmlFor="operator-note">
                <span><MessageSquareText aria-hidden="true" size={14} /> Operator note <em>optional</em></span>
                <textarea
                  id="operator-note"
                  maxLength={500}
                  onChange={(event) => setOperatorNote(event.target.value)}
                  placeholder="Describe the inspection, mitigation, or handoff."
                  rows={3}
                  value={operatorNote}
                />
              </label>
            ) : null}

            <div className={styles.detailActions}>
              <Button onClick={() => inspectUav(selectedAlert.uav_id)} size="sm" variant="ghost">Inspect UAV</Button>
              {selectedAlert.status === 'open' ? (
                <Button onClick={() => changeStatus('acknowledged')} size="sm" variant="secondary">Acknowledge</Button>
              ) : null}
              {selectedAlert.status !== 'resolved' ? (
                <Button onClick={() => changeStatus('resolved')} size="sm" variant="success">Resolve alert</Button>
              ) : null}
              {selectedAlert.status === 'resolved' ? <Badge tone="green">Resolution retained in history</Badge> : null}
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}

function SummaryCard({
  count,
  icon,
  label,
  tone,
}: {
  count: number;
  icon: ReactNode;
  label: string;
  tone: 'red' | 'amber' | 'green';
}) {
  return (
    <Card className={styles.summaryCard}>
      <span className={cn(styles.summaryIcon, styles[`summaryIcon${tone}`])}>{icon}</span>
      <div>
        <strong>{count}</strong>
        <p>{label}</p>
      </div>
    </Card>
  );
}

export default AlertsView;
