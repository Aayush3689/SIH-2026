'use client';

import { useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Activity,
  Battery,
  ChevronRight,
  CircleDot,
  Edit3,
  Filter,
  Gauge,
  Plane,
  Plus,
  Radio,
  Search,
  Signal,
  Wrench,
} from 'lucide-react';
import { toast } from 'sonner';
import { useApp } from '@/components/layout/app-providers';
import type { MissionStatus, Uav } from '@/types/uav';
import { Badge, Button, Card, CardHeader, EmptyState, Input, Modal, ProgressBar, SectionHeading, Select, Tabs } from '@/components/ui';
import { cn } from '@/lib/utils';

const statusTone = {
  active: 'green',
  warning: 'amber',
  critical: 'red',
  offline: 'gray',
} as const;

const statusCopy = {
  active: 'Connected',
  warning: 'Degraded',
  critical: 'At risk',
  offline: 'Offline',
} as const;

export function FleetView() {
  const router = useRouter();
  const { openAddUav, selectedUav, selectedUavId, setSelectedUavId, updateUav, uavs } = useApp();
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [tab, setTab] = useState('all');
  const [editing, setEditing] = useState<Uav | null>(null);
  const [editName, setEditName] = useState('');
  const [editModel, setEditModel] = useState('');
  const [editMission, setEditMission] = useState<MissionStatus>('idle');

  const filtered = useMemo(() => {
    const term = query.trim().toLowerCase();
    return uavs.filter((uav) => {
      const matchesText = !term || (uav.id + ' ' + uav.callsign + ' ' + uav.name + ' ' + uav.engine_id).toLowerCase().includes(term);
      const matchesStatus = statusFilter === 'all' || uav.status === statusFilter;
      const matchesTab =
        tab === 'all' ||
        (tab === 'airborne' && !['idle', 'maintenance'].includes(uav.mission_status)) ||
        (tab === 'attention' && ['warning', 'critical'].includes(uav.status)) ||
        (tab === 'maintenance' && uav.mission_status === 'maintenance');
      return matchesText && matchesStatus && matchesTab;
    });
  }, [query, statusFilter, tab, uavs]);

  const startEdit = (uav: Uav) => {
    setEditing(uav);
    setEditName(uav.name);
    setEditModel(uav.model);
    setEditMission(uav.mission_status);
  };

  const saveEdit = () => {
    if (!editing) return;
    if (!editName.trim()) {
      toast.error('UAV name is required.');
      return;
    }
    updateUav(editing.id, { name: editName.trim(), model: editModel.trim() || editing.model, mission_status: editMission });
    setEditing(null);
  };

  const selected = selectedUav || uavs[0];

  return (
    <div className="page-enter fleet-page">
      <SectionHeading
        action={<Button onClick={openAddUav}><Plus size={16} /> Add UAV</Button>}
        description="Inventory, operating state, engine context, and source readiness for every aircraft."
        eyebrow="Aircraft inventory"
        title="UAV Fleet"
      />

      <div className="fleet-summary">
        <div><span>Registered</span><b>{uavs.length}</b><small>Aircraft in console</small></div>
        <div><span>In flight</span><b className="value-cyan">{uavs.filter((uav) => !['idle', 'maintenance'].includes(uav.mission_status)).length}</b><small>Active mission profiles</small></div>
        <div><span>Needs attention</span><b className="value-amber">{uavs.filter((uav) => ['warning', 'critical'].includes(uav.status)).length}</b><small>Degraded or at risk</small></div>
        <div><span>Source coverage</span><b className="value-good">{uavs.filter((uav) => uav.gateway_id || uav.status === 'active').length}/{uavs.length}</b><small>Connected or simulated</small></div>
      </div>

      <div className="fleet-layout">
        <Card className="fleet-table-card">
          <CardHeader
            action={<Badge dot tone="blue">Demo fleet</Badge>}
            subtitle="Select a row to inspect its operational context"
            title="Aircraft inventory"
          />
          <div className="fleet-tabs-wrap">
            <Tabs
              active={tab}
              onChange={setTab}
              tabs={[
                { id: 'all', label: 'All', count: uavs.length },
                { id: 'airborne', label: 'Airborne', count: uavs.filter((uav) => !['idle', 'maintenance'].includes(uav.mission_status)).length },
                { id: 'attention', label: 'Attention', count: uavs.filter((uav) => ['warning', 'critical'].includes(uav.status)).length },
                { id: 'maintenance', label: 'Maintenance', count: uavs.filter((uav) => uav.mission_status === 'maintenance').length },
              ]}
            />
          </div>
          <div className="fleet-filters">
            <label className="inline-search">
              <Search size={15} />
              <input onChange={(event) => setQuery(event.target.value)} placeholder="Search ID, callsign, engine…" value={query} />
            </label>
            <Select aria-label="Filter by status" className="fleet-status-select" onChange={(event) => setStatusFilter(event.target.value)} value={statusFilter}>
              <option value="all">All states</option>
              <option value="active">Connected</option>
              <option value="warning">Degraded</option>
              <option value="critical">At risk</option>
              <option value="offline">Offline</option>
            </Select>
          </div>
          {filtered.length ? (
            <div className="table-scroll">
              <table className="console-table fleet-table">
                <thead>
                  <tr>
                    <th>Aircraft</th>
                    <th>Mission</th>
                    <th>Engine</th>
                    <th>Telemetry source</th>
                    <th>Link</th>
                    <th>State</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((uav) => (
                    <tr className={cn(uav.id === selectedUavId && 'is-selected')} key={uav.id} onClick={() => setSelectedUavId(uav.id)}>
                      <td>
                        <div className="aircraft-cell">
                          <span className={'aircraft-icon aircraft-icon--' + uav.status}><Plane size={15} /></span>
                          <span><strong>{uav.id}</strong><small>{uav.callsign} · {uav.name}</small></span>
                        </div>
                      </td>
                      <td><span className="mission-cell">{uav.mission_status.replace('-', ' ')}</span></td>
                      <td><span className="mono-cell">{uav.engine_id}</span><small>{uav.model}</small></td>
                      <td><span className="source-cell"><Radio size={13} /> {uav.gateway_id || 'Awaiting source'}</span></td>
                      <td><span className="link-cell"><Signal size={13} />{uav.signal_strength_pct}%</span></td>
                      <td><Badge dot tone={statusTone[uav.status]}>{statusCopy[uav.status]}</Badge></td>
                      <td>
                        <div className="row-actions">
                          <button aria-label={'Edit ' + uav.id} onClick={(event) => { event.stopPropagation(); startEdit(uav); }} type="button"><Edit3 size={14} /></button>
                          <button aria-label={'Monitor ' + uav.id} onClick={(event) => { event.stopPropagation(); setSelectedUavId(uav.id); router.push('/live-monitoring'); }} type="button"><ChevronRight size={16} /></button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              icon={<Filter size={21} />}
              title="No UAVs match these filters"
              description="Try clearing the search or status filter, or register a new aircraft."
              action={<Button onClick={openAddUav} size="sm"><Plus size={14} /> Add UAV</Button>}
            />
          )}
        </Card>

        {selected ? (
          <Card className="fleet-detail-card">
            <CardHeader
              action={<Badge dot tone={statusTone[selected.status]}>{statusCopy[selected.status]}</Badge>}
              subtitle={selected.callsign + ' · ' + selected.model}
              title={selected.id}
            />
            <div className="fleet-detail-body">
              <div className="detail-hero">
                <div className={'detail-hero-icon detail-hero-icon--' + selected.status}><Plane size={25} /></div>
                <div><strong>{selected.name}</strong><span>{selected.mission_status.replace('-', ' ')}</span></div>
              </div>
              <div className="detail-location">
                <CircleDot size={15} />
                <span><b>{selected.position.altitude_ft.toLocaleString()} ft</b><small>{selected.speed_kts} kt · heading {selected.heading_deg}°</small></span>
              </div>
              <div className="detail-health">
                <div><span><Gauge size={14} /> Engine</span><b>{selected.engine_id}</b></div>
                <div><span><Battery size={14} /> Battery</span><b>{selected.battery_pct}%</b></div>
              </div>
              <ProgressBar label="Telemetry signal" tone={selected.signal_strength_pct > 80 ? 'green' : selected.signal_strength_pct > 50 ? 'amber' : 'red'} value={selected.signal_strength_pct} />
              <div className="detail-source">
                <span><Radio size={14} /> Source</span>
                <b>{selected.gateway_id || 'No gateway assigned'}</b>
              </div>
              <div className="detail-actions">
                <Button onClick={() => router.push('/live-monitoring')} size="sm"><Activity size={14} /> Monitor</Button>
                <Button onClick={() => router.push('/engine-health')} size="sm" variant="secondary"><Gauge size={14} /> Health</Button>
                <Button onClick={() => startEdit(selected)} size="sm" variant="ghost"><Edit3 size={14} /> Edit</Button>
              </div>
            </div>
          </Card>
        ) : null}
      </div>

      <Modal
        description={editing ? 'Update the displayed metadata for this demo aircraft.' : undefined}
        onClose={() => setEditing(null)}
        open={Boolean(editing)}
        size="md"
        title={editing ? 'Edit ' + editing.id : 'Edit UAV'}
      >
        <div className="edit-uav-form">
          <Input label="UAV name" onChange={(event) => setEditName(event.target.value)} value={editName} />
          <Input label="Engine model" onChange={(event) => setEditModel(event.target.value)} value={editModel} />
          <Select label="Mission state" onChange={(event) => setEditMission(event.target.value as MissionStatus)} value={editMission}>
            <option value="patrolling">Patrolling</option>
            <option value="en-route">En route</option>
            <option value="returning">Returning</option>
            <option value="idle">Idle</option>
            <option value="maintenance">Maintenance</option>
          </Select>
          <div className="modal-form-footer">
            <Button onClick={() => setEditing(null)} variant="ghost">Cancel</Button>
            <Button onClick={saveEdit}><Wrench size={15} /> Save details</Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
