'use client';

import { type ReactNode, useMemo, useState } from 'react';
import {
  Activity,
  Cable,
  CircleAlert,
  Map,
  Pencil,
  Play,
  Power,
  Search,
  ServerCog,
  Trash2,
  Wifi,
} from 'lucide-react';
import { toast } from 'sonner';
import { useApp } from '@/components/layout/app-providers';
import { Badge, Button, Card, EmptyState, IconButton, Input, Modal, SectionHeading, Select } from '@/components/ui';
import { demoService } from '@/lib/services';
import { cn, formatTimestamp } from '@/lib/utils';
import type { DataSourceStatus, GatewayDataSource } from '@/types/gateway';
import styles from './gateway-view.module.css';

type StatusFilter = 'all' | DataSourceStatus;
type GatewayPanel = 'details' | 'mapping' | 'edit' | null;

const statusLabels: Record<DataSourceStatus, string> = {
  connected: 'Connected',
  connecting: 'Connecting',
  error: 'Attention',
  disabled: 'Disabled',
};

const statusTones: Record<DataSourceStatus, 'green' | 'blue' | 'amber' | 'gray'> = {
  connected: 'green',
  connecting: 'blue',
  error: 'amber',
  disabled: 'gray',
};

function sourceTypeLabel(type: GatewayDataSource['type']): string {
  return type === 'REAL_UAV' ? 'Real UAV' : type === 'GCS' ? 'GCS' : 'Simulator';
}

function sourceIcon(type: GatewayDataSource['type']) {
  if (type === 'SIMULATOR') return <Activity aria-hidden="true" size={16} />;
  if (type === 'GCS') return <ServerCog aria-hidden="true" size={16} />;
  return <Wifi aria-hidden="true" size={16} />;
}

/**
 * Gateway operations view. Adapter mutations are intentionally local demo-state
 * changes; a future backend adapter service can replace this state boundary.
 */
export function GatewayView() {
  const { openAddUav, uavs } = useApp();
  const [sources, setSources] = useState<GatewayDataSource[]>(() => demoService.listDataSources());
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [typeFilter, setTypeFilter] = useState<'all' | GatewayDataSource['type']>('all');
  const [testingId, setTestingId] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<GatewayDataSource | null>(null);
  const [panel, setPanel] = useState<GatewayPanel>(null);
  const [removeTarget, setRemoveTarget] = useState<GatewayDataSource | null>(null);
  const [editName, setEditName] = useState('');
  const [editProtocol, setEditProtocol] = useState('');

  const visibleSources = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return sources.filter((source) => {
      const matchesQuery =
        !normalizedQuery ||
        [source.id, source.name, source.uav_id ?? '', source.protocol, source.type]
          .join(' ')
          .toLowerCase()
          .includes(normalizedQuery);
      const matchesStatus = statusFilter === 'all' || source.status === statusFilter;
      const matchesType = typeFilter === 'all' || source.type === typeFilter;
      return matchesQuery && matchesStatus && matchesType;
    });
  }, [query, sources, statusFilter, typeFilter]);

  const summary = useMemo(
    () => ({
      connected: sources.filter((source) => source.status === 'connected').length,
      active: sources.filter((source) => source.status !== 'disabled').length,
      rate: sources.reduce((total, source) => total + (source.status === 'disabled' ? 0 : source.packet_rate_hz), 0),
      attention: sources.filter((source) => source.status === 'error').length,
    }),
    [sources],
  );

  const updateSource = (id: string, update: (source: GatewayDataSource) => GatewayDataSource) => {
    setSources((current) => current.map((source) => (source.id === id ? update(source) : source)));
    setSelectedSource((current) => {
      if (current && current.id === id) return update(current);
      return current;
    });
  };

  const openPanel = (source: GatewayDataSource, nextPanel: Exclude<GatewayPanel, null>) => {
    setSelectedSource(source);
    setPanel(nextPanel);
  };

  const openEdit = (source: GatewayDataSource) => {
    setSelectedSource(source);
    setEditName(source.name);
    setEditProtocol(source.protocol);
    setPanel('edit');
  };

  const saveEdit = () => {
    if (!selectedSource) return;
    if (!editName.trim()) {
      toast.error('Data source name is required.');
      return;
    }
    updateSource(selectedSource.id, (current) => ({
      ...current,
      name: editName.trim(),
      protocol: editProtocol as GatewayDataSource['protocol'],
    }));
    toast.success('Data source configuration saved.', { description: 'Updated in the local demo session.' });
    closePanel();
  };

  const closePanel = () => {
    setPanel(null);
    setSelectedSource(null);
  };

  const handleToggle = (source: GatewayDataSource) => {
    const enabling = source.status === 'disabled';
    updateSource(source.id, (current) => ({
      ...current,
      status: enabling ? 'connected' : 'disabled',
      last_packet_at: enabling ? '2026-09-09T06:15:00.000Z' : current.last_packet_at,
    }));
    toast.success(enabling ? 'Data source enabled.' : 'Data source disabled.', {
      description: source.name,
    });
  };

  const handleTest = (source: GatewayDataSource) => {
    if (source.status === 'disabled') {
      toast.error('Enable this data source before testing its connection.');
      return;
    }

    setTestingId(source.id);
    window.setTimeout(() => {
      const testedLatency = source.type === 'SIMULATOR' ? 8 : source.latency_ms ?? 48;
      updateSource(source.id, (current) => ({
        ...current,
        status: 'connected',
        latency_ms: testedLatency,
        last_packet_at: '2026-09-09T06:15:00.000Z',
      }));
      setTestingId(null);
      toast.success('Connection test passed.', {
        description: `${source.name} responded in ${testedLatency} ms.`,
      });
    }, 700);
  };

  const confirmRemoval = () => {
    if (!removeTarget) return;

    setSources((current) => current.filter((source) => source.id !== removeTarget.id));
    if (selectedSource?.id === removeTarget.id) closePanel();
    toast.success('Data source removed from the demo gateway.', { description: removeTarget.name });
    setRemoveTarget(null);
  };

  const selectedUav = selectedSource?.uav_id
    ? uavs.find((uav) => uav.id === selectedSource.uav_id)
    : undefined;
  const selectedMappings = selectedSource?.uav_id
    ? demoService.listMappings().filter((mapping) => mapping.uav_id === selectedSource.uav_id)
    : [];

  return (
    <div className={styles.gatewayView}>
      <SectionHeading
        eyebrow="Gateway operations"
        title="Gateway Data Sources"
        description="Monitor adapter connectivity, packet flow, and telemetry mappings."
        action={
          <Button onClick={openAddUav} size="sm">
            <Cable aria-hidden="true" size={15} />
            Add Data Source
          </Button>
        }
      />

      <section aria-label="Gateway summary" className={styles.summaryGrid}>
        <SummaryCard icon={<Wifi size={17} />} label="Connected sources" value={String(summary.connected)} hint={`${sources.length} configured`} tone="green" />
        <SummaryCard icon={<ServerCog size={17} />} label="Active adapters" value={String(summary.active)} hint="Enabled for routing" tone="blue" />
        <SummaryCard icon={<Activity size={17} />} label="Data rate" value={`${summary.rate} Hz`} hint="Aggregate packet flow" tone="cyan" />
        <SummaryCard icon={<CircleAlert size={17} />} label="Needs attention" value={String(summary.attention)} hint="Adapters with an error" tone="amber" />
      </section>

      <Card className={styles.managementCard}>
        <div className={styles.tableHeader}>
          <div>
            <h2>Adapter management</h2>
            <p>{visibleSources.length} of {sources.length} data sources shown</p>
          </div>
          <div className={styles.filters}>
            <label className={styles.searchField}>
              <Search aria-hidden="true" size={15} />
              <span className={styles.srOnly}>Search data sources</span>
              <input
                aria-label="Search data sources"
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search source, UAV, protocol"
                type="search"
                value={query}
              />
            </label>
            <label className={styles.selectField}>
              <span className="sr-only">Filter by status</span>
              <select aria-label="Filter by status" onChange={(event) => setStatusFilter(event.target.value as StatusFilter)} value={statusFilter}>
                <option value="all">All statuses</option>
                <option value="connected">Connected</option>
                <option value="connecting">Connecting</option>
                <option value="error">Needs attention</option>
                <option value="disabled">Disabled</option>
              </select>
            </label>
            <label className={styles.selectField}>
              <span className="sr-only">Filter by source type</span>
              <select
                aria-label="Filter by source type"
                onChange={(event) => setTypeFilter(event.target.value as 'all' | GatewayDataSource['type'])}
                value={typeFilter}
              >
                <option value="all">All types</option>
                <option value="GCS">GCS</option>
                <option value="REAL_UAV">Real UAV</option>
                <option value="SIMULATOR">Simulator</option>
              </select>
            </label>
          </div>
        </div>

        {visibleSources.length ? (
          <div className={styles.tableScroll}>
            <table className={styles.sourceTable}>
              <thead>
                <tr>
                  <th>Data source</th>
                  <th>Type / adapter</th>
                  <th>Assigned UAV</th>
                  <th>Connection health</th>
                  <th>Data rate</th>
                  <th>Mappings</th>
                  <th>Status</th>
                  <th><span className={styles.srOnly}>Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {visibleSources.map((source) => {
                  const uav = source.uav_id ? uavs.find((item) => item.id === source.uav_id) : undefined;
                  const isTesting = testingId === source.id;
                  return (
                    <tr key={source.id}>
                      <td>
                        <button className={styles.sourceName} onClick={() => openPanel(source, 'details')} type="button">
                          <span className={cn(styles.sourceIcon, styles[`sourceIcon${source.type}`])}>{sourceIcon(source.type)}</span>
                          <span>
                            <strong>{source.name}</strong>
                            <small>{source.id}</small>
                          </span>
                        </button>
                      </td>
                      <td>
                        <strong className={styles.compactValue}>{sourceTypeLabel(source.type)}</strong>
                        <small>{source.protocol}</small>
                      </td>
                      <td>
                        <strong className={styles.compactValue}>{uav?.callsign ?? source.uav_id ?? 'Unassigned'}</strong>
                        <small>{uav?.engine_id ?? 'No engine assigned'}</small>
                      </td>
                      <td>
                        <strong className={styles.compactValue}>{source.latency_ms === null ? '—' : `${source.latency_ms} ms`}</strong>
                        <small>{source.last_packet_at ? `Last packet ${formatTimestamp(source.last_packet_at)}` : 'No packets received'}</small>
                      </td>
                      <td><span className={styles.numeric}>{source.status === 'disabled' ? '0 Hz' : `${source.packet_rate_hz} Hz`}</span></td>
                      <td><span className={styles.numeric}>{source.mapping_count}</span></td>
                      <td><Badge dot tone={statusTones[source.status]}>{statusLabels[source.status]}</Badge></td>
                      <td>
                        <div className={styles.actions}>
                          <Button loading={isTesting} onClick={() => handleTest(source)} size="sm" variant="ghost">
                            <Play aria-hidden="true" size={13} />
                            Test
                          </Button>
                          <IconButton label={'Edit ' + source.name} onClick={() => openEdit(source)}>
                            <Pencil aria-hidden="true" size={15} />
                          </IconButton>
                          <IconButton label={source.status === 'disabled' ? `Enable ${source.name}` : `Disable ${source.name}`} onClick={() => handleToggle(source)}>
                            <Power aria-hidden="true" size={15} />
                          </IconButton>
                          <IconButton label={`View mapping for ${source.name}`} onClick={() => openPanel(source, 'mapping')}>
                            <Map aria-hidden="true" size={15} />
                          </IconButton>
                          <IconButton label={`Remove ${source.name}`} onClick={() => setRemoveTarget(source)}>
                            <Trash2 aria-hidden="true" size={15} />
                          </IconButton>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            icon={<Search size={20} />}
            title="No data sources match these filters"
            description="Clear or change a filter to see configured gateway adapters."
            action={<Button onClick={() => { setQuery(''); setStatusFilter('all'); setTypeFilter('all'); }} size="sm" variant="secondary">Clear filters</Button>}
          />
        )}
      </Card>

      <Modal
        description="Connection configuration and the most recent adapter activity."
        onClose={closePanel}
        open={panel === 'details' && Boolean(selectedSource)}
        size="md"
        title={selectedSource?.name ?? 'Data source details'}
      >
        {selectedSource ? (
          <div className={styles.details}>
            <div className={styles.detailTopline}>
              <Badge dot tone={statusTones[selectedSource.status]}>{statusLabels[selectedSource.status]}</Badge>
              <span>{selectedSource.id}</span>
            </div>
            <dl className={styles.detailGrid}>
              <div><dt>Source type</dt><dd>{sourceTypeLabel(selectedSource.type)}</dd></div>
              <div><dt>Adapter</dt><dd>{selectedSource.protocol}</dd></div>
              <div><dt>Assigned UAV</dt><dd>{selectedUav ? `${selectedUav.callsign} · ${selectedUav.id}` : 'Unassigned'}</dd></div>
              <div><dt>Engine</dt><dd>{selectedUav?.engine_id ?? '—'}</dd></div>
              <div><dt>Latency</dt><dd>{selectedSource.latency_ms === null ? 'No measurement' : `${selectedSource.latency_ms} ms`}</dd></div>
              <div><dt>Packet rate</dt><dd>{selectedSource.packet_rate_hz} Hz</dd></div>
              <div><dt>Latest packet</dt><dd>{selectedSource.last_packet_at ? formatTimestamp(selectedSource.last_packet_at) : 'No packets received'}</dd></div>
              <div><dt>Mapped fields</dt><dd>{selectedSource.mapping_count}</dd></div>
            </dl>
            <div className={styles.detailActions}>
              <Button onClick={() => handleTest(selectedSource)} loading={testingId === selectedSource.id} size="sm" variant="secondary">Test connection</Button>
              <Button onClick={() => openEdit(selectedSource)} size="sm" variant="ghost">Edit configuration</Button>
              <Button onClick={() => setPanel('mapping')} size="sm" variant="ghost">View mapping</Button>
            </div>
          </div>
        ) : null}
      </Modal>

      <Modal
        description="Changes are applied only to this frontend demo session."
        onClose={closePanel}
        open={panel === 'edit' && Boolean(selectedSource)}
        size="md"
        title={selectedSource ? 'Edit ' + selectedSource.name : 'Edit data source'}
      >
        {selectedSource ? (
          <div className={styles.details}>
            <Input label="Data source name" onChange={(event) => setEditName(event.target.value)} value={editName} />
            <Select label="Adapter protocol" onChange={(event) => setEditProtocol(event.target.value)} value={editProtocol}>
              {['TCP', 'UDP', 'REST API', 'MAVLink', 'Own Protocol', 'CAN'].map((protocol) => <option key={protocol}>{protocol}</option>)}
            </Select>
            <div className={styles.detailActions}>
              <Button onClick={closePanel} size="sm" variant="secondary">Cancel</Button>
              <Button onClick={saveEdit} size="sm">Save configuration</Button>
            </div>
          </div>
        ) : null}
      </Modal>

      <Modal
        description="Current raw-field to platform-parameter assignments for this source."
        onClose={closePanel}
        open={panel === 'mapping' && Boolean(selectedSource)}
        size="md"
        title={selectedSource ? `${selectedSource.name} mapping` : 'Data mapping'}
      >
        {selectedSource ? (
          <div className={styles.mappingPanel}>
            {selectedMappings.length ? (
              <ul className={styles.mappingList}>
                {selectedMappings.map((mapping) => (
                  <li key={mapping.id}>
                    <code>{mapping.source_field}</code>
                    <span aria-hidden="true">→</span>
                    <code>{mapping.target_field}</code>
                    <Badge tone="green">Valid</Badge>
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState
                icon={<Map size={20} />}
                title="No finalized mapping"
                description="This adapter has not yet saved telemetry field mappings in the demo data."
              />
            )}
            <Button onClick={closePanel} size="sm" variant="secondary">Close</Button>
          </div>
        ) : null}
      </Modal>

      <Modal
        description="This removes the adapter from the local demo session. It does not contact any physical gateway."
        onClose={() => setRemoveTarget(null)}
        open={Boolean(removeTarget)}
        size="sm"
        title="Remove data source?"
      >
        <div className={styles.confirmation}>
          <p><strong>{removeTarget?.name}</strong> will no longer appear in gateway operations.</p>
          <div className={styles.detailActions}>
            <Button onClick={() => setRemoveTarget(null)} size="sm" variant="secondary">Cancel</Button>
            <Button onClick={confirmRemoval} size="sm" variant="danger">Remove source</Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

function SummaryCard({
  icon,
  label,
  value,
  hint,
  tone,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  hint: string;
  tone: 'green' | 'blue' | 'cyan' | 'amber';
}) {
  return (
    <Card className={styles.summaryCard}>
      <span className={cn(styles.summaryIcon, styles[`summaryIcon${tone}`])}>{icon}</span>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        <small>{hint}</small>
      </div>
    </Card>
  );
}

export default GatewayView;
