'use client';

import { useMemo, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import {
  Activity,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Database,
  Gauge,
  MonitorCog,
  Network,
  Plane,
  PlugZap,
  Radar,
  Radio,
  ShieldCheck,
  Waves,
} from 'lucide-react';
import { toast } from 'sonner';
import { demoService } from '@/lib/services';
import { useApp } from '@/components/layout/app-providers';
import {
  Badge,
  Button,
  Card,
  Input,
  Modal,
  Select,
} from '@/components/ui';
import { cn } from '@/lib/utils';

type SourceType = 'GCS' | 'REAL UAV' | 'SIMULATOR';
type ConnectionState =
  | 'Idle'
  | 'Validating'
  | 'Connecting'
  | 'Receiving Data'
  | 'Connected'
  | 'Timeout'
  | 'Connection Refused'
  | 'Authentication Failed'
  | 'No Data'
  | 'Parser Error';

interface CanInterface {
  interface: string;
  type: string;
  status: string;
  bitrate: number;
  framesPerSecond: number;
  busLoadPercent: number;
  source: string;
}

const steps = ['Details', 'Source', 'Adapter', 'Connection', 'Data Preview', 'Mapping', 'Validate', 'Ready'];
const requiredTargets = [
  { id: 'engine-rpm', name: 'Engine RPM', unit: 'rpm', raw: 'engine_rpm', required: true },
  { id: 'oil-temperature', name: 'Oil Temperature', unit: '°C', raw: 'oil_temp', required: true },
  { id: 'oil-pressure', name: 'Oil Pressure', unit: 'kPa', raw: 'oil_press', required: true },
  { id: 'egt', name: 'Exhaust Gas Temperature', unit: '°C', raw: 'egt_sensor', required: true },
  { id: 'fuel-flow', name: 'Fuel Flow', unit: 'L/h', raw: 'fuel_rate', required: true },
  { id: 'vibration', name: 'Vibration', unit: 'mm/s', raw: 'vibration_x', required: true },
  { id: 'battery', name: 'Battery Voltage', unit: 'V', raw: '', required: false },
];

const rawRows = [
  { key: 'engine_rpm', value: '5,128 rpm', unit: 'rpm', source: 'CAN 0x182' },
  { key: 'oil_temp', value: '88.4 °C', unit: '°C', source: 'CAN 0x191' },
  { key: 'oil_press', value: '381.8 kPa', unit: 'kPa', source: 'CAN 0x192' },
  { key: 'egt_sensor', value: '642.1 °C', unit: '°C', source: 'CAN 0x1A4' },
  { key: 'fuel_rate', value: '13.7 L/h', unit: 'L/h', source: 'CAN 0x1B0' },
  { key: 'vibration_x', value: '3.42 mm/s', unit: 'mm/s', source: 'CAN 0x1C5' },
  { key: 'battery_voltage', value: '25.18 V', unit: 'V', source: 'CAN 0x205' },
];

const sources: Array<{
  type: SourceType;
  title: string;
  description: string;
  icon: typeof Radio;
  protocols: string[];
}> = [
  {
    type: 'GCS',
    title: 'Ground Control Station',
    description: 'Route decoded flight telemetry from an existing ground-control feed.',
    icon: Radio,
    protocols: ['TCP', 'UDP', 'REST API', 'MAVLink', 'Own Protocol'],
  },
  {
    type: 'REAL UAV',
    title: 'Physical UAV',
    description: 'Configure a managed backend adapter for a live aircraft telemetry stream.',
    icon: Plane,
    protocols: ['TCP', 'UDP', 'REST API', 'MAVLink', 'Own Protocol'],
  },
  {
    type: 'SIMULATOR',
    title: 'UAV / Engine Simulator',
    description: 'Run deterministic CAN and engine telemetry for a safe operational demo.',
    icon: MonitorCog,
    protocols: ['Simulator CAN', 'Playback telemetry'],
  },
];

const adapterOptions = ['TCP', 'UDP', 'REST API', 'MAVLink', 'Own Protocol'];
const outcomes: ConnectionState[] = [
  'Connected',
  'Timeout',
  'Connection Refused',
  'Authentication Failed',
  'No Data',
  'Parser Error',
];

const connectionCopy: Record<Exclude<ConnectionState, 'Idle' | 'Validating' | 'Connecting' | 'Receiving Data' | 'Connected'>, string> = {
  Timeout: 'Connection timed out. Check the host and port and try again.',
  'Connection Refused': 'Connection refused. Verify the source is running.',
  'Authentication Failed': 'Authentication failed. Verify the access token and try again.',
  'No Data': 'Connected, but no telemetry data was received.',
  'Parser Error': 'Incoming data could not be mapped to the required parameters.',
};

function toneForConnection(state: ConnectionState) {
  if (state === 'Connected') return 'green' as const;
  if (state === 'Idle') return 'gray' as const;
  if (state === 'Validating' || state === 'Connecting' || state === 'Receiving Data') return 'blue' as const;
  return 'red' as const;
}

export function AddUavWizard() {
  const { addUav, closeAddUav, isAddUavOpen } = useApp();
  const reduceMotion = useReducedMotion();
  const [step, setStep] = useState(0);
  const [source, setSource] = useState<SourceType | null>(null);
  const [adapter, setAdapter] = useState('MAVLink');
  const [connectionState, setConnectionState] = useState<ConnectionState>('Idle');
  const [testOutcome, setTestOutcome] = useState<ConnectionState>('Connected');
  const [canInterfaces, setCanInterfaces] = useState<CanInterface[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [selectedCan, setSelectedCan] = useState('');
  const [simulationRunning, setSimulationRunning] = useState(true);
  const [speed, setSpeed] = useState('1×');
  const [mapping, setMapping] = useState<Record<string, string>>(() =>
    Object.fromEntries(requiredTargets.map((target) => [target.id, target.raw])),
  );
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [form, setForm] = useState({
    name: '',
    id: '',
    engineId: '',
    engineModel: '',
    manufacturer: '',
    missionProfile: 'Inspection',
    description: '',
    host: '127.0.0.1',
    port: '14550',
    endpoint: '/telemetry',
    timeout: '5000',
    retry: '3',
    token: '',
    protocolName: '',
    parserFormat: 'JSON',
  });

  const selectedCanInterface = canInterfaces.find((item) => item.interface === selectedCan);
  const mappedRequired = requiredTargets.filter((target) => target.required && mapping[target.id]).length;
  const requiredCount = requiredTargets.filter((target) => target.required).length;
  const mappingValid = mappedRequired === requiredCount;
  const telemetry = useMemo(() => demoService.getTelemetry('UAV-001').slice(-1)[0], []);

  const resetAndClose = () => {
    closeAddUav();
    window.setTimeout(() => setStep(0), 180);
  };

  const setFormValue = (field: keyof typeof form, value: string) => {
    setForm((current) => ({ ...current, [field]: value }));
    if (errors[field]) {
      setErrors((current) => ({ ...current, [field]: '' }));
    }
  };

  const validateDetails = () => {
    const nextErrors: Record<string, string> = {};
    if (!form.name.trim()) nextErrors.name = 'UAV name is required.';
    if (!form.id.trim()) nextErrors.id = 'UAV ID is required.';
    if (!form.engineId.trim()) nextErrors.engineId = 'Engine ID is required.';
    if (!form.engineModel.trim()) nextErrors.engineModel = 'Engine model is required.';
    if (!form.manufacturer.trim()) nextErrors.manufacturer = 'Manufacturer is required.';
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) {
      toast.error('Complete the required UAV details before continuing.');
      return false;
    }
    return true;
  };

  const validateConnection = () => {
    if (source === 'SIMULATOR') {
      if (!selectedCan) {
        toast.error('Select a simulator CAN interface before continuing.');
        return false;
      }
      return true;
    }
    const nextErrors: Record<string, string> = {};
    if (adapter !== 'REST API' && !form.host.trim()) nextErrors.host = 'Enter a valid host or IP address.';
    const port = Number(form.port);
    if (adapter !== 'REST API' && (!Number.isInteger(port) || port < 1 || port > 65535)) {
      nextErrors.port = 'Port must be between 1 and 65535.';
    }
    if (adapter === 'REST API' && !form.endpoint.trim()) nextErrors.endpoint = 'Endpoint URL is required.';
    if (adapter === 'Own Protocol' && !form.protocolName.trim()) nextErrors.protocolName = 'Protocol name is required.';
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) {
      toast.error('Review the connection configuration.');
      return false;
    }
    if (connectionState !== 'Connected') {
      toast.error('Test the connection successfully before continuing.');
      return false;
    }
    return true;
  };

  const runCanScan = () => {
    setIsScanning(true);
    setCanInterfaces([]);
    window.setTimeout(() => {
      const detected = demoService.listCanInterfaces() as CanInterface[];
      setCanInterfaces(detected);
      setSelectedCan(detected[0]?.interface || '');
      setIsScanning(false);
      toast.success('Detected simulator CAN interfaces.', {
        description: 'Demo discovery completed locally. No operating-system interface was accessed.',
      });
    }, 1300);
  };

  const runConnectionTest = () => {
    if (source !== 'SIMULATOR') {
      const port = Number(form.port);
      if (!form.host.trim() || !Number.isInteger(port) || port < 1 || port > 65535) {
        validateConnection();
        return;
      }
    } else if (!selectedCan) {
      toast.error('Select a simulator CAN interface before testing.');
      return;
    }

    setConnectionState('Validating');
    window.setTimeout(() => setConnectionState('Connecting'), 350);
    window.setTimeout(() => setConnectionState('Receiving Data'), 750);
    window.setTimeout(() => {
      setConnectionState(testOutcome);
      if (testOutcome === 'Connected') {
        toast.success('Connection established.', {
          description: 'Demo adapter is receiving deterministic telemetry.',
        });
      } else {
        toast.error(connectionCopy[testOutcome as keyof typeof connectionCopy] || 'Connection test failed.');
      }
    }, 1200);
  };

  const moveNext = () => {
    if (step === 0 && !validateDetails()) return;
    if (step === 1 && !source) {
      toast.error('Choose a data source to continue.');
      return;
    }
    if (step === 2 && source !== 'SIMULATOR' && !adapter) {
      toast.error('Choose an adapter or protocol to continue.');
      return;
    }
    if (step === 2 && source === 'SIMULATOR' && !selectedCan) {
      toast.error('Run demo discovery and select a CAN interface.');
      return;
    }
    if (step === 3 && !validateConnection()) return;
    if (step === 5 && !mappingValid) {
      toast.error('Map all required parameters before validation.');
      return;
    }
    if (step === 6) {
      const saved = addUav({
        name: form.name,
        id: form.id,
        engineId: form.engineId,
        engineModel: form.engineModel,
        manufacturer: form.manufacturer,
        missionProfile: form.missionProfile,
        description: form.description,
        source: source || 'GCS',
        adapter: source === 'SIMULATOR' ? 'Simulator CAN' : adapter,
      });
      if (!saved) return;
    }
    setStep((current) => Math.min(current + 1, steps.length - 1));
  };

  const renderBody = () => {
    if (step === 0) {
      return (
        <div className="wizard-form">
          <div className="wizard-callout">
            <Database size={18} />
            <span>Register the aircraft and engine before configuring its telemetry data source.</span>
          </div>
          <div className="form-grid form-grid--two">
            <Input
              error={errors.name}
              label="UAV name"
              onChange={(event) => setFormValue('name', event.target.value)}
              placeholder="e.g. Falcon Five"
              value={form.name}
            />
            <Input
              error={errors.id}
              label="UAV ID"
              onChange={(event) => setFormValue('id', event.target.value.toUpperCase())}
              placeholder="e.g. UAV-005"
              value={form.id}
            />
            <Input
              error={errors.engineId}
              label="Engine ID"
              onChange={(event) => setFormValue('engineId', event.target.value.toUpperCase())}
              placeholder="e.g. ENG-F05"
              value={form.engineId}
            />
            <Input
              error={errors.engineModel}
              label="Engine model"
              onChange={(event) => setFormValue('engineModel', event.target.value)}
              placeholder="e.g. AE-240 EFI"
              value={form.engineModel}
            />
            <Input
              error={errors.manufacturer}
              label="Manufacturer"
              onChange={(event) => setFormValue('manufacturer', event.target.value)}
              placeholder="e.g. AeroSentinel"
              value={form.manufacturer}
            />
            <Select
              label="Mission profile"
              onChange={(event) => setFormValue('missionProfile', event.target.value)}
              value={form.missionProfile}
            >
              <option>Inspection</option>
              <option>Survey</option>
              <option>Patrol</option>
              <option>Delivery</option>
              <option>Training</option>
            </Select>
          </div>
          <label className="field">
            <span className="field-label">Operator note <span className="optional">(optional)</span></span>
            <textarea
              className="input textarea"
              onChange={(event) => setFormValue('description', event.target.value)}
              placeholder="Mission or data-source context for this aircraft"
              rows={3}
              value={form.description}
            />
          </label>
        </div>
      );
    }

    if (step === 1) {
      return (
        <div className="source-grid">
          {sources.map((item) => {
            const Icon = item.icon;
            const selected = source === item.type;
            return (
              <button
                className={cn('source-card', selected && 'source-card--selected')}
                key={item.type}
                onClick={() => setSource(item.type)}
                type="button"
              >
                <span className="source-card-check">{selected ? <Check size={15} /> : null}</span>
                <span className="source-card-icon"><Icon size={31} strokeWidth={1.45} /></span>
                <strong>{item.type}</strong>
                <span className="source-card-title">{item.title}</span>
                <span className="source-card-copy">{item.description}</span>
                <span className="source-card-divider" />
                <span className="source-card-caption">Available adapters / protocols</span>
                <span className="source-card-protocols">{item.protocols.join(' · ')}</span>
              </button>
            );
          })}
        </div>
      );
    }

    if (step === 2 && source === 'SIMULATOR') {
      return (
        <div className="simulator-step">
          <div className="wizard-callout">
            <MonitorCog size={18} />
            <span>Simulator discovery is deterministic demo data. The browser is not accessing OS CAN interfaces.</span>
          </div>
          <div className="scan-header">
            <div>
              <span className="micro-label">Simulator CAN discovery</span>
              <h3>Detected simulator CAN interfaces</h3>
              <p>Choose the interface that will provide the demo telemetry stream.</p>
            </div>
            <Button loading={isScanning} onClick={runCanScan} variant="secondary">
              <Radar size={16} />
              {isScanning ? 'Scanning…' : 'Auto scan'}
            </Button>
          </div>
          {isScanning ? (
            <div className="scan-state">
              <motion.div
                animate={reduceMotion ? {} : { scale: [1, 1.22, 1], opacity: [0.7, 0.18, 0.7] }}
                className="scan-radar"
                transition={{ duration: 1.15, repeat: Infinity }}
              />
              <Radar size={26} />
              <strong>Scanning local simulator interfaces…</strong>
              <span>This takes a moment in demo mode.</span>
            </div>
          ) : canInterfaces.length === 0 ? (
            <div className="scan-empty">
              <Radar size={21} />
              <span>Run an auto scan to load deterministic simulator interfaces.</span>
            </div>
          ) : (
            <div className="can-list">
              {canInterfaces.map((item) => {
                const selected = item.interface === selectedCan;
                return (
                  <button
                    className={cn('can-row', selected && 'can-row--selected')}
                    key={item.interface}
                    onClick={() => setSelectedCan(item.interface)}
                    type="button"
                  >
                    <span className="can-icon"><Network size={17} /></span>
                    <span className="can-name">
                      <strong>{item.interface}</strong>
                      <small>{item.type} · {item.source}</small>
                    </span>
                    <span className="can-detail"><b>{(item.bitrate / 1000).toFixed(0)} kbps</b><small>bitrate</small></span>
                    <span className="can-detail"><b>{item.framesPerSecond}</b><small>frames/s</small></span>
                    <span className="can-detail"><b>{item.busLoadPercent}%</b><small>bus load</small></span>
                    <Badge dot tone={item.status === 'CONNECTED' ? 'green' : 'blue'}>{item.status}</Badge>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      );
    }

    if (step === 2) {
      return (
        <div className="adapter-step">
          <div className="adapter-intro">
            <PlugZap size={22} />
            <h3>Choose an adapter protocol</h3>
            <p>The backend adapter will own connectivity. This demo only simulates its configuration and data flow.</p>
          </div>
          <div className="adapter-grid">
            {adapterOptions.map((option) => (
              <button
                className={cn('adapter-card', adapter === option && 'adapter-card--selected')}
                key={option}
                onClick={() => setAdapter(option)}
                type="button"
              >
                <span className="adapter-dot">{adapter === option ? <Check size={12} /> : null}</span>
                <span>
                  <strong>{option}</strong>
                  <small>
                    {option === 'MAVLink'
                      ? 'Frame-aware telemetry adapter'
                      : option === 'REST API'
                        ? 'Polling or streaming HTTP endpoint'
                        : option === 'Own Protocol'
                          ? 'Custom parser and framing'
                          : 'Managed transport adapter'}
                  </small>
                </span>
              </button>
            ))}
          </div>
          {source === 'REAL UAV' ? (
            <div className="safety-note">
              <ShieldCheck size={17} />
              <span>Live aircraft connections are configured through a future backend adapter. Verify operating approval before enabling one.</span>
            </div>
          ) : null}
        </div>
      );
    }

    if (step === 3) {
      return (
        <div className="wizard-form">
          {source === 'SIMULATOR' ? (
            <>
              <div className="connection-overview">
                <div className="connection-overview-icon"><MonitorCog size={22} /></div>
                <div>
                  <strong>{selectedCanInterface?.interface || 'Simulator interface'} telemetry source</strong>
                  <p>Configure playback for the selected deterministic simulator stream.</p>
                </div>
                <Badge dot tone="green">Demo ready</Badge>
              </div>
              <div className="form-grid form-grid--three">
                <Select label="Playback speed" onChange={(event) => setSpeed(event.target.value)} value={speed}>
                  <option>0.5×</option>
                  <option>1×</option>
                  <option>2×</option>
                  <option>4×</option>
                </Select>
                <Select label="Telemetry profile" defaultValue="Nominal">
                  <option>Nominal</option>
                  <option>Rising vibration</option>
                  <option>Thermal stress</option>
                </Select>
                <div className="field">
                  <span className="field-label">Simulation</span>
                  <Button
                    onClick={() => {
                      setSimulationRunning((value) => !value);
                      toast.info(simulationRunning ? 'Simulation paused.' : 'Simulation resumed.');
                    }}
                    variant={simulationRunning ? 'success' : 'secondary'}
                  >
                    <Activity size={15} />
                    {simulationRunning ? 'Stop simulation' : 'Start simulation'}
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="connection-config-heading">
                <div>
                  <span className="micro-label">{source} · {adapter}</span>
                  <h3>Connection configuration</h3>
                  <p>Fields below describe what will be passed to the future backend adapter.</p>
                </div>
                <Badge tone="blue">Demo adapter</Badge>
              </div>
              <div className="form-grid form-grid--two">
                {adapter === 'REST API' ? (
                  <Input
                    className="span-two"
                    error={errors.endpoint}
                    label="Endpoint URL"
                    onChange={(event) => setFormValue('endpoint', event.target.value)}
                    placeholder="https://adapter.example/telemetry"
                    value={form.endpoint}
                  />
                ) : (
                  <>
                    <Input
                      error={errors.host}
                      label="Host / IP address"
                      onChange={(event) => setFormValue('host', event.target.value)}
                      placeholder="192.168.1.120"
                      value={form.host}
                    />
                    <Input
                      error={errors.port}
                      label="Port"
                      onChange={(event) => setFormValue('port', event.target.value)}
                      placeholder="14550"
                      value={form.port}
                    />
                  </>
                )}
                <Input
                  label="Connection timeout (ms)"
                  onChange={(event) => setFormValue('timeout', event.target.value)}
                  value={form.timeout}
                />
                <Input
                  label="Retry attempts"
                  onChange={(event) => setFormValue('retry', event.target.value)}
                  value={form.retry}
                />
                <Input
                  className={adapter === 'Own Protocol' ? '' : 'span-two'}
                  label="Access token <optional>"
                  onChange={(event) => setFormValue('token', event.target.value)}
                  placeholder="Optional adapter credential"
                  type="password"
                  value={form.token}
                />
                {adapter === 'Own Protocol' ? (
                  <>
                    <Input
                      error={errors.protocolName}
                      label="Protocol name"
                      onChange={(event) => setFormValue('protocolName', event.target.value)}
                      placeholder="e.g. ACME-TLM"
                      value={form.protocolName}
                    />
                    <Select
                      className="span-two"
                      label="Parser format"
                      onChange={(event) => setFormValue('parserFormat', event.target.value)}
                      value={form.parserFormat}
                    >
                      <option>JSON</option>
                      <option>Binary frame</option>
                      <option>Delimited text</option>
                    </Select>
                  </>
                ) : null}
              </div>
            </>
          )}
          <Card className="test-panel">
            <div className="test-panel-top">
              <div>
                <span className="micro-label">Connection test</span>
                <h3>Validate telemetry availability</h3>
              </div>
              <Badge dot tone={toneForConnection(connectionState)}>{connectionState}</Badge>
            </div>
            <div className="test-panel-controls">
              <Select
                label="Demo test result"
                onChange={(event) => setTestOutcome(event.target.value as ConnectionState)}
                value={testOutcome}
              >
                {outcomes.map((outcome) => <option key={outcome}>{outcome}</option>)}
              </Select>
              <Button
                loading={connectionState === 'Validating' || connectionState === 'Connecting' || connectionState === 'Receiving Data'}
                onClick={runConnectionTest}
                variant="primary"
              >
                <Waves size={16} />
                {connectionState === 'Connected' ? 'Test again' : 'Test connection'}
              </Button>
            </div>
            {connectionState === 'Connected' ? (
              <div className="test-result test-result--success">
                <CheckCircle2 size={18} />
                <span><strong>Receiving telemetry</strong><small>32 ms latency · 128 packets/s · last packet just now</small></span>
              </div>
            ) : connectionState !== 'Idle' && !['Validating', 'Connecting', 'Receiving Data'].includes(connectionState) ? (
              <div className="test-result test-result--error">
                <CircleAlert size={18} />
                <span><strong>{connectionState}</strong><small>{connectionCopy[connectionState as keyof typeof connectionCopy]}</small></span>
              </div>
            ) : (
              <p className="test-helper">Choose a demo result to inspect success and failure states. A real adapter will be called by the backend later.</p>
            )}
          </Card>
        </div>
      );
    }

    if (step === 4) {
      return (
        <div className="preview-step">
          <div className="preview-header">
            <div>
              <span className="micro-label">Incoming telemetry</span>
              <h3>Raw data preview</h3>
              <p>Live values are generated from deterministic frontend mock telemetry.</p>
            </div>
            <Badge dot tone={simulationRunning || source !== 'SIMULATOR' ? 'green' : 'amber'}>
              {simulationRunning || source !== 'SIMULATOR' ? 'Demo data live' : 'Simulation paused'}
            </Badge>
          </div>
          <div className="preview-summary">
            <span><b>{source === 'SIMULATOR' ? selectedCan || 'can0' : adapter}</b> source</span>
            <span><b>{source === 'SIMULATOR' ? speed : '128'} {source === 'SIMULATOR' ? 'playback' : 'packets/s'}</b></span>
            <span><b>0x182–0x205</b> raw frames</span>
            <span><b>{telemetry?.flight_id || 'FLT-260909-A1'}</b> flight</span>
          </div>
          <div className="raw-preview-grid">
            {rawRows.map((row) => (
              <div className="raw-preview-row" key={row.key}>
                <span className="raw-preview-key">{row.key}</span>
                <span className="raw-preview-value">{row.value}</span>
                <span className="raw-preview-meta">{row.source}</span>
                <Badge tone="green">Valid</Badge>
              </div>
            ))}
          </div>
          <div className="preview-footer">
            <Activity size={16} />
            <span>Frame count <b>18,420</b> · Payload last updated <b>just now</b> · Mapping target <b>platform telemetry v1</b></span>
          </div>
        </div>
      );
    }

    if (step === 5) {
      return (
        <div className="mapping-step">
          <div className="mapping-topline">
            <div>
              <span className="micro-label">Normalization workspace</span>
              <h3>Map incoming telemetry</h3>
              <p>Connect raw source fields to standard platform parameters.</p>
            </div>
            <Badge tone={mappingValid ? 'green' : 'amber'}>{mappedRequired} / {requiredCount} required mapped</Badge>
          </div>
          <div className="mapping-workspace">
            <div className="mapping-column">
              <div className="mapping-column-heading"><span>Incoming raw data</span><span>Current value</span></div>
              {rawRows.map((row) => (
                <div className="mapping-raw-row" key={row.key}>
                  <span><b>{row.key}</b><small>{row.source} · {row.unit}</small></span>
                  <strong>{row.value}</strong>
                  <i />
                </div>
              ))}
            </div>
            <div className="mapping-links" aria-hidden="true">
              {requiredTargets.map((target, index) => (
                mapping[target.id] ? (
                  <motion.span
                    animate={{ opacity: 1, scaleX: 1 }}
                    className="mapping-link"
                    initial={{ opacity: 0, scaleX: 0 }}
                    key={target.id}
                    style={{ top: 54 + index * 54 }}
                    transition={{ duration: reduceMotion ? 0 : 0.25 }}
                  />
                ) : null
              ))}
            </div>
            <div className="mapping-column mapping-column--target">
              <div className="mapping-column-heading"><span>Standardized parameter</span><span>Mapping</span></div>
              {requiredTargets.map((target) => (
                <div className="mapping-target-row" key={target.id}>
                  <span>
                    <b>{target.name}</b>
                    <small>{target.unit} · {target.required ? 'Required' : 'Optional'}</small>
                  </span>
                  <select
                    aria-label={'Map ' + target.name}
                    className="input mapping-select"
                    onChange={(event) => setMapping((current) => ({ ...current, [target.id]: event.target.value }))}
                    value={mapping[target.id]}
                  >
                    <option value="">Unmapped</option>
                    {rawRows.map((row) => <option key={row.key} value={row.key}>{row.key}</option>)}
                  </select>
                </div>
              ))}
            </div>
          </div>
          <div className="mapping-inline-summary">
            <span><b>{requiredCount}</b> Required parameters</span>
            <span><b>{Object.values(mapping).filter(Boolean).length}</b> Mapped parameters</span>
            <span className={mappingValid ? 'value-good' : 'value-bad'}><b>{requiredCount - mappedRequired}</b> Unmapped required</span>
            <span><b>0</b> Invalid parameters</span>
          </div>
        </div>
      );
    }

    if (step === 6) {
      return (
        <div className="validate-step">
          <div className="validation-mark"><CheckCircle2 size={34} /></div>
          <h3>Mapping validation</h3>
          <p>Review the normalized telemetry configuration before creating the demo data source.</p>
          <div className="validation-grid">
            <div><span>Required parameters</span><b>{requiredCount}</b></div>
            <div><span>Mapped parameters</span><b className="value-good">{Object.values(mapping).filter(Boolean).length}</b></div>
            <div><span>Unmapped required</span><b className={mappingValid ? 'value-good' : 'value-bad'}>{requiredCount - mappedRequired}</b></div>
            <div><span>Invalid parameters</span><b className="value-good">0</b></div>
          </div>
          <Card className={cn('validation-result', mappingValid && 'validation-result--good')}>
            {mappingValid ? <CheckCircle2 size={20} /> : <CircleAlert size={20} />}
            <div>
              <strong>{mappingValid ? 'Mapping is ready to save' : 'Mapping validation failed'}</strong>
              <p>{mappingValid ? 'All required telemetry parameters are mapped to the platform schema.' : 'Map all required parameters before validation.'}</p>
            </div>
          </Card>
        </div>
      );
    }

    return (
      <div className="ready-step">
        <div className="ready-icon"><CheckCircle2 size={40} /></div>
        <Badge tone="green">Configuration complete</Badge>
        <h3>Gateway Data Source Ready</h3>
        <p>{form.id || 'Your UAV'} can now stream deterministic demo telemetry into the operations console.</p>
        <div className="ready-details">
          <span><small>Source</small><b>{source}</b></span>
          <span><small>Adapter</small><b>{source === 'SIMULATOR' ? 'Simulator CAN' : adapter}</b></span>
          <span><small>Connection</small><b className="value-good">Connected</b></span>
          <span><small>Mappings</small><b>{Object.values(mapping).filter(Boolean).length} fields</b></span>
        </div>
      </div>
    );
  };

  return (
    <Modal
      description={step === 7 ? 'The demo data source has been added locally.' : 'Configure a frontend-local mock source that can be replaced by a backend adapter later.'}
      onClose={resetAndClose}
      open={isAddUavOpen}
      size="full"
      title={step === 7 ? 'Data source ready' : 'Add UAV'}
    >
      <div className="wizard">
        <nav aria-label="Add UAV progress" className="wizard-steps">
          {steps.map((label, index) => (
            <div className={cn('wizard-step', index === step && 'wizard-step--active', index < step && 'wizard-step--complete')} key={label}>
              <span>{index < step ? <Check size={12} /> : index + 1}</span>
              <b>{label}</b>
            </div>
          ))}
        </nav>
        <AnimatePresence mode="wait">
          <motion.div
            animate={{ opacity: 1, x: 0 }}
            className="wizard-content"
            initial={{ opacity: 0, x: reduceMotion ? 0 : 10 }}
            key={step}
            transition={{ duration: reduceMotion ? 0 : 0.18 }}
          >
            {renderBody()}
          </motion.div>
        </AnimatePresence>
        <footer className="wizard-footer">
          <div>
            {step < 7 ? <span className="wizard-progress-label">Step {step + 1} of {steps.length}</span> : null}
          </div>
          <div className="wizard-actions">
            {step > 0 && step < 7 ? (
              <Button onClick={() => setStep((current) => current - 1)} variant="ghost">
                <ChevronLeft size={16} /> Back
              </Button>
            ) : null}
            {step < 7 ? (
              <Button
                disabled={step === 5 && !mappingValid}
                onClick={moveNext}
              >
                {step === 6 ? 'Save data source' : 'Continue'} <ChevronRight size={16} />
              </Button>
            ) : (
              <Button
                onClick={() => {
                  resetAndClose();
                  window.location.assign('/live-monitoring');
                }}
              >
                <Gauge size={16} /> Go to Live Monitoring
              </Button>
            )}
          </div>
        </footer>
      </div>
    </Modal>
  );
}
