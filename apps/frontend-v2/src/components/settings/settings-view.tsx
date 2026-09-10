'use client';

import { type FormEvent, useEffect, useMemo, useState } from 'react';
import {
  BellRing,
  CheckCircle2,
  CloudCog,
  Eye,
  Gauge,
  MonitorCog,
  RefreshCcw,
  Save,
  ShieldCheck,
  SlidersHorizontal,
  UserRound,
} from 'lucide-react';
import { toast } from 'sonner';
import { type EnvironmentMode, useApp } from '@/components/layout/app-providers';
import { Badge, Button, Card, CardHeader, Input, SectionHeading, Select, Toggle } from '@/components/ui';
import styles from './settings-view.module.css';

interface ProfileSettings {
  operatorName: string;
  email: string;
  role: string;
  callsign: string;
}

interface UiSettings {
  refreshInterval: string;
  telemetryDensity: 'compact' | 'comfortable';
  defaultDataView: 'normalized' | 'raw';
  showSignalBadges: boolean;
  reduceVisualUpdates: boolean;
}

interface NotificationSettings {
  criticalAlerts: boolean;
  healthWarnings: boolean;
  connectionChanges: boolean;
  readinessDigest: boolean;
}

interface SettingsDraft {
  profile: ProfileSettings;
  ui: UiSettings;
  notifications: NotificationSettings;
}

interface StoredSettings {
  version: 1;
  environment: EnvironmentMode;
  settings: SettingsDraft;
  savedAt: string;
}

type SettingsField = 'operatorName' | 'email' | 'role' | 'refreshInterval';
type SettingsErrors = Partial<Record<SettingsField, string>>;

const storageKey = 'aerosentinel.operator-settings.v1';
const refreshIntervals = ['1', '2', '5', '10', '30'];

const defaultSettings: SettingsDraft = {
  profile: {
    operatorName: 'Alex Operator',
    email: 'alex.operator@aerosentinel.local',
    role: 'Flight Engineer',
    callsign: 'AERO-OPS-01',
  },
  ui: {
    refreshInterval: '2',
    telemetryDensity: 'compact',
    defaultDataView: 'normalized',
    showSignalBadges: true,
    reduceVisualUpdates: false,
  },
  notifications: {
    criticalAlerts: true,
    healthWarnings: true,
    connectionChanges: true,
    readinessDigest: false,
  },
};

function cloneSettings(settings: SettingsDraft): SettingsDraft {
  return {
    profile: { ...settings.profile },
    ui: { ...settings.ui },
    notifications: { ...settings.notifications },
  };
}

function isSettingsDraft(value: unknown): value is SettingsDraft {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<SettingsDraft>;
  const profile = candidate.profile;
  const ui = candidate.ui;
  const notifications = candidate.notifications;

  return Boolean(
    profile &&
      ui &&
      notifications &&
      typeof profile.operatorName === 'string' &&
      typeof profile.email === 'string' &&
      typeof profile.role === 'string' &&
      typeof profile.callsign === 'string' &&
      typeof ui.refreshInterval === 'string' &&
      (ui.telemetryDensity === 'compact' || ui.telemetryDensity === 'comfortable') &&
      (ui.defaultDataView === 'normalized' || ui.defaultDataView === 'raw') &&
      typeof ui.showSignalBadges === 'boolean' &&
      typeof ui.reduceVisualUpdates === 'boolean' &&
      typeof notifications.criticalAlerts === 'boolean' &&
      typeof notifications.healthWarnings === 'boolean' &&
      typeof notifications.connectionChanges === 'boolean' &&
      typeof notifications.readinessDigest === 'boolean',
  );
}

function readStoredSettings(): StoredSettings | null {
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return null;

    const candidate: unknown = JSON.parse(raw);
    if (!candidate || typeof candidate !== 'object') return null;
    const stored = candidate as Partial<StoredSettings>;
    if (
      stored.version !== 1 ||
      (stored.environment !== 'Demo' && stored.environment !== 'Live') ||
      !isSettingsDraft(stored.settings) ||
      typeof stored.savedAt !== 'string'
    ) {
      return null;
    }

    return {
      version: 1,
      environment: stored.environment,
      settings: cloneSettings(stored.settings),
      savedAt: stored.savedAt,
    };
  } catch {
    return null;
  }
}

function formatSavedAt(timestamp: string | null): string {
  if (!timestamp) return 'Not yet saved in this browser';

  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return 'Saved locally';

  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    month: 'short',
    day: 'numeric',
  }).format(date);
}

export function SettingsView() {
  const { environment, setEnvironment } = useApp();
  const [settings, setSettings] = useState<SettingsDraft>(() => cloneSettings(defaultSettings));
  const [savedSettings, setSavedSettings] = useState<SettingsDraft>(() => cloneSettings(defaultSettings));
  const [environmentDraft, setEnvironmentDraft] = useState<EnvironmentMode>(environment);
  const [savedEnvironment, setSavedEnvironment] = useState<EnvironmentMode>(environment);
  const [lastSavedAt, setLastSavedAt] = useState<string | null>(null);
  const [errors, setErrors] = useState<SettingsErrors>({});
  const [hasLoaded, setHasLoaded] = useState(false);

  useEffect(() => {
    const stored = readStoredSettings();
    if (stored) {
      setSettings(stored.settings);
      setSavedSettings(stored.settings);
      setEnvironmentDraft(stored.environment);
      setSavedEnvironment(stored.environment);
      setLastSavedAt(stored.savedAt);
      setEnvironment(stored.environment);
    }
    setHasLoaded(true);
  }, [setEnvironment]);

  const isDirty = useMemo(
    () =>
      environmentDraft !== savedEnvironment ||
      JSON.stringify(settings) !== JSON.stringify(savedSettings),
    [environmentDraft, savedEnvironment, savedSettings, settings],
  );

  const clearError = (field: SettingsField) => {
    setErrors((current) => ({ ...current, [field]: undefined }));
  };

  const updateProfile = (field: keyof ProfileSettings, value: string) => {
    setSettings((current) => ({
      ...current,
      profile: { ...current.profile, [field]: value },
    }));
    if (field === 'operatorName' || field === 'email' || field === 'role') clearError(field);
  };

  const updateUi = <Field extends keyof UiSettings>(field: Field, value: UiSettings[Field]) => {
    setSettings((current) => ({
      ...current,
      ui: { ...current.ui, [field]: value },
    }));
    if (field === 'refreshInterval') clearError('refreshInterval');
  };

  const updateNotifications = <Field extends keyof NotificationSettings>(
    field: Field,
    value: NotificationSettings[Field],
  ) => {
    setSettings((current) => ({
      ...current,
      notifications: { ...current.notifications, [field]: value },
    }));
  };

  const validate = (): SettingsErrors => {
    const nextErrors: SettingsErrors = {};
    if (settings.profile.operatorName.trim().length < 2) {
      nextErrors.operatorName = 'Enter an operator name with at least 2 characters.';
    }
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(settings.profile.email.trim())) {
      nextErrors.email = 'Enter a valid notification email address.';
    }
    if (!settings.profile.role.trim()) {
      nextErrors.role = 'Select or enter an operational role.';
    }
    if (!refreshIntervals.includes(settings.ui.refreshInterval)) {
      nextErrors.refreshInterval = 'Choose a refresh interval between 1 and 30 seconds.';
    }
    return nextErrors;
  };

  const saveSettings = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const nextErrors = validate();
    setErrors(nextErrors);

    if (Object.keys(nextErrors).length > 0) {
      toast.error('Review the highlighted settings before saving.');
      return;
    }

    const savedAt = new Date().toISOString();
    const saved = cloneSettings(settings);
    const record: StoredSettings = {
      version: 1,
      environment: environmentDraft,
      settings: saved,
      savedAt,
    };

    try {
      window.localStorage.setItem(storageKey, JSON.stringify(record));
    } catch {
      toast.warning('Settings were applied for this session, but could not be stored locally.');
    }

    setEnvironment(environmentDraft);
    setSavedSettings(saved);
    setSavedEnvironment(environmentDraft);
    setLastSavedAt(savedAt);
    toast.success('Settings saved.', {
      description:
        environmentDraft === 'Demo'
          ? 'Demo mode and console preferences are active.'
          : 'Live mode preferences are active when the backend is available.',
    });
  };

  const resetChanges = () => {
    setSettings(cloneSettings(savedSettings));
    setEnvironmentDraft(savedEnvironment);
    setErrors({});
    toast.info('Unsaved settings were reverted.');
  };

  return (
    <div className={'page-enter ' + styles.view}>
      <SectionHeading
        action={
          <div className={styles.headingActions}>
            <Badge dot tone={environmentDraft === 'Demo' ? 'blue' : 'green'}>
              {environmentDraft} mode
            </Badge>
            <Button disabled={!isDirty} type="submit" form="operator-settings-form">
              <Save size={16} /> Save changes
            </Button>
          </div>
        }
        description="Manage operator preferences, console behavior, and notification defaults for this browser."
        eyebrow="Operator console"
        title="Settings"
      />

      <form id="operator-settings-form" onSubmit={saveSettings}>
        <div className={styles.layout}>
          <div className={styles.stack}>
            <Card className={styles.card}>
              <CardHeader
                subtitle="Used in notifications, operator notes, and the shared console header."
                title={<span className={styles.cardTitle}><UserRound size={16} /> Operator profile</span>}
              />
              <div className={styles.cardBody}>
                <div className={styles.profileGrid}>
                  <Input
                    error={errors.operatorName}
                    label="Operator name"
                    onChange={(event) => updateProfile('operatorName', event.target.value)}
                    value={settings.profile.operatorName}
                  />
                  <Input
                    error={errors.email}
                    label="Notification email"
                    onChange={(event) => updateProfile('email', event.target.value)}
                    type="email"
                    value={settings.profile.email}
                  />
                  <Select
                    error={errors.role}
                    label="Operational role"
                    onChange={(event) => updateProfile('role', event.target.value)}
                    value={settings.profile.role}
                  >
                    <option value="">Select a role</option>
                    <option>Flight Engineer</option>
                    <option>Mission Operator</option>
                    <option>Maintenance Lead</option>
                    <option>System Administrator</option>
                  </Select>
                  <Input
                    hint="Shown on operator notes and exported reports."
                    label="Operator callsign"
                    onChange={(event) => updateProfile('callsign', event.target.value.toUpperCase())}
                    value={settings.profile.callsign}
                  />
                </div>
              </div>
            </Card>

            <Card className={styles.card}>
              <CardHeader
                subtitle="Keep live telemetry readable without exposing implementation-only controls."
                title={<span className={styles.cardTitle}><SlidersHorizontal size={16} /> Console preferences</span>}
              />
              <div className={styles.cardBody}>
                <div className={styles.preferenceGrid}>
                  <Select
                    error={errors.refreshInterval}
                    hint="Applies to demo telemetry and supported live views."
                    label="Default refresh interval"
                    onChange={(event) => updateUi('refreshInterval', event.target.value)}
                    value={settings.ui.refreshInterval}
                  >
                    {refreshIntervals.map((interval) => (
                      <option key={interval} value={interval}>{interval} second{interval === '1' ? '' : 's'}</option>
                    ))}
                  </Select>
                  <Select
                    label="Telemetry density"
                    onChange={(event) => updateUi('telemetryDensity', event.target.value as UiSettings['telemetryDensity'])}
                    value={settings.ui.telemetryDensity}
                  >
                    <option value="compact">Compact operations view</option>
                    <option value="comfortable">Comfortable review view</option>
                  </Select>
                  <Select
                    label="Default adapter display"
                    onChange={(event) => updateUi('defaultDataView', event.target.value as UiSettings['defaultDataView'])}
                    value={settings.ui.defaultDataView}
                  >
                    <option value="normalized">Normalized platform values</option>
                    <option value="raw">Raw source values</option>
                  </Select>
                </div>
                <div className={styles.toggleList}>
                  <Toggle
                    checked={settings.ui.showSignalBadges}
                    description="Show connection and telemetry quality labels where available."
                    label="Display signal health badges"
                    onChange={(checked) => updateUi('showSignalBadges', checked)}
                  />
                  <Toggle
                    checked={settings.ui.reduceVisualUpdates}
                    description="Limit non-essential value pulses during continuous telemetry updates."
                    label="Reduce visual update motion"
                    onChange={(checked) => updateUi('reduceVisualUpdates', checked)}
                  />
                </div>
              </div>
            </Card>

            <Card className={styles.card}>
              <CardHeader
                action={<Badge tone="blue">Operator alerts</Badge>}
                subtitle="Choose which operational changes should surface as desktop-ready notifications."
                title={<span className={styles.cardTitle}><BellRing size={16} /> Notification preferences</span>}
              />
              <div className={styles.cardBody + ' ' + styles.toggleList}>
                <Toggle
                  checked={settings.notifications.criticalAlerts}
                  description="Critical engine, safety, and mission-impacting alerts."
                  label="Critical alerts"
                  onChange={(checked) => updateNotifications('criticalAlerts', checked)}
                />
                <Toggle
                  checked={settings.notifications.healthWarnings}
                  description="Pre-fault, low-RUL, and sustained health warning changes."
                  label="Engine health warnings"
                  onChange={(checked) => updateNotifications('healthWarnings', checked)}
                />
                <Toggle
                  checked={settings.notifications.connectionChanges}
                  description="Gateway disconnects, recovered adapters, and stale telemetry."
                  label="Connection state changes"
                  onChange={(checked) => updateNotifications('connectionChanges', checked)}
                />
                <Toggle
                  checked={settings.notifications.readinessDigest}
                  description="A compact readiness summary when the console is first opened."
                  label="Mission readiness digest"
                  onChange={(checked) => updateNotifications('readinessDigest', checked)}
                />
              </div>
            </Card>
          </div>

          <aside className={styles.sidebar}>
            <Card className={styles.card}>
              <CardHeader
                subtitle="Choose how the console obtains operational data."
                title={<span className={styles.cardTitle}><CloudCog size={16} /> Environment mode</span>}
              />
              <div className={styles.cardBody}>
                <div aria-label="Environment mode" className={styles.modeSelector} role="radiogroup">
                  {(['Demo', 'Live'] as EnvironmentMode[]).map((mode) => {
                    const selected = environmentDraft === mode;
                    return (
                      <button
                        aria-checked={selected}
                        className={styles.modeOption + (selected ? ' ' + styles.modeOptionSelected : '')}
                        key={mode}
                        onClick={() => setEnvironmentDraft(mode)}
                        role="radio"
                        type="button"
                      >
                        {mode === 'Demo' ? <MonitorCog size={18} /> : <Gauge size={18} />}
                        <span><strong>{mode}</strong><small>{mode === 'Demo' ? 'Deterministic local data' : 'Backend-aware data flow'}</small></span>
                        {selected ? <CheckCircle2 aria-hidden="true" size={16} /> : null}
                      </button>
                    );
                  })}
                </div>
                <div className={styles.environmentNote}>
                  <ShieldCheck size={16} />
                  <span>
                    {environmentDraft === 'Demo'
                      ? 'Demo mode simulates data locally. It never claims to connect directly to an aircraft.'
                      : 'Live mode applies this preference now; compatible screens use backend services when available.'}
                  </span>
                </div>
              </div>
            </Card>

            <Card className={styles.card}>
              <CardHeader
                subtitle="Changes are stored only in this browser until a profile API is available."
                title={<span className={styles.cardTitle}><Eye size={16} /> Save status</span>}
              />
              <div className={styles.cardBody}>
                <dl className={styles.statusList}>
                  <div><dt>Last saved</dt><dd>{hasLoaded ? formatSavedAt(lastSavedAt) : 'Loading preferences…'}</dd></div>
                  <div><dt>Active mode</dt><dd><Badge dot tone={environment === 'Demo' ? 'blue' : 'green'}>{environment}</Badge></dd></div>
                  <div><dt>Pending changes</dt><dd className={isDirty ? styles.pending : styles.saved}>{isDirty ? 'Ready to save' : 'All changes saved'}</dd></div>
                </dl>
                <div className={styles.sidebarActions}>
                  <Button disabled={!isDirty} onClick={resetChanges} type="button" variant="secondary">
                    <RefreshCcw size={15} /> Revert changes
                  </Button>
                  <Button type="submit">
                    <Save size={15} /> Save settings
                  </Button>
                </div>
              </div>
            </Card>
          </aside>
        </div>
      </form>
    </div>
  );
}

export default SettingsView;
