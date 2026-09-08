'use client';

import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useState,
} from 'react';
import { Toaster, toast } from 'sonner';
import type { Alert, AlertStatus } from '@/types/alerts';
import type { Uav, UavStatus } from '@/types/uav';
import { demoService } from '@/lib/services';

export type EnvironmentMode = 'Demo' | 'Live';

export interface AddUavRecord {
  name: string;
  id: string;
  engineId: string;
  engineModel: string;
  manufacturer: string;
  missionProfile: string;
  description?: string;
  source: 'GCS' | 'REAL UAV' | 'SIMULATOR';
  adapter: string;
}

interface AppContextValue {
  uavs: Uav[];
  alerts: Alert[];
  selectedUavId: string;
  selectedUav: Uav | undefined;
  environment: EnvironmentMode;
  isAddUavOpen: boolean;
  setSelectedUavId: (id: string) => void;
  setEnvironment: (environment: EnvironmentMode) => void;
  openAddUav: () => void;
  closeAddUav: () => void;
  addUav: (record: AddUavRecord) => boolean;
  updateUav: (id: string, changes: Partial<Pick<Uav, 'name' | 'model' | 'mission_status'>>) => void;
  updateAlertStatus: (id: string, status: AlertStatus, note?: string) => void;
}

const AppContext = createContext<AppContextValue | null>(null);

const sourceToStatus = (source: AddUavRecord['source']): UavStatus =>
  source === 'SIMULATOR' ? 'active' : 'warning';

export function AppProviders({ children }: { children: ReactNode }) {
  const [uavs, setUavs] = useState<Uav[]>(() => demoService.listUavs());
  const [alerts, setAlerts] = useState<Alert[]>(() => demoService.listAlerts());
  const [selectedUavId, setSelectedUavId] = useState('UAV-001');
  const [environment, setEnvironment] = useState<EnvironmentMode>('Demo');
  const [isAddUavOpen, setIsAddUavOpen] = useState(false);

  const addUav = useCallback(
    (record: AddUavRecord) => {
      if (uavs.some((uav) => uav.id.toLowerCase() === record.id.trim().toLowerCase())) {
        toast.error('A UAV with this ID already exists.');
        return false;
      }

      const newUav: Uav = {
        id: record.id.trim(),
        callsign: record.name.trim().toUpperCase().replace(/\s+/g, '-').slice(0, 15) || record.id.trim(),
        name: record.name.trim(),
        model: record.engineModel.trim() || 'Unspecified engine model',
        manufacturer: record.manufacturer.trim(),
        engine_model: record.engineModel.trim(),
        mission_profile: record.missionProfile,
        description: record.description?.trim() || undefined,
        status: sourceToStatus(record.source),
        mission_status: 'idle',
        position: { latitude: 28.628, longitude: 77.218, altitude_ft: 0 },
        heading_deg: 0,
        speed_kts: 0,
        battery_pct: 100,
        signal_strength_pct: record.source === 'SIMULATOR' ? 100 : 86,
        engine_id: record.engineId.trim(),
        gateway_id: null,
        last_seen: new Date().toISOString(),
      };

      setUavs((current) => [...current, newUav]);
      setSelectedUavId(newUav.id);
      toast.success('UAV data source added to the demo fleet.', {
        description: newUav.id + ' is ready for Live Monitoring.',
      });
      return true;
    },
    [uavs],
  );

  const updateAlertStatus = useCallback((id: string, status: AlertStatus, note?: string) => {
    setAlerts((current) =>
      current.map((alert) =>
        alert.id === id
          ? {
              ...alert,
              status,
              acknowledged_at:
                status === 'open'
                  ? null
                  : alert.acknowledged_at || new Date().toISOString(),
              message: note?.trim() ? alert.message + ' Operator note: ' + note.trim() : alert.message,
            }
          : alert,
      ),
    );

    if (status === 'acknowledged') {
      toast.success('Alert acknowledged.');
    }
    if (status === 'resolved') {
      toast.success('Alert resolved and retained in history.');
    }
  }, []);

  const updateUav = useCallback((id: string, changes: Partial<Pick<Uav, 'name' | 'model' | 'mission_status'>>) => {
    setUavs((current) => current.map((uav) => (uav.id === id ? { ...uav, ...changes } : uav)));
    toast.success('UAV details saved.', { description: id + ' has been updated in the demo fleet.' });
  }, []);

  const value = useMemo<AppContextValue>(
    () => ({
      uavs,
      alerts,
      selectedUavId,
      selectedUav: uavs.find((uav) => uav.id === selectedUavId),
      environment,
      isAddUavOpen,
      setSelectedUavId,
      setEnvironment,
      openAddUav: () => setIsAddUavOpen(true),
      closeAddUav: () => setIsAddUavOpen(false),
      addUav,
      updateUav,
      updateAlertStatus,
    }),
    [addUav, alerts, environment, isAddUavOpen, selectedUavId, uavs, updateAlertStatus, updateUav],
  );

  return (
    <AppContext.Provider value={value}>
      {children}
      <Toaster closeButton position="bottom-right" richColors theme="dark" />
    </AppContext.Provider>
  );
}

export function useApp() {
  const value = useContext(AppContext);
  if (!value) {
    throw new Error('useApp must be used inside AppProviders.');
  }
  return value;
}
