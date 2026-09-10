import type { Uav, Alert, Adapter, DataSourceConnection } from "@/lib/types";
import {
  mockUavs,
  mockAlerts,
  mockAdapters,
  mockConnections,
} from "@/data/mock";

let useMock = true;

export function setDemoMode(demo: boolean) {
  useMock = demo;
}

export function isDemoMode() {
  return useMock;
}

export async function getUavs(): Promise<Uav[]> {
  return mockUavs;
}

export async function getUav(id: string): Promise<Uav | undefined> {
  return mockUavs.find((u) => u.id === id);
}

export async function getAlerts(): Promise<Alert[]> {
  return mockAlerts;
}

export async function getAdapters(): Promise<Adapter[]> {
  return mockAdapters;
}

export async function getConnections(): Promise<DataSourceConnection[]> {
  return mockConnections;
}
