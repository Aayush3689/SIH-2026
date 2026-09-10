import { getHealth as getHealthFromApi, getInference as getInferenceFromApi } from "@/lib/api";
import { mockAlerts } from "@/lib/mock/alerts";
import { mockAnalytics, type FleetAnalytics } from "@/lib/mock/analytics";
import { getMockCanFrames, mockCanInterfaces } from "@/lib/mock/can";
import { mockGatewayDataSources, mockGateways } from "@/lib/mock/gateways";
import { getMockInference } from "@/lib/mock/inference";
import {
  mockIncomingTelemetryFields,
  mockMappingValidation,
  mockPlatformTelemetryParameters,
  mockTelemetryMappings,
  mockUavGatewayMappings,
} from "@/lib/mock/mappings";
import { getMockTelemetry, mockTelemetry } from "@/lib/mock/telemetry";
import { mockFleetSummary, mockUavs } from "@/lib/mock/uavs";
import type { Alert } from "@/types/alerts";
import type { ApiClientOptions } from "@/lib/api";
import type {
  CanFrame,
  CanInterface,
  Gateway,
  GatewayDataSource,
  IncomingTelemetryField,
  MappingValidationSummary,
  PlatformTelemetryParameter,
  TelemetryFieldMapping,
  UavGatewayMapping,
} from "@/types/gateway";
import type { HealthResponse, InferenceResponse } from "@/types/inference";
import type { TelemetrySample, TelemetrySeries } from "@/types/telemetry";
import type { FleetSummary, Uav } from "@/types/uav";

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

/**
 * Deterministic data surface for the dashboard. It is synchronous by design so
 * server and client components can consume the demo data without loading state.
 */
export interface DemoService {
  listUavs(): Uav[];
  getTelemetry(uavId: string): TelemetrySample[];
  listTelemetry(): TelemetrySeries[];
  listGateways(): Gateway[];
  listDataSources(): GatewayDataSource[];
  listAlerts(): Alert[];
  /** Telemetry source-to-platform mappings for the Data Mapping page. */
  listMappings(): TelemetryFieldMapping[];
  /** Vehicle-to-gateway assignments for connectivity views. */
  listGatewayMappings(): UavGatewayMapping[];
  listIncomingTelemetryFields(): IncomingTelemetryField[];
  listPlatformTelemetryParameters(): PlatformTelemetryParameter[];
  getMappingValidation(): MappingValidationSummary;
  listCanInterfaces(): CanInterface[];
  listCanFrames(uavId?: string): CanFrame[];
  getInference(samples: TelemetrySample[]): InferenceResponse;
  getAnalytics(): FleetAnalytics;
  getFleetSummary(): FleetSummary;
}

export const demoService: DemoService = {
  listUavs: () => clone(mockUavs),
  getTelemetry: (uavId) => clone(getMockTelemetry(uavId)),
  listTelemetry: () => clone(mockTelemetry),
  listGateways: () => clone(mockGateways),
  listDataSources: () => clone(mockGatewayDataSources),
  listAlerts: () => clone(mockAlerts),
  listMappings: () => clone(mockTelemetryMappings),
  listGatewayMappings: () => clone(mockUavGatewayMappings),
  listIncomingTelemetryFields: () => clone(mockIncomingTelemetryFields),
  listPlatformTelemetryParameters: () => clone(mockPlatformTelemetryParameters),
  getMappingValidation: () => clone(mockMappingValidation),
  listCanInterfaces: () => clone(mockCanInterfaces),
  listCanFrames: (uavId) => clone(getMockCanFrames(uavId)),
  getInference: (samples) => getMockInference(samples),
  getAnalytics: () => clone(mockAnalytics),
  getFleetSummary: () => clone(mockFleetSummary),
};

/** The only live backend surface currently documented for the frontend. */
export interface AiBackendService {
  getHealth(options?: ApiClientOptions): Promise<HealthResponse>;
  getInference(samples: TelemetrySample[], options?: ApiClientOptions): Promise<InferenceResponse>;
}

export const aiBackendService: AiBackendService = {
  getHealth: (options) => getHealthFromApi(options),
  getInference: (samples, options) => getInferenceFromApi(samples, options),
};

export type { FleetAnalytics } from "@/lib/mock/analytics";
export type {
  CanFrame,
  CanInterface,
  Gateway,
  GatewayDataSource,
  IncomingTelemetryField,
  MappingValidationSummary,
  PlatformTelemetryParameter,
  TelemetryFieldMapping,
  UavGatewayMapping,
} from "@/types/gateway";
