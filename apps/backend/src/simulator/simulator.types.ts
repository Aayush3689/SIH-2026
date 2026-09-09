export interface EngineState {
  rpm: number;
  throttlePositionPct: number;
  engineLoadPct: number;

  chtC: number;
  egtC: number;
  oilTemperatureC: number;

  oilPressureKpa: number;
  fuelFlowLph: number;
  fuelPressureKpa: number;

  intakeManifoldPressureKpa: number;

  engineVibrationMmS: number;

  batteryVoltageV: number;
  alternatorOutputA: number;

  injectionTimingDegBtdc: number;

  ambientTempC: number;
  plannedAltitudeFt: number;
  airspeedKts: number;

  elapsedS: number;
}

export interface TelemetrySample extends EngineState {
  timestamp: string;
  engineId: string;
  uavId: string;
}

export interface SimulatorConfig {
  uavId: string;
  engineId: string;
  canInterface: string;
  rateHz: number;
}

export interface CanFrameData {
  id: number;
  data: Buffer;
}
