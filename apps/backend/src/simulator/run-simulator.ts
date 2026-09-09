import { SimulatorInstance } from "./simulator.instance";
import { EngineState } from "./simulator.types";

const initialEngineState: EngineState = {
  // ------------------------------------------------------------
  // Engine operating parameters
  // ------------------------------------------------------------

  rpm: 3500,
  throttlePositionPct: 50,
  engineLoadPct: 50,

  // ------------------------------------------------------------
  // Thermal parameters
  // ------------------------------------------------------------

  chtC: 145,
  egtC: 690,
  oilTemperatureC: 92,

  // ------------------------------------------------------------
  // Lubrication / fuel system
  // ------------------------------------------------------------

  oilPressureKpa: 380,
  fuelFlowLph: 18,
  fuelPressureKpa: 295,

  // ------------------------------------------------------------
  // Air / intake
  // ------------------------------------------------------------

  intakeManifoldPressureKpa: 82,

  // ------------------------------------------------------------
  // Mechanical
  // ------------------------------------------------------------

  engineVibrationMmS: 1.8,

  // ------------------------------------------------------------
  // Electrical
  // ------------------------------------------------------------

  batteryVoltageV: 27.6,
  alternatorOutputA: 16,

  // ------------------------------------------------------------
  // Injection
  // ------------------------------------------------------------

  injectionTimingDegBtdc: 18,

  // ------------------------------------------------------------
  // Environment / mission context
  // ------------------------------------------------------------

  ambientTempC: 28,
  plannedAltitudeFt: 5000,
  airspeedKts: 85,

  // ------------------------------------------------------------
  // Simulation time
  // ------------------------------------------------------------

  elapsedS: 0,
};

const simulator = new SimulatorInstance(
  {
    uavId: "UAV-001",
    engineId: "ENG-001",
    canInterface: "vcan0",
    rateHz: 1,
  },
  initialEngineState
);

try {
  simulator.start();
} catch (error) {
  console.error(
    "[Simulator] Failed to start:",
    error
  );

  process.exit(1);
}

// --------------------------------------------------------------
// Graceful shutdown
// --------------------------------------------------------------

const shutdown = (signal: string): void => {
  console.log(
    `\n[Simulator] Received ${signal}. Shutting down...`
  );

  simulator.stop();

  process.exit(0);
};

process.on(
  "SIGINT",
  () => shutdown("SIGINT")
);

process.on(
  "SIGTERM",
  () => shutdown("SIGTERM")
);