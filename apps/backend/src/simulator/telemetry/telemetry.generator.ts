import {
  EngineState,
  TelemetrySample,
} from "../simulator.types";

export class TelemetryGenerator {
  generate(
    state: EngineState,
    uavId: string,
    engineId: string
  ): TelemetrySample {
    return {
      timestamp: new Date().toISOString(),

      uavId,
      engineId,

      rpm: Number(state.rpm.toFixed(2)),
      throttlePositionPct: Number(
        state.throttlePositionPct.toFixed(2)
      ),
      chtC: Number(state.chtC.toFixed(2)),
      egtC: Number(state.egtC.toFixed(2)),

      oilPressureKpa: Number(
        state.oilPressureKpa.toFixed(2)
      ),
      oilTemperatureC: Number(
        state.oilTemperatureC.toFixed(2)
      ),

      fuelFlowLph: Number(
        state.fuelFlowLph.toFixed(2)
      ),
      fuelPressureKpa: Number(
        state.fuelPressureKpa.toFixed(2)
      ),

      intakeManifoldPressureKpa: Number(
        state.intakeManifoldPressureKpa.toFixed(2)
      ),

      engineVibrationMmS: Number(
        state.engineVibrationMmS.toFixed(2)
      ),

      batteryVoltageV: Number(
        state.batteryVoltageV.toFixed(2)
      ),
      alternatorOutputA: Number(
        state.alternatorOutputA.toFixed(2)
      ),

      injectionTimingDegBtdc: Number(
        state.injectionTimingDegBtdc.toFixed(2)
      ),

      engineLoadPct: Number(
        state.engineLoadPct.toFixed(2)
      ),

      ambientTempC: Number(
        state.ambientTempC.toFixed(2)
      ),

      plannedAltitudeFt: Number(
        state.plannedAltitudeFt.toFixed(2)
      ),

      airspeedKts: Number(
        state.airspeedKts.toFixed(2)
      ),

      elapsedS: Number(
        state.elapsedS.toFixed(2)
      ),
    };
  }
}