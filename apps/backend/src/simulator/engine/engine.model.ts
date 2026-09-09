import { EngineState } from "../simulator.types";

export class EngineModel {
  update(state: EngineState): EngineState {
    const targetRpm = this.calculateTargetRpm(
      state.throttlePositionPct
    );

    const rpm = this.moveTowards(
      state.rpm,
      targetRpm,
      0.12
    );

    const engineLoadPct =
      this.calculateEngineLoad(
        state.throttlePositionPct
      );

    const fuelFlowLph =
      this.calculateFuelFlow(
        engineLoadPct
      );

    const egtC =
      this.calculateEgt(
        engineLoadPct,
        state.ambientTempC
      );

    const chtC =
      this.calculateCht(
        engineLoadPct,
        state.ambientTempC
      );

    const oilPressureKpa =
      this.calculateOilPressure(
        rpm
      );

    const oilTemperatureC =
      this.calculateOilTemperature(
        engineLoadPct,
        state.ambientTempC
      );

    const fuelPressureKpa =
      this.calculateFuelPressure(
        engineLoadPct
      );

    const intakeManifoldPressureKpa =
      this.calculateManifoldPressure(
        state.throttlePositionPct
      );

    const engineVibrationMmS =
      this.calculateVibration(
        rpm,
        engineLoadPct
      );

    const alternatorOutputA =
      this.calculateAlternatorOutput(
        engineLoadPct
      );

    return {
      ...state,

      rpm,
      engineLoadPct,

      fuelFlowLph,
      egtC,
      chtC,

      oilPressureKpa,
      oilTemperatureC,

      fuelPressureKpa,
      intakeManifoldPressureKpa,

      engineVibrationMmS,
      alternatorOutputA,
    };
  }

  private calculateTargetRpm(
    throttlePositionPct: number
  ): number {
    const idleRpm = 1800;
    const maximumOperatingRpm = 5200;

    const throttle =
      this.clamp(
        throttlePositionPct,
        0,
        100
      );

    return (
      idleRpm +
      (throttle / 100) *
        (maximumOperatingRpm - idleRpm)
    );
  }

  private calculateEngineLoad(
    throttlePositionPct: number
  ): number {
    const throttle =
      this.clamp(
        throttlePositionPct,
        0,
        100
      );

    return this.clamp(
      throttle * 0.95,
      0,
      100
    );
  }

  private calculateFuelFlow(
    engineLoadPct: number
  ): number {
    const baseFlowLph = 6;
    const flowPerLoadPct = 0.24;

    return (
      baseFlowLph +
      engineLoadPct * flowPerLoadPct
    );
  }

  private calculateEgt(
    engineLoadPct: number,
    ambientTempC: number
  ): number {
    const baseEgtC = 520;
    const loadContribution = 3.5;

    return (
      baseEgtC +
      engineLoadPct * loadContribution +
      (ambientTempC - 25) * 0.35
    );
  }

  private calculateCht(
    engineLoadPct: number,
    ambientTempC: number
  ): number {
    const baseChtC = 105;
    const loadContribution = 1.1;

    return (
      baseChtC +
      engineLoadPct * loadContribution +
      (ambientTempC - 25) * 0.25
    );
  }

  private calculateOilPressure(
    rpm: number
  ): number {
    const basePressureKpa = 250;
    const rpmContribution = 0.035;

    return (
      basePressureKpa +
      rpm * rpmContribution
    );
  }

  private calculateOilTemperature(
    engineLoadPct: number,
    ambientTempC: number
  ): number {
    const baseOilTemperatureC = 70;
    const loadContribution = 0.45;
    const ambientContribution =
      (ambientTempC - 25) * 0.20;

    return (
      baseOilTemperatureC +
      engineLoadPct * loadContribution +
      ambientContribution
    );
  }

  private calculateFuelPressure(
    engineLoadPct: number
  ): number {
    const basePressureKpa = 270;
    const loadContribution = 0.5;

    return (
      basePressureKpa +
      engineLoadPct * loadContribution
    );
  }

  private calculateManifoldPressure(
    throttlePositionPct: number
  ): number {
    const minimumPressureKpa = 55;
    const pressureRangeKpa = 40;

    const throttle =
      this.clamp(
        throttlePositionPct,
        0,
        100
      );

    return (
      minimumPressureKpa +
      (throttle / 100) *
        pressureRangeKpa
    );
  }

  private calculateVibration(
    rpm: number,
    engineLoadPct: number
  ): number {
    const baseVibration = 1.2;
    const loadContribution = 0.012;

    const rpmDeviation =
      Math.abs(rpm - 3500);

    const rpmContribution =
      Math.min(
        0.8,
        rpmDeviation / 5000
      );

    return (
      baseVibration +
      engineLoadPct * loadContribution +
      rpmContribution
    );
  }

  private calculateAlternatorOutput(
    engineLoadPct: number
  ): number {
    const baseOutputA = 8;
    const loadContribution = 0.16;

    return (
      baseOutputA +
      engineLoadPct * loadContribution
    );
  }

  private moveTowards(
    current: number,
    target: number,
    factor: number
  ): number {
    return (
      current +
      (target - current) * factor
    );
  }

  private clamp(
    value: number,
    minimum: number,
    maximum: number
  ): number {
    return Math.max(
      minimum,
      Math.min(
        maximum,
        value
      )
    );
  }
}