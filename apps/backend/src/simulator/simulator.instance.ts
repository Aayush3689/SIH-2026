import * as can from "socketcan";

import { EngineModel } from "./engine/engine.model";
import { EngineStateStore } from "./engine/engine.state";
import { TelemetryGenerator } from "./telemetry/telemetry.generator";
import { CanEncoder } from "./can/can.encoder";

import {
  EngineState,
  SimulatorConfig,
} from "./simulator.types";

export class SimulatorInstance {
  private readonly config: SimulatorConfig;

  private readonly engineState: EngineStateStore;
  private readonly engineModel: EngineModel;
  private readonly telemetryGenerator: TelemetryGenerator;
  private readonly canEncoder: CanEncoder;

  private channel: can.RawChannel | null = null;
  private timer: NodeJS.Timeout | null = null;

  constructor(
    config: SimulatorConfig,
    initialState: EngineState
  ) {
    this.config = config;

    this.engineState =
      new EngineStateStore(initialState);

    this.engineModel =
      new EngineModel();

    this.telemetryGenerator =
      new TelemetryGenerator();

    this.canEncoder =
      new CanEncoder();
  }

  // ============================================================
  // Start simulation
  // ============================================================

  start(): void {
    if (this.timer !== null) {
      throw new Error(
        `Simulator for UAV '${this.config.uavId}' is already running.`
      );
    }

    if (
      !Number.isFinite(this.config.rateHz) ||
      this.config.rateHz <= 0
    ) {
      throw new Error(
        "Simulator rate must be greater than 0 Hz."
      );
    }

    // ----------------------------------------------------------
    // SocketCAN raw channel
    // ----------------------------------------------------------

    this.channel = can.createRawChannel(
      this.config.canInterface,
      true
    );

    this.channel.start();

    // ----------------------------------------------------------
    // Simulation timer
    // ----------------------------------------------------------

    const intervalMs =
      1000 / this.config.rateHz;

    this.timer = setInterval(
      () => {
        try {
          this.tick();
        } catch (error) {
          console.error(
            `[Simulator] Tick failed for UAV '${this.config.uavId}':`,
            error
          );
        }
      },
      intervalMs
    );

    console.log(
      `[Simulator] Started | ` +
      `UAV=${this.config.uavId} | ` +
      `Engine=${this.config.engineId} | ` +
      `Interface=${this.config.canInterface} | ` +
      `Rate=${this.config.rateHz}Hz`
    );
  }

  // ============================================================
  // Stop simulation
  // ============================================================

  stop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }

    if (this.channel !== null) {
      this.channel.stop();
      this.channel = null;
    }

    console.log(
      `[Simulator] Stopped | UAV=${this.config.uavId}`
    );
  }

  // ============================================================
  // Simulation tick
  // ============================================================

  private tick(): void {
    if (this.channel === null) {
      return;
    }

    // ----------------------------------------------------------
    // 1. Read current state
    // ----------------------------------------------------------

    const currentState =
      this.engineState.getState();

    // ----------------------------------------------------------
    // 2. Calculate next engine state
    // ----------------------------------------------------------

    const nextState =
      this.engineModel.update(
        currentState
      );

    // ----------------------------------------------------------
    // 3. Store updated state
    // ----------------------------------------------------------

    this.engineState.updateState(
      nextState
    );

    // ----------------------------------------------------------
    // 4. Advance simulation time
    // ----------------------------------------------------------

    const elapsedSeconds =
      1 / this.config.rateHz;

    this.engineState.updateElapsedTime(
      elapsedSeconds
    );

    // ----------------------------------------------------------
    // 5. Generate telemetry sample
    // ----------------------------------------------------------

    const telemetry =
      this.telemetryGenerator.generate(
        this.engineState.getState(),
        this.config.uavId,
        this.config.engineId
      );

    // ----------------------------------------------------------
    // 6. Convert telemetry into CAN frames
    // ----------------------------------------------------------

    const frames =
      this.canEncoder.encode(
        telemetry
      );

    // ----------------------------------------------------------
    // 7. Send CAN frames to interface
    // ----------------------------------------------------------

    for (const frame of frames) {
      this.channel.send({
        id: frame.id,
        data: frame.data,
      });
    }

    // ----------------------------------------------------------
    // 8. Simulator debug output
    // ----------------------------------------------------------

    console.log(
      `[Simulator] ` +
      `${telemetry.engineId} | ` +
      `t=${telemetry.elapsedS.toFixed(1)}s | ` +
      `RPM=${telemetry.rpm.toFixed(0)} | ` +
      `EGT=${telemetry.egtC.toFixed(1)} | ` +
      `CHT=${telemetry.chtC.toFixed(1)} | ` +
      `OilP=${telemetry.oilPressureKpa.toFixed(1)}`
    );
  }

  // ============================================================
  // Current status
  // ============================================================

  isRunning(): boolean {
    return this.timer !== null;
  }

  getConfig(): SimulatorConfig {
    return {
      ...this.config,
    };
  }

  getCurrentState(): EngineState {
    return this.engineState.getState();
  }
}