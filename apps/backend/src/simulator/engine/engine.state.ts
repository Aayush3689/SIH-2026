import { EngineState } from "../simulator.types";

export class EngineStateStore {
  private state: EngineState;
  constructor(intialState: EngineState) {
    this.state = { ...intialState };
  }

  getState() {
    return { ...this.state };
  }

  updateState(updates: Partial<EngineState>): void {
    this.state = {
      ...this.state,
      ...updates,
    };
  }

  updateElapsedTime(seconds: number): void {
    if (!Number.isFinite(seconds) || seconds < 0) {
      throw new Error(
        "Elapsed time increment must be a non-negative finite number.",
      );
    }

    this.state.elapsedS += seconds;
  }

  reset(initialState: EngineState): void {
    this.state = { ...initialState };
  }
}
