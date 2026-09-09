import {
  CanFrameData,
  TelemetrySample,
} from "../simulator.types";

export class CanEncoder {
  encode(
    telemetry: TelemetrySample
  ): CanFrameData[] {
    return [
      this.encodeEngineOperating(telemetry),
      this.encodeThermal(telemetry),
      this.encodePressure(telemetry),
      this.encodeElectrical(telemetry),
      this.encodeEnvironment(telemetry),
      this.encodeRuntime(telemetry),
    ];
  }

  // ============================================================
  // 0x100 - Engine operating parameters
  //
  // Byte 0-1 : RPM
  // Byte 2-3 : Throttle Position × 10
  // Byte 4-5 : Engine Load × 10
  // Byte 6-7 : Fuel Flow × 100
  // ============================================================

  private encodeEngineOperating(
    telemetry: TelemetrySample
  ): CanFrameData {
    const data = Buffer.alloc(8);

    data.writeUInt16BE(
      this.toUInt16(telemetry.rpm),
      0
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.throttlePositionPct * 10
      ),
      2
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.engineLoadPct * 10
      ),
      4
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.fuelFlowLph * 100
      ),
      6
    );

    return {
      id: 0x100,
      data,
    };
  }

  // ============================================================
  // 0x101 - Thermal parameters
  //
  // Byte 0-1 : EGT × 10
  // Byte 2-3 : CHT × 10
  // Byte 4-5 : Oil Temperature × 10
  // Byte 6-7 : Reserved
  // ============================================================

  private encodeThermal(
    telemetry: TelemetrySample
  ): CanFrameData {
    const data = Buffer.alloc(8);

    data.writeInt16BE(
      this.toInt16(
        telemetry.egtC * 10
      ),
      0
    );

    data.writeInt16BE(
      this.toInt16(
        telemetry.chtC * 10
      ),
      2
    );

    data.writeInt16BE(
      this.toInt16(
        telemetry.oilTemperatureC * 10
      ),
      4
    );

    return {
      id: 0x101,
      data,
    };
  }

  // ============================================================
  // 0x102 - Pressure / fuel parameters
  //
  // Byte 0-1 : Oil Pressure × 10
  // Byte 2-3 : Fuel Pressure × 10
  // Byte 4-5 : Intake Manifold Pressure × 10
  // Byte 6-7 : Reserved
  // ============================================================

  private encodePressure(
    telemetry: TelemetrySample
  ): CanFrameData {
    const data = Buffer.alloc(8);

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.oilPressureKpa * 10
      ),
      0
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.fuelPressureKpa * 10
      ),
      2
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.intakeManifoldPressureKpa * 10
      ),
      4
    );

    return {
      id: 0x102,
      data,
    };
  }

  // ============================================================
  // 0x103 - Mechanical / electrical parameters
  //
  // Byte 0-1 : Vibration × 100
  // Byte 2-3 : Battery Voltage × 100
  // Byte 4-5 : Alternator Output × 10
  // Byte 6-7 : Reserved
  // ============================================================

  private encodeElectrical(
    telemetry: TelemetrySample
  ): CanFrameData {
    const data = Buffer.alloc(8);

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.engineVibrationMmS * 100
      ),
      0
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.batteryVoltageV * 100
      ),
      2
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.alternatorOutputA * 10
      ),
      4
    );

    return {
      id: 0x103,
      data,
    };
  }

  // ============================================================
  // 0x104 - Injection / environment parameters
  //
  // Byte 0-1 : Injection Timing × 10
  // Byte 2-3 : Ambient Temperature × 10
  // Byte 4-5 : Planned Altitude
  // Byte 6-7 : Airspeed × 10
  // ============================================================

  private encodeEnvironment(
    telemetry: TelemetrySample
  ): CanFrameData {
    const data = Buffer.alloc(8);

    data.writeInt16BE(
      this.toInt16(
        telemetry.injectionTimingDegBtdc * 10
      ),
      0
    );

    data.writeInt16BE(
      this.toInt16(
        telemetry.ambientTempC * 10
      ),
      2
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.plannedAltitudeFt
      ),
      4
    );

    data.writeUInt16BE(
      this.toUInt16(
        telemetry.airspeedKts * 10
      ),
      6
    );

    return {
      id: 0x104,
      data,
    };
  }

  // ============================================================
  // 0x105 - Runtime / protocol information
  //
  // Byte 0-3 : Elapsed Seconds
  // Byte 4-5 : Reserved
  // Byte 6-7 : Protocol Version
  // ============================================================

  private encodeRuntime(
    telemetry: TelemetrySample
  ): CanFrameData {
    const data = Buffer.alloc(8);

    data.writeUInt32BE(
      this.toUInt32(
        telemetry.elapsedS
      ),
      0
    );

    // Protocol version = 1
    data.writeUInt16BE(
      1,
      6
    );

    return {
      id: 0x105,
      data,
    };
  }

  // ============================================================
  // Numeric safety helpers
  // ============================================================

  private toUInt16(
    value: number
  ): number {
    if (!Number.isFinite(value)) {
      return 0;
    }

    return Math.max(
      0,
      Math.min(
        65535,
        Math.round(value)
      )
    );
  }

  private toUInt32(
    value: number
  ): number {
    if (!Number.isFinite(value)) {
      return 0;
    }

    return Math.max(
      0,
      Math.min(
        4294967295,
        Math.round(value)
      )
    );
  }

  private toInt16(
    value: number
  ): number {
    if (!Number.isFinite(value)) {
      return 0;
    }

    return Math.max(
      -32768,
      Math.min(
        32767,
        Math.round(value)
      )
    );
  }
}