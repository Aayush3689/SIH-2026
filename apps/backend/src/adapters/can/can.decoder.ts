import {
  CanFrame,
  DecodedCanFrame,
} from "./can.types";

export class CanDecoder {
  decode(
    frame: CanFrame
  ): DecodedCanFrame {
    const {
      nodeId,
      messageType,
    } = this.parseCanId(frame.id);

    if (
      frame.data.length !== 8
    ) {
      throw new Error(
        `Invalid CAN payload length for ` +
          `CAN ID 0x${frame.id.toString(16)}. ` +
          `Expected 8 bytes, received ` +
          `${frame.data.length}.`
      );
    }

    switch (messageType) {
      case 0x01:
        return {
          canId: frame.id,
          nodeId,
          messageType,
          fields:
            this.decodeEngineOperating(
              frame.data
            ),
        };

      case 0x02:
        return {
          canId: frame.id,
          nodeId,
          messageType,
          fields:
            this.decodeThermal(
              frame.data
            ),
        };

      case 0x03:
        return {
          canId: frame.id,
          nodeId,
          messageType,
          fields:
            this.decodePressure(
              frame.data
            ),
        };

      case 0x04:
        return {
          canId: frame.id,
          nodeId,
          messageType,
          fields:
            this.decodeElectrical(
              frame.data
            ),
        };

      case 0x05:
        return {
          canId: frame.id,
          nodeId,
          messageType,
          fields:
            this.decodeEnvironment(
              frame.data
            ),
        };

      case 0x06:
        return {
          canId: frame.id,
          nodeId,
          messageType,
          fields:
            this.decodeRuntime(
              frame.data
            ),
        };

      default:
        throw new Error(
          `Unsupported CAN message type ` +
            `0x${messageType.toString(16)}.`
        );
    }
  }

  private parseCanId(
    canId: number
  ): {
    nodeId: number;
    messageType: number;
  } {
    if (
      !Number.isInteger(canId) ||
      canId < 0 ||
      canId > 0x7ff
    ) {
      throw new Error(
        `Invalid standard CAN ID: ${canId}.`
      );
    }

    const offset =
      canId - 0x100;

    if (offset < 0) {
      throw new Error(
        `Unsupported CAN ID: ` +
          `0x${canId.toString(16)}.`
      );
    }

    const nodeId =
      Math.floor(offset / 0x10);

    const messageType =
      offset % 0x10;

    if (
      nodeId < 1 ||
      nodeId > 127
    ) {
      throw new Error(
        `Invalid CAN node ID: ${nodeId}.`
      );
    }

    return {
      nodeId,
      messageType,
    };
  }

  private decodeEngineOperating(
    data: Buffer
  ): Record<string, number> {
    return {
      rpm:
        data.readUInt16BE(0),

      throttlePositionPct:
        data.readUInt16BE(2) / 10,

      engineLoadPct:
        data.readUInt16BE(4) / 10,

      fuelFlowLph:
        data.readUInt16BE(6) / 100,
    };
  }

  private decodeThermal(
    data: Buffer
  ): Record<string, number> {
    return {
      egtC:
        data.readInt16BE(0) / 10,

      chtC:
        data.readInt16BE(2) / 10,

      oilTemperatureC:
        data.readInt16BE(4) / 10,
    };
  }

  private decodePressure(
    data: Buffer
  ): Record<string, number> {
    return {
      oilPressureKpa:
        data.readUInt16BE(0) / 10,

      fuelPressureKpa:
        data.readUInt16BE(2) / 10,

      intakeManifoldPressureKpa:
        data.readUInt16BE(4) / 10,
    };
  }

  private decodeElectrical(
    data: Buffer
  ): Record<string, number> {
    return {
      engineVibrationMmS:
        data.readUInt16BE(0) / 100,

      batteryVoltageV:
        data.readUInt16BE(2) / 100,

      alternatorOutputA:
        data.readUInt16BE(4) / 10,
    };
  }

  private decodeEnvironment(
    data: Buffer
  ): Record<string, number> {
    return {
      injectionTimingDegBtdc:
        data.readInt16BE(0) / 10,

      ambientTempC:
        data.readInt16BE(2) / 10,

      plannedAltitudeFt:
        data.readUInt16BE(4),

      airspeedKts:
        data.readUInt16BE(6) / 10,
    };
  }

  private decodeRuntime(
    data: Buffer
  ): Record<string, number> {
    return {
      elapsedS:
        data.readUInt32BE(0),

      protocolVersion:
        data.readUInt16BE(6),
    };
  }
}