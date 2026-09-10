import net from "node:net";
import {
  RemoteCanFrame,
  CanFrameHandler,
} from "./can.types";

export interface RemoteCanAdapterConfig {
  host: string;
  port: number;
  reconnect?: boolean;
  reconnectDelayMs?: number;
}

export class RemoteCanAdapter {
  private readonly config: Required<RemoteCanAdapterConfig>;

  private socket: net.Socket | null = null;
  private connected = false;

  private reconnectTimer: NodeJS.Timeout | null = null;

  private buffer = "";

  private readonly frameHandlers =
    new Set<CanFrameHandler>();

  constructor(
    config: RemoteCanAdapterConfig
  ) {
    this.config = {
      reconnect: true,
      reconnectDelayMs: 3000,
      ...config,
    };
  }

  connect(): void {
    if (this.socket !== null) {
      return;
    }

    console.log(
      `[RemoteCAN] Connecting to ` +
        `${this.config.host}:${this.config.port}...`
    );

    const socket = new net.Socket();

    this.socket = socket;

    socket.setKeepAlive(true);

    socket.on("connect", () => {
      this.connected = true;

      console.log(
        `[RemoteCAN] Connected to ` +
          `${this.config.host}:${this.config.port}`
      );
    });

    socket.on("data", (chunk: Buffer) => {
      this.handleData(chunk);
    });

    socket.on("error", (error) => {
      console.error(
        "[RemoteCAN] Socket error:",
        error.message
      );
    });

    socket.on("close", () => {
      this.connected = false;
      this.socket = null;

      console.log(
        "[RemoteCAN] Connection closed."
      );

      this.scheduleReconnect();
    });

    socket.connect(
      this.config.port,
      this.config.host
    );
  }

  disconnect(): void {
    this.clearReconnectTimer();

    this.config.reconnect = false;

    if (this.socket === null) {
      return;
    }

    this.socket.destroy();

    this.socket = null;
    this.connected = false;

    console.log(
      "[RemoteCAN] Disconnected."
    );
  }

  onFrame(
    handler: CanFrameHandler
  ): () => void {
    this.frameHandlers.add(handler);

    return () => {
      this.frameHandlers.delete(handler);
    };
  }

  isConnected(): boolean {
    return this.connected;
  }

  private handleData(
    chunk: Buffer
  ): void {
    this.buffer += chunk.toString("utf8");

    let newlineIndex: number;

    while (
      (newlineIndex =
        this.buffer.indexOf("\n")) !== -1
    ) {
      const rawMessage =
        this.buffer
          .slice(0, newlineIndex)
          .trim();

      this.buffer =
        this.buffer.slice(
          newlineIndex + 1
        );

      if (!rawMessage) {
        continue;
      }

      this.handleMessage(rawMessage);
    }
  }

  private handleMessage(
    rawMessage: string
  ): void {
    let parsed: unknown;

    try {
      parsed =
        JSON.parse(rawMessage);
    } catch (error) {
      console.error(
        "[RemoteCAN] Invalid JSON received:",
        error
      );

      return;
    }

    if (
      !this.isValidCanFrame(parsed)
    ) {
      console.error(
        "[RemoteCAN] Invalid CAN message received."
      );

      return;
    }

    for (
      const handler of this.frameHandlers
    ) {
      try {
        handler(parsed);
      } catch (error) {
        console.error(
          "[RemoteCAN] Frame handler failed:",
          error
        );
      }
    }
  }

  private isValidCanFrame(
    value: unknown
  ): value is RemoteCanFrame {
    if (
      typeof value !== "object" ||
      value === null
    ) {
      return false;
    }

    const message =
      value as Record<string, unknown>;

    return (
      message.type === "can" &&
      typeof message.id === "number" &&
      Number.isInteger(message.id) &&
      message.id >= 0 &&
      message.id <= 0x7ff &&
      typeof message.data === "string" &&
      /^[0-9a-fA-F]*$/.test(
        message.data
      ) &&
      message.data.length % 2 === 0 &&
      typeof message.timestamp === "number" &&
      Number.isFinite(
        message.timestamp
      )
    );
  }

  private scheduleReconnect(): void {
    if (!this.config.reconnect) {
      return;
    }

    if (
      this.reconnectTimer !== null
    ) {
      return;
    }

    console.log(
      `[RemoteCAN] Reconnecting in ` +
        `${this.config.reconnectDelayMs}ms...`
    );

    this.reconnectTimer =
      setTimeout(() => {
        this.reconnectTimer = null;

        if (this.socket === null) {
          this.connect();
        }
      }, this.config.reconnectDelayMs);
  }

  private clearReconnectTimer(): void {
    if (
      this.reconnectTimer === null
    ) {
      return;
    }

    clearTimeout(
      this.reconnectTimer
    );

    this.reconnectTimer = null;
  }
}