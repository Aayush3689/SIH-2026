import net from "node:net";
import * as can from "socketcan";

export interface CanTcpGatewayConfig {
  canInterface: string;
  host: string;
  port: number;
}

interface CanTcpMessage {
  type: "can";
  id: number;
  data: string;
  timestamp: number;
}

export class CanTcpGateway {
  private readonly config: CanTcpGatewayConfig;

  private channel: can.RawChannel | null = null;
  private server: net.Server | null = null;

  private readonly clients = new Set<net.Socket>();

  constructor(config: CanTcpGatewayConfig) {
    this.config = config;
  }

  start(): void {
    if (this.server !== null) {
      throw new Error(
        "CAN TCP Gateway is already running."
      );
    }

    // ----------------------------------------------------------
    // SocketCAN channel
    // ----------------------------------------------------------

    this.channel = can.createRawChannel(
      this.config.canInterface,
      true
    );

    this.channel.addListener(
      "onMessage",
      (message) => {
        this.handleCanFrame(message);
      }
    );

    this.channel.start();

    // ----------------------------------------------------------
    // TCP server
    // ----------------------------------------------------------

    this.server = net.createServer(
      (socket) => {
        this.handleClient(socket);
      }
    );

    this.server.on(
      "error",
      (error) => {
        console.error(
          "[CAN Gateway] TCP server error:",
          error
        );
      }
    );

    this.server.listen(
      this.config.port,
      this.config.host,
      () => {
        console.log(
          `[CAN Gateway] Started | ` +
          `CAN=${this.config.canInterface} | ` +
          `TCP=${this.config.host}:${this.config.port}`
        );
      }
    );
  }

  stop(): void {
    // ----------------------------------------------------------
    // Stop SocketCAN
    // ----------------------------------------------------------

    if (this.channel !== null) {
      this.channel.stop();
      this.channel = null;
    }

    // ----------------------------------------------------------
    // Close connected TCP clients
    // ----------------------------------------------------------

    for (const client of this.clients) {
      client.destroy();
    }

    this.clients.clear();

    // ----------------------------------------------------------
    // Stop TCP server
    // ----------------------------------------------------------

    if (this.server !== null) {
      this.server.close();
      this.server = null;
    }

    console.log(
      "[CAN Gateway] Stopped."
    );
  }

  getClientCount(): number {
    return this.clients.size;
  }

  isRunning(): boolean {
    return (
      this.server !== null &&
      this.channel !== null
    );
  }

  private handleClient(
    socket: net.Socket
  ): void {
    const remoteAddress =
      `${socket.remoteAddress ?? "unknown"}:` +
      `${socket.remotePort ?? "unknown"}`;

    this.clients.add(socket);

    console.log(
      `[CAN Gateway] TCP client connected: ${remoteAddress}`
    );

    socket.setKeepAlive(true);

    socket.on(
      "error",
      (error) => {
        console.error(
          `[CAN Gateway] Client error ${remoteAddress}:`,
          error.message
        );
      }
    );

    socket.on(
      "close",
      () => {
        this.clients.delete(socket);

        console.log(
          `[CAN Gateway] TCP client disconnected: ${remoteAddress}`
        );
      }
    );
  }

  private handleCanFrame(
    frame: can.CanMessage
  ): void {
    const message: CanTcpMessage = {
      type: "can",
      id: frame.id,
      data: frame.data.toString("hex"),
      timestamp: Date.now(),
    };

    const payload =
      `${JSON.stringify(message)}\n`;

    for (const client of this.clients) {
      if (client.destroyed) {
        this.clients.delete(client);
        continue;
      }

      client.write(payload);
    }
  }
}