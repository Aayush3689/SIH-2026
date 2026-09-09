declare module "socketcan" {
  export interface CanMessage {
    id: number;
    data: Buffer;
    ext?: boolean;
    rtr?: boolean;
  }

  export interface RawChannel {
    start(): void;

    stop(): void;

    send(message: CanMessage): void;

    addListener(
      event: "onMessage",
      listener: (message: CanMessage) => void
    ): this;

    on(
      event: "onMessage",
      listener: (message: CanMessage) => void
    ): this;
  }

  export function createRawChannel(
    interfaceName: string,
    receiveOwnMessages?: boolean
  ): RawChannel;
}