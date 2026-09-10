export interface CanFrame {
  id: number;
  data: Buffer;
}

export interface RemoteCanFrame {
  type: "can";
  id: number;
  data: string;
  timestamp: number;
}

export interface DecodedCanFrame {
  canId: number;
  nodeId: number;
  messageType: number;
  fields: Record<string, number>;
}

export type CanFrameHandler = (
  frame: RemoteCanFrame
) => void;