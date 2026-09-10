import {
  RemoteCanAdapter,
} from "./remote-can.adapter";

import {
  CanDecoder,
} from "./can.decoder";

const adapter =
  new RemoteCanAdapter({
    host: "56.228.75.94",
    port: 5001,
    reconnect: true,
    reconnectDelayMs: 3000,
  });

const decoder =
  new CanDecoder();

adapter.onFrame((frame) => {
  try {
    const decoded =
      decoder.decode({
        id: frame.id,
        data: Buffer.from(
          frame.data,
          "hex"
        ),
      });

    console.log(
      "\n=============================="
    );

    console.log(
      `CAN ID      : 0x${frame.id.toString(16)}`
    );

    console.log(
      `Node ID     : ${decoded.nodeId}`
    );

    console.log(
      `Message Type: 0x${decoded.messageType.toString(16)}`
    );

    console.log(
      "Decoded Fields:",
      decoded.fields
    );

    console.log(
      "==============================\n"
    );
  } catch (error) {
    console.error(
      "[CAN Pipeline] Decode failed:",
      error
    );
  }
});

adapter.connect();

process.on("SIGINT", () => {
  console.log(
    "\n[CAN Pipeline] Shutting down..."
  );

  adapter.disconnect();

  process.exit(0);
});

process.on("SIGTERM", () => {
  adapter.disconnect();

  process.exit(0);
});