import { RemoteCanAdapter } from "./remote-can.adapter";

const adapter =
  new RemoteCanAdapter({
    host: "56.228.75.94",
    port: 5001,
    reconnect: true,
    reconnectDelayMs: 3000,
  });

adapter.onFrame(
  (frame) => {
    console.log(
      "[RemoteCAN] Received frame:",
      frame
    );
  }
);

adapter.connect();

process.on(
  "SIGINT",
  () => {
    console.log(
      "\n[RemoteCAN] Shutting down..."
    );

    adapter.disconnect();

    process.exit(0);
  }
);