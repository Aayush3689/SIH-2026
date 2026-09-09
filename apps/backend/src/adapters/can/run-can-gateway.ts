import { CanTcpGateway } from "./can.gateway";

const gateway =
  new CanTcpGateway({
    canInterface: "vcan0",
    host: "0.0.0.0",
    port: 5001,
  });

try {
  gateway.start();
} catch (error) {
  console.error(
    "[CAN Gateway] Failed to start:",
    error
  );

  process.exit(1);
}

const shutdown = (
  signal: string
): void => {
  console.log(
    `\n[CAN Gateway] Received ${signal}.`
  );

  gateway.stop();

  process.exit(0);
};

process.on(
  "SIGINT",
  () => shutdown("SIGINT")
);

process.on(
  "SIGTERM",
  () => shutdown("SIGTERM")
);