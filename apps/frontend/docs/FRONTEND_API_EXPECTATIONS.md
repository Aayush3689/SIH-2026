# Frontend API expectations

The application currently runs in deterministic demo mode. Page components consume typed services from `src/lib/api/`; they must not call backend endpoints or import mock data directly when a service is available.

## Endpoints

| Method | Path | Purpose | Environment |
| --- | --- | --- | --- |
| GET | `/health` | Render an optional AI-service health indicator. | Live only; demo returns a local healthy result. |
| POST | `/api/v1/inference` | Obtain anomaly, fault and RUL inference for an engine telemetry window. | Live only; demo uses a local deterministic result. |

## Inference request

`src/lib/api/inference.ts` owns the inference client. The live client must send `{ telemetry: TelemetrySample[] }`, where a sample has `timestamp`, `engine_id`, `flight_id`, `elapsed_s`, `rpm`, `throttle_position_pct`, `cht_c`, `egt_c`, `oil_pressure_kpa`, `oil_temperature_c`, `fuel_flow_lph`, `fuel_pressure_kpa`, `intake_manifold_pressure_kpa`, `engine_vibration_mm_s`, `battery_voltage_v`, `alternator_output_a`, `injection_timing_deg_btdc`, `engine_load_pct`, `ambient_temp_c`, `planned_altitude_ft`, and `airspeed_kts`.

## Inference response and UI mapping

The backend responds with `{ success, data }`. `data` contains `engine_id`, `flight_id`, `timestamp`, `anomaly[]`, `fault[]`, and `rul`.

- `anomaly.state`, probability map and confidence feed Live Monitoring and Engine Health.
- `fault.fault`, probability map and confidence are presented as a model prediction, never as a confirmed fault.
- `rul` feeds Live Monitoring, Engine Health, Fleet and Analytics risk states.
- `rul: null` means: “RUL is not available yet. Collect at least 8 telemetry samples.” Do not calculate or display an invented numeric RUL.

## Demo implementation

`src/lib/api/inference.ts` currently supplies the mock inference implementation, backed by `src/data/mock/index.ts`. It exposes `runInference`, `getRollupDiagnosis`, and `getDashboardAlerts`.

## Edge behaviour

- Invalid telemetry: retain the latest visible telemetry and surface “Telemetry validation failed. Check the incoming data.”
- Unavailable inference: retain telemetry and surface “AI inference is temporarily unavailable.” with retry.
- No telemetry/stale telemetry: show an informational status and last-update time.
- Future backend model or status endpoints must be documented here before UI components use them.
