# Frontend API expectations

The frontend currently has exactly two live AI-backend contracts. Configure their
shared origin with `NEXT_PUBLIC_AI_API_BASE_URL`; the local default is
`http://localhost:8000`. Do not add or call undocumented API routes.

## 1. Health check

`GET /health`

Expected successful response:

```json
{
  "status": "ok",
  "service": "ai-backend"
}
```

Use this only to report backend reachability. The frontend client exposes it as
`getHealth()` from `@/lib/api` (or `aiBackendService.getHealth()`).

## 2. Inference

`POST /api/v1/inference`

Send a JSON request containing the exact `telemetry` array shape below:

```json
{
  "telemetry": [
    {
      "timestamp": "2026-09-09T06:14:25.000Z",
      "engine_id": "ENG-F01",
      "flight_id": "FLT-260909-A1",
      "elapsed_s": 55,
      "rpm": 5220,
      "throttle_position_pct": 51.2,
      "cht_c": 173.1,
      "egt_c": 641.5,
      "oil_pressure_kpa": 379.3,
      "oil_temperature_c": 89.3,
      "fuel_flow_lph": 13.9,
      "fuel_pressure_kpa": 290.4,
      "intake_manifold_pressure_kpa": 112.1,
      "engine_vibration_mm_s": 3.4,
      "battery_voltage_v": 25.04,
      "alternator_output_a": 39.8,
      "injection_timing_deg_btdc": 18.4,
      "engine_load_pct": 66.4,
      "ambient_temp_c": 29.3,
      "planned_altitude_ft": 4850,
      "airspeed_kts": 68
    }
  ]
}
```

Expected successful response shape:

```ts
{
  success: true;
  data: {
    engine_id: string;
    flight_id: string;
    timestamp: string;
    anomaly: Array<{
      state: "NORMAL" | "PRE_FAULT" | "ACTIVE_FAULT";
      raw_label: number;
      probabilities: {
        NORMAL: number;
        PRE_FAULT: number;
        ACTIVE_FAULT: number;
      };
      confidence: number;
    }>;
    fault: Array<{
      fault:
        | "abnormal_vibration"
        | "coking_degradation"
        | "combustion_instability"
        | "injector_abnormality"
        | "lubrication_issue"
        | "misfire"
        | "overheating_trend"
        | "sensor_drift";
      probabilities: Record<string, number>;
      confidence: number;
    }>;
    rul: number[] | null;
  };
}
```

The live client is `getInference(telemetry)` from `@/lib/api` (or
`aiBackendService.getInference(telemetry)`). The shared request type is
`InferenceRequest` and is intentionally limited to `{ telemetry: TelemetrySample[] }`.

## RUL display rule

`data.rul` is numerical only when it is an array. A null RUL is not zero, an
error, or an estimate that the frontend should calculate. When `data.rul === null`,
the UI must show this exact message:

> RUL is not available yet. Collect at least 8 telemetry samples.

Only display numerical RUL values when the backend supplies a non-null array.

## Client boundaries and consuming screens

There are two intentionally separate client paths:

| Mode | Entry point | Use |
| --- | --- | --- |
| Live | `src/lib/api/health.ts` → `getHealth()` | Calls `GET /health` |
| Live | `src/lib/api/inference.ts` → `getInference(telemetry)` | Calls `POST /api/v1/inference` |
| Live facade | `src/lib/services.ts` → `aiBackendService` | Typed wrapper around the two live clients |
| Demo | `src/lib/services.ts` → `demoService` | Deterministic frontend-local fleet, telemetry, alerts, mappings, CAN, analytics, and inference data |

Dashboard consumes fleet and alert summaries from `demoService`. Live Monitoring and
Engine Health consume `anomaly`, `fault`, `confidence`, `probabilities`, and `rul`
from the inference result. Any inferred fault must be presented as predicted or
suspected, not as a confirmed fault. The RUL rule above applies in every consuming
screen.

Pages should use the service surface rather than import arrays from `src/lib/mock`
directly. Switching mock/live inference must therefore not require a page rewrite.

## Error behavior

Both live clients throw `ApiRequestError` for network failures, non-2xx responses,
and malformed successful responses. Consumers should show a friendly inline error
or toast and retain any previously visible telemetry; they must not fabricate an
inference result or a numerical RUL. `data.rul === null` is an informational
ready-state, not an API error.

## Versioning note

The current backend contract has no separately advertised API version beyond the
documented inference path. Keep request/response changes backward compatible, add
new optional fields where possible, and update this document, shared TypeScript
types, and both mock/live clients together before consuming a future contract.
