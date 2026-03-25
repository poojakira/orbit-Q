# Ingestion Module

gRPC / REST services that accept raw satellite telemetry (power, thermal, ADCS, comms) and normalize it into a common event schema.

## Available Endpoints
- `POST /api/v1/telemetry`: Core high-frequency ingestion endpoint for raw spacecraft states.
- `GET /api/v1/health`: Alive check for the ingestion load balancer.

## Example Payload

```json
{
  "timestamp": "2026-03-20T21:30:00Z",
  "subsystem": "EPS",
  "metrics": {
    "battery_voltage": 8.2,
    "solar_panel_current": 1.4,
    "temperature_c": 12.5
  }
}
```

```bash
curl -X POST http://localhost:8080/api/v1/telemetry \
     -H "Content-Type: application/json" \
     -d '{"timestamp":"2026-03-20T21:30","subsystem":"EPS","metrics":{"battery_voltage":8.2}}'
```
