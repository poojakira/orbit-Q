# Orchestrator Module

The Orchestrator acts as the central rules engine. It fuses anomaly outputs from the models with absolute mission rules before dispatching events into `/ML_ALERTS` for downstream routing via Webhooks or PagerDuty.

## Handling Rules Engine

Rules are codified inside `ml_orchestrator.py`. They operate on rolling multi-variate metrics extracted by `feature_processor.py`.

### Mission Code Examples

If thermal ranges breach standard limits alongside an anomaly spike:

```python
# Simplified Rule Example
if telemetry['temperature_c'] > 85.0 and telemetry['battery_voltage'] < 7.2:
    trigger_alert("CRITICAL_THERMAL_SAG", priority=1)
elif anomaly_flag == "ANOMALY_SENSITIVE" and telemetry['solar_panel_current'] < 0.2:
    trigger_alert("ECLIPSE_ANOMALY", priority=2)
```
