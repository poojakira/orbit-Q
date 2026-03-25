"""
Security & Auth Hardening Module
Provides token validation for telemetry streams, secrets management via
environment variables, and an audit trail logger for key events.
"""
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Secrets (always from env, never hardcoded) ───────────────────────────────
_SIGNING_SECRET = os.getenv("ORBIT_Q_SIGNING_SECRET", "change-me-in-production")
_TOKEN_TTL_SECONDS = int(os.getenv("ORBIT_Q_TOKEN_TTL", "3600"))


# ── Token Validation ─────────────────────────────────────────────────────────
def generate_stream_token(satellite_id: str, timestamp: Optional[int] = None) -> str:
    """
    Generate an HMAC-SHA256 token for a satellite telemetry stream.

    Format: ``{satellite_id}:{timestamp}:{signature}``
    """
    ts = timestamp or int(time.time())
    payload = f"{satellite_id}:{ts}"
    sig = hmac.new(
        _SIGNING_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload}:{sig}"


def validate_stream_token(token: str) -> bool:
    """
    Return True if the token is valid (correct HMAC) and not expired.
    Constant-time comparison prevents timing attacks.
    """
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return False
        satellite_id, ts_str, provided_sig = parts
        ts = int(ts_str)

        if time.time() - ts > _TOKEN_TTL_SECONDS:
            log.warning("Token for %s is expired (age=%ds)", satellite_id, int(time.time() - ts))
            return False

        payload = f"{satellite_id}:{ts_str}"
        expected_sig = hmac.new(
            _SIGNING_SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_sig, provided_sig)
    except Exception as exc:
        log.error("Token validation error: %s", exc)
        return False


# ── Audit Trail ───────────────────────────────────────────────────────────────
_AUDIT_LOG_PATH = os.getenv("ORBIT_Q_AUDIT_LOG", "audit.log")


def audit(event: str, satellite_id: str = "unknown", extra: Optional[dict] = None) -> None:
    """
    Append a structured audit event to the audit log file.
    Events: TELEMETRY_INGESTED | ANOMALY_DETECTED | MODEL_RETRAINED | AUTH_FAIL
    """
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": event,
        "satellite_id": satellite_id,
        **(extra or {}),
    }
    log.info("AUDIT | %s", record)
    try:
        with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(str(record) + "\n")
    except OSError as exc:
        log.error("Could not write audit log: %s", exc)


# ── Webhook Alerting ─────────────────────────────────────────────────────────
def send_alert_webhook(message: str, channel: str = "slack") -> bool:
    """Send an anomaly alert to the configured webhook (Slack or PagerDuty sim)."""
    try:
        import requests  # type: ignore

        if channel == "slack":
            url = os.getenv("SLACK_WEBHOOK_URL", "")
            if not url:
                log.info("[SIM] Slack alert: %s", message)
                return True
            r = requests.post(url, json={"text": message}, timeout=5)
            r.raise_for_status()
            return True

        elif channel == "pagerduty":
            routing_key = os.getenv("PAGERDUTY_ROUTING_KEY", "")
            if not routing_key:
                log.info("[SIM] PagerDuty alert: %s", message)
                return True
            url = "https://events.pagerduty.com/v2/enqueue"
            payload = {
                "routing_key": routing_key,
                "event_action": "trigger",
                "payload": {
                    "summary": message,
                    "severity": "critical",
                    "source": "orbit-q",
                },
            }
            r = requests.post(url, json=payload, timeout=5)
            r.raise_for_status()
            return True

    except Exception as exc:
        log.error("Webhook send failed (%s): %s", channel, exc)
        return False

    return False
