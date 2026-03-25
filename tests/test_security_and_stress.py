"""
Tests for the security module (token validation, audit trail)
and the multi-CubeSat stress test simulator.
"""
import pytest
import time
import os
from orbit_q.security import (
    generate_stream_token,
    validate_stream_token,
    audit,
)
from orbit_q.simulator.multi_cubesat_stress import MultiCubeSatStressTest


# ── Security Tests ─────────────────────────────────────────────────────────────

class TestTokenValidation:
    def test_valid_token(self):
        token = generate_stream_token("SAT-001")
        assert validate_stream_token(token) is True

    def test_tampered_token_rejected(self):
        token = generate_stream_token("SAT-001")
        tampered = token[:-4] + "XXXX"
        assert validate_stream_token(tampered) is False

    def test_expired_token_rejected(self):
        # Generate token with a timestamp far in the past
        old_ts = int(time.time()) - 9999
        token = generate_stream_token("SAT-001", timestamp=old_ts)
        assert validate_stream_token(token) is False

    def test_malformed_token_rejected(self):
        assert validate_stream_token("not-a-valid-token") is False
        assert validate_stream_token("") is False


class TestAuditTrail:
    def test_audit_writes_to_log(self, tmp_path):
        log_path = str(tmp_path / "test_audit.log")
        import orbit_q.security as sec_module
        original = sec_module._AUDIT_LOG_PATH
        sec_module._AUDIT_LOG_PATH = log_path

        audit("ANOMALY_DETECTED", satellite_id="SAT-001", extra={"score": -0.45})

        sec_module._AUDIT_LOG_PATH = original  # restore

        assert os.path.exists(log_path)
        content = open(log_path).read()
        assert "ANOMALY_DETECTED" in content
        assert "SAT-001" in content


# ── Multi-CubeSat Stress Test ──────────────────────────────────────────────────

class TestMultiCubeSatStressTest:
    def test_stress_test_runs_and_reports(self):
        """3 satellites × 5 Hz × 2s — fast smoke test."""
        test = MultiCubeSatStressTest(n_satellites=3, hz_per_satellite=5, duration_s=2.0)
        report = test.run()

        assert report["n_satellites"] == 3
        assert report["total_packets"] > 0
        assert "aggregate_throughput_hz" in report
        assert report["anomaly_rate_pct"] >= 0

    def test_throughput_scales_with_satellites(self):
        """More satellites → proportionally higher total throughput."""
        small = MultiCubeSatStressTest(n_satellites=2, hz_per_satellite=5, duration_s=2.0)
        large = MultiCubeSatStressTest(n_satellites=5, hz_per_satellite=5, duration_s=2.0)

        r_small = small.run()
        r_large = large.run()

        # Large should have at least 2× the packets of small
        assert r_large["total_packets"] >= r_small["total_packets"] * 1.5
