"""
Automatic Retraining Pipeline
Monitors anomaly detection performance (precision drift) and triggers
a model retrain when the rolling anomaly rate deviates from the baseline.
"""

import logging
import time
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Watches inference anomaly rates and triggers ensemble retraining
    whenever the rate exceeds 'drift_threshold' above the calibrated
    baseline, indicating concept drift or sensor degradation.
    """

    def __init__(
        self,
        engine,
        baseline_anomaly_rate: float = 0.05,
        drift_threshold: float = 0.10,
        window_size: int = 200,
    ):
        self.engine = engine
        self.baseline_rate = baseline_anomaly_rate
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self._recent_labels: list = []
        self.retrain_count = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, labels: np.ndarray) -> None:
        """Append latest prediction labels and prune the rolling window."""
        self._recent_labels.extend(labels.tolist())
        if len(self._recent_labels) > self.window_size:
            self._recent_labels = self._recent_labels[-self.window_size :]

    def check_and_retrain(self, X: np.ndarray) -> bool:
        """Return True if retraining was triggered."""
        if len(self._recent_labels) < self.window_size:
            return False  # not enough data yet

        current_rate = sum(1 for l in self._recent_labels if l == -1) / len(self._recent_labels)
        drift = current_rate - self.baseline_rate

        log.info(
            "Retraining check | baseline=%.3f | current=%.3f | drift=%.3f",
            self.baseline_rate,
            current_rate,
            drift,
        )

        if drift >= self.drift_threshold:
            log.warning(
                "🔁 Drift %.3f >= threshold %.3f — triggering retraining",
                drift,
                self.drift_threshold,
            )
            self._retrain(X)
            return True

        return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _retrain(self, X: np.ndarray) -> None:
        try:
            self.engine.train(X)
            self.retrain_count += 1
            self._recent_labels.clear()  # reset window post-retrain
            log.info("✅ Retraining complete (count=%d)", self.retrain_count)
        except Exception as exc:
            log.error("Retraining failed: %s", exc)


if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from orbit_q.engine.ml_engine import AnomalyEngine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    engine = AnomalyEngine()
    pipeline = RetrainingPipeline(engine, drift_threshold=0.05)

    # Simulate streaming data with injected drift
    X_baseline = np.random.normal(0, 1, (500, 5))
    engine.train(X_baseline)
    log.info("Baseline training done.")

    for epoch in range(5):
        X_stream = np.random.normal(0, 1.5 + epoch * 0.5, (200, 5))  # drift
        preds, _ = engine.predict(X_stream)
        pipeline.record(preds)
        retrained = pipeline.check_and_retrain(X_stream)
        log.info("Epoch %d | anomalies=%d | retrained=%s", epoch, (preds == -1).sum(), retrained)
        time.sleep(0.1)
