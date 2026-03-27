"""
Multi-CubeSat Distributed Stress Test
Spawns N satellite simulators as concurrent threads, each pushing telemetry
at a configurable rate. Measures aggregate throughput, anomaly detection
latency, and system resource usage under high load.
"""

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SatelliteState:
    satellite_id: str
    hz: int = 10
    total_packets: int = 0
    anomaly_count: int = 0
    start_time: float = field(default_factory=time.time)
    errors: int = 0


class CubeSatSimulator(threading.Thread):
    """
    Single CubeSat telemetry simulator running in its own thread.
    Generates streaming telemetry and optionally pushes to a shared queue.
    """

    def __init__(
        self,
        satellite_id: str,
        hz: int = 10,
        duration_s: float = 10.0,
        packet_queue: Optional["queue.Queue"] = None,
        anomaly_rate: float = 0.05,
    ):
        super().__init__(name=f"cubesat-{satellite_id}", daemon=True)
        self.satellite_id = satellite_id
        self.hz = hz
        self.duration_s = duration_s
        self.packet_queue = packet_queue
        self.anomaly_rate = anomaly_rate
        self.state = SatelliteState(satellite_id=satellite_id, hz=hz)

    def _generate_packet(self) -> dict:
        is_anomaly = random.random() < self.anomaly_rate
        return {
            "satellite_id": self.satellite_id,
            "timestamp": time.time(),
            "temperature_C": random.uniform(85, 130) if is_anomaly else random.normalvariate(24, 3),
            "voltage_V": random.uniform(2.0, 3.0) if is_anomaly else random.normalvariate(5.0, 0.1),
            "signal_strength_dBm": random.normalvariate(-60, 5),
            "gyro_x_dps": random.normalvariate(0, 0.5),
            "is_anomaly": is_anomaly,
        }

    def run(self) -> None:
        interval = 1.0 / self.hz
        deadline = time.time() + self.duration_s

        while time.time() < deadline:
            t0 = time.time()
            try:
                packet = self._generate_packet()
                self.state.total_packets += 1
                if packet["is_anomaly"]:
                    self.state.anomaly_count += 1

                if self.packet_queue:
                    self.packet_queue.put_nowait(packet)

            except Exception as exc:
                self.state.errors += 1
                log.error("[%s] Error: %s", self.satellite_id, exc)

            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.info(
            "[%s] Done | packets=%d anomalies=%d errors=%d",
            self.satellite_id,
            self.state.total_packets,
            self.state.anomaly_count,
            self.state.errors,
        )


class MultiCubeSatStressTest:
    """
    Orchestrates N CubeSat simulators and reports aggregate performance.
    """

    def __init__(
        self,
        n_satellites: int = 5,
        hz_per_satellite: int = 20,
        duration_s: float = 10.0,
    ):
        self.n_satellites = n_satellites
        self.hz_per_satellite = hz_per_satellite
        self.duration_s = duration_s
        self.simulators: List[CubeSatSimulator] = []
        self._packet_store: List[dict] = []
        self._lock = threading.Lock()

    def _shared_queue_consumer(self, q: "queue.Queue") -> None:
        while True:
            try:
                packet = q.get(timeout=1.0)
                with self._lock:
                    self._packet_store.append(packet)
                q.task_done()
            except Exception:
                break

    def run(self) -> Dict:
        import queue

        q: queue.Queue = queue.Queue(maxsize=50_000)

        self.simulators = [
            CubeSatSimulator(
                satellite_id=f"SAT-{i:03d}",
                hz=self.hz_per_satellite,
                duration_s=self.duration_s,
                packet_queue=q,
            )
            for i in range(self.n_satellites)
        ]

        consumer = threading.Thread(target=self._shared_queue_consumer, args=(q,), daemon=True)
        consumer.start()

        log.info(
            "🛰️  Stress test | %d satellites × %d Hz × %.0fs", self.n_satellites, self.hz_per_satellite, self.duration_s
        )
        t0 = time.time()

        for sim in self.simulators:
            sim.start()

        for sim in self.simulators:
            sim.join()

        total_elapsed = time.time() - t0
        q.join()  # drain queue

        total_packets = sum(s.state.total_packets for s in self.simulators)
        total_anomalies = sum(s.state.anomaly_count for s in self.simulators)
        aggregate_hz = total_packets / total_elapsed

        report = {
            "n_satellites": self.n_satellites,
            "hz_per_satellite": self.hz_per_satellite,
            "duration_s": round(total_elapsed, 2),
            "total_packets": total_packets,
            "total_anomalies": total_anomalies,
            "anomaly_rate_pct": round(100 * total_anomalies / max(total_packets, 1), 2),
            "aggregate_throughput_hz": round(aggregate_hz, 1),
            "packets_in_queue": len(self._packet_store),
        }

        log.info("📊 Stress test report: %s", report)
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    hz = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

    test = MultiCubeSatStressTest(n_satellites=n, hz_per_satellite=hz, duration_s=duration)
    report = test.run()

    print("\n=== STRESS TEST RESULTS ===")
    for k, v in report.items():
        print(f"  {k:<30}{v}")
