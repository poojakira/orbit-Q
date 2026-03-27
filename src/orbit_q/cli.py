"""
orbit-q CLI — entry point for all platform commands.

Commands
--------
orbit-q simulator      Run the multi-fault-injection CubeSat telemetry simulator
orbit-q orchestrator   Run the ML orchestrator (fetches telemetry, trains ensemble)
orbit-q dashboard      Launch the Streamlit operator dashboard
orbit-q benchmark      Run latency / throughput benchmark at configurable Hz
"""

import argparse
import sys
import subprocess
import os
import importlib.util


def _pkg_dir() -> str:
    spec = importlib.util.find_spec("orbit_q")
    if spec and spec.origin:
        return os.path.dirname(str(spec.origin))
    return os.path.join(os.path.dirname(__file__))


def cmd_simulator(args):
    script = os.path.join(_pkg_dir(), "simulator", "mock_telemetry.py")
    subprocess.run([sys.executable, script], check=False)


def cmd_orchestrator(args):
    script = os.path.join(_pkg_dir(), "orchestrator", "ml_orchestrator.py")
    subprocess.run([sys.executable, script], check=False)


def cmd_dashboard(args):
    script = os.path.join(_pkg_dir(), "dashboard", "dashboard.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", script, "--server.port", str(args.port)],
        check=False,
    )


def cmd_benchmark(args):
    import time
    import numpy as np
    from orbit_q.engine.ml_engine import AnomalyEngine

    print(f"\n🛰️  Orbit-Q Benchmark  |  {args.hz} Hz  |  {args.seconds}s window")
    print("=" * 60)

    n = args.hz * args.seconds
    X = np.random.normal(0, 1, (n, 5))
    engine = AnomalyEngine()

    t0 = time.perf_counter()
    engine.train(X)
    train_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    preds, scores = engine.predict(X)
    infer_ms = (time.perf_counter() - t0) * 1000

    latency_us = (infer_ms / n) * 1000
    throughput = n / (infer_ms / 1000)
    anomaly_pct = int((preds == -1).sum()) / n * 100

    print(f"  Samples           : {n:,}")
    print(f"  Train time        : {train_ms:.1f} ms")
    print(f"  Infer time        : {infer_ms:.1f} ms")
    print(f"  Latency/detection : {latency_us:.2f} µs")
    print(f"  Throughput        : {throughput:,.0f} samples/s")
    print(f"  Anomaly rate      : {anomaly_pct:.1f}%")
    print("=" * 60)


def cmd_stress_test(args):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from orbit_q.simulator.multi_cubesat_stress import MultiCubeSatStressTest

    print(f"\n🛰️  Multi-CubeSat Stress Test | {args.satellites} satellites × {args.hz} Hz × {args.duration}s")
    print("=" * 60)
    test = MultiCubeSatStressTest(
        n_satellites=args.satellites,
        hz_per_satellite=args.hz,
        duration_s=args.duration,
    )
    report = test.run()
    print("\n📊 Results:")
    for k, v in report.items():
        print(f"  {k:<30}{v}")
    print("=" * 60)


def cmd_retrain(args):
    import logging
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from orbit_q.engine.ml_engine import AnomalyEngine
    from orbit_q.mlflow_tracking.retraining_pipeline import RetrainingPipeline

    print("🔁 Manual retraining triggered...")
    engine = AnomalyEngine()
    pipeline = RetrainingPipeline(engine)

    # Retrain on synthetic data (replace with real telemetry fetch in production)
    X = np.random.normal(0, 1, (500, 5))
    engine.train(X)
    print("✅ Retraining complete. Model logged to MLflow.")


def main():
    parser = argparse.ArgumentParser(
        prog="orbit-q",
        description="🛰️  Orbit-Q ML Telemetry Platform CLI",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("simulator", help="Run CubeSat telemetry simulator (with fault injection)")
    subs.add_parser("orchestrator", help="Run ML orchestrator daemon")

    dash = subs.add_parser("dashboard", help="Launch Streamlit operator dashboard")
    dash.add_argument("--port", type=int, default=8501)

    bench = subs.add_parser("benchmark", help="Run latency/throughput benchmark")
    bench.add_argument("--hz", type=int, default=100, help="Simulated stream rate (Hz)")
    bench.add_argument("--seconds", type=int, default=10, help="Window size (seconds)")

    stress = subs.add_parser("stress-test", help="Multi-CubeSat distributed stress test")
    stress.add_argument("--satellites", type=int, default=10, help="Number of CubeSat simulators")
    stress.add_argument("--hz", type=int, default=50, help="Hz per satellite")
    stress.add_argument("--duration", type=float, default=10.0, help="Test duration (seconds)")

    subs.add_parser("retrain", help="Manually trigger anomaly model retraining")

    args = parser.parse_args()

    dispatch = {
        "simulator": cmd_simulator,
        "orchestrator": cmd_orchestrator,
        "dashboard": cmd_dashboard,
        "benchmark": cmd_benchmark,
        "stress-test": cmd_stress_test,
        "retrain": cmd_retrain,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
