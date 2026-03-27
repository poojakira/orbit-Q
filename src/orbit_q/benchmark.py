import time
import numpy as np
from orbit_q.engine.ml_engine import AnomalyEngine


def run_benchmark():
    print("🚀 Orbit-Q Performance Benchmark")
    print("--------------------------------")

    engine = AnomalyEngine()

    # 1. Throughput & Latency Test (Simulating 200 Hz for 10 seconds)
    hz = 200
    seconds = 10
    total_samples = hz * seconds

    print(f"Generating {total_samples} samples for Latency/Throughput test...")
    dummy_data = np.random.normal(loc=0, scale=1, size=(total_samples, 5))

    print("Training Model...")
    t0 = time.time()
    engine.train(dummy_data)
    train_time = time.time() - t0
    print(f"Training Time: {train_time:.4f} seconds")

    print(f"Running Inference on {total_samples} samples...")
    t0 = time.time()
    engine.predict(dummy_data)
    infer_time = time.time() - t0

    latency_per_sample = (infer_time / total_samples) * 1000  # in ms
    throughput = total_samples / infer_time

    print(f"Total Inference Time: {infer_time:.4f} seconds")
    print(f"Latency per Detection: {latency_per_sample:.4f} ms")
    print(f"Throughput: {throughput:.2f} samples/second")
    print("--------------------------------")
    print("✅ Benchmark Complete")


if __name__ == "__main__":
    run_benchmark()
