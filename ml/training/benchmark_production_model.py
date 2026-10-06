"""
SolarSentinel AI - Production Model Reproducible Benchmark Runner
Measures forward inference, Grad-CAM generation, and total pipeline latency on CPU.
Hardware-dependent: Latency measurements reflect host CPU capabilities.
"""

import argparse
import json
import os
import platform
import time
import sys
from typing import Dict, Any, List
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress excessive TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from ml.inference.preprocessing import preprocess_for_vgg16
from ml.explainability.gradcam import GradCAMExplainer


def get_hardware_info(iterations: int, warmup_iterations: int) -> Dict[str, Any]:
    """Extracts platform and environment metadata."""
    return {
        "cpu": platform.processor() or platform.machine() or "Unknown CPU",
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "tensorflow_version": keras.__version__,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "hardware_notice": "Latency results are hardware-dependent and reflect CPU execution."
    }


def compute_statistics(latencies: List[float]) -> Dict[str, float]:
    """Calculates min, max, mean, p50 (median), and p95 latencies in milliseconds."""
    arr = np.array(latencies, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2)
    }


def run_benchmark(
    model_path: str = "ml/models/solar_sentinel_vgg16.keras",
    class_names_path: str = "ml/metadata/class_names.json",
    output_path: str = "ml/metadata/performance_benchmark.json",
    iterations: int = 30,
    warmup: int = 5
) -> Dict[str, Any]:
    """Executes reproducible latency benchmarking for inference and Grad-CAM."""
    print(f"[*] Loading model from {model_path}...")
    load_start = time.perf_counter()
    model = keras.models.load_model(model_path)
    model_load_ms = round((time.perf_counter() - load_start) * 1000.0, 2)
    print(f"[+] Model loaded in {model_load_ms} ms.")

    classes = []
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            data = json.load(f)
            classes = data if isinstance(data, list) else data.get("classes", [])

    explainer = GradCAMExplainer(model=model, class_names=classes)

    # Synthetic realistic solar panel test image (244x244 RGB)
    np.random.seed(42)
    test_img = Image.fromarray(np.random.randint(40, 200, size=(244, 244, 3), dtype=np.uint8))
    preprocessed_arr = preprocess_for_vgg16(test_img, target_size=(244, 244))

    print(f"[*] Warming up for {warmup} iterations...")
    for _ in range(warmup):
        _ = model(preprocessed_arr, training=False)
        _ = explainer.explain(test_img, target_class_index=0)

    print(f"[*] Running {iterations} timed benchmark iterations...")
    inference_times = []
    gradcam_times = []
    pipeline_times = []

    for i in range(iterations):
        t0 = time.perf_counter()
        preds = model(preprocessed_arr, training=False)
        t1 = time.perf_counter()
        
        top_idx = int(np.argmax(preds[0]))
        _ = explainer.explain(test_img, target_class_index=top_idx)
        t2 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000.0
        gc_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        inference_times.append(inf_ms)
        gradcam_times.append(gc_ms)
        pipeline_times.append(total_ms)

        if (i + 1) % 10 == 0 or (i + 1) == iterations:
            print(f"    Completed {i + 1}/{iterations} iterations (recent pipeline: {total_ms:.1f}ms)")

    benchmark_data = {
        "hardware": get_hardware_info(iterations, warmup),
        "model_loading_time_ms": model_load_ms,
        "inference_latency_ms": compute_statistics(inference_times),
        "gradcam_latency_ms": compute_statistics(gradcam_times),
        "total_pipeline_latency_ms": compute_statistics(pipeline_times)
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"[+] Benchmark successfully saved to {output_path}")
    print(f"    Inference Mean: {benchmark_data['inference_latency_ms']['mean']} ms")
    print(f"    Grad-CAM Mean:  {benchmark_data['gradcam_latency_ms']['mean']} ms")
    print(f"    Pipeline Mean:  {benchmark_data['total_pipeline_latency_ms']['mean']} ms")

    return benchmark_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SolarSentinel AI Benchmark Runner")
    parser.add_argument("--iterations", type=int, default=30, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, warmup=args.warmup)
