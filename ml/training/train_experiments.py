"""
SolarSentinel AI - Experiment Runner & Registry Aggregator
Orchestrates controlled VGG16 experiments, tracks hyperparameters,
and aggregates validation metrics into a unified experiment registry.
"""
import os
import sys
import json
import csv
import glob
from typing import Dict, List, Any
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "ml", "experiments", "phase9_vgg16")
REGISTRY_CSV = os.path.join(EXPERIMENTS_DIR, "experiment_results.csv")
REGISTRY_JSON = os.path.join(EXPERIMENTS_DIR, "experiment_results.json")


def aggregate_experiment_results(experiments_dir: str = EXPERIMENTS_DIR) -> List[Dict[str, Any]]:
    """Scans all experiment subdirectories and compiles validation benchmarks."""
    results = []
    metric_files = glob.glob(os.path.join(experiments_dir, "*", "val_metrics.json"))

    for fpath in sorted(metric_files):
        exp_name = os.path.basename(os.path.dirname(fpath))
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        acc = data.get("accuracy") or data.get("val_accuracy", 0.0)
        macro_f1 = data.get("macro_f1", 0.0)
        weighted_f1 = data.get("weighted_f1", 0.0)
        phys_rec = data.get("physical_damage_recall", 0.0)
        phys_f1 = data.get("physical_damage_f1", 0.0)

        entry = {
            "experiment_id": exp_name,
            "validation_accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "physical_damage_recall": round(float(phys_rec), 4),
            "physical_damage_f1": round(float(phys_f1), 4)
        }
        results.append(entry)

    # Sort descending by macro_f1 and accuracy
    results.sort(key=lambda x: (x["validation_accuracy"], x["macro_f1"]), reverse=True)

    # Save to CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(REGISTRY_CSV, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Save to JSON
    with open(REGISTRY_JSON, "w", encoding="utf-8") as jsonfile:
        json.dump(results, jsonfile, indent=2)

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SOLARSENTINEL AI - AGGREGATING EXPERIMENT REGISTRY")
    print("=" * 60)
    res = aggregate_experiment_results()
    for r in res:
        print(f"[{r['experiment_id']}] Val Acc: {r['validation_accuracy']*100:.2f}% | Macro F1: {r['macro_f1']:.4f} | Phys Rec: {r['physical_damage_recall']*100:.1f}%")
    print(f"\nSaved summary to: {REGISTRY_CSV}")
