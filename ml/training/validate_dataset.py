"""
SolarSentinel AI - Dataset Validation Utility
Validates dataset presence, directory hierarchy, class names, image file integrity,
corrupt file detection, and class balance analysis.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def check_image_integrity(file_path: str) -> Tuple[bool, str]:
    """Verify that an image file can be completely decoded without corruption."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        # Verify does not decode all pixels; re-open to test decompression
        with Image.open(file_path) as img:
            img.load()
            if img.mode not in ("RGB", "RGBA", "L"):
                return False, f"Unsupported color mode: {img.mode}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def validate_dataset(data_dir: str, expected_classes: List[str] = None) -> Dict:
    """
    Validates a dataset directory against production criteria.
    Returns structured analysis dictionary.
    """
    report = {
        "dataset_exists": False,
        "data_dir": os.path.abspath(data_dir),
        "class_directories_found": [],
        "expected_classes": expected_classes or [],
        "missing_expected_classes": [],
        "unexpected_classes": [],
        "total_images": 0,
        "class_counts": {},
        "corrupt_images": [],
        "imbalance_metrics": {},
        "valid": False,
        "summary": ""
    }

    if not os.path.exists(data_dir):
        report["summary"] = f"Dataset directory does not exist: {data_dir}"
        return report

    report["dataset_exists"] = True

    # Check child folders
    subdirs = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    subdirs.sort()
    report["class_directories_found"] = subdirs

    if expected_classes:
        exp_set = set(expected_classes)
        found_set = set(subdirs)
        report["missing_expected_classes"] = sorted(list(exp_set - found_set))
        report["unexpected_classes"] = sorted(list(found_set - exp_set))

    if not subdirs:
        report["summary"] = f"No class subdirectories found in {data_dir}"
        return report

    total_valid_imgs = 0
    class_counts = {}
    corrupt_files = []

    for c in subdirs:
        c_path = os.path.join(data_dir, c)
        files = [
            f for f in os.listdir(c_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]
        
        valid_for_class = 0
        for f in files:
            full_path = os.path.join(c_path, f)
            is_valid, reason = check_image_integrity(full_path)
            if is_valid:
                valid_for_class += 1
            else:
                corrupt_files.append({"file": full_path, "error": reason})

        class_counts[c] = valid_for_class
        total_valid_imgs += valid_for_class

    report["total_images"] = total_valid_imgs
    report["class_counts"] = class_counts
    report["corrupt_images"] = corrupt_files

    # Calculate class imbalance metrics
    if total_valid_imgs > 0 and len(class_counts) > 0:
        counts = list(class_counts.values())
        min_c = min(counts)
        max_c = max(counts)
        avg_c = total_valid_imgs / len(counts)
        imbalance_ratio = (max_c / min_c) if min_c > 0 else float("inf")
        report["imbalance_metrics"] = {
            "min_count": min_c,
            "max_count": max_c,
            "mean_count": round(avg_c, 1),
            "imbalance_ratio": round(imbalance_ratio, 2),
            "severe_imbalance": imbalance_ratio > 3.0
        }

    # Validation criteria
    has_classes = len(subdirs) >= 2
    no_corrupt = len(corrupt_files) == 0
    has_data = total_valid_imgs > 0
    no_missing_expected = len(report.get("missing_expected_classes", [])) == 0 if expected_classes else True

    report["valid"] = has_classes and no_corrupt and has_data and no_missing_expected

    if report["valid"]:
        report["summary"] = (
            f"Dataset is valid: {len(subdirs)} classes, {total_valid_imgs} total images, "
            f"0 corrupt images. Imbalance ratio: {report['imbalance_metrics']['imbalance_ratio']}."
        )
    else:
        issues = []
        if not has_data:
            issues.append("Zero images found")
        if corrupt_files:
            issues.append(f"{len(corrupt_files)} corrupt images detected")
        if not no_missing_expected:
            issues.append(f"Missing classes: {report['missing_expected_classes']}")
        report["summary"] = f"Dataset validation failed: {'; '.join(issues)}"

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate SolarSentinel AI Dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("data", "raw"),
        help="Path to raw dataset directory"
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default="",
        help="Optional path to output validation JSON"
    )
    args = parser.parse_args()

    # Load canonical classes if metadata exists
    expected_classes = None
    meta_path = os.path.join("ml", "metadata", "class_names.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                expected_classes = meta.get("classes")
        except Exception:
            pass

    print(f"\n==========================================")
    print(f"SolarSentinel AI - Dataset Validation Utility")
    print(f"Target Directory: {os.path.abspath(args.data_dir)}")
    print(f"==========================================\n")

    report = validate_dataset(args.data_dir, expected_classes=expected_classes)

    print(f"Dataset Exists: {report['dataset_exists']}")
    if not report["dataset_exists"]:
        print(f"\n[ALERT] Dataset not found at: {args.data_dir}")
        print(f"Instructions: Please place class folders inside {args.data_dir}")
        print(f"See data/README.md for directory structure details.\n")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(report, f, indent=2)
        sys.exit(0)

    print(f"Classes Found ({len(report['class_directories_found'])}): {report['class_directories_found']}")
    if report["missing_expected_classes"]:
        print(f"Missing Expected Classes: {report['missing_expected_classes']}")
    if report["unexpected_classes"]:
        print(f"Unexpected Additional Classes: {report['unexpected_classes']}")

    print(f"\nPer-Class Distribution:")
    for c, cnt in report["class_counts"].items():
        print(f"  - {c}: {cnt} images")

    print(f"\nTotal Valid Images: {report['total_images']}")
    print(f"Corrupt Images Detected: {len(report['corrupt_images'])}")

    if report["imbalance_metrics"]:
        print(f"\nImbalance Analysis:")
        print(f"  - Min Class Size: {report['imbalance_metrics']['min_count']}")
        print(f"  - Max Class Size: {report['imbalance_metrics']['max_count']}")
        print(f"  - Imbalance Ratio: {report['imbalance_metrics']['imbalance_ratio']}")

    print(f"\nStatus: {'VALID' if report['valid'] else 'INVALID'}")
    print(f"Summary: {report['summary']}\n")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed JSON report saved to: {args.json_out}")


if __name__ == "__main__":
    main()
