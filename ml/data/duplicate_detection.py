"""
SolarSentinel AI - Duplicate Detection & Hash Collision Audit
Computes SHA-256 checksums for all dataset images to detect identical files,
identifies intra-class and cross-class duplicates, and verifies that no
identical image crosses the train/val/test partition boundary.
"""
import os
import sys
import json
import hashlib
from collections import defaultdict
from typing import Dict, List, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "Faulty_solar_panel")
SPLIT_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata", "split_manifest_70_15_15.json")
OUTPUT_AUDIT_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata", "duplicate_audit.json")


def compute_file_hash(filepath: str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_duplicates(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    manifest_path: str = SPLIT_MANIFEST_PATH
) -> Dict[str, Any]:
    """Scan dataset, find duplicates by SHA-256, and check split leakages."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    hash_to_files: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    total_scanned = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for root, _, files in os.walk(dataset_dir):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/")
                cls_name = os.path.basename(root)
                f_hash = compute_file_hash(abs_path)
                hash_to_files[f_hash].append({
                    "path": rel_path,
                    "filename": fname,
                    "class": cls_name
                })
                total_scanned += 1

    duplicate_groups = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    num_unique_hashes = len(hash_to_files)
    num_duplicate_instances = sum(len(f) - 1 for f in duplicate_groups.values())

    intra_class_dups = []
    cross_class_dups = []

    for h, files in duplicate_groups.items():
        classes = {f["class"] for f in files}
        entry = {
            "sha256": h,
            "count": len(files),
            "classes": sorted(list(classes)),
            "files": [f["path"] for f in files]
        }
        if len(classes) == 1:
            intra_class_dups.append(entry)
        else:
            cross_class_dups.append(entry)

    split_leakage_detected = False
    leakage_details = []

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fp:
            manifest = json.load(fp)

        file_to_split = {}
        for split_name in ["train", "val", "test"]:
            items = manifest.get(split_name, [])
            for item in items:
                p = item if isinstance(item, str) else item.get("path", "")
                norm_p = p.replace("\\", "/")
                file_to_split[norm_p] = split_name
                file_to_split[os.path.basename(norm_p)] = split_name

        for h, files in duplicate_groups.items():
            splits_seen = set()
            for f in files:
                sp = file_to_split.get(f["path"]) or file_to_split.get(f["filename"])
                if sp:
                    splits_seen.add(sp)
            if len(splits_seen) > 1:
                split_leakage_detected = True
                leakage_details.append({
                    "sha256": h,
                    "splits_involved": sorted(list(splits_seen)),
                    "files": [f["path"] for f in files]
                })

    report = {
        "dataset_directory": dataset_dir.replace("\\", "/"),
        "total_images_scanned": total_scanned,
        "unique_images_by_hash": num_unique_hashes,
        "duplicate_hash_groups": len(duplicate_groups),
        "total_duplicate_redundant_files": num_duplicate_instances,
        "intra_class_duplicate_groups": len(intra_class_dups),
        "cross_class_duplicate_groups": len(cross_class_dups),
        "split_manifest_checked": os.path.exists(manifest_path),
        "split_leakage_detected": split_leakage_detected,
        "split_leakage_count": len(leakage_details),
        "split_leakages": leakage_details,
        "cross_class_duplicates": cross_class_dups,
        "sample_intra_class_duplicates": intra_class_dups[:10]
    }

    os.makedirs(os.path.dirname(OUTPUT_AUDIT_PATH), exist_ok=True)
    with open(OUTPUT_AUDIT_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    return report


if __name__ == "__main__":
    rep = detect_duplicates()
    print("=" * 60)
    print("SOLARSENTINEL AI - DUPLICATE DETECTION AUDIT")
    print("=" * 60)
    print(f"Total Images Scanned: {rep['total_images_scanned']}")
    print(f"Unique Hashes:        {rep['unique_images_by_hash']}")
    print(f"Duplicate Groups:     {rep['duplicate_hash_groups']}")
    print(f"Redundant Files:      {rep['total_duplicate_redundant_files']}")
    print(f"Intra-Class Dups:     {rep['intra_class_duplicate_groups']}")
    print(f"Cross-Class Dups:     {rep['cross_class_duplicate_groups']}")
    print(f"Split Leakage:        {rep['split_leakage_detected']} ({rep['split_leakage_count']} cases)")
    print(f"Report saved to:      {OUTPUT_AUDIT_PATH}")
    print("=" * 60)
