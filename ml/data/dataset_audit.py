# SolarSentinel AI - Dataset Audit Tool
import os, sys, json
from collections import defaultdict
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'Faulty_solar_panel')

def audit_dataset(dataset_dir=DEFAULT_DATASET_DIR):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError('Dataset dir not found: ' + dataset_dir)

    classes = [d for d in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, d))]
    total_images = 0
    corrupted_images = []
    class_counts = defaultdict(int)
    resolutions = defaultdict(int)
    channels = defaultdict(int)

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        for root, _, files in os.walk(cls_dir):
            for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        p = os.path.join(root, f)
                        total_images += 1
                        try:
                            with Image.open(p) as img:
                                w, h = img.size
                                resolutions[str(w) + 'x' + str(h)] += 1
                                channels[img.mode] += 1
                            class_counts[cls] += 1
                        except Exception as e:
                            corrupted_images.append({'path': p, 'error': str(e)})

    report = {
        'total_images': total_images,
        'num_classes': len(classes),
        'classes': classes,
        'class_counts': dict(class_counts),
        'corrupted_count': len(corrupted_images),
        'corrupted_files': corrupted_images,
        'top_resolutions': sorted(resolutions.items(), key=lambda x: x[1], reverse=True)[:5],
        'channel_modes': dict(channels)
    }

    out_path = os.path.join(PROJECT_ROOT, 'ml', 'metadata', 'dataset_audit.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fp:
        json.dump(report, fp, indent=2)

    return report

if __name__ == '__main__':
    rep = audit_dataset()
    print('Dataset Audit Completed:')
    print('Total Images:', rep['total_images'])
    print('Corrupted:', rep['corrupted_count'])
    print('Class Counts:', rep['class_counts'])