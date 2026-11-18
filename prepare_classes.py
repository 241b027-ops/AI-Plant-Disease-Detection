import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Generate class_names.json in the sorted folder order used by the notebook."
)
parser.add_argument("--dataset", required=True, help="Path to PlantVillage-Dataset/raw/color")
parser.add_argument("--output", default="class_names.json", help="Output JSON path")
args = parser.parse_args()

dataset = Path(args.dataset)
if not dataset.is_dir():
    raise SystemExit(f"Dataset folder not found: {dataset}")

classes = sorted(p.name for p in dataset.iterdir() if p.is_dir())
if not classes:
    raise SystemExit("No class folders found.")

Path(args.output).write_text(json.dumps(classes, indent=2), encoding="utf-8")
print(f"Saved {len(classes)} classes to {args.output}")
for i, name in enumerate(classes):
    print(i, name)
