import json
import os
from collections import defaultdict

JSON_PATH = "project-export.json"
TARGET_LABEL = "Ground cover/Bare soil"  # Set to None to print all labels and their matching images

with open(JSON_PATH, "r", encoding="utf-8") as f:
    tasks = json.load(f)

label_to_images = defaultdict(set)

for task in tasks:
    task_id = task.get("id")
    inner_id = task.get("inner_id")
    raw_name = os.path.basename(task["data"].get("image", ""))
    # Strip Label Studio's 8-character upload hash if present
    if "-" in raw_name and len(raw_name.split("-")[0]) == 8:
        raw_name = raw_name.split("-", 1)[1]

    entry = (task_id, inner_id, raw_name)

    for ann in task.get("annotations", []):
        for item in ann.get("result", []):
            # Check Taxonomy labels
            if item.get("type") == "taxonomy":
                for path in item.get("value", {}).get("taxonomy", []):
                    label_to_images[path[-1]].add(entry)          # Leaf name (e.g., "Car")
                    label_to_images["/".join(path)].add(entry)    # Full path (e.g., "Vehicles/Car")

            # Also check PolygonLabels directly in case any region didn't use Taxonomy
            elif item.get("type") == "polygonlabels":
                for lbl in item.get("value", {}).get("polygonlabels", []):
                    label_to_images[lbl].add(entry)

if TARGET_LABEL:
    matches = sorted(label_to_images.get(TARGET_LABEL, []), key=lambda x: (x[0] is None, x[0]))
    print(f"Images containing '{TARGET_LABEL}' ({len(matches)}):")
    for tid, iid, img in matches:
        print(f"  - ID: {tid} | Inner ID: {iid} | Image: {img}")
else:
    for lbl, entries in sorted(label_to_images.items()):
        print(f"\n{lbl} ({len(entries)} images):")
        for tid, iid, img in sorted(entries, key=lambda x: (x[0] is None, x[0])):
            print(f"  - ID: {tid} | Inner ID: {iid} | Image: {img}")