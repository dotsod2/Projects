# This script cleans your json file
import json

JSON_PATH = "project-export.json"
CLEAN_JSON_PATH = "project-export-cleaned.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    tasks = json.load(f)

removed_orphans = 0
removed_cancelled = 0

for task in tasks:
    # 1. Keep only active (non-cancelled) annotations
    original_anns = task.get("annotations", [])
    active_anns = [a for a in original_anns if not a.get("was_cancelled", False)]
    removed_cancelled += len(original_anns) - len(active_anns)
    task["annotations"] = active_anns

    # 2. Within each active annotation, remove taxonomy items that have no matching polygon/brush shape
    for ann in task["annotations"]:
        results = ann.get("result", [])
        valid_shape_ids = {
            r.get("id") for r in results
            if r.get("type") in ("polygonlabels", "brushlabels", "rectanglelabels")
        }

        cleaned_results = []
        for r in results:
            if r.get("type") == "taxonomy" and r.get("id") not in valid_shape_ids:
                removed_orphans += 1
                continue
            cleaned_results.append(r)

        ann["result"] = cleaned_results

with open(CLEAN_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)

print(f"Cleaned JSON saved to '{CLEAN_JSON_PATH}'.")
print(f"Removed {removed_cancelled} cancelled annotations and {removed_orphans} orphaned taxonomy labels.")