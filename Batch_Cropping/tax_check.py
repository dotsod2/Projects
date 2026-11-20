# This script outputs the exact taxonomy labels used in annotation.
import json

with open("project-export-cleaned.json", "r", encoding="utf-8") as f:
    tasks = json.load(f)

labels = {
    "/".join(p)
    for t in tasks
    for a in t.get("annotations", [])
    for r in a.get("result", []) if r.get("type") == "taxonomy"
    for p in r.get("value", {}).get("taxonomy", [])
}
print("Found taxonomy labels:", sorted(labels))