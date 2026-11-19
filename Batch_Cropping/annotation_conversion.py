# This script should convert brush labels into polygon labels
# Input: JSON file
# Output: JSON file

import copy
import json
import cv2
import numpy as np
from label_studio_sdk.converter.brush import decode_rle

INPUT_FILE = "project-export.json"
OUTPUT_FILE = "20250820_project-polygons.json"

# Adjust between 0.002 (detailed) and 0.006 (lighter/fewer vertices)
SIMPLIFY_EPSILON = 0.003

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    tasks = json.load(f)

converted_regions = 0

for task in tasks:
    for ann in task.get("annotations", []):
        results = ann.get("result", [])

        # Map region IDs to any linked per-region taxonomy results
        taxonomy_by_id = {}
        for res in results:
            if res.get("type") == "taxonomy":
                taxonomy_by_id.setdefault(res["id"], []).append(res)

        new_results = []
        handled_taxonomy_ids = set()

        for res in results:
            # Skip taxonomy entries here; they are attached alongside their parent region
            if res.get("type") == "taxonomy":
                continue

            # Keep non-brush regions (e.g., already-drawn polygons) untouched
            if res.get("type") != "brushlabels":
                new_results.append(res)
                if res.get("id") in taxonomy_by_id:
                    new_results.extend(taxonomy_by_id[res["id"]])
                    handled_taxonomy_ids.add(res["id"])
                continue

            val = res["value"]
            w, h = res["original_width"], res["original_height"]
            orig_id = res["id"]

            # Decode RLE into a 2D binary alpha mask
            mask = decode_rle(val["rle"]).reshape((h, w, 4))[:, :, 3]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, SIMPLIFY_EPSILON * peri, True)
                if len(approx) >= 3:
                    valid_contours.append(approx)

            # Create a polygon (and duplicate linked taxonomy) for each contour
            for idx, approx in enumerate(valid_contours):
                new_id = orig_id if idx == 0 else f"{orig_id}_{idx}"
                points = [
                    [round(float(pt[0][0]) / w * 100, 2), round(float(pt[0][1]) / h * 100, 2)]
                    for pt in approx
                ]

                new_results.append({
                    "id": new_id,
                    "from_name": "polygon_labels",  # Matches <PolygonLabels name="polygon_labels">
                    "to_name": res["to_name"],
                    "type": "polygonlabels",
                    "original_width": w,
                    "original_height": h,
                    "value": {
                        "points": points,
                        "polygonlabels": val["brushlabels"]
                    }
                })

                # Attach matching per-region taxonomy classification to this polygon
                if orig_id in taxonomy_by_id:
                    for tax_res in taxonomy_by_id[orig_id]:
                        tax_copy = copy.deepcopy(tax_res)
                        tax_copy["id"] = new_id
                        new_results.append(tax_copy)

                converted_regions += 1

            handled_taxonomy_ids.add(orig_id)

        # Preserve any standalone taxonomy entries not tied to a deleted brush
        for tid, tax_list in taxonomy_by_id.items():
            if tid not in handled_taxonomy_ids:
                new_results.extend(tax_list)

        ann["result"] = new_results

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)

print(f"Converted {converted_regions} polygon regions into {OUTPUT_FILE}.")