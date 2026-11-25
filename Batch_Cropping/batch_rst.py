# This script rasterizes polygons from MULTIPLE Label Studio JSON exports
# and MULTIPLE image folders onto single-channel uint8 masks using taxonomy labels
# Filters cancelled annotations, splits images into train/val/test sets,
# tiles large images into overlapping 512x512 patches, and saves class_map.json

import json
import os
import random
import cv2
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Add as many (prefix, json_path, img_dir) batches as you want to combine
SOURCES = [
    {
        "prefix": "batch1",
        "json_path": "20250506_annotations.json",
        "img_dir": "raw_images",
    },
    {
        "prefix": "batch2",
        "json_path": "20250828_annotations.json",
        "img_dir": "raw_images_batch2",
    },
    # {
    #     "prefix": "batch3",
    #     "json_path": "project3-export.json",
    #     "img_dir": "raw_images_batch3",
    # },
]

OUTPUT_DIR = "dataset_combined_512"     # Final combined output dataset directory

# Map your Taxonomy labels to integer IDs (0 is reserved for background)
CLASS_MAP = {
    "Ground cover": 1,
    "Vegetation": 2,
}
USE_FULL_TAXONOMY_PATH = False          # True -> matches "Category/Ground cover", False -> matches "Ground cover"

TILE_SIZE = 512                         # Divisible by 32 (works for U-Net, DeepLab, SegFormer)
STRIDE = 384                            # 128px overlap between tiles
SPLIT_RATIOS = (0.8, 0.1, 0.1)          # Train / Val / Test
RANDOM_SEED = 42


# ==========================================
# 2. HELPER: RASTERIZE POLYGONS + TAXONOMY
# ==========================================
def rasterize_task(task):
    # Skip cancelled annotations and grab the latest active annotation
    active_anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled", False)]
    if not active_anns:
        return None

    results = active_anns[-1].get("result", [])
    if not results:
        return None

    # Link region IDs to their Taxonomy label
    region_to_class = {}
    for item in results:
        if item.get("type") == "taxonomy":
            paths = item.get("value", {}).get("taxonomy", [])
            if paths:
                label_key = "/".join(paths[0]) if USE_FULL_TAXONOMY_PATH else paths[0][-1]
                region_to_class[item["id"]] = label_key

    # Pull original image dimensions from the first polygon item (safer than results[0])
    shape_items = [r for r in results if r.get("type") == "polygonlabels"]
    if not shape_items:
        return None

    orig_w = shape_items[0]["original_width"]
    orig_h = shape_items[0]["original_height"]
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    for item in shape_items:
        region_id = item.get("id")
        label_key = region_to_class.get(region_id)
        if label_key is None:
            poly_names = item.get("value", {}).get("polygonlabels", [])
            label_key = poly_names[0] if poly_names else None

        class_id = CLASS_MAP.get(label_key)
        if class_id is None:
            print(f"Skipping unmapped label: {label_key}")
            continue

        # Convert Label Studio 0-100 percentage points to absolute pixel coordinates
        points = item["value"]["points"]
        pixel_pts = np.array(
            [[int(round(x * orig_w / 100.0)), int(round(y * orig_h / 100.0))] for x, y in points],
            dtype=np.int32
        )
        cv2.fillPoly(mask, [pixel_pts], color=class_id)

    return mask


# ==========================================
# 3. HELPER: SLICE OR PAD TO 512x512
# ==========================================
def save_tiles(img, mask, unique_base_name, out_img_dir, out_mask_dir):
    h, w = mask.shape
    tile_idx = 0

    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):
            # Pull window back if it hits the right/bottom image border
            y1 = min(y, max(0, h - TILE_SIZE))
            x1 = min(x, max(0, w - TILE_SIZE))
            y2 = min(y1 + TILE_SIZE, h)
            x2 = min(x1 + TILE_SIZE, w)

            img_tile = img[y1:y2, x1:x2]
            mask_tile = mask[y1:y2, x1:x2]

            # Pad if original image was smaller than TILE_SIZE
            if img_tile.shape[0] < TILE_SIZE or img_tile.shape[1] < TILE_SIZE:
                pad_h = TILE_SIZE - img_tile.shape[0]
                pad_w = TILE_SIZE - img_tile.shape[1]
                img_tile = cv2.copyMakeBorder(img_tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                mask_tile = cv2.copyMakeBorder(mask_tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=255)

            # Skip tiles that only contain background (0) and ignore-padding (255)
            valid_pixels = mask_tile[mask_tile != 255]
            if valid_pixels.size == 0 or np.all(valid_pixels == 0):
                continue

            cv2.imwrite(os.path.join(out_img_dir, f"{unique_base_name}_tile_{tile_idx}.png"), img_tile)
            cv2.imwrite(os.path.join(out_mask_dir, f"{unique_base_name}_tile_{tile_idx}.png"), mask_tile)
            tile_idx += 1

    return tile_idx


# ==========================================
# 4. MAIN PIPELINE
# ==========================================
valid_samples = []

for src in SOURCES:
    prefix = src["prefix"]
    json_path = src["json_path"]
    img_dir = src["img_dir"]

    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # Match tasks to local image files
    for task in tasks:
        if not task.get("annotations"):
            continue
        raw_name = os.path.basename(task["data"].get("image", ""))
        if "-" in raw_name and len(raw_name.split("-")[0]) == 8:
            raw_name = raw_name.split("-", 1)[1]
        base_name = os.path.splitext(raw_name)[0]

        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            candidate = os.path.join(img_dir, base_name + ext)
            if os.path.exists(candidate):
                valid_samples.append((f"{prefix}_{base_name}", candidate, task))
                break

# Shuffle and split BEFORE tiling to prevent data leakage
random.seed(RANDOM_SEED)
random.shuffle(valid_samples)

n_total = len(valid_samples)
n_train = int(n_total * SPLIT_RATIOS[0])
n_val = int(n_total * SPLIT_RATIOS[1])

splits = {
    "train": valid_samples[:n_train],
    "val": valid_samples[n_train:n_train + n_val],
    "test": valid_samples[n_train + n_val:],
}

for split_name, samples in splits.items():
    out_img_dir = os.path.join(OUTPUT_DIR, split_name, "images")
    out_mask_dir = os.path.join(OUTPUT_DIR, split_name, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    total_tiles = 0
    for base_name, img_path, task in samples:
        mask = rasterize_task(task)
        if mask is None:
            continue
        img = cv2.imread(img_path)
        if img.shape[:2] != mask.shape[:2]:
            print(f"Dimension mismatch on {base_name}: img={img.shape[:2]}, mask={mask.shape[:2]}")
            continue
        total_tiles += save_tiles(img, mask, base_name, out_img_dir, out_mask_dir)

    print(f"{split_name.upper()}: {len(samples)} source images -> {total_tiles} tiles")

# Save reverse class map for reference during training/inference
id_to_class = {0: "background", **{v: k for k, v in CLASS_MAP.items()}}
with open(os.path.join(OUTPUT_DIR, "class_map.json"), "w", encoding="utf-8") as f:
    json.dump(id_to_class, f, indent=2)

print("Dataset prep complete!")