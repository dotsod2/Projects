# This script rasterizes polygons onto single-channel
# uint8 masks using taxonomy lables
# Splits images into training validation and test sets
# Tiles large images into overlapping 512x512 patches
# Saves the class map for training

import json
import os
import random
import cv2
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
JSON_PATH = "project-export-cleaned.json"       # Your Label Studio JSON export
RAW_IMG_DIR = "raw_images"              # Folder containing your original images
OUTPUT_DIR = "dataset_512"              # Final output dataset directory

# Map your Taxonomy labels to integer IDs (0 is reserved for background)
CLASS_MAP = {
    "Ground cover": 1,
    "Vegetation": 2
}
USE_FULL_TAXONOMY_PATH = False          # True -> matches "Vehicles/Car", False -> matches "Car"

TILE_SIZE = 512                         # Divisible by 32 (works for U-Net, DeepLab, SegFormer)
STRIDE = 384                            # 128px overlap between tiles
SPLIT_RATIOS = (0.8, 0.1, 0.1)          # Train / Val / Test
RANDOM_SEED = 42

# ==========================================
# 2. HELPER: RASTERIZE POLYGONS + TAXONOMY
# ==========================================
def rasterize_task(task):
    results = task.get("annotations", [{}])[0].get("result", [])
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

    orig_w = results[0]["original_width"]
    orig_h = results[0]["original_height"]
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    for item in results:
        if item.get("type") != "polygonlabels":
            continue

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
def save_tiles(img, mask, base_name, out_img_dir, out_mask_dir):
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

            cv2.imwrite(os.path.join(out_img_dir, f"{base_name}_tile_{tile_idx}.png"), img_tile)
            cv2.imwrite(os.path.join(out_mask_dir, f"{base_name}_tile_{tile_idx}.png"), mask_tile)
            tile_idx += 1

    return tile_idx

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    tasks = json.load(f)

# Match tasks to local image files
valid_samples = []
for task in tasks:
    if not task.get("annotations"):
        continue
    raw_name = os.path.basename(task["data"].get("image", ""))
    if "-" in raw_name and len(raw_name.split("-")[0]) == 8:
        raw_name = raw_name.split("-", 1)[1]
    base_name = os.path.splitext(raw_name)[0]

    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = os.path.join(RAW_IMG_DIR, base_name + ext)
        if os.path.exists(candidate):
            valid_samples.append((base_name, candidate, task))
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