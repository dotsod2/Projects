# This script should perform cropping of images in a folder
import json, os

JSON_PATH = "20250506_annotations.json"  # Point to one of your JSONs
IMG_DIR = "raw_images"                     # Point to its matching folder

with open(JSON_PATH, "r", encoding="utf-8") as f:
    tasks = json.load(f)

print("Sample task['data'] entries from JSON:")
for t in tasks[:3]:
    print(" ", t.get("data"))

print(f"\nSample files actually inside '{IMG_DIR}':")
for root, _, files in os.walk(IMG_DIR):
    for fn in files[:5]:
        print(" ", os.path.join(root, fn))
    break