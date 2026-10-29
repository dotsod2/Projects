# This script should append a given append string to each file in a folder
import os
import folium
from collections import defaultdict
import shutil
import re
# folder initialization
input_folder = '/home/dotsod/Documents/2025-05-29_Renamed'
output_folder = '/home/dotsod/Documents/2025-05-29_Renamed'
# looping through to store images
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg') and '_' in f]
renamed = 0
appendage = "20250529"
# Loop to append appendage to each image
for filename in image_files:
    img_path = os.path.join(input_folder, filename)

    new_filename = f"{appendage}_{filename}"
    shutil.copy2(img_path, os.path.join(output_folder, new_filename))
    renamed += 1

print(f"Renamed {renamed} images")