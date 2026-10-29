import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

folder_path = "./Ashlynn"
output_pdf = "Ringham_Ashlynn.pdf"

# Find all HEIC files (case-insensitive)
heic_files = [
    os.path.join(folder_path, f) 
    for f in os.listdir(folder_path) 
    if f.lower().endswith(('.heic', '.heif'))
]

# Sort files to ensure chronological or alphabetical order
heic_files.sort()

if heic_files:
    # Open and convert the first image
    first_image = Image.open(heic_files[0]).convert("RGB")
    
    # Open and convert the rest of the images
    remaining_images = [
        Image.open(img).convert("RGB") 
        for img in heic_files[1:]
    ]
    
    # Save everything into a single multi-page PDF
    first_image.save(output_pdf, save_all=True, append_images=remaining_images)
    print(f"Successfully created {output_pdf}")
else:
    print("No HEIC files found in the specified directory.")