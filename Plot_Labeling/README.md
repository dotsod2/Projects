
# 🛰️ Image Plot Labeling Using EXIF GPS and Plot GeoJSON

This repository provides a Python tool for automatically associating geotagged JPG images with field plots using GPS coordinates stored in image EXIF metadata and plot geometries provided in a `.geojson` file.

For each image containing valid GPS metadata, the script identifies the nearest plot, prefixes the original filename with the corresponding `Plot_No`, and copies the renamed image to an output directory.

The script also generates an interactive HTML map for visually inspecting plot boundaries, image locations, and image-to-plot associations.

---

## 📌 What It Does

- Reads GPS latitude and longitude from image EXIF metadata.
- Loads plot boundaries from a `.geojson` file.
- Converts plot geometries to EPSG:4326 for comparison with GPS coordinates.
- Determines the nearest plot to each image GPS location using spatial distance.
- Prefixes each image filename with the corresponding plot number (e.g., `3_IMG_0045.JPG`).
- Copies renamed images to a separate output directory, leaving the original images unchanged.
- Generates an interactive Folium HTML map showing:
  - Plot boundaries
  - GPS locations of images
  - Plot-specific colors
  - Image filenames
  - Image-to-plot associations
  - A plot legend

---

## 📂 Example Folder Structure

```text
/Plot_Labeling/
├── 2025-05-29_Transect/
│   └── Transect/
│       ├── IMG_0001.JPG
│       ├── IMG_0002.JPG
│       └── ...
├── RRR2_ALLCOMPONENTS_ANDRADE_plots_20240809.geojson
├── label_by_gps.py
└── Plot_labeling_NEW/
    └── 2025-05-29_Renamed/
        ├── 3_IMG_0001.JPG
        ├── 5_IMG_0002.JPG
        └── 2025-05-29_interactive_map.html
```

---

## 🧭 How It Works

### 1. Identify Input Images

The script searches the configured input directory for files that end in `.jpg` (case-insensitive) and contain an underscore (`_`) in their filenames.

### 2. Extract GPS Coordinates

For each eligible image, the script reads the following EXIF GPS fields:

- `GPSLatitude`
- `GPSLatitudeRef`
- `GPSLongitude`
- `GPSLongitudeRef`

Latitude and longitude are converted from degrees, minutes, and seconds (DMS) into decimal degrees. The latitude and longitude reference fields determine the signs of the coordinates for the northern or southern and eastern or western hemispheres.

Images with missing or incomplete GPS information are skipped.

### 3. Load Plot Geometries

The script loads the specified GeoJSON file using GeoPandas:

```python
plots = gpd.read_file(geojson_path).to_crs(epsg=4326)
```

The GeoJSON file must contain plot geometries and a `Plot_No` attribute identifying each plot. The plot geometries are converted to EPSG:4326 to match the geographic coordinate system used by the image GPS coordinates.

### 4. Find the Nearest Plot

Each image's GPS coordinates are converted into a Shapely `Point`. The script calculates the distance between that point and every plot geometry, then selects the plot with the minimum distance.

Conceptually, the process is:

```text
Image GPS coordinates
        |
        v
   Geographic point
        |
        v
Calculate distance to every plot
        |
        v
Select plot with minimum distance
        |
        v
Assign the plot number to the image
```

**Important:** The script assigns the nearest plot based on spatial distance. It does not explicitly check whether the GPS point falls inside a plot polygon. Consequently, an image may be assigned to a plot even if its GPS point is outside that plot's boundary.

### 5. Prefix the Image Filename

Once a plot has been assigned, its plot number is added to the beginning of the original filename.

For example:

```text
Original filename:
IMG_0045.JPG

Assigned plot:
3

Output filename:
3_IMG_0045.JPG
```

The original filename is preserved after the plot-number prefix.

### 6. Copy the Renamed Image

The script uses `shutil.copy2()` to copy each renamed image into the configured output directory.

The original image is not moved, modified, or deleted.

For example:

```text
Original:
2025-05-29_Transect/Transect/IMG_0045.JPG

Output copy:
2025-05-29_Renamed/3_IMG_0045.JPG
```

### 7. Generate an Interactive Map

If at least one image has valid GPS information, the script automatically generates an interactive map using Folium.

The map includes:

- Colored plot boundaries
- GPS locations of processed images
- Plot-specific colors
- Clickable image markers
- Plot popups listing associated image filenames
- A plot legend

The map is centered using the mean latitude and longitude of the successfully processed image locations. It uses OpenStreetMap tiles and starts at zoom level 18.

---

## 🗺️ Interactive Map Features

### Plot Boundaries

Each plot boundary is displayed using a color assigned to its plot number. Hovering over a plot displays its plot number.

### Image Locations

Each GPS-tagged image is represented by a small circular marker. The marker uses the same color as its assigned plot.

Clicking an image marker displays the corresponding output filename.

### Plot Popups

Clicking a plot boundary opens a popup containing the plot number and a list of image filenames associated with that plot.

For example:

```text
Plot 7

7_IMG_0012.JPG
7_IMG_0013.JPG
7_IMG_0014.JPG
7_IMG_0015.JPG
```

### Legend

A legend in the bottom-left corner of the map identifies the color associated with each plot.

### Image Counts

The script prints the number of successfully processed images associated with each plot, making it easier to inspect the distribution of images across the study area.

---


## 💻 How to Run

### 1. Configure the Paths

Open `plot_labeling.py` and edit the configuration section:

```python
input_folder = '/path/to/drone/images/'
geojson_path = '/path/to/plots.geojson'
output_folder = '/path/to/output_folder/'
```

- `input_folder`: Directory containing the source JPG images.
- `geojson_path`: Path to the GeoJSON file containing plot geometries and the `Plot_No` attribute.
- `output_folder`: Directory where renamed image copies and the interactive HTML map will be saved.

### 2. Run the Script

Execute the following command from the directory containing `plot_labeling.py`:

```bash
python plot_labeling.py
```

The script automatically creates the output directory if it does not already exist.

---

## 🔍 Dependencies

Install the required Python packages using:

```bash
pip install geopandas shapely folium piexif matplotlib
```

The script also uses standard Python libraries that are included with Python:

- `os`
- `re`
- `shutil`
- `collections`

These standard libraries do not require separate installation.

---

## 🖼️ Output Files

### Renamed Images

Images with valid GPS metadata are copied into the configured output directory.

Example:

```text
Plot_labeling_NEW/2025-05-29_Renamed/
├── 1_IMG_0001.JPG
├── 1_IMG_0002.JPG
├── 2_IMG_0003.JPG
├── 3_IMG_0004.JPG
└── ...
```

The number at the beginning of each filename represents the plot assigned by the nearest-plot spatial-distance calculation.

### Interactive HTML Map

The script generates an HTML map named using the date found in the input path.

Example:

```text
2025-05-29_interactive_map.html
```

The map is saved in the configured output directory and can be opened in a web browser for spatial inspection and quality control.

---

## 📊 Console Output

The script reports the number of images successfully processed and the number skipped because of missing or incomplete GPS information.

Example:

```text
Renamed 842 images | Skipped 31 without GPS.
```

It also prints the number of images associated with each plot:

```text
Image count per plot:
Plot 1: 42 images
Plot 2: 38 images
Plot 3: 51 images
```

Finally, it reports the path of the generated interactive map:

```text
Interactive map saved at: /path/to/output_folder/2025-05-29_interactive_map.html
```

If no images contain valid GPS information, the script reports that no GPS-tagged images were visualized and does not generate the map.

---

## ⚠️ Important Notes and Limitations

- Only `.jpg` files, including `.JPG` files, with an underscore (`_`) in their filenames are processed.
- Images without complete EXIF GPS information are skipped.
- The GeoJSON file must contain a field named `Plot_No`.
- Multiple images can be associated with the same plot.
- The script uses the nearest plot based on spatial distance; it does not perform a point-in-polygon check.
- The original images remain unchanged because the script copies renamed versions into the output directory.
- Plot geometries are converted to EPSG:4326 before spatial comparison.
- The date used in the output map filename is extracted from the input path. If no date is found, the script uses `unknown_date`.
- The accuracy of image-to-plot assignments depends partly on the accuracy of the GPS coordinates stored in the image EXIF metadata.
- The script does not validate GPS accuracy or automatically flag points that are far from all plot boundaries.
- Existing files with the same output filenames may be overwritten when copied.
- The script assumes that the GeoJSON geometries are valid and that the `Plot_No` field contains usable plot identifiers.

---

## 📌 Why This Matters

Automatically associating geotagged images with field plots provides a reproducible way to organize spatial imagery for downstream analysis.

Plot-level image labeling can support:

- Plot-wise image organization
- Alignment between field measurements and imagery
- Semantic segmentation and annotation workflows
- Vegetation analysis
- Machine learning and computer vision dataset preparation
- Spatial quality control and visualization
- Plot-level environmental monitoring
- UAV-based phenotyping and ecological analysis

The interactive map provides an additional visual quality-control step by allowing users to inspect image locations and their assigned plots spatially.

---

## 🧪 Research Reproducibility

Given a directory of geotagged JPG images, a GeoJSON file containing plot boundaries and the `Plot_No` attribute, and appropriately configured paths, the script automates image organization and generates a corresponding spatial visualization.

The resulting renamed images and interactive map can support reproducible data organization and subsequent plot-level environmental analysis.

---

## 🙌 Acknowledgments

This tool uses the following open-source Python packages:

- **GeoPandas** for geospatial data handling
- **Shapely** for geometric operations
- **Folium** for interactive mapping
- **piexif** for reading image EXIF metadata
- **Matplotlib** for generating plot colors

---

*Built for research automation and reproducibility in UAV-based environmental monitoring and field-plot image analysis.*
