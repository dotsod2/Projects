import os
import piexif
import geopandas as gpd
from shapely.geometry import Point
import matplotlib as mpl
import matplotlib.colors as mcolors
import folium
from collections import defaultdict
from folium import Popup, IFrame
import shutil
import re


# CONFIGURATION
input_folder = 'C:/Users/deldo/Projects/Plot_Labeling/input_images'
geojson_path = 'C:/Users/deldo/Projects/Plot_Labeling/RRR2_ALLCOMPONENTS_ANDRADE_plots_20240809.geojson'
output_folder = 'C:/Users/deldo/Projects/Plot_Labeling_NEW/2025-05-29_Renamed/'
os.makedirs(output_folder, exist_ok=True)


# Extract date from parent folder name (which contains the date)
parent_folder = os.path.basename(os.path.normpath(os.path.dirname(input_folder)))
match = re.search(r'\d{4}-\d{2}-\d{2}', input_folder)
date_prefix = match.group(0) if match else "unknown_date"


# Load plots GeoJSON. Here 'plots' is a GeoDataFrame (a table).
plots = gpd.read_file(geojson_path).to_crs(epsg=4326)

def get_gps_coords(img_path):
    try:
        exif_dict = piexif.load(img_path)
        gps = exif_dict.get("GPS", {})
        lat = gps.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
        lon = gps.get(piexif.GPSIFD.GPSLongitude)
        lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)
        if not all([lat, lat_ref, lon, lon_ref]):
            return None, None

        def dms_to_deg(dms):
            d, m, s = [val[0] / val[1] for val in dms]
            return d + m / 60 + s / 3600

        latitude = dms_to_deg(lat) * (-1 if lat_ref.decode() == 'S' else 1)
        longitude = dms_to_deg(lon) * (-1 if lon_ref.decode() == 'W' else 1)
        return latitude, longitude
    except Exception:
        return None, None

renamed = 0
skipped = 0
points_data = []

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg') and '_' in f]
for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    lat, lon = get_gps_coords(img_path)
    if lat is None or lon is None:
        skipped += 1
        continue

    point = Point(lon, lat)
    # Find closest plot index 
    distances = plots.distance(point)
    closest_plot_idx = distances.idxmin()
    plot_no = plots.loc[closest_plot_idx, 'Plot_No']

    # Ensure plot_no is string for consistency
    plot_no = str(plot_no).strip()

    new_filename = f"{plot_no}_{filename}"
    shutil.copy2(img_path, os.path.join(output_folder, new_filename))
    renamed += 1

    points_data.append({
        'geometry': point,
        'plot_no': plot_no,
        'filename': new_filename
    })

print(f"Renamed {renamed} images | Skipped {skipped} without GPS.")

if points_data:
    plot_nos = sorted(set(p['plot_no'] for p in points_data))
    cmap = mpl.colormaps['tab20'].resampled(len(plot_nos))
    plot_color_map = {plot: mcolors.to_hex(cmap(i)) for i, plot in enumerate(plot_nos)}

    mean_lat = sum(p['geometry'].y for p in points_data) / len(points_data)
    mean_lon = sum(p['geometry'].x for p in points_data) / len(points_data)
    #m = folium.Map(location=[mean_lat, mean_lon], zoom_start=17, tiles='OpenStreetMap')
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=18, tiles='OpenStreetMap', max_zoom=25)

    # Group filenames by plot_no
    plot_images = defaultdict(list)
    for item in points_data:
        plot_images[item['plot_no']].append(item['filename'])

    # DEBUG: print how many images per plot
    print("Image count per plot:")
    for plot, files in plot_images.items():
        print(f"Plot {plot}: {len(files)} images")

    # Add plot boundaries with popup
    for _, row in plots.iterrows():
        plot_no = str(row['Plot_No']).strip()  # ensure string type and strip spaces
        color = plot_color_map.get(plot_no, '#888')
        filenames = plot_images.get(plot_no, [])

        if filenames:
            html_content = f"<b>Plot {plot_no}</b><br><ul>" + "".join(f"<li>{f}</li>" for f in filenames) + "</ul>"
        else:
            html_content = f"<b>Plot {plot_no}</b><br><i>No images found.</i>"
        
        
        #The scrollable popup window
        iframe = IFrame(html_content, width=300, height=200)
        popup = Popup(iframe, max_width=320)

        # Fix style_function by wrapping in lambda factory
        def style_func_factory(col):
            return lambda feature: {
                'fillColor': col,
                'color': col,
                'weight': 2,
                'fillOpacity': 0.2,
            }

        folium.GeoJson(
            row['geometry'],
            name=f"Plot {plot_no}",
            style_function=style_func_factory(color),
            tooltip=f"Plot {plot_no}",
            popup=popup
        ).add_to(m)

    # Add image points
    for item in points_data:
        lat, lon = item['geometry'].y, item['geometry'].x
        label = item['filename']
        plot_no = item['plot_no']
        color = plot_color_map[plot_no]

        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=Popup(label, parse_html=True)
        ).add_to(m)

    # Add legend
    legend_html = '<div style="position: fixed; bottom: 20px; left: 20px; width: 200px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px;"><b>&nbsp;Plot Legend</b><br>'
    for plot, color in plot_color_map.items():
        legend_html += f'&nbsp;<i style="background:{color};width:10px;height:10px;display:inline-block;"></i> Plot {plot}<br>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    
    #Saving the HTML map
    map_filename = f"{date_prefix}_interactive_map.html"
    map_path = os.path.join(output_folder, map_filename)
    m.save(map_path)
    print(f" Interactive map saved at: {map_path}")

else:
    print(" No GPS-tagged images were visualized.")
