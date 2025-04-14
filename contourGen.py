import streamlit as st
import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import tempfile
from tqdm import tqdm

# === Helper: Convert color names to hex (streamlit requires hex codes) ===
import matplotlib.colors as mcolors

def color_name_to_hex(color_name):
    return mcolors.to_hex(mcolors.CSS4_COLORS.get(color_name, color_name))

# === Classification color and label map ===
default_classification_styles = {
    0:  ("Created, never classified", "lightgray"),
    1:  ("Unclassified", "gray"),
    2:  ("Ground", "white"),
    3:  ("Low Vegetation", "lightgreen"),
    4:  ("Medium Vegetation", "green"),
    5:  ("High Vegetation", "darkgreen"),
    6:  ("Building", "slategray"),
    7:  ("Low Point (noise)", "red"),
    8:  ("Model Key-point (mass point)", "orange"),
    9:  ("Water", "blue"),
    10: ("Rail", "purple"),
    11: ("Road Surface", "brown"),
    12: ("Reserved", "pink"),
    13: ("Wire Guard", "cyan"),
    14: ("Wire Conductor", "deepskyblue"),
    15: ("Transmission Tower", "gold"),
    16: ("Wire Structure Connector", "coral"),
    17: ("Bridge Deck", "sienna"),
    18: ("High Noise", "crimson")
}

st.title("LiDAR Contour Generator")
st.markdown("Upload a LAZ file and customize contour output.")

uploaded_file = st.file_uploader("Upload a LAZ file", type=["laz"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".laz") as temp_file:
        temp_file.write(uploaded_file.read())
        laz_path = temp_file.name

    las = laspy.read(laz_path)
    st.success("LAZ file loaded successfully!")

    unique_classes = np.unique(las.classification)
    classification_styles = default_classification_styles.copy()

    selected_classes = st.multiselect(
        "Select classifications to include:",
        options=unique_classes,
        format_func=lambda x: f"Class {x}: {classification_styles.get(x, ('Unknown', 'yellow'))[0]}"
    )

    custom_colors = {}
    st.subheader("Customize colors (optional):")
    for cls in selected_classes:
        classification, default_color_name = classification_styles.get(cls, ("Unknown", "yellow"))
        default_color_hex = color_name_to_hex(default_color_name)
        user_color = st.color_picker(f"Class {cls} ({classification})", default_color_hex)
        custom_colors[cls] = (classification, user_color)

    line_width = st.slider("Line width", 0.1, 5.0, 0.6)
    grid_spacing = st.slider("Grid spacing (meters)", 0.5, 10.0, 1.0)

    if st.button("Generate Contour Map"):
        fig, ax = plt.subplots(figsize=(20, 10), facecolor='black')
        ax.set_facecolor('black')

        progress_bar = st.progress(0)

        for i, class_code in enumerate(selected_classes):
            classification, color = custom_colors[class_code]
            mask = las.classification == class_code
            x = las.x[mask]
            y = las.y[mask]
            z = las.z[mask]

            if len(z) < 100:
                continue

            x_range = np.arange(np.min(x), np.max(x), grid_spacing)
            y_range = np.arange(np.min(y), np.max(y), grid_spacing)
            grid_x, grid_y = np.meshgrid(x_range, y_range)
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
            grid_z = np.ma.masked_invalid(grid_z)

            if grid_z.mask.all():
                continue

            levels = np.arange(np.floor(np.min(z)), np.ceil(np.max(z)), 0.2)
            contours = ax.contour(grid_x, grid_y, grid_z, levels=levels, colors=color, linewidths=line_width)
            ax.clabel(contours, fmt='%.2f', colors='white', fontsize=6, inline=True)

            progress_bar.progress((i + 1) / len(selected_classes))

        ax.axis('off')
        output_path = os.path.join(tempfile.gettempdir(), "webapp_contour_output.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()

        st.success("Contour image generated!")
        st.image(output_path, caption="Contour Output", use_container_width=True)
        with open(output_path, "rb") as file:
            st.download_button(label="Download Image", data=file, file_name="contour_output.png", mime="image/png")
