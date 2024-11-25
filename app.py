import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans

# Loading all datasets
skin_tone_data = pd.read_csv('skin_tone_with_palettes.csv')
clothing_data = pd.read_csv('clothing_recommendations.csv')
makeup_data = pd.read_csv('makeup_recommendations.csv')
design_data = pd.read_csv('design_palettes.csv')

# Parsing RGB values
def parse_rgb(rgb_string):
    """Convert RGB string into a tuple of integers, or return None if invalid."""
    try:
        return tuple(map(int, rgb_string.strip("()").split(",")))
    except (ValueError, AttributeError):
        return None

# Cleaning and validating skin tone data
skin_tone_data['RGB_Tuple'] = skin_tone_data['RGB_Tuple'].apply(parse_rgb)
skin_tone_data = skin_tone_data[skin_tone_data['RGB_Tuple'].notnull()]

# Detecting dominant color using k-means
def extract_dominant_color(image, k=5):
    """Extract dominant color from image using k-means."""
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[kmeans.labels_[0]]
    return tuple(map(int, dominant_color))

# Matching skin tone in all datasets
def match_skin_tone_all(input_rgb):
    input_rgb = np.array(input_rgb)
    distances = skin_tone_data['RGB_Tuple'].apply(
        lambda x: np.linalg.norm(input_rgb - np.array(x)) if isinstance(x, tuple) else float('inf')
    )
    closest_index = distances.idxmin()
    matched_tone = skin_tone_data.iloc[closest_index]

    # Fetching recommendations from all datasets
    subcategory = matched_tone["Subcategory"]
    clothing_rec = clothing_data[clothing_data["Subcategory"] == subcategory]
    makeup_rec = makeup_data[makeup_data["Subcategory"] == subcategory]
    design_rec = design_data[design_data["Subcategory"] == subcategory]

    return matched_tone, clothing_rec, makeup_rec, design_rec

# Safely evaluating palette strings
def safe_eval_palette(palette_string):
    """Safely evaluate a palette string and return a list of tuples."""
    try:
        return eval(palette_string)
    except:
        return []

# Rendering all color swatches from a list of RGB values
def render_color_palette(colors, label=""):
    """Display a palette of colors as swatches."""
    if not colors or len(colors) == 0:
        st.markdown(f"### {label}: No Colors Available")
        return
    st.markdown(f"### {label}")
    palette_html = ""
    for color in colors:
        if color:  # Check if color is valid
            palette_html += f"""
            <div style="width: 50px; height: 50px; background-color: rgb{color}; display: inline-block; 
                        border: 1px solid black; margin: 5px;"></div>
            """
    st.markdown(palette_html, unsafe_allow_html=True)

# Replacing RGB descriptions in text with swatches
def replace_rgb_with_swatch(text):
    """Replace RGB references in text with inline color swatches."""
    import re
    rgb_pattern = r"RGB\s*\((\d+),\s*(\d+),\s*(\d+)\)"
    
    def replace_match(match):
        color_rgb = tuple(map(int, match.groups()))
        return f"[{inline_color_swatch(color_rgb)}]"
    
    return re.sub(rgb_pattern, replace_match, text)

# Inline color swatch for text
def inline_color_swatch(color_rgb):
    """Render a small color swatch inline with text."""
    return f"""<span style="display: inline-block; width: 20px; height: 20px; 
               background-color: rgb{color_rgb}; border: 1px solid black; margin: 0 5px;"></span>"""

# Streamlit App
st.title("Comprehensive Color Theory Analysis")
st.write("""
Capture a photo to detect your skin tone and get personalized recommendations for:
- Clothing (seasonal and day/night palettes)
- Makeup (natural and bold looks)
- Design palettes
- Wardrobe planning
""")

# Camera Input
image_file = st.camera_input("Take a picture")

if image_file:
    # Read image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Detecting dominant skin tone
    dominant_rgb = extract_dominant_color(image)
    st.subheader("Detected Skin Tone")
    render_color_palette([dominant_rgb], label="Detected Skin Tone")

    # Matching skin tone across datasets
    matched_tone, clothing_rec, makeup_rec, design_rec = match_skin_tone_all(dominant_rgb)

    # Displaying matched tone
    st.subheader("Matched Skin Tone")
    render_color_palette([matched_tone["RGB_Tuple"]], label=f"Matched Tone: {matched_tone['Subcategory']}")

    # Clothing Recommendations
    st.subheader("Clothing Recommendations")
    for _, row in clothing_rec.iterrows():
        recommendation_text = replace_rgb_with_swatch(row["Clothing_Recommendations"])
        st.markdown(recommendation_text, unsafe_allow_html=True)

    # Makeup Recommendations
    st.subheader("Makeup Recommendations")
    for _, row in makeup_rec.iterrows():
        recommendation_text = replace_rgb_with_swatch(row["Makeup_Recommendations"])
        st.markdown(recommendation_text, unsafe_allow_html=True)

    # Design Palettes
    st.subheader("Design Palettes")
    palette_types = ["Monochromatic_Palette", "Complementary_Palette", "Analogous_Palette", "Triadic_Palette"]
    for palette_type in palette_types:
        palette_colors = safe_eval_palette(design_rec.iloc[0][palette_type]) if palette_type in design_rec.columns else []
        render_color_palette(palette_colors, label=f"{palette_type.replace('_', ' ').title()}")

    # Wardrobe Palette
    st.subheader("Wardrobe Palette")
    monochromatic_palette = safe_eval_palette(design_rec.iloc[0].get("Monochromatic_Palette", "[]"))
    complementary_palette = safe_eval_palette(design_rec.iloc[0].get("Complementary_Palette", "[]"))
    wardrobe_palette = monochromatic_palette + complementary_palette
    render_color_palette(wardrobe_palette, label="Wardrobe Palette")



    # Export Recommendations
    combined_data = pd.concat([clothing_rec, makeup_rec, design_rec], axis=1)
    st.download_button(
        label="Download Full Recommendations",
        data=combined_data.to_csv(index=False),
        file_name="comprehensive_recommendations.csv",
        mime="text/csv",
    )
