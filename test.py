import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from colorsys import rgb_to_hsv, hsv_to_rgb
import streamlit as st
import random

# Loading Skin Tone Dataset
skin_tone_file = "skintonedetailed.xlsx"
skin_tone_data = pd.read_excel(skin_tone_file)

skin_tone_data['R'], skin_tone_data['G'], skin_tone_data['B'] = zip(
    *skin_tone_data['RGB_Values'].str.split(',').apply(lambda x: map(int, x))
)
skin_tone_data['Hex_Value'] = skin_tone_data['Hex_Value'].str.strip()

# Loading Makeup Dataset
makeup_file = "makeupdetailed.csv"
makeup_data = pd.read_csv(makeup_file)

makeup_data['R'], makeup_data['G'], makeup_data['B'] = zip(
    *makeup_data['RGB_Value'].str.strip('()').str.split(',').apply(lambda x: map(int, x))
)

# Defining Seasonal Palette Rules(based on color theory)
seasonal_palette_rules = {
    "Spring": {"hue_shift": 10, "saturation_factor": 1.2, "brightness_factor": 1.1},
    "Summer": {"hue_shift": -10, "saturation_factor": 0.9, "brightness_factor": 1.0},
    "Autumn": {"hue_shift": -5, "saturation_factor": 1.1, "brightness_factor": 0.9},
    "Winter": {"hue_shift": -20, "saturation_factor": 1.0, "brightness_factor": 1.2},
}

# Utility Functions
def find_closest_skin_tone(detected_rgb):
    """Find the closest matching skin tone from the dataset."""
    distances = euclidean_distances([detected_rgb], skin_tone_data[['R', 'G', 'B']])
    closest_index = np.argmin(distances)
    closest_tone = skin_tone_data.iloc[closest_index]
    return closest_tone, round(100 - distances[0][closest_index], 2)


def detect_skin_tone_with_landmarks(image):
    """Detect skin tone using Mediapipe face landmarks."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    # Extract key landmarks for forehead, cheeks, and chin
    skin_regions = []
    for face_landmarks in results.multi_face_landmarks:
        landmark_indices = [10, 338, 297, 332, 284]
        for idx in landmark_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1])
            y = int(face_landmarks.landmark[idx].y * image.shape[0])
            skin_regions.append(image[y, x])

    if not skin_regions:
        return None

    # Calculate median RGB
    skin_regions = np.array(skin_regions)
    dominant_color = np.median(skin_regions, axis=0).astype(int)
    return dominant_color


def generate_diverse_palette(base_rgb, season_rules, num_colors=10):
    """Generate a diverse seasonal color palette."""
    base_h, base_s, base_v = rgb_to_hsv(*[x / 255.0 for x in base_rgb])
    hue_shift = season_rules["hue_shift"] / 360.0
    saturation_factor = season_rules["saturation_factor"]
    brightness_factor = season_rules["brightness_factor"]

    palette = []
    used_hues = set()

    while len(palette) < num_colors:
        h = (base_h + hue_shift + random.uniform(-0.6, 0.6)) % 1.0
        s = min(max(base_s * saturation_factor + random.uniform(-0.3, 0.3), 0.4), 1.0)
        v = min(max(base_v * brightness_factor + random.uniform(-0.3, 0.3), 0.4), 1.0)
        color = tuple(int(c * 255) for c in hsv_to_rgb(h, s, v))

        if h not in used_hues and all(np.linalg.norm(np.array(color) - np.array(existing_color)) > 75 for existing_color in palette):
            palette.append(color)
            used_hues.add(h)

    return palette


def recommend_makeup(detected_rgb, category, limit):
    """Recommend makeup shades based on detected skin tone."""
    category_data = makeup_data[makeup_data['Product_Category'] == category]
    distances = euclidean_distances([detected_rgb], category_data[['R', 'G', 'B']])
    category_data['Distance'] = distances.flatten()
    recommended_shades = category_data.sort_values(by="Distance").drop_duplicates(
        subset=['R', 'G', 'B']
    ).head(limit)
    return recommended_shades


# Streamlit App UI
st.set_page_config(page_title="Skin Tone and Color Recommendations", layout="wide")
st.title("Skin Tone and Color Recommendations")

# File upload or camera input
image_file = st.file_uploader("Upload an image")
if not image_file:
    image_file = st.camera_input("Or take a picture")

if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Detect skin tone
    detected_rgb = detect_skin_tone_with_landmarks(image)
    if detected_rgb is not None:
        detected_rgb = detected_rgb.tolist()
        closest_tone, confidence = find_closest_skin_tone(detected_rgb)

        # Tabbed Interface
        tab1, tab2, tab3 = st.tabs(["Skin Tone", "Seasonal Colors", "Makeup Recommendations"])

        with tab1:
            st.write(f"### Detected Skin Tone: {closest_tone['Skin_Tone_Name']} (Category: {closest_tone['Category']})")
            st.write(f"### Confidence: {confidence}%")
            st.markdown(
                f"""
                <div style="width: 100px; height: 100px; background-color: {closest_tone['Hex_Value']}; border-radius: 50%; margin: 20px auto;"></div>
                """,
                unsafe_allow_html=True,
            )

        with tab2:
            st.subheader("Seasonal Color Recommendations")

            # Dropdown for season selection
            selected_season = st.selectbox(
                "Select a Season:",
                list(seasonal_palette_rules.keys())
            )

            # Display selected season's palette
            if selected_season:
                st.write(f"### {selected_season} Palette")
                rules = seasonal_palette_rules[selected_season]
                seasonal_palette = generate_diverse_palette(
                    [closest_tone['R'], closest_tone['G'], closest_tone['B']], rules, num_colors=10
                )
                for color in seasonal_palette:
                    hex_color = f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}".upper()
                    st.markdown(
                        f"""
                        <div style="display: inline-block; width: 50px; height: 50px; background-color: {hex_color}; border-radius: 50%; margin: 5px;"></div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab3:
            st.subheader("Makeup Recommendations")

            # Foundation Recommendations
            st.write("### Foundation Shades")
            foundation_recommendations = recommend_makeup(detected_rgb, "Foundation Shade", 4)
            for _, row in foundation_recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 50px; height: 50px; background-color: {row['Hex_Value']}; border-radius: 50%; margin-right: 10px;"></div>
                        <span style="font-size: 16px;">{row['Product_Name']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Blush Recommendations
            st.write("### Blush Shades")
            blush_recommendations = recommend_makeup(detected_rgb, "Blush Shade", 3)
            for _, row in blush_recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 50px; height: 50px; background-color: {row['Hex_Value']}; border-radius: 50%; margin-right: 10px;"></div>
                        <span style="font-size: 16px;">{row['Product_Name']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Lipstick Recommendations
            st.write("### Lipstick Shades")
            lipstick_recommendations = recommend_makeup(detected_rgb, "Lipstick Shade", 5)
            for _, row in lipstick_recommendations.iterrows():
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="width: 50px; height: 50px; background-color: {row['Hex_Value']}; border-radius: 50%; margin-right: 10px;"></div>
                        <span style="font-size: 16px;">{row['Product_Name']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.error("Skin tone could not be detected. Please try with another image.")
else:
    st.info("Please upload or capture an image to start.")
