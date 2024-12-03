import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from colorsys import rgb_to_hsv, hsv_to_rgb
import streamlit as st
import random

st.set_page_config(page_title="Skin Tone and Color Recommendations", layout="wide")

# Load Datasets
@st.cache_data
def load_datasets():
    skin_tone_data = pd.read_excel("skintonedetailed.xlsx")
    makeup_data = pd.read_csv("makeupdetailed.csv")
    skin_tone_data['R'], skin_tone_data['G'], skin_tone_data['B'] = zip(
        *skin_tone_data['RGB_Values'].str.split(',').apply(lambda x: map(int, x))
    )
    skin_tone_data['Hex_Value'] = skin_tone_data['Hex_Value'].str.strip()

    makeup_data['R'], makeup_data['G'], makeup_data['B'] = zip(
        *makeup_data['RGB_Value'].str.strip('()').str.split(',').apply(lambda x: map(int, x))
    )
    return skin_tone_data, makeup_data

skin_tone_data, makeup_data = load_datasets()

# Utility Functions
def color_difference(color1, color2):
    """Calculate Euclidean distance between two RGB colors."""
    return np.linalg.norm(np.array(color1) - np.array(color2))

def find_closest_skin_tone(detected_rgb):
    """Find the closest matching skin tone from the dataset."""
    distances = euclidean_distances([detected_rgb], skin_tone_data[['R', 'G', 'B']])
    closest_index = np.argmin(distances)
    closest_tone = skin_tone_data.iloc[closest_index]
    return closest_tone, round(100 - distances[0][closest_index], 2)

def detect_skin_tone_with_landmarks(image):
    """Detect skin tone using Mediapipe face landmarks."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    # Extract landmarks for skin tone analysis
    skin_regions = []
    for face_landmarks in results.multi_face_landmarks:
        landmark_indices = [10, 338, 297, 332, 284]
        for idx in landmark_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1])
            y = int(face_landmarks.landmark[idx].y * image.shape[0])
            skin_regions.append(image[y, x])

    if not skin_regions:
        return None

    skin_regions = np.array(skin_regions)
    dominant_color = np.median(skin_regions, axis=0).astype(int)
    return dominant_color

@st.cache_data
def generate_seasonal_palette(season, base_rgb, global_used_colors, num_colors=15, min_diff=50):
    """Generate diverse seasonal palettes with global uniqueness."""
    seasonal_adjustments = {
        "Spring": {"hue_shift": 0.1, "saturation_shift": 0.3, "brightness_shift": 0.2},
        "Summer": {"hue_shift": -0.7, "saturation_shift": 0.21, "brightness_shift": 0.25},
        "Autumn": {"hue_shift": -0.04, "saturation_shift": 0.5, "brightness_shift": 0.9},
        "Winter": {"hue_shift": 0.3, "saturation_shift": 0.2, "brightness_shift": 0.12},
    }

    # Convert base color to HSV
    r, g, b = [x / 255.0 for x in base_rgb]
    h, s, v = rgb_to_hsv(r, g, b)
    palette = []
    used_colors = set()

    # Retrieve seasonal adjustments
    adjustments = seasonal_adjustments.get(season, {})

    for _ in range(num_colors * 20):  # Ensure enough attempts for diverse colors
        new_h = (h + adjustments["hue_shift"] + random.uniform(-0.2, 0.2)) % 1.0
        new_s = min(max(s + adjustments["saturation_shift"] + random.uniform(-0.2, 0.2), 0.3), 1.0)
        new_v = min(max(v + adjustments["brightness_shift"] + random.uniform(-0.2, 0.2), 0.4), 2.0)
        new_color = tuple(int(c * 255) for c in hsv_to_rgb(new_h, new_s, new_v))

        # Ensure the color is unique globally and within the current palette
        if (
            all(color_difference(new_color, existing_color) > min_diff for existing_color in used_colors)
            and all(color_difference(new_color, existing_color) > min_diff for existing_color in global_used_colors)
        ):
            palette.append(new_color)
            used_colors.add(new_color)
            global_used_colors.add(new_color)

        if len(palette) >= num_colors:  # Stop once the required number of colors is generated
            break

    return palette



def recommend_makeup(detected_rgb, category, limit=5):
    """Recommend makeup shades based on detected skin tone."""
    category_data = makeup_data[makeup_data['Product_Category'] == category].copy()
    distances = euclidean_distances([detected_rgb], category_data[['R', 'G', 'B']])
    category_data['Distance'] = distances.flatten()
    return category_data.nsmallest(limit, 'Distance')

# Streamlit App
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
        tab1, tab2, tab3 = st.tabs(["Skin Tone", "Seasonal Colors", "Makeup Suggestions"])

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

            # Dropdown for selecting season
            selected_season = st.selectbox(
                "Select a Season:",
                ["Spring", "Summer", "Autumn", "Winter"]
            )
            global_used_colors = set()
            # Generate and display palette
            palette = generate_seasonal_palette(selected_season, detected_rgb, global_used_colors, num_colors=10)

            grid_cols = st.columns(5)
            for idx, color in enumerate(palette):
                hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}".upper()
                col = grid_cols[idx % 5]
                col.markdown(
                    f"""
                    <div style="background-color: {hex_color}; width: 100%; height: 120px; 
                    border-radius: 10px; margin: 10px auto; display: flex; justify-content: center; align-items: center; font-weight: bold; color: #fff; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                        {hex_color}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab3:
            st.subheader("Makeup Suggestions")
            show_makeup = st.radio(
                 "Would you like to see makeup recommendations?",
                 ("Yes", "No"),
                 index=1  # Default to "No"
              )

        if show_makeup == "Yes":
            # Display makeup recommendations
            for category in ["Foundation Shade", "Blush Shade", "Lipstick Shade"]:
                st.write(f"### {category}")
                recommendations = recommend_makeup(detected_rgb, category, limit=5)
                grid_cols = st.columns(4)
                for idx, row in recommendations.iterrows():
                    hex_color = row['Hex_Value']
                    product_name = row['Product_Name']
                    col = grid_cols[idx % 4]
                    col.markdown(
                        f"""
                        <div style="background-color: {hex_color}; width: 100%; height: 140px; 
                        border-radius: 15px; margin: 10px auto; display: flex; flex-direction: column; justify-content: center; align-items: center; font-weight: bold; color: #fff; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
                            <span style="font-size: 14px; margin-bottom: 5px;">{product_name}</span>
                            <span style="font-size: 12px;">{hex_color}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    else:
        st.error("Skin tone could not be detected. Please try with another image.")
else:
    st.info("Please upload or capture an image to start.")
