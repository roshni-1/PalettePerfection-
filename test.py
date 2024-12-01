import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from colorsys import rgb_to_hsv, hsv_to_rgb
from webcolors import rgb_to_name, rgb_to_hex
import streamlit as st
import random

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the expanded skin tone dataset
dataset_path = 'expanded_skin_tone_dataset.csv'
skin_tone_df = pd.read_csv(dataset_path)

# Extract skin tone RGB values and names
skin_tone_rgb_values = skin_tone_df['RGB_Value'].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()
skin_tone_names = skin_tone_df['Skin_Tone_Name'].tolist()

# Load the makeup dataset
makeup_dataset_path = 'makeup_dataset.csv'
makeup_df = pd.read_csv(makeup_dataset_path)

# Variables for dominant skin tone and thresholding
fixed_skin_tone = None  # Fixed dominant tone
change_threshold = 20  # Large threshold for extreme changes
last_dominant_color = None
skin_tone_buffer = []  # Buffer for smoothing
buffer_size = 5

# Function to calculate color difference
def color_difference(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

# Function to get the closest color name
def get_closest_skin_tone(detected_rgb):
    min_diff = float('inf')
    closest_tone_name = None
    closest_rgb = None
    
    for i, rgb in enumerate(skin_tone_rgb_values):
        diff = color_difference(detected_rgb, rgb)
        if diff < min_diff:
            min_diff = diff
            closest_tone_name = skin_tone_names[i]
            closest_rgb = rgb
    
    return closest_tone_name, closest_rgb

# Function to smooth the detected skin tone
def smooth_skin_tone(detected_color):
    global skin_tone_buffer
    skin_tone_buffer.append(detected_color)
    if len(skin_tone_buffer) > buffer_size:
        skin_tone_buffer.pop(0)
    smoothed_color = np.mean(skin_tone_buffer, axis=0).astype(int)
    return smoothed_color

# Function to detect dominant skin tone
def detect_skin_tone(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([35, 180, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = cv2.bitwise_and(face_roi, face_roi, mask=mask)
        reshaped_skin = skin_pixels.reshape((-1, 3))
        reshaped_skin = reshaped_skin[~np.all(reshaped_skin == 0, axis=1)]

        if len(reshaped_skin) > 100:
            kmeans = KMeans(n_clusters=3, random_state=0)
            kmeans.fit(reshaped_skin)
            dominant_color = np.mean(kmeans.cluster_centers_, axis=0).astype(int)
            smoothed_color = smooth_skin_tone(dominant_color)
            closest_tone_name, closest_rgb = get_closest_skin_tone(smoothed_color)
            return closest_tone_name, closest_rgb

    return None, None

# Function to generate color palettes from a color wheel
def generate_palette_from_wheel(rgb, scheme="complementary", num_colors=10, min_diff=40):
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = rgb_to_hsv(r, g, b)

    palettes = []
    used_colors = set()
    for i in range(num_colors * 10):
        if scheme == "complementary":
            new_h = (h + 0.5 + random.uniform(-0.1, 0.1)) % 1.0
        elif scheme == "analogous":
            new_h = (h + random.uniform(-0.2, 0.2)) % 1.0
        elif scheme == "triadic":
            new_h = (h + (i % 3) * 0.333 + random.uniform(-0.1, 0.1)) % 1.0
        else:
            new_h = (h + random.uniform(-0.15, 0.15)) % 1.0

        new_color = tuple(int(x * 255) for x in hsv_to_rgb(new_h, min(max(s + random.uniform(-0.1, 0.1), 0), 1), min(max(v + random.uniform(-0.1, 0.1), 0), 1)))
        if all(color_difference(new_color, existing_color) > min_diff for existing_color in used_colors):
            used_colors.add(new_color)
            palettes.append(new_color)
        if len(palettes) >= num_colors:
            break

    return palettes

# Function to get seasonal color palette based on skin tone
def get_seasonal_palette(skin_tone_rgb, season):
    r, g, b = [x / 255.0 for x in skin_tone_rgb]
    h, s, v = rgb_to_hsv(r, g, b)
    palettes = []

    if season == "Light Spring":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "complementary", num_colors=10))
    elif season == "True Spring":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "triadic", num_colors=10))
    elif season == "Warm Spring":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "complementary", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
    elif season == "Light Summer":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "complementary", num_colors=10))
    elif season == "True Summer":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "complementary", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "triadic", num_colors=10))
    elif season == "Soft Summer":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "triadic", num_colors=10))
    elif season == "Soft Autumn":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "triadic", num_colors=10))
    elif season == "True Autumn":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "triadic", num_colors=10))
    elif season == "Deep Autumn":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "complementary", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "triadic", num_colors=10))
    elif season == "Deep Winter":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "complementary", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "triadic", num_colors=10))
    elif season == "True Winter":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "triadic", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
    elif season == "Cool Winter":
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone_rgb, "complementary", num_colors=10))

    return list(dict.fromkeys(palettes))  # Remove duplicates while preserving order

# Function to display makeup recommendations
def display_makeup_recommendations(skin_tone_name, detected_rgb):
    """Display makeup recommendations based on detected skin tone."""
    if skin_tone_name is not None:
        # Get makeup recommendations from the dataset based on the closest match
        recommendations = makeup_df.copy()
        recommendations['RGB_Value'] = recommendations['RGB_Value'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        recommendations['Color_Diff'] = recommendations['RGB_Value'].apply(lambda x: color_difference(x, detected_rgb))
        sorted_recommendations = recommendations.sort_values(by='Color_Diff').head(5)  # Get top 5 closest matches
        
        if not sorted_recommendations.empty:
            st.write("### Makeup Recommendations")
            for _, row in sorted_recommendations.iterrows():
                st.write(f"**Product Type**: {row['Product_Type']}")
                st.write(f"**Product Shade Name**: {row['Product_Shade_Name']}")
                st.write(f"**Undertone**: {row['Undertone']}")
                st.write(f"**RGB Value**: {row['RGB_Value']}")
                color_rgb = row['RGB_Value']
                st.markdown(
                    f"<div class='color-box' style='background-color: rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}); width: 50px; height: 50px; display: inline-block; margin: 5px;'></div>",
                    unsafe_allow_html=True
                )
        else:
            st.write("No makeup recommendations available for the closest match found.")
    else:
        st.write("Skin tone could not be detected accurately.")

# Streamlit app
st.title("Automatic Skin Tone Detection and Color Recommendations")
st.write("Capture a photo or upload one to detect your skin tone and get personalized color recommendations.")

# Camera Input or File Upload
image_file = st.file_uploader("Upload an image")
if not image_file:
    image_file = st.camera_input("Or take a picture")

if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Detect skin tone
    skin_tone_name, skin_tone_rgb = detect_skin_tone(image)

    if skin_tone_rgb is not None:
        # Display detected skin tone
        st.markdown(f"### Detected Skin Tone: {skin_tone_name} (RGB: {skin_tone_rgb})")
        st.markdown(
            f"<div class='color-box' style='background-color: rgb({skin_tone_rgb[0]}, {skin_tone_rgb[1]}, {skin_tone_rgb[2]}); width: 50px; height: 50px; display: inline-block; margin: 5px;'></div>",
            unsafe_allow_html=True
        )

        # Dropdown for season selection
        season = st.selectbox("\U0001F308 Select a season for recommended palettes:", [
            "Light Spring", "True Spring", "Warm Spring",
            "Light Summer", "True Summer", "Soft Summer",
            "Soft Autumn", "True Autumn", "Deep Autumn",
            "Deep Winter", "True Winter", "Cool Winter"
        ])

        # Display seasonal palettes
        st.subheader(f"\U0001F3A8 Recommended Seasonal Palette for {season}")
        seasonal_palette = get_seasonal_palette(skin_tone_rgb, season)
        st.markdown("<div class='palette-container'>", unsafe_allow_html=True)
        for color in seasonal_palette:
            color_style = f"rgb({color[0]}, {color[1]}, {color[2]})"
            st.markdown(
                f"<div class='color-box' style='background-color: {color_style}; width: 50px; height: 50px; display: inline-block; margin: 5px;'></div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Display makeup recommendations
        st.subheader("\U0001F484 Personalized Makeup Recommendations")
        display_makeup_recommendations(skin_tone_name, skin_tone_rgb)
    else:
        st.error("Could not detect a face or skin tone. Please try again.")
else:
    st.info("\U0001F4F7 Please upload or capture an image to start.")
