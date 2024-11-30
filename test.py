import cv2
import numpy as np
from sklearn.cluster import KMeans
from colorsys import rgb_to_hsv, hsv_to_rgb
from webcolors import rgb_to_name, rgb_to_hex
import streamlit as st
import random

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables for dominant skin tone and thresholding
fixed_skin_tone = None  # Fixed dominant tone
change_threshold = 20  # Large threshold for extreme changes
last_dominant_color = None
skin_tone_buffer = []  # Buffer for smoothing
buffer_size = 5

# Makeup recommendations database
makeup_recommendations = {
    "fair": {
        "foundation": [
            {"name": "Ivory", "color": (230, 225, 220)},
            {"name": "Porcelain", "color": (240, 235, 230)},
            {"name": "Light Neutral", "color": (220, 215, 210)},
            {"name": "Fair Beige", "color": (245, 240, 235)},
            {"name": "Alabaster", "color": (250, 245, 240)},
            {"name": "Snow", "color": (255, 250, 245)},
            {"name": "Cool Ivory", "color": (235, 230, 225)},
            {"name": "Fair Pearl", "color": (255, 245, 240)},
            {"name": "Soft Porcelain", "color": (242, 237, 232)},
            {"name": "Shell", "color": (240, 230, 225)}
        ],
        "blush": [
            {"name": "Soft Pink", "color": (255, 182, 193)},
            {"name": "Rose", "color": (255, 174, 185)},
            {"name": "Blushing Pink", "color": (255, 192, 203)},
            {"name": "Baby Pink", "color": (255, 200, 210)},
            {"name": "Pale Peach", "color": (255, 218, 185)}
        ],
        "lipstick": [
            {"name": "Peach", "color": (255, 160, 122)},
            {"name": "Light Pink", "color": (255, 182, 193)},
            {"name": "Nude Pink", "color": (240, 180, 170)},
            {"name": "Pale Coral", "color": (255, 210, 200)},
            {"name": "Petal Pink", "color": (255, 192, 203)}
        ],
        "eyeshadow": [
            {"name": "Champagne Shimmer", "color": (255, 239, 219)},
            {"name": "Light Taupe", "color": (210, 180, 140)},
            {"name": "Soft Lavender", "color": (230, 230, 250)},
            {"name": "Cool Silver", "color": (192, 192, 192)},
            {"name": "Pale Pink Shimmer", "color": (255, 228, 225)}  # Complementary cool tone
        ],
        "season": "Summer",  # Cool and soft colors
        "makeup_looks": [
            {"name": "Daytime Dewy Look", "description": "A fresh and natural look with light blush and pastel lipstick for a radiant glow."},
            {"name": "Evening Glam", "description": "Use a bold pink lipstick with shimmery eyeshadow to create a cool, glamorous evening look."}
        ]
    },
    "light": {
        "foundation": [
            {"name": "Light Beige", "color": (230, 220, 200)},
            {"name": "Natural Beige", "color": (220, 205, 190)},
            {"name": "Light Yellow Undertone", "color": (225, 215, 190)},
            {"name": "Vanilla", "color": (240, 225, 210)},
            {"name": "Buff Beige", "color": (235, 215, 200)},
            {"name": "Warm Sand", "color": (245, 220, 200)},
            {"name": "Golden Ivory", "color": (240, 225, 195)},
            {"name": "Cream", "color": (245, 230, 215)},
            {"name": "Light Honey", "color": (235, 220, 200)},
            {"name": "Soft Beige", "color": (230, 215, 190)}
        ],
        "blush": [
            {"name": "Rose", "color": (255, 105, 180)},
            {"name": "Coral", "color": (255, 127, 80)},
            {"name": "Petal Pink", "color": (255, 182, 193)},
            {"name": "Warm Peach", "color": (255, 150, 130)},
            {"name": "Light Apricot", "color": (255, 190, 150)}
        ],
        "lipstick": [
            {"name": "Coral", "color": (255, 69, 0)},
            {"name": "Light Red", "color": (255, 99, 71)},
            {"name": "Peachy Nude", "color": (255, 160, 122)},
            {"name": "Rosy Beige", "color": (240, 150, 135)},
            {"name": "Soft Coral", "color": (255, 127, 100)}
        ],
        "eyeshadow": [
            {"name": "Golden Peach", "color": (255, 223, 186)},
            {"name": "Bronze", "color": (205, 127, 50)},
            {"name": "Soft Pink", "color": (255, 182, 193)},
            {"name": "Warm Gold", "color": (255, 215, 0)},
            {"name": "Peach Shimmer", "color": (255, 218, 185)}  # Complementary warm tone
        ],
        "season": "Spring",  # Warm and bright colors
        "makeup_looks": [
            {"name": "Spring Bloom Look", "description": "Light and bright makeup using coral blush and peachy lipstick for a cheerful daytime look."},
            {"name": "Golden Hour Glow", "description": "Apply bronze eyeshadow with warm coral lipstick for a radiant, sunset-inspired evening look."}
        ]
    }
}

# Function to calculate color difference
def color_difference(color1, color2):
    print(f"Calculating color difference between {color1} and {color2}")
    return np.linalg.norm(np.array(color1) - np.array(color2))

# Function to get the closest color name
def get_color_name(rgb):
    print(f"Getting color name for RGB: {rgb}")
    rgb = tuple(map(int, rgb))
    try:
        return rgb_to_name(rgb)
    except ValueError:
        return rgb_to_hex(rgb)

# Function to generate color palettes from a color wheel
def generate_palette_from_wheel(rgb, scheme="complementary", num_colors=10, min_diff=40):
    print(f"Generating palette from wheel with RGB: {rgb}, scheme: {scheme}, num_colors: {num_colors}")
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = rgb_to_hsv(r, g, b)

    palettes = []
    used_colors = set()  # Track used colors to avoid duplicates
    for i in range(num_colors * 10):  # Generate more colors to ensure diversity and reduce repetition
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

    print(f"Generated palette: {palettes}")
    return palettes

# Function to smooth the detected skin tone
def smooth_skin_tone(detected_color):
    print(f"Smoothing detected skin tone with color: {detected_color}")
    global skin_tone_buffer
    skin_tone_buffer.append(detected_color)
    if len(skin_tone_buffer) > buffer_size:
        skin_tone_buffer.pop(0)
    smoothed_color = np.mean(skin_tone_buffer, axis=0).astype(int)
    print(f"Smoothed skin tone: {smoothed_color}")
    return smoothed_color

# Function to detect dominant skin tone
def detect_skin_tone(frame):
    print("Detecting skin tone")
    global fixed_skin_tone, last_dominant_color

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    print(f"Detected faces: {faces}")

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = cv2.bitwise_and(face_roi, face_roi, mask=mask)
        reshaped_skin = skin_pixels.reshape((-1, 3))
        reshaped_skin = reshaped_skin[~np.all(reshaped_skin == 0, axis=1)]
        print(f"Extracted skin pixels: {len(reshaped_skin)}")

        if len(reshaped_skin) > 0:
            kmeans = KMeans(n_clusters=1, random_state=0)
            kmeans.fit(reshaped_skin)
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            print(f"Dominant skin tone detected: {dominant_color}")

            # Smooth the detected skin tone
            smoothed_color = smooth_skin_tone(dominant_color)

            # If no fixed skin tone or a significant change occurs, update
            if fixed_skin_tone is None or color_difference(smoothed_color, fixed_skin_tone) > change_threshold:
                fixed_skin_tone = smoothed_color
                print(f"Updated fixed skin tone: {fixed_skin_tone}")
                return fixed_skin_tone

    return None

# Function to get seasonal color palette based on skin tone
def get_seasonal_palette(skin_tone, season):
    print(f"Getting seasonal palette for skin tone: {skin_tone}, season: {season}")
    r, g, b = [x / 255.0 for x in skin_tone]
    h, s, v = rgb_to_hsv(r, g, b)
    palettes = []

    if season == "Light Spring":
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "complementary", num_colors=10))
    elif season == "True Spring":
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "triadic", num_colors=10))
    elif season == "Warm Spring":
        palettes.extend(generate_palette_from_wheel(skin_tone, "complementary", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
    elif season == "Light Summer":
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "complementary", num_colors=10))
    elif season == "True Summer":
        palettes.extend(generate_palette_from_wheel(skin_tone, "complementary", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "triadic", num_colors=10))
    elif season == "Soft Summer":
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "triadic", num_colors=10))
    elif season == "Soft Autumn":
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "triadic", num_colors=10))
    elif season == "True Autumn":
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "triadic", num_colors=10))
    elif season == "Deep Autumn":
        palettes.extend(generate_palette_from_wheel(skin_tone, "complementary", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "triadic", num_colors=10))
    elif season == "Deep Winter":
        palettes.extend(generate_palette_from_wheel(skin_tone, "complementary", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "triadic", num_colors=10))
    elif season == "True Winter":
        palettes.extend(generate_palette_from_wheel(skin_tone, "triadic", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
    elif season == "Cool Winter":
        palettes.extend(generate_palette_from_wheel(skin_tone, "analogous", num_colors=10))
        palettes.extend(generate_palette_from_wheel(skin_tone, "complementary", num_colors=10))

    print(f"Generated seasonal palette: {palettes}")
    return list(dict.fromkeys(palettes))  # Remove duplicates while preserving order

# Function to display makeup recommendations
def display_makeup_recommendations(skin_tone):
    if skin_tone is not None:
        # Determine skin category
        r, g, b = skin_tone
        if r > 200 and g > 180 and b > 170:
            skin_category = "fair"
        elif r > 180 and g > 150 and b > 130:
            skin_category = "light"
        elif r > 160 and g > 130 and b > 110:
            skin_category = "medium"
        elif r > 120 and g > 100 and b > 80:
            skin_category = "tan"
        else:
            skin_category = "deep"

        print(f"Detected skin category: {skin_category}")
        recommendations = makeup_recommendations.get(skin_category, {})

        st.subheader("Makeup Recommendations")
        if recommendations:
            for category, items in recommendations.items():
                if isinstance(items, list):  # Skip non-list items like "season" and "makeup_looks"
                    st.write(f"### {category.capitalize()}")
                    for item in items:
                        if 'color' in item:
                            st.markdown(f"<div style='background-color: rgb{item['color']}; width: 50px; height: 50px; display: inline-block; margin: 5px;'></div>", unsafe_allow_html=True)
                            st.write(item["name"])

            # Display makeup looks
            if "makeup_looks" in recommendations:
                st.write("### Suggested Makeup Looks")
                for look in recommendations["makeup_looks"]:
                    st.write(f"**{look['name']}**: {look['description']}")

# Function to visualize palettes
def display_palettes(skin_tone):
    if skin_tone is not None:
        st.write(f"Detected Skin Tone (RGB): {skin_tone} ({get_color_name(skin_tone)})")

        # Dropdown for season selection
        season = st.selectbox("Select a season to view recommended colors:", [
            "Light Spring", "True Spring", "Warm Spring", "Light Summer", "True Summer", "Soft Summer",
            "Soft Autumn", "True Autumn", "Deep Autumn", "Deep Winter", "True Winter", "Cool Winter"
        ])
        seasonal_palette = get_seasonal_palette(skin_tone, season)

        st.subheader(f"Colors Recommended for {season}")
        for color in seasonal_palette:
            st.markdown(f"<div style='background-color: rgb{color}; width: 50px; height: 50px; display: inline-block; margin: 5px;'></div>", unsafe_allow_html=True)

        display_makeup_recommendations(skin_tone)

# Streamlit app
st.title("Automatic Skin Tone Detection and Color Recommendations")
st.write("Capture a photo to detect your skin tone and get personalized color recommendations.")

# Camera Input
image_file = st.camera_input("Take a picture")

if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    skin_tone = detect_skin_tone(image)

    display_palettes(skin_tone)
