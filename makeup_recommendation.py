import cv2
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
from PIL import Image
import tempfile

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Makeup recommendations database (updated with foundation colors based on RGB guidelines)
makeup_recommendations = {
    "fair": {
        "foundation": [
            {"name": "Ivory", "color": (230, 225, 220)},
            {"name": "Porcelain", "color": (240, 235, 230)},
            {"name": "Light Neutral", "color": (220, 215, 210)},
            {"name": "Fair Beige", "color": (245, 240, 235)},
            {"name": "Alabaster", "color": (250, 245, 240)}
        ],
        "blush": [
            {"name": "Soft Pink", "color": (255, 182, 193)},
            {"name": "Rose", "color": (255, 174, 185)},
            {"name": "Blushing Pink", "color": (255, 192, 203)}
        ],
        "lipstick": [
            {"name": "Peach", "color": (255, 160, 122)},
            {"name": "Light Pink", "color": (255, 182, 193)},
            {"name": "Nude Pink", "color": (240, 180, 170)}
        ]
    },
    "light": {
        "foundation": [
            {"name": "Light Beige", "color": (230, 220, 200)},
            {"name": "Natural Beige", "color": (220, 205, 190)},
            {"name": "Light Yellow Undertone", "color": (225, 215, 190)},
            {"name": "Vanilla", "color": (240, 225, 210)},
            {"name": "Buff Beige", "color": (235, 215, 200)}
        ],
        "blush": [
            {"name": "Rose", "color": (255, 105, 180)},
            {"name": "Coral", "color": (255, 127, 80)},
            {"name": "Petal Pink", "color": (255, 182, 193)}
        ],
        "lipstick": [
            {"name": "Coral", "color": (255, 69, 0)},
            {"name": "Light Red", "color": (255, 99, 71)},
            {"name": "Peachy Nude", "color": (255, 160, 122)}
        ]
    },
    "medium": {
        "foundation": [
            {"name": "Warm Beige", "color": (200, 195, 190)},
            {"name": "Honey Beige", "color": (195, 180, 160)},
            {"name": "Medium Neutral", "color": (190, 180, 170)},
            {"name": "Golden Beige", "color": (205, 185, 165)},
            {"name": "Tan Beige", "color": (210, 190, 175)}
        ],
        "blush": [
            {"name": "Apricot", "color": (255, 140, 105)},
            {"name": "Peach", "color": (255, 218, 185)},
            {"name": "Sunset Glow", "color": (255, 165, 135)}
        ],
        "lipstick": [
            {"name": "Mauve", "color": (199, 21, 133)},
            {"name": "Berry", "color": (186, 85, 211)},
            {"name": "Copper Rose", "color": (205, 92, 92)}
        ]
    },
    "tan": {
        "foundation": [
            {"name": "Honey", "color": (180, 160, 140)},
            {"name": "Caramel", "color": (175, 140, 120)},
            {"name": "Tan Yellow Undertone", "color": (180, 165, 140)},
            {"name": "Warm Honey", "color": (190, 155, 130)},
            {"name": "Golden Tan", "color": (200, 170, 150)}
        ],
        "blush": [
            {"name": "Peachy Bronze", "color": (205, 92, 92)},
            {"name": "Terracotta", "color": (210, 105, 30)},
            {"name": "Golden Apricot", "color": (210, 120, 90)}
        ],
        "lipstick": [
            {"name": "Warm Red", "color": (233, 150, 122)},
            {"name": "Terracotta", "color": (178, 76, 57)},
            {"name": "Burnt Sienna", "color": (204, 85, 0)}
        ]
    },
    "deep": {
        "foundation": [
            {"name": "Mocha", "color": (150, 110, 100)},
            {"name": "Espresso", "color": (120, 85, 75)},
            {"name": "Deep Neutral", "color": (140, 120, 110)},
            {"name": "Chestnut", "color": (130, 90, 70)},
            {"name": "Cocoa", "color": (115, 80, 70)}
        ],
        "blush": [
            {"name": "Raisin", "color": (165, 42, 42)},
            {"name": "Plum", "color": (142, 69, 133)},
            {"name": "Berry Wine", "color": (123, 45, 68)}
        ],
        "lipstick": [
            {"name": "Deep Plum", "color": (128, 0, 128)},
            {"name": "Burgundy", "color": (128, 0, 32)},
            {"name": "Chocolate Cherry", "color": (85, 25, 25)}
        ]
    }
}

def determine_skin_category(avg_color):
    """Determine the skin category based on the average color."""
    r, g, b = avg_color
    if r > 200 and g > 180 and b > 170:
        return "fair"
    elif r > 180 and g > 150 and b > 130:
        return "light"
    elif r > 160 and g > 130 and b > 110:
        return "medium"
    elif r > 120 and g > 100 and b > 80:
        return "tan"
    else:
        return "deep"

def extract_dominant_skin_tone(image):
    # Convert the image to RGB (if needed)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert the image to numpy array
    frame = np.array(image)
    
    # Convert the frame to BGR (OpenCV format)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]
        
        # Convert the face region to HSV color space
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        
        # Create a mask for skin detection
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply the mask to extract skin pixels
        skin_pixels = cv2.bitwise_and(face_roi, face_roi, mask=mask)
        
        # Reshape the skin region into a 2D array of RGB values
        reshaped_skin = skin_pixels.reshape((-1, 3))
        reshaped_skin = reshaped_skin[~np.all(reshaped_skin == 0, axis=1)]  # Remove black pixels
        
        if len(reshaped_skin) > 0:  # Ensure there are enough skin pixels
            # Apply K-means clustering to find the dominant skin tone
            kmeans = KMeans(n_clusters=1, random_state=0)
            kmeans.fit(reshaped_skin)
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            
            # Determine the skin category
            skin_category = determine_skin_category(dominant_color)
            
            # Get makeup recommendations
            recommendations = makeup_recommendations.get(skin_category, {})
            return dominant_color, recommendations
    
    return None, {}

def capture_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        # Convert the captured frame to an image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    else:
        return None

def main():
    st.title("Skin Tone Detection and Makeup Recommendation")
    st.write("Upload an image or take a real-time photo to detect your skin tone and get personalized makeup recommendations.")
    
    # Option to take a real-time photo
    if st.button("Take a Photo"):
        captured_image = capture_photo()
        if captured_image is not None:
            st.image(captured_image, caption='Captured Image', use_column_width=True)
            # Extract dominant skin tone and get makeup recommendations
            dominant_color, recommendations = extract_dominant_skin_tone(captured_image)
            if dominant_color is not None:
                st.write(f"Dominant Skin Tone (RGB): {dominant_color}")
                
                # Display makeup recommendations with color swatches
                st.write("### Makeup Recommendations:")
                
                st.write("#### Foundation:")
                for foundation in recommendations.get("foundation", []):
                    st.markdown(f'<div style="background-color: rgb{foundation["color"]}; width: 100px; height: 50px;"></div>', unsafe_allow_html=True)
                    st.write(foundation["name"])
                
                st.write("#### Blush:")
                for blush in recommendations.get("blush", []):
                    st.markdown(f'<div style="background-color: rgb{blush["color"]}; width: 100px; height: 50px;"></div>', unsafe_allow_html=True)
                    st.write(blush["name"])
                
                st.write("#### Lipstick:")
                for lipstick in recommendations.get("lipstick", []):
                    st.markdown(f'<div style="background-color: rgb{lipstick["color"]}; width: 100px; height: 50px;"></div>', unsafe_allow_html=True)
                    st.write(lipstick["name"])
            else:
                st.write("No face detected or unable to determine skin tone. Please try again.")
        else:
            st.write("Failed to capture photo. Please try again.")
    
    # Option to upload an image
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Extract dominant skin tone and get makeup recommendations
        dominant_color, recommendations = extract_dominant_skin_tone(image)
        
        if dominant_color is not None:
            st.write(f"Dominant Skin Tone (RGB): {dominant_color}")
            
            # Display makeup recommendations with color swatches
            st.write("### Makeup Recommendations:")
            
            st.write("#### Foundation:")
            for foundation in recommendations.get("foundation", []):
                st.markdown(f'<div style="background-color: rgb{foundation["color"]}; width: 100px; height: 50px;"></div>', unsafe_allow_html=True)
                st.write(foundation["name"])
            
            st.write("#### Blush:")
            for blush in recommendations.get("blush", []):
                st.markdown(f'<div style="background-color: rgb{blush["color"]}; width: 100px; height: 50px;"></div>', unsafe_allow_html=True)
                st.write(blush["name"])
            
            st.write("#### Lipstick:")
            for lipstick in recommendations.get("lipstick", []):
                st.markdown(f'<div style="background-color: rgb{lipstick["color"]}; width: 100px; height: 50px;"></div>', unsafe_allow_html=True)
                st.write(lipstick["name"])
        else:
            st.write("No face detected or unable to determine skin tone. Please try a different image.")

if __name__ == "__main__":
    main()
