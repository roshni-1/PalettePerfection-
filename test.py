## Streamlit  working 2 

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import re

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

skin_tone_data = pd.read_csv('skin_tone_with_palettes.csv')
makeup_recommendations = pd.read_csv('makeup_recommendations.csv')
clothing_recommendations = pd.read_csv('clothing_recommendations.csv')

def detect_face_and_skin_tone(image):
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        
        hsv_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
        
        skin_pixels = hsv_face[mask > 0]
        average_skin_tone = np.mean(skin_pixels, axis=0) if len(skin_pixels) > 0 else [0, 0, 0]
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        st.write(f"Average Skin Tone (HSV): {average_skin_tone}")
        
        skin_tone_data[['R', 'G', 'B']] = skin_tone_data['Approx. RGB (R,G,B)'].str.extract(r'\((\d+), (\d+), (\d+)\)').astype(float)
        skin_tone_data['Distance'] = ((skin_tone_data['R'] - average_skin_tone[0]) ** 2 + 
                                      (skin_tone_data['G'] - average_skin_tone[1]) ** 2 + 
                                      (skin_tone_data['B'] - average_skin_tone[2]) ** 2) ** 0.5
        closest_match = skin_tone_data.loc[skin_tone_data['Distance'].idxmin()]
        
        clothing_colors = clothing_recommendations[clothing_recommendations['Subcategory'] == closest_match['Subcategory']]['Clothing_Recommendations'].values
        makeup_colors = makeup_recommendations[makeup_recommendations['Subcategory'] == closest_match['Subcategory']]['Makeup_Recommendations'].values
        
        st.subheader('Clothing Color Recommendations:')
        clothing_colors_grid = []
        for color in clothing_colors:
            color_match = re.search(r'\((\d+), (\d+), (\d+)\)', color)
            if color_match:
                color_rgb = tuple(map(int, color_match.groups()))
                hex_color = '#%02x%02x%02x' % color_rgb
                clothing_colors_grid.append(f"<div style='width:50px; height:50px; background-color:{hex_color}; margin: 5px; display: inline-block;'></div>")
        st.markdown("".join(clothing_colors_grid), unsafe_allow_html=True)
        
        st.subheader('Makeup Color Recommendations:')
        makeup_types = ['Foundation', 'Eyeshadow', 'Blush', 'Lipstick']
        makeup_colors_grid = []
        for makeup_type, color in zip(makeup_types, makeup_colors):
            color_match = re.search(r'\((\d+), (\d+), (\d+)\)', color)
            if color_match:
                color_rgb = tuple(map(int, color_match.groups()))
                hex_color = '#%02x%02x%02x' % color_rgb
                makeup_colors_grid.append(f"<div style='display: flex; align-items: center; margin-bottom: 10px;'><span style='margin-right: 10px;'>{makeup_type}:</span><div style='width:50px; height:50px; background-color:{hex_color};'></div></div>")
        st.markdown("".join(makeup_colors_grid), unsafe_allow_html=True)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

st.title('Skin Tone Detection and Color Recommendations')

if st.button('Capture Real-Time Picture'):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cap.release()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            st.image(image, caption='Captured Image', use_column_width=True)
            
            result_img = detect_face_and_skin_tone(image)
            
            st.image(result_img, caption='Detected Faces and Skin Tone', use_column_width=True)
    else:
        st.error("Unable to access the camera.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    result_img = detect_face_and_skin_tone(image)
    
    st.image(result_img, caption='Detected Faces and Skin Tone', use_column_width=True)
