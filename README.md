# Overview of the Skin Tone Detection and Color Recommendations Project

### Project Summary
This project aims to create a user-friendly application that detects the user's skin tone and provides personalized recommendations for makeup and clothing colors. The application uses machine learning, color theory, and advanced image processing techniques to ensure accurate color suggestions, tailored for different skin tones and seasonal preferences.. 

### Features Implemented
1. **Skin Tone Detection**:
   - The application allows users to either upload an image or capture a real-time picture using their webcam.
   - A Haar Cascade model is used to detect the user's face, and K-means clustering is used to extract the dominant skin tone from the facial region. The brightest skin tone is selected as the representative skin tone.

2. **Clothing Color Recommendations**:
   - Based on the user's skin tone, a set of clothing colors is recommended.
   - The user can also select the ongoing season (Spring, Summer, Autumn, Winter), and the clothing colors are adjusted based on seasonal preferences.
   - Recommendations include a variety of color theory-based suggestions: complementary, analogous, and triadic colors.

3. **Makeup Color Recommendations**:
   - The application provides makeup recommendations for foundation, blush, lipstick, and eyeshadow.
   - Foundation shades are suggested as the closest match along with slightly lighter and darker variations.
   - Blush, lipstick, and eyeshadow colors are suggested based on warm, cool, or neutral undertones.

4. **Color Visualization**:
   - Recommended colors are displayed as visual color swatches instead of numeric RGB values, making it easier for users to visualize the suggestions.

### Technical Approach
- **Image Processing**:
  - OpenCV is used for image preprocessing, including face detection and color conversion to HSV (hue, saturation, value) space.
  - K-means clustering is applied to extract dominant colors from the face region.

- **Seasonal Recommendations**:
  - The user selects the current season, and the application provides season-appropriate clothing colors.
  - Predefined seasonal colors are used for Spring, Summer, Autumn, and Winter, making the clothing recommendations more contextually relevant.

- **Streamlit Interface**:
  - The application uses Streamlit to create an interactive web interface where users can upload images, take real-time photos, and view color recommendations.
  - Users can easily interact with the tool to obtain tailored color suggestions.

### Current Challenges
- **Accuracy of Skin Tone Detection**: The use of Haar Cascade and basic K-means clustering provides moderate accuracy but may need further refinement for better skin tone classification.
- **Makeup Recommendations Consistency**: The lipstick and foundation colors sometimes produce incorrect recommendations, such as dark shades for foundation or unusual shades for lipstick.

### Next Steps for Enhancement
- **Improve Skin Tone Detection**: Implement a deep learning model for more accurate face and skin tone detection.
- **Seasonal Recommendations Based on User Data**: Instead of static recommendations, provide more dynamic and personalized suggestions.
- **Interactive User Feedback**: Add a feedback system where users can provide input on the accuracy of the recommendations, which can be used to improve the model.

![image](https://github.com/user-attachments/assets/519344df-30ad-4c26-8b8f-0fc9aad13f23)

- **Virtual Try-On**: Implement an AR-based feature for users to virtually try on clothing and makeup colors.

This project is intended to provide accessible solution for users to find personalized fashion and makeup color suggestions. It combines machine learning, image processing, and color theory to make the process intuitive and engaging.

