
# Skin Tone and Color Recommendations App

This application detects a user's skin tone from an uploaded image, recommends seasonal color palettes, and provides personalized makeup suggestions based on their skin tone.

## Features
1. **Skin Tone Detection**:
   - Uses Mediapipe to detect facial landmarks and analyze the skin tone.
   - Matches the detected tone with a predefined dataset of skin tones.

2. **Seasonal Color Recommendations**:
   - Generates diverse and unique color palettes for each season (Spring, Summer, Autumn, Winter).
   - Displays color swatches with HEX values in a grid layout.

3. **Makeup Suggestions**:
   - Recommends foundation, blush, and lipstick shades that match the detected skin tone.
   - Optionally displays makeup recommendations based on user input.

4. **User-Friendly Interface**:
   - Interactive tabs for skin tone, seasonal colors, and makeup suggestions.
   - Clean and responsive design using Streamlit.

## Requirements
To run this application, install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/roshni-1/PalettePerfection-app.git
   ```
2. Navigate to the project directory:
   ```bash
   cd skin-tone-recommendation-app
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Upload an image or use the camera input to detect your skin tone.

## File Structure
- **app.py**: Main application file.
- **requirements.txt**: List of required Python libraries.
- **skintonedetailed.xlsx**: Dataset containing detailed skin tone information.
- **makeupdetailed.csv**: Dataset containing makeup product details.

## Datasets
- `skintonedetailed.xlsx`: Contains skin tone names, RGB values, HEX values, and categories.
- `makeupdetailed.csv`: Contains makeup product details, including RGB values and HEX codes for foundation, blush, and lipstick.

## Application

https://github.com/user-attachments/assets/76ed82cc-c017-4521-99a3-89f699b7f4dc

## Future Enhancements
- Adding event-based color recommendations.
- Include additional makeup categories and product filters.
- AR-Based Virtual Try-On.

## Acknowledgments
- **Mediapipe**: For facial landmark detection.
- **Streamlit**: For creating an interactive web application with attractive UI
- **Color Theory Resources**: For guiding seasonal palette generation.






