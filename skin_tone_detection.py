import cv2
import numpy as np
from sklearn.cluster import KMeans

# Loading Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initializing variables
dominant_colors = []
max_frames = 10  # Number of frames to average over
last_output_color = None
change_threshold = 10  # Minimum change in color to update output

def color_difference(color1, color2):
    """Calculating Euclidean distance between two colors."""
    return np.sqrt(np.sum((color1 - color2) ** 2))

def extract_dominant_skin_tone(frame):
    global dominant_colors, last_output_color
    
    # Converting frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
        # Extracting face region
        face_roi = frame[y:y + h, x:x + w]
        
        # Converting face region to HSV color space
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        
        # Defining skin color range in HSV
        lower_skin = np.array([0, 10, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        
        # Creating mask for skin detection
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Applying mask to extract skin pixels
        skin_pixels = cv2.bitwise_and(face_roi, face_roi, mask=mask)
        
        # Reshaping skin region into 2D array of RGB values
        reshaped_skin = skin_pixels.reshape((-1, 3))
        reshaped_skin = reshaped_skin[~np.all(reshaped_skin == 0, axis=1)]  # Removing black pixels
        
        if len(reshaped_skin) > 0:  # Ensuring there are enough skin pixels
            # Applying K-means clustering to find dominant skin tone
            kmeans = KMeans(n_clusters=1, random_state=0)
            kmeans.fit(reshaped_skin)
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            
            # Adding to list of dominant colors
            dominant_colors.append(dominant_color)
            
            # Keeping only last 'max_frames' colors
            if len(dominant_colors) > max_frames:
                dominant_colors.pop(0)
            
            # Calculating average dominant color
            avg_color = np.mean(dominant_colors, axis=0).astype(int)
            
            # Checking for significant change
            if last_output_color is None or color_difference(avg_color, last_output_color) > change_threshold:
                last_output_color = avg_color
                
                # Displaying smoothed & updated dominant skin tone
                print(f"Updated Dominant Skin Tone (RGB): {avg_color}")
                swatch = np.zeros((100, 100, 3), dtype=np.uint8)
                swatch[:, :] = avg_color
                cv2.imshow("Dominant Skin Tone", swatch)
    
    return frame

# To open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extracting dominant skin tone
    processed_frame = extract_dominant_skin_tone(frame)
    
    # Displaying original frame
    cv2.imshow("Webcam Feed", processed_frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
