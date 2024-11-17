import os
import cv2
from deepface import DeepFace
import webbrowser

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture object to read from the webcam using DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Define playlists for different emotions
playlists = {
    'happy': 'https://www.youtube.com/watch?v=Km2k94sZczU',
    'sad': 'https://www.youtube.com/watch?v=SQ1ED8-tBpE',
    'angry': 'https://www.youtube.com/watch?v=pJAXt1D68IE',
    'neutral': 'https://www.youtube.com/watch?v=Km2k94sZczU',
    'surprise': 'https://www.youtube.com/watch?v=Iz0_OC4Y-dc',
    'fear': 'https://www.youtube.com/watch?v=Iz0_OC4Y-dc',
    'disgust': 'https://www.youtube.com/watch?v=Iz0_OC4Y-dc'
}

# Flag to track if emotion is detected
emotion_detected = False

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    
    # Check if the frame is captured correctly
    if not ret:
        print("Failed to capture image")
        break
    
    # Convert the frame to grayscale as face detection works better on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face region of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Use DeepFace to analyze the face and predict the emotion
        result = DeepFace.analyze(roi_color, actions=['emotion'], enforce_detection=False)
        
        # Check if result is a list and get the first element
        if isinstance(result, list):
            result = result[0]
        
        # Get the dominant emotion
        dominant_emotion = result['dominant_emotion']

        print('Emotion is: ',dominant_emotion)
        
        # Put the emotion text above the rectangle
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Open the corresponding playlist in the web browser
        if dominant_emotion in playlists and not emotion_detected:
            webbrowser.open(playlists[dominant_emotion])
            emotion_detected = True  # Set the flag to stop further scanning
    
    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)
    
    # Break the loop if 'q' key is pressed or emotion is detected
    if cv2.waitKey(1) & 0xFF == ord('q') or emotion_detected:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
