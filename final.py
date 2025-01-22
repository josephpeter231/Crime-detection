from keras.models import load_model
import cv2
import numpy as np
import time
import pygame
from pymongo import MongoClient
from datetime import datetime

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the trained model
model = load_model("keras_Model.h5", compile=False)

# Load the class labels
class_names = open("labels.txt", "r").readlines()

# Initialize the webcam (0 for default camera, 1 for external)
camera = cv2.VideoCapture(0)

# Initialize pygame mixer for sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.mp3")

# MongoDB setup
uri = "mongodb+srv://josephpeterjece2021:AJ9Hg6xTtQBUCoGr@cluster1.xaacunv.mongodb.net/CrimeDetection?retryWrites=true&w=majority"
client = MongoClient(uri)
db = client["CrimeDetection"]
collection = db["CrimeDetails"]

last_checked_time = time.time()

while True:
    # Capture frame from the webcam
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the image to be larger for display
    image_resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

    # Display the webcam feed
    cv2.imshow("Webcam Image", image_resized)

    # Check every 5 seconds
    if time.time() - last_checked_time >= 5:
        last_checked_time = time.time()

        # Prepare the image for prediction
        image_input = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_array = np.asarray(image_input, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1  # Normalize to [-1, 1]

        # Make a prediction
        prediction = model.predict(image_array)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Print the result
        print(f"Class: {class_name}, Confidence Score: {confidence_score * 100:.2f}%")

        # Check for crime detection with confidence above 90%
        if confidence_score > 0.9:
            alarm_sound.play()

            # Store the details in MongoDB
            crime_detail = {
                "class_name": class_name,
                "confidence_score": round(confidence_score * 100, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            collection.insert_one(crime_detail)
            print("Crime details saved to MongoDB:", crime_detail)

    # Exit on 'Esc' key press
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close all windows
camera.release()
cv2.destroyAllWindows()
