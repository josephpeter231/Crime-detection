import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound

# Load the model and labels
model = load_model("old.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def play_alarm_sound():
    """Plays an alarm sound."""
    playsound('alarm.mp3')

def preprocess_frame(frame):
    """Preprocess a single video frame for prediction."""
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
    frame = (frame / 127.5) - 1  # Normalize the frame
    return frame

def main():
    # Open the webcam (0 is the default camera index)
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Display the video feed
        cv2.imshow("Video Feed", frame)

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Predict using the model
        prediction = model.predict(processed_frame, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Get the class label
        confidence_score = prediction[0][index] * 100

        # Display the prediction and confidence score
        print(f"Prediction: {class_name}, Confidence: {confidence_score:.2f}%")

        # Trigger an alarm if the confidence score is high enough
        if confidence_score > 80:  # Adjust the threshold as needed
            print("ALARM! Match detected!")
            play_alarm_sound()

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
