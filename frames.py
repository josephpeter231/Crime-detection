import cv2
import os

def video_to_frames(video_path):
    # Create 'frames' directory if it doesn't exist
    if not os.path.exists('frames_harrasment'):
        os.makedirs('frames_harrasment')
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save each frame as an image
        cv2.imwrite(f'frames_harrasment/frame_{count:04d}.jpg', frame)
        count += 1
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_to_frames('harrasment.mp4')
