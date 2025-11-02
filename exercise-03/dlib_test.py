import cv2
from utils import get_face_with_landmarks, get_landmark_image

print("Starting live demo... Press 'q' to quit.")

# Initialize the webcam. 0 is usually the default built-in webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam. Check if it is connected.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip the frame horizontally. This creates a more intuitive
    # "mirror" effect for the user.
    frame = cv2.flip(frame, 1)

    # Display the resulting frame
    # landmark_image = get_face_with_landmarks(frame)
    landmark_image = get_landmark_image(frame)
    cv2.imshow('Landmark Image', landmark_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
