import cv2
import dlib
import numpy as np

# This map tells us which points to connect to draw the lines, the dlib model returns 68 individual points (x, y)
# For example to draw the jawline, connect point 0 to point 1, then point 1 to point 2, and so on, up to point 16
LANDMARK_MAP = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_tip": list(range(31, 36)),
    "right_eye": list(range(36, 42)) + [36],
    "left_eye": list(range(42, 48)) + [42],
    "outer_mouth": list(range(48, 60)) + [48],
    "inner_mouth": list(range(60, 68)) + [60]
}

# --- Initialize dlib ---
try:
    # dlib's built-in face detector
    detector = dlib.get_frontal_face_detector()
    # pre-trained landmark predictor model
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError:
    print("Error: Could not find 'shape_predictor_68_face_landmarks.dat'")
    print("Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()


def get_landmark_image(frame, target_size=(256, 256)):
    """
    Detects landmarks in a frame and draws them onto a new, blank image.
    """
    # Create a blank image (all black)
    landmark_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Convert frame to grayscale for dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    if not faces:
        # If no face is found, return the blank image
        return landmark_image

    # --- Use the first face found ---
    face = faces[0]
    shape = predictor(gray, face)

    # Get coords of the bounding box for scaling
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

    # We need to scale the landmarks from the webcam frame
    # to fit our target 256x256 image.

    # 1. Get landmarks as a numpy array
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # 2. Normalize coordinates relative to the face bounding box
    # This makes the position independent
    coords_normalized = (coords - [x, y]) / [w, h]

    # 3. Scale normalized coords to the target image size
    # We add a small padding (0.9) to ensure it fits
    coords_scaled = (coords_normalized * [target_size[0] * 0.9, target_size[1] * 0.9]
                     + [target_size[0] * 0.05, target_size[1] * 0.05])  # Add a 5% margin
    coords_scaled = coords_scaled.astype("int")

    # --- Draw the landmarks ---
    for group in LANDMARK_MAP.values():
        points = coords_scaled[group]
        # 'isClosed=False' for open lines like jaw and eyebrows
        is_closed = group[0] == group[-1]
        cv2.polylines(landmark_image, [points], isClosed=is_closed, color=(0, 0, 255), thickness=2)

    return landmark_image

def get_face_with_landmarks(frame, target_size=(256, 256)):
    """
    Detects a face in the frame, crops it, resizes it to target_size,
    and then draws the facial landmarks on it.
    """

    # --- Create a fallback blank image ---
    # This is returned if no face is detected.
    fallback_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # --- Detect Face ---
    # Convert frame to grayscale for dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    if not faces:
        # If no face is found, return the blank fallback image
        return fallback_image

    # --- Use the first face found ---
    face = faces[0]
    shape = predictor(gray, face)

    # Get coords of the bounding box
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())

    # --- 1. Create the new canvas ---
    # Crop the exact face bounding box from the original frame
    # Add a check to prevent crashing if the box is out of bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

    face_crop = frame[y1:y2, x1:x2]

    # Check if crop is valid (sometimes dlib gives empty boxes)
    if face_crop.size == 0:
        return fallback_image

    # Resize this cropped face to be our canvas
    # This is what we will draw on
    output_image = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_AREA)

    # --- 2. Get and scale landmarks ---
    # (This logic is identical to before)

    # 1. Get landmarks as a numpy array (relative to the full frame)
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # 2. Normalize coordinates (relative to the face bounding box)
    #    (coords - [x, y]) makes them relative to the top-left corner of the box
    #    / [w, h] scales them to a [0, 1] range inside the box
    coords_normalized = (coords - [x, y]) / [w, h]

    # 3. Scale normalized coords to the target image size
    #    This maps the [0, 1] coordinates to the [0, 256] pixel range
    #    (The padding logic is kept to match the original function's output)
    coords_scaled = (coords_normalized * [target_size[0] * 0.9, target_size[1] * 0.9]
                     + [target_size[0] * 0.05, target_size[1] * 0.05])
    coords_scaled = coords_scaled.astype("int")

    # --- 3. Draw the landmarks ---
    # (This logic is identical, but uses the new 'output_image' canvas)
    for group in LANDMARK_MAP.values():
        points = coords_scaled[group]
        is_closed = group[0] == group[-1]

        # *** THIS IS THE MAIN CHANGE ***
        # Draw on the 'output_image' (the face) instead of 'landmark_image' (the black)
        cv2.polylines(output_image, [points], isClosed=is_closed, color=(255, 255, 255), thickness=1)

    # Return the face image with landmarks drawn on it
    return output_image