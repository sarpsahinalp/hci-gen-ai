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
    Detects a face in the frame, RESIZES THE ENTIRE FRAME,
    and then draws the SCALED landmarks on it.
    """

    # --- Get original frame size ---
    # We need this to calculate the scaling ratio
    orig_h, orig_w, _ = frame.shape

    # --- Create a fallback blank image ---
    fallback_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # --- Detect Face ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        return fallback_image

    # --- Use the first face found ---
    face = faces[0]
    shape = predictor(gray, face)

    # --- 2. Get landmarks (unscaled) ---
    coords = np.zeros((68, 2), dtype="float32") # Use float for scaling
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # --- 3. Scale the coordinates ---
    # NEW: Calculate the scaling ratios
    scale_x = target_size[0] / orig_w
    scale_y = target_size[1] / orig_h

    # NEW: Apply the scaling to all coordinates
    coords_scaled = coords.copy()
    coords_scaled[:, 0] = coords[:, 0] * scale_x
    coords_scaled[:, 1] = coords[:, 1] * scale_y

    # Convert to integer for drawing
    coords_scaled = coords_scaled.astype("int")

    # --- 4. Draw the scaled landmarks ---
    for group in LANDMARK_MAP.values():
        # Use the NEW scaled points
        points = coords_scaled[group]
        is_closed = group[0] == group[-1]

        cv2.polylines(fallback_image, [points], isClosed=is_closed, color=(255, 255, 255), thickness=1) # Changed to green

    # Return the 256x256 image with lines drawn on it
    return fallback_image

def get_face_with_landmarks(frame, target_size=(256, 256)):
    """
    Detects a face in the frame, RESIZES THE ENTIRE FRAME,
    and then draws the SCALED landmarks on it.
    """

    # --- Get original frame size ---
    # We need this to calculate the scaling ratio
    orig_h, orig_w, _ = frame.shape

    # --- Create a fallback blank image ---
    fallback_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # --- Detect Face ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        return fallback_image

    # --- Use the first face found ---
    face = faces[0]
    shape = predictor(gray, face)

    # --- 1. Create the new canvas ---
    # Resize the *entire* frame to be our canvas
    # This will squash the image, as in your original code.
    output_image = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    # --- 2. Get landmarks (unscaled) ---
    coords = np.zeros((68, 2), dtype="float32") # Use float for scaling
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # --- 3. Scale the coordinates ---
    # NEW: Calculate the scaling ratios
    scale_x = target_size[0] / orig_w
    scale_y = target_size[1] / orig_h

    # NEW: Apply the scaling to all coordinates
    coords_scaled = coords.copy()
    coords_scaled[:, 0] = coords[:, 0] * scale_x
    coords_scaled[:, 1] = coords[:, 1] * scale_y

    # Convert to integer for drawing
    coords_scaled = coords_scaled.astype("int")

    # --- 4. Draw the scaled landmarks ---
    for group in LANDMARK_MAP.values():
        # Use the NEW scaled points
        points = coords_scaled[group]
        is_closed = group[0] == group[-1]

        cv2.polylines(output_image, [points], isClosed=is_closed, color=(0, 255, 0), thickness=1) # Changed to green

    # Return the 256x256 image with lines drawn on it
    return output_image