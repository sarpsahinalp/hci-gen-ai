# --- 1. Import Necessary Libraries ---

import cv2  # OpenCV: For capturing webcam feed and all image processing
import torch  # PyTorch: For running our deep learning model
import numpy as np  # NumPy: For numerical operations and creating image arrays
import torchvision.transforms as transforms  # For pre-processing images into PyTorch tensors
from PIL import Image  # PIL (Pillow): For easy conversion between OpenCV images and PyTorch transforms

# --- 2. Import Your Custom Files ---

# Imports the get_landmark_image function from your utils.py file
from utils import get_landmark_image
# Imports the GeneratorUNet class definition from your model.py file
from model import GeneratorUNet

# --- 3. Configuration and Setup ---

# Set the path to your trained model file
# This is the .pth file you got from training pix2pix (Exercise 1)
GENERATOR_PATH = "generator.pth"

# The image size (height and width) your pix2pix model was trained on
IMAGE_SIZE = 256

# Automatically select the best available device: GPU ("cuda") if possible, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- 4. Load the Trained pix2pix Model ---

# Create an instance of the pix2pix Generator
# It takes 3-channel (RGB) images as input and outputs 3-channel (RGB) images
model = GeneratorUNet(in_channels=3, out_channels=3)

# Load the saved weights from your trained .pth file
# map_location=DEVICE ensures the model loads onto the correct device (GPU or CPU)
model.load_state_dict(torch.load(GENERATOR_PATH, map_location=DEVICE))

# Move the model to the selected device (GPU/CPU)
model.to(DEVICE)

# !!! IMPORTANT !!!
# Set the model to "evaluation mode"
# This disables layers like Dropout and Batch Normalization, which behave
# differently during training. This is crucial for getting correct predictions.
model.eval()

# --- 5. Define Image Pre-processing Transform ---

# This 'transform' pipeline must be EXACTLY the same as the one used
# to train your pix2pix model.
transform = transforms.Compose([
    # 1. Resize the input image to the size the model expects (256x256)
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # 2. Convert the image from a PIL Image to a PyTorch Tensor
    # This also scales pixel values from [0, 255] to [0.0, 1.0]
    transforms.ToTensor(),

    # 3. Normalize the tensor. The pix2pix repo normalizes images to the
    # range [-1, 1] instead of [0, 1].
    # This (value - 0.5) / 0.5 operation achieves that.
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# --- 6. Define Helper Function for Post-processing ---

def tensor_to_cv2(tensor):
    """
    Converts a PyTorch output tensor (from the model) back into an
    OpenCV-displayable image (NumPy array).
    """
    # 1. Get the tensor off the GPU and onto the CPU
    tensor = tensor.clone().detach().cpu()

    # 2. Un-normalize the tensor: from [-1, 1] back to [0, 1]
    tensor = (tensor * 0.5) + 0.5

    # 3. Convert from PyTorch's tensor format (C, H, W)
    #    to NumPy/OpenCV's image format (H, W, C)
    img = tensor.numpy()
    img = np.transpose(img, (1, 2, 0))

    # 4. Scale pixel values from [0.0, 1.0] to [0, 255]
    #    and change data type to 8-bit integer (standard for images)
    img = (img * 255).astype(np.uint8)

    # 5. Convert color from RGB (used by PIL/PyTorch)
    #    to BGR (used by OpenCV for display)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# --- 7. Main Webcam and Inference Loop ---

print("Starting live demo... Press 'q' to quit.")

# Initialize the webcam. 0 is usually the default built-in webcam.
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Cannot open webcam. Check if it is connected.")
    exit()

# Start an infinite loop to process video frames
while True:

    # --- A. Capture Frame ---

    # Read one frame from the webcam
    # 'ret' is a boolean (True/False) if the read was successful
    # 'frame' is the image itself (a NumPy array)
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip the frame horizontally. This creates a more intuitive
    # "mirror" effect for the user.
    frame = cv2.flip(frame, 1)

    # Resize the original webcam frame to our standard size for easy display
    frame_display = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

    # --- B. Get Landmark Image (Exercise 2) ---

    # Call your function from utils.py
    # This takes the full-size webcam frame and returns a 256x256
    # black image with the white landmarks drawn on it.
    # This 'landmark_image' is the INPUT to our pix2pix model.
    landmark_image = get_landmark_image(frame, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    # --- C. Pre-process for PyTorch ---

    # The 'transform' pipeline expects a PIL Image.
    # We must convert our OpenCV image (NumPy array) to a PIL Image.
    # Note: Convert from BGR (OpenCV) to RGB (PIL) first.
    pil_image = Image.fromarray(cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB))

    # Apply the transforms: Resize -> ToTensor -> Normalize
    # The .unsqueeze(0) adds a "batch dimension" (N=1)
    # Final shape is [1, 3, 256, 256], which the model expects.
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    # --- D. Run Model Inference (Exercise 3) ---

    # Use 'torch.no_grad()' to tell PyTorch not to calculate gradients.
    # This saves memory and is much, much faster.
    with torch.no_grad():
        # This is the "forward pass"
        # We feed the landmark tensor into the model...
        # ...and get a new tensor representing the generated face.
        output_tensor = model(input_tensor)

    # --- E. Post-process for Display ---

    # The output is a batch (size 1), so get the first item [0]
    # Then, convert the output tensor back to an OpenCV image
    generated_face = tensor_to_cv2(output_tensor[0])

    # --- F. Display the Results ---

    # Create a combined image to show everything side-by-side
    # np.hstack stacks arrays horizontally
    combined_view = np.hstack((frame_display, landmark_image, generated_face))

    # Show the combined view in an OpenCV window
    cv2.imshow("Deep-Fake Demo (Original | Landmarks | Generated) - Press 'q' to quit", combined_view)

    # --- G. Check for Quit Condition ---

    # Wait for 1 millisecond for a key press.
    # If the pressed key is 'q', break out of the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 8. Cleanup ---

print("Shutting down...")
# Release the webcam so other applications can use it
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()