import cv2
import numpy as np
import time
import onnxruntime
from PIL import Image
from torchvision import transforms

# Path to the ONNX model
onnx_model_path = '/home/aimaster/person_reid/person_re.onnx'

# Load the ONNX model using ONNX Runtime
session = onnxruntime.InferenceSession(onnx_model_path)

# Image preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(image_path):
    """Load and preprocess an image for the ONNX model."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((480, 640))  # Resize the image to 240x240
    img = transform(img).unsqueeze(0).numpy()  # Add batch dimension and convert to numpy array
    return img

def compute_similarity(img1, img2):
    """Compute the similarity using the ONNX model."""
    inputs = {
        'person1': img1,
        'person2': img2
    }
    outputs = session.run(None, inputs)
    similarity = outputs[0][0][0]  # Extract the similarity score from the output
    return similarity

# Load the first image (reference image)
reference_image_path = "/home/aimaster/person_reid/Untitled0.jpeg"
reference_img = load_image(reference_image_path)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize FPS calculation
prev_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.5  # Normalized threshold

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize the webcam frame to 240x240
    frame_resized = cv2.resize(frame, (480, 640))

    # Convert OpenCV BGR frame to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = transform(frame_pil).unsqueeze(0).numpy()

    # Compute similarity
    similarity = compute_similarity(reference_img, frame_tensor)

    # Normalize similarity score
    normalized_similarity = (similarity - 0) / (1 - 0)  # Normalizing to [0, 1], assuming similarity is in [0, 1]

    # Determine color based on normalized similarity threshold
    color = (0, 255, 0) if normalized_similarity >= threshold else (0, 0, 255)  # Green if similarity >= threshold, red otherwise

    # Display similarity score and normalized similarity on the frame
    cv2.putText(frame_resized, f"Similarity: {similarity:.2f}", (50, 50), font, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame_resized, f"Norm Sim: {normalized_similarity:.2f}", (50, 100), font, 1, color, 2, cv2.LINE_AA)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame_resized, f"FPS: {fps:.2f}", (50, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame with annotations
    cv2.imshow('Webcam Feed', frame_resized)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
