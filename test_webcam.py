import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import time
from model.backbone import SiameseNetwork

# Load the Siamese network model
model = SiameseNetwork(backbone="resnet50")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'checkpoint/siamese_network_weights_epoch_20.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# BCE loss criterion
criterion = nn.BCELoss()

# Image preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(image_path):
    """Load and preprocess an image for the model."""
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

def compute_similarity(img1, img2):
    """Compute the similarity and BCE loss between two images."""
    img1 = img1.to(device)
    img2 = img2.to(device)

    with torch.no_grad():
        similarity = model(img1, img2)
    
    return similarity.item()

# Load the first image (reference image)
reference_image_path = "Here give image path"
reference_img = load_image(reference_image_path).to(device)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize FPS calculation
prev_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.3  # Normalized BCE threshold

bce_values = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert OpenCV BGR frame to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

    # Compute similarity
    similarity = compute_similarity(reference_img, frame_tensor)
    similarity_tensor = torch.tensor([similarity], dtype=torch.float32)  # Shape: (1,)
    actual_label_tensor = torch.tensor([1.0], dtype=torch.float32)  # Assuming the actual label is positive for comparison

    # Compute BCE loss
    BCE = criterion(similarity_tensor, actual_label_tensor)
    bce_values.append(BCE.item())

    # Normalize BCE loss
    if len(bce_values) > 1:
        min_bce = min(bce_values)
        max_bce = max(bce_values)
        normalized_bce = (BCE.item() - min_bce) / (max_bce - min_bce + 1e-6)  # Adding epsilon to avoid division by zero
    else:
        normalized_bce = 0.0

    # Determine color based on normalized threshold
    color = (0, 255, 0) if normalized_bce <= threshold else (0, 0, 255)  # Green if low BCE (high similarity), red otherwise

    # Display similarity score and normalized BCE on the frame
    cv2.putText(frame, f"Similarity: {similarity:.2f}", (50, 50), font, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Norm BCE: {normalized_bce:.2f}", (50, 100), font, 1, color, 2, cv2.LINE_AA)

    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (50, 150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame with annotations
    cv2.imshow('Webcam Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
