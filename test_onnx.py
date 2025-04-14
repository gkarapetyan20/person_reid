import cv2
import numpy as np
import time
import onnxruntime
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import torch.nn as nn
import torch

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Path to the ONNX model
onnx_model_path = '/home/aimaster/person_reid/person_re.onnx'

# Load the ONNX model using ONNX Runtimebn
session = onnxruntime.InferenceSession(onnx_model_path)

# Image preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    img = img.numpy()  # Convert to numpy array for ONNX model
    return img

def compute_similarity(image1_path, image2_path):
    # Load and preprocess images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Prepare the inputs for the model
    inputs = {
        'person1': img1,
        'person2': img2
    }

    # Run the model to get the similarity score
    outputs = session.run(None, inputs)
    similarity = outputs[0][0][0]  # Extract the similarity score from the output

    return similarity

# Path to the CSV file
csv_file = 'test_images/images.csv'  
data = pd.read_csv(csv_file)

y_true = []
y_pred = []
similarities = []  
threshold = 0.5  # You can set a threshold to decide if the images are similar or not
bce_values = []
criterion = nn.BCELoss()

# Iterate over the dataset
for idx, row in data.iterrows():
    image1_path = os.path.join('test_images/Folder_1', row['Folder_1'])
    image2_path = os.path.join('test_images/Folder_2', row['Folder_2'])
    actual_label = row['actual']

    # Compute similarity between the two images
    similarity = compute_similarity(image1_path, image2_path)

    # Compute BCE loss (Binary Cross Entropy) manually
    # similarity_tensor = np.array([similarity], dtype=np.float32)  # Convert to NumPy array
    # actual_label_tensor = np.array([actual_label], dtype=np.float32)  # Convert to NumPy array
    similarity_tensor = torch.tensor([similarity], dtype=torch.float32)  # Shape: (1,)
    actual_label_tensor = torch.tensor([actual_label], dtype=torch.float32)  # Shape: (1,)

    BCE = criterion(similarity_tensor, actual_label_tensor)
    bce_values.append(BCE.item())

    y_true.append(actual_label)
    similarities.append(similarity)  # Store the similarity for MAP

min_bce = min(bce_values)
max_bce = max(bce_values)
normalized_bce_values = [(bce - min_bce) / (max_bce - min_bce) for bce in bce_values]

# Predict based on the normalized BCE values
y_pred = []  # Ensure this list starts empty
print(normalized_bce_values)
for idx, norm_bce in enumerate(normalized_bce_values):
    if norm_bce > 0.3:  # Use a threshold suited for normalized values
        predicted_label = 1
    else:
        predicted_label = 0
    y_pred.append(predicted_label)

# Print true and predicted labels for inspection
for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
    print(f"Index: {idx}, True Label: {true_label}, Predicted Label: {pred_label}")

# Calculate precision, recall, f1-score, and accuracy
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Print the evaluation metrics
print(f"\nEvaluation Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
