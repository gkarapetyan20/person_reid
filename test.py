import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from model.backbone import SiameseNetwork

model = SiameseNetwork(backbone="resnet50")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'checkpoint/siamese_network_weights_epoch_20.pth' 
model.load_state_dict(torch.load(model_path, map_location=device))  

model.to(device)  
model.eval()  

transform = transforms.Compose([
    transforms.ToTensor(),

])

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  
    return img

def compute_similarity(image1_path, image2_path):
    # Load and preprocess images
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

    # Move images to the same device as the model
    img1 = img1.to(device)
    img2 = img2.to(device)

    # Forward pass to get similarity
    with torch.no_grad():
        similarity = model(img1, img2)

    return similarity.item()



csv_file = 'test_images/images.csv'  
data = pd.read_csv(csv_file)

y_true = []
y_pred = []
similarities = []  
criterion = nn.BCELoss()

threshold = 0.5  # You can set a threshold to decide if the images are similar or not
bce_values = []
for idx, row in data.iterrows():
    image1_path = os.path.join('test_images/Folder_1', row['Folder_1'])
    image2_path = os.path.join('test_images/Folder_2', row['Folder_2'])
    actual_label = row['actual']

    # Compute similarity between the two images
    similarity = compute_similarity(image1_path, image2_path)
    similarity_tensor = torch.tensor([similarity], dtype=torch.float32)  # Shape: (1,)
    actual_label_tensor = torch.tensor([actual_label], dtype=torch.float32)  # Shape: (1,)

    # Compute BCE loss
    BCE = criterion(similarity_tensor, actual_label_tensor)
    bce_values.append(BCE.item())

    y_true.append(actual_label)
    similarities.append(similarity)  # Store the similarity for MAP

min_bce = min(bce_values)
max_bce = max(bce_values)
normalized_bce_values = [(bce - min_bce) / (max_bce - min_bce) for bce in bce_values]


y_pred = []  # Ensure this list starts empty
for idx, norm_bce in enumerate(normalized_bce_values):  # Ensure alignment with y_true
    if norm_bce > 0.3:  # Use a threshold suited for normalized values
        predicted_label = 1
    else:
        predicted_label = 0
    y_pred.append(predicted_label)


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






