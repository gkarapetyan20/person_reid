import torch
import time
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
# from model.backbone import SiameseNetwork
# from gender_classification.test_single import check_gender
from CNN_with.test_single import check_gender
start_time = time.time()
import torchvision.models as models
from model.backbone import SiameseNetwork

model = SiameseNetwork(backbone="resnet50")

#ALEXNET-------------------
# class SiameseNetwork(nn.Module):
#     def __init__(self, backbone="vgg16"):
#         super().__init__()

#         if backbone not in models.__dict__:
#             raise Exception(f"No model named {backbone} exists in torchvision.models.")


#         # model = models.__dict__[backbone](pretrained=True)
#         # model = models.__dict__[backbone](weights=AlexNet_Weights.DEFAULT)
#         # model = models.alexnet(pretrained=True)
#         # self.backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
#         # out_features = list(model.modules())[-1].out_features

#         model = models.alexnet(pretrained=True)
#         features = nn.Sequential(*list(model.features), model.avgpool, nn.Flatten())
#         self.backbone = features
#         out_features = 256 * 6 * 6 
#         self.cls_head = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(out_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),

#             nn.Dropout(p=0.5),
#             nn.Linear(512, 64),
#             nn.BatchNorm1d(64),
#             nn.Sigmoid(),
#             nn.Dropout(p=0.5),

#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, img1, img2):
#         feat1 = self.backbone(img1)
#         feat2 = self.backbone(img2)

#         combined_features = feat1 * feat2
#         output = self.cls_head(combined_features)
#         return output


# # Load the Siamese Network model
# model = SiameseNetwork(backbone='alexnet')


#dENSENET -------------------------
# class SiameseNetwork(nn.Module):
#     def __init__(self, backbone="densenet121"):
#         super().__init__()

#         if backbone not in models.__dict__:
#             raise Exception(f"No model named {backbone} exists in torchvision.models.")

#         model = models.densenet121(pretrained=True)
#         self.backbone = nn.Sequential(
#             model.features,
#             nn.AdaptiveAvgPool2d((1, 1)),  # to get a fixed-size output
#             nn.Flatten()
#         )

#         # DenseNet121's output after AdaptiveAvgPool2d and Flatten is 1024
#         out_features = 1024  
#         self.cls_head = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(out_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),

#             nn.Dropout(p=0.5),
#             nn.Linear(512, 64),
#             nn.BatchNorm1d(64),
#             nn.Sigmoid(),
#             nn.Dropout(p=0.5),

#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, img1, img2):
#         feat1 = self.backbone(img1)
#         feat2 = self.backbone(img2)

#         combined_features = feat1 * feat2
#         output = self.cls_head(combined_features)
#         return output


# model = SiameseNetwork(backbone='densenet121')








# class SiameseNetwork(nn.Module):
#     def __init__(self, backbone="vgg16"):
#         super().__init__()

#         if backbone not in models.__dict__:
#             raise Exception(f"No model named {backbone} exists in torchvision.models.")

#         model = models.vgg16(pretrained=True)
#         features = nn.Sequential(*list(model.features), model.avgpool, nn.Flatten())
#         self.backbone = features
#         out_features = 512 * 7 * 7 
#         self.cls_head = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(out_features, 512),
#             nn.BatchNorm10d(512),
#             nn.ReLU(),

#             nn.Dropout(p=0.5),
#             nn.Linear(512, 64),
#             nn.BatchNorm1d(64),
#             nn.Sigmoid(),
#             nn.Dropout(p=0.5),

#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, img1, img2):
#         feat1 = self.backbone(img1)
#         feat2 = self.backbone(img2)

#         combined_features = feat1 * feat2
#         output = self.cls_head(combined_features)
#         return output
# model = SiameseNetwork(backbone='vgg16')



# class SiameseNetwork(nn.Module):
#     def __init__(self, backbone="regnet_y_400mf"):
#         super().__init__()

#         if backbone not in models.__dict__:
#             raise Exception(f"No model named {backbone} exists in torchvision.models.")

#         model = models.regnet_y_400mf(pretrained=True)
        
#         # RegNet parts before classification head
#         self.backbone = nn.Sequential(
#             model.stem,
#             model.trunk_output,
#             nn.AdaptiveAvgPool2d((1, 1)),  # global pooling
#             nn.Flatten()
#         )

#         out_features = model.fc.in_features  # typically 440 for regnet_y_400mf
#         self.cls_head = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(out_features, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),

#             nn.Dropout(p=0.5),
#             nn.Linear(512, 64),
#             nn.BatchNorm1d(64),
#             nn.Sigmoid(),
#             nn.Dropout(p=0.5),

#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, img1, img2):
#         feat1 = self.backbone(img1)
#         feat2 = self.backbone(img2)

#         combined_features = feat1 * feat2
#         output = self.cls_head(combined_features)
#         return output


# model = SiameseNetwork(backbone='regnet_y_400mf')











device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'checkpoint/siamese_network_weights_epoch_20.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_image(image_path):
    """Loads and preprocesses an image."""
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

def compute_similarity(image1_path, image2_path):
    """Computes similarity score between two images using the Siamese network."""
    img1 = load_image(image1_path).to(device)
    img2 = load_image(image2_path).to(device)

    with torch.no_grad():
        similarity = model(img1, img2)

    return similarity.item()

# Load test image pairs
csv_file = '/home/aimaster/person_reid/test_images/images.csv'
data = pd.read_csv(csv_file)

y_true = []
similarities = []
bce_values = []

criterion = nn.BCELoss()





bce_values = []
y_true = []
similarities = []

for idx, row in data.iterrows():
    image1_path = os.path.join('test_images/Folder_1', row['Folder_1'])
    image2_path = os.path.join('test_images/Folder_2', row['Folder_2'])
    actual_label = int(row['actual'])  # Ensure it's an integer (0 or 1)
    
    gender = check_gender(image1_path, image2_path)
    if gender:
        similarity = compute_similarity(image1_path, image2_path)
        similarity_tensor = torch.tensor([similarity], dtype=torch.float32)
        actual_label_tensor = torch.tensor([actual_label], dtype=torch.float32)

        # Compute BCE loss
        BCE = criterion(similarity_tensor, actual_label_tensor)
        bce_values.append(BCE.item())
        y_true.append(actual_label)
        similarities.append(similarity)
    else:
        similarities.append(0)
        y_true.append(0)  # Ensure y_true has a value
        bce_values.append(0)  # Append 0 to maintain consistent list length







# Normalize BCE values
min_bce, max_bce = min(bce_values), max(bce_values)
normalized_bce_values = [(bce - min_bce) / (max_bce - min_bce) for bce in bce_values]

# Generate predictions based on threshold
threshold = 0.4 # Adjust threshold if needed
y_pred = [1 if norm_bce > threshold else 0 for norm_bce in normalized_bce_values]

# Print individual predictions
for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
    print(f"Index: {idx}, True Label: {true_label}, Predicted Label: {pred_label}")

# Compute evaluation metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Print results
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# ========================== CMC and mAP Calculation ==========================

def compute_cmc_map(y_true, similarities):
    """
    Compute the CMC (Cumulative Matching Characteristic) and mAP (Mean Average Precision).

    :param y_true: List of true labels (0 or 1).
    :param similarities: List of similarity scores.
    :return: CMC curve (list), mAP score (float).
    """
    num_queries = len(y_true)
    cmc_curve = np.zeros(num_queries)  # CMC curve initialization
    average_precisions = []  # Store AP for each query

    # Sort indices by similarity scores in descending order
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_y_true = np.array(y_true)[sorted_indices]

    # Compute CMC curve
    for rank, label in enumerate(sorted_y_true):
        if label == 1:
            cmc_curve[rank:] = 1  # Mark all ranks from here as successful
            break

    # Compute mAP
    correct_ranks = np.where(sorted_y_true == 1)[0]
    if len(correct_ranks) > 0:
        precisions = [(i + 1) / (rank + 1) for i, rank in enumerate(correct_ranks)]
        average_precisions.append(np.mean(precisions))  # AP for this query

    mAP_score = np.mean(average_precisions) if average_precisions else 0.0

    return cmc_curve, mAP_score

# Compute CMC and mAP
cmc_curve, mAP_score = compute_cmc_map(y_true, y_pred)

# Print results
# print(f"\nCMC Curve: {cmc_curve}")
print(f"Mean Average Precision (mAP): {mAP_score:.4f}")
print(f"Process Time = {time.time() - start_time}")