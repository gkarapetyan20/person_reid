# import torch
# from torchvision import transforms, models
# from PIL import Image
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import pandas as pd
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# import numpy as np
# from model.backbone import SiameseNetwork

# model = SiameseNetwork(backbone="resnet50")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_path = 'checkpoint/siamese_network_weights_epoch_20.pth' 
# model.load_state_dict(torch.load(model_path, map_location=device))  

# model.to(device)  
# model.eval()  

# transform = transforms.Compose([
#     transforms.ToTensor(),

# ])

# def load_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     img = transform(img).unsqueeze(0)  
#     return img

# def compute_similarity(image1_path, image2_path):
#     # Load and preprocess images
#     img1 = load_image(image1_path)
#     img2 = load_image(image2_path)

#     # Move images to the same device as the model
#     img1 = img1.to(device)
#     img2 = img2.to(device)

#     # Forward pass to get similarity
#     with torch.no_grad():
#         similarity = model(img1, img2)

#     return similarity.item()



# csv_file = 'test_images/images.csv'  
# data = pd.read_csv(csv_file)

# y_true = []
# y_pred = []
# similarities = []  
# criterion = nn.BCELoss()

# threshold = 0.5  # You can set a threshold to decide if the images are similar or not
# bce_values = []
# for idx, row in data.iterrows():
#     image1_path = os.path.join('test_images/Folder_1', row['Folder_1'])
#     image2_path = os.path.join('test_images/Folder_2', row['Folder_2'])
#     actual_label = row['actual']

#     # Compute similarity between the two images
#     similarity = compute_similarity(image1_path, image2_path)
#     similarity_tensor = torch.tensor([similarity], dtype=torch.float32)  # Shape: (1,)
#     actual_label_tensor = torch.tensor([actual_label], dtype=torch.float32)  # Shape: (1,)

#     # Compute BCE loss
#     BCE = criterion(similarity_tensor, actual_label_tensor)
#     bce_values.append(BCE.item())

#     y_true.append(actual_label)
#     similarities.append(similarity)  # Store the similarity for MAP

# min_bce = min(bce_values)
# max_bce = max(bce_values)
# normalized_bce_values = [(bce - min_bce) / (max_bce - min_bce) for bce in bce_values]


# y_pred = []  # Ensure this list starts empty
# print(normalized_bce_values)
# for idx, norm_bce in enumerate(normalized_bce_values):  # Ensure alignment with y_true
#     if norm_bce > 0.45:  # Use a threshold suited for normalized values
#         predicted_label = 1
#     else:
#         predicted_label = 0
#     y_pred.append(predicted_label)


# for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
#     print(f"Index: {idx}, True Label: {true_label}, Predicted Label: {pred_label}")


# # Calculate precision, recall, f1-score, and accuracy
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)
# accuracy = accuracy_score(y_true, y_pred)

# # Print the evaluation metrics
# print(f"\nEvaluation Metrics:")
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Precision: {precision * 100:.2f}%")
# print(f"Recall: {recall * 100:.2f}%")
# print(f"F1 Score: {f1 * 100:.2f}%")




import torch
import time
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from model.backbone import SiameseNetwork
# from gender_classification.test_single import check_gender
from CNN_with.test_single import check_gender
start_time = time.time()

# Load the Siamese Network model
model = SiameseNetwork(backbone="resnet50")
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
csv_file = 'test_images/images.csv'
data = pd.read_csv(csv_file)

y_true = []
similarities = []
bce_values = []

criterion = nn.BCELoss()



# for idx, row in data.iterrows():
#     image1_path = os.path.join('/home/aimaster/re_identification/seamens_nn/test_label/Folder_1', row['Folder_1'])  # Provide the folder path for image 1
#     image2_path = os.path.join('/home/aimaster/re_identification/seamens_nn/test_label/Folder_2', row['Folder_2'])  # Provide the folder path for image 2
#     actual_label = row['actual']

#     gender = check_gender(image1_path , image2_path)
#     if gender:
#         auto = compare_images(autoencoder , image1_path ,image2_path , transform_autoencoder )
#         similarity = compute_similarity(image1_path, image2_path)
#         mean = 2 / (1/auto + 1/similarity)
#         # print(mean)
#         # predicted_label = 1 if similarity > threshold else 0
#         predicted_label = 1 if mean > threshold else 0

#         y_true.append(actual_label)
#         y_pred.append(predicted_label)
#         similarities.append(similarity)  

#         print(f"Image Pair: {row['Folder_1']} & {row['Folder_2']} - Similarity: {mean:.4f}, Predicted: {predicted_label}, Actual: {actual_label}")

#     else:
#         predicted_label == 0
#         y_true.append(actual_label)
#         y_pred.append(predicted_label)


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



#_____________________________________________________WIDTHOUT GENDER 

    # gender = check_gender(image1_path , image2_path)
    # if gender:

    # similarity = compute_similarity(image1_path, image2_path)
    # similarity_tensor = torch.tensor([similarity], dtype=torch.float32)
    # actual_label_tensor = torch.tensor([actual_label], dtype=torch.float32)

    #         # Compute BCE loss
    # BCE = criterion(similarity_tensor, actual_label_tensor)
    # bce_values.append(BCE.item())
    # y_true.append(actual_label)
    # similarities.append(similarity)
    # else:
    #     similarities.append(0)










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