import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import random
from PIL import Image
import torch.nn.functional as F

class PairedImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None, max_images=3000):
        self.dataset_path = dataset_path
        self.transform = transform
        self.max_images = max_images  
        self.X, self.labels = self.load_data()
        
    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        filenames = sorted(os.listdir(self.dataset_path))  # Get all filenames in the directory

        X = []
        labels = []

        for filename in filenames[:self.max_images]:  # Limit to the first max_images
            img_path = os.path.join(self.dataset_path, filename)

            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(img_path).convert('RGB')

                try:
                    label = int(filename[:4])  # Assuming label is the first 4 characters of the filename
                except ValueError:
                    print(f"Warning: Invalid label in file {filename}")
                    continue  # Skip files with invalid labels

                X.append(transform(img))
                labels.append(label)

        X = torch.stack(X)  # Stack all images into a single tensor
        labels = torch.tensor(labels, dtype=torch.int64)  # Create a tensor of labels

        return X, labels
    
    def create_input_pairs(self, idx1, idx2):
        img1 = self.X[idx1]
        img2 = self.X[idx2]
        label1 = self.labels[idx1]
        label2 = self.labels[idx2]

        if str(label1.item())[:4] == str(label2.item())[:4]:
            return img1, img2, 1  # Same class
        else:
            return img1, img2, 0  # Different class

    def __getitem__(self, idx):
        positive_pairs = [(i, j) for i in range(len(self.X)) for j in range(i+1, len(self.X)) if str(self.labels[i].item())[:4] == str(self.labels[j].item())[:4]]
        negative_pairs = [(i, j) for i in range(len(self.X)) for j in range(i+1, len(self.X)) if str(self.labels[i].item())[:4] != str(self.labels[j].item())[:4]]
        
        # Sample from both positive and negative pairs equally
        if random.random() < 0.5:
            idx1, idx2 = random.choice(positive_pairs)
            label = 1
        else:
            idx1, idx2 = random.choice(negative_pairs)
            label = 0

        img1, img2, _ = self.create_input_pairs(idx1, idx2)
        
        return img1, img2, label


    
    def __len__(self):

        return len(self.X) * (len(self.X) - 1) // 2  # Combinations of 2 from n items
    
    def __getitem__(self, idx):

        idx1, idx2 = random.sample(range(len(self.X)), 2)
        img1, img2, label = self.create_input_pairs(idx1, idx2)
        
        return img1, img2, label



import torch
import torch.nn as nn
import torchvision.models as models
# from torchvision.models import AlexNet_Weights


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



class SiameseNetwork(nn.Module):
    def __init__(self, backbone="densenet121"):
        super().__init__()

        if backbone not in models.__dict__:
            raise Exception(f"No model named {backbone} exists in torchvision.models.")

        model = models.densenet121(pretrained=True)
        self.backbone = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),  # to get a fixed-size output
            nn.Flatten()
        )

        # DenseNet121's output after AdaptiveAvgPool2d and Flatten is 1024
        out_features = 1024  
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        combined_features = feat1 * feat2
        output = self.cls_head(combined_features)
        return output







dataset_path = '/home/user/Desktop/re_id/seamens/new_train'  
transform = transforms.Compose([
    transforms.ToTensor(),

])

# Create dataset and dataloader
paired_dataset = PairedImageDataset(dataset_path, transform=transform,max_images=400)
train_loader = DataLoader(paired_dataset, batch_size=32, shuffle=True)
# Setup device, model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork(backbone="vgg16").to(device)
# criterion = ContrastiveLossCosine().to(device)
criterion = torch.nn.BCELoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
# scheduler = CosineAnnealingLR(optimizer, T_max=70)


# Training loop
num_epochs = 250

# Open the text file in append mode ('a') to write the loss values
with open('epoch_loss_values.txt', 'a') as f:
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input1, input2, labels) in enumerate(train_loader):
            input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)

            # Reshape labels to match output shape
            labels = labels.unsqueeze(1).float()
            # print(labels)

            # Forward pass
            output1 = model(input1, input2)  # Get embeddings for input1

            # Compute loss
            loss = criterion(output1, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)
        print("*" * 50)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        print("*" * 50)

        # Write the loss to the file for each epoch
        f.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}\n')

        # Save model weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'siamese_network_weights_epoch_densNet{epoch+1}.pth')
            print(f'Model weights saved to siamese_network_weights_epoch_{epoch+1}.pth')

        # Step the scheduler after each epoch
        # scheduler.step()

    # Optionally, save final model weights after the last epoch
    torch.save(model.state_dict(), 'siamese_network_weights_final_densNet.pth')
    print('Model weights saved to siamese_network_weights_final.pth')

