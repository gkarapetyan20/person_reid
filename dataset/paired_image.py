import os
import torch
from torchvision import transforms
from torch.utils.data import  Dataset
import random
from PIL import Image


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

        filenames = sorted(os.listdir(self.dataset_path))  

        X = []
        labels = []

        for filename in filenames[:self.max_images]:
            img_path = os.path.join(self.dataset_path, filename)

            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(img_path).convert('RGB')

                try:
                    label = int(filename[:4])  
                except ValueError:
                    print(f"Warning: Invalid label in file {filename}")
                    continue  

                X.append(transform(img))
                labels.append(label)

        X = torch.stack(X)  
        labels = torch.tensor(labels, dtype=torch.int64)  

        return X, labels
    
    def create_input_pairs(self, idx1, idx2):
        img1 = self.X[idx1]
        img2 = self.X[idx2]
        label1 = self.labels[idx1]
        label2 = self.labels[idx2]

        if str(label1.item())[:4] == str(label2.item())[:4]:
            return img1, img2, 1  
        else:
            return img1, img2, 0 

    def __getitem__(self, idx):
        positive_pairs = [(i, j) for i in range(len(self.X)) for j in range(i+1, len(self.X)) if str(self.labels[i].item())[:4] == str(self.labels[j].item())[:4]]
        negative_pairs = [(i, j) for i in range(len(self.X)) for j in range(i+1, len(self.X)) if str(self.labels[i].item())[:4] != str(self.labels[j].item())[:4]]
        
        if random.random() < 0.5:
            idx1, idx2 = random.choice(positive_pairs)
            label = 1
        else:
            idx1, idx2 = random.choice(negative_pairs)
            label = 0

        img1, img2, _ = self.create_input_pairs(idx1, idx2)
        
        return img1, img2, label


    
    def __len__(self):

        return len(self.X) * (len(self.X) - 1) // 2  
    
    def __getitem__(self, idx):

        idx1, idx2 = random.sample(range(len(self.X)), 2)
        img1, img2, label = self.create_input_pairs(idx1, idx2)
        
        return img1, img2, label
