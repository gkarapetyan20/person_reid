import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder_path" ,type=str, help="dataset path for mean and std")
args = parser.parse_args()

class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.imgs[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.ToTensor(),
])


dataset = SimpleDataset(root_dir=args.folder_path, transform=transform)

def calculate_mean_std(dataset):
    num_samples = len(dataset)
    sum_channel = torch.zeros(3)
    sum_channel_squared = torch.zeros(3)

    for inputs in dataset:
        sum_channel += torch.mean(inputs, dim=(1, 2))
        sum_channel_squared += torch.mean(inputs**2, dim=(1, 2))

    mean = sum_channel / num_samples
    std = torch.sqrt(sum_channel_squared / num_samples - mean**2)
    return mean, std

mean, std = calculate_mean_std(dataset)

print("Mean of the dataset:")
print(mean)
print("\nStandard Deviation of the dataset:")
print(std)
