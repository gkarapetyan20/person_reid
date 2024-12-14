import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.paired_image import PairedImageDataset
from model.backbone import SiameseNetwork
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path" ,type=str, help="dataset path ")
parser.add_argument("--epoch" , type=int,default=70)
parser.add_argument("--backbone" , type=str , default='resnet50')
parser.add_argument("--learning_rate" , type=float, default=1e-4)
parser.add_argument("--batch_size" , type=int,default=4)
args = parser.parse_args()


transform = transforms.Compose([
    transforms.ToTensor(),

])

paired_dataset = PairedImageDataset(args.dataset_path, transform=transform,max_images=11)
train_loader = DataLoader(paired_dataset, batch_size=args.batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseNetwork(backbone=args.backbone).to(device)
criterion = torch.nn.BCELoss()

optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)



with open('epoch_loss_values.txt', 'a') as f:
    for epoch in range(args.epoch):
        total_loss = 0
        for batch_idx, (input1, input2, labels) in enumerate(train_loader):
            input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)

            labels = labels.unsqueeze(1).float()

            output1 = model(input1, input2)  

            loss = criterion(output1, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{args.epoch}], Loss: {avg_loss:.4f}')

        f.write(f'Epoch [{epoch+1}/{args.epoch}], Loss: {avg_loss:.4f}\n')

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'siamese_network_weights_epoch_{epoch+1}.pth')
            print(f'Model weights saved to siamese_network_weights_epoch_{epoch+1}.pth')


