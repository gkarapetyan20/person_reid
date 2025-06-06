import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class VGG_New(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG_New, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 1st Downsampling

            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2nd Downsampling

            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 3rd Downsampling

            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4th Downsampling

            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 5th Downsampling
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = VGG_New()    
model.load_state_dict(torch.load('/home/aimaster/person_reid/CNN_with/vgg_epoch_30_good.pth',map_location=torch.device('cpu') ) )
model.eval()



# Unstructured Pruning ---------------------------

# for layer in model.features:
#     if isinstance(layer,nn.Conv2d):
#         prune.l1_unstructured(layer,name="weight" , amount=0.9)


# for layer in model.features:
#     if isinstance(layer, nn.Conv2d):
#         prune.remove(layer, 'weight')  # Now it's permanently pruned

# # Save the pruned model
# torch.save(model.state_dict(), "pruned_model_0_9.pth")




# Structured Pruning (remove 90% of filters/channels)
for layer in model.features:
    if isinstance(layer, nn.Conv2d):
        prune.ln_structured(layer, name="weight", amount=0.5, n=2, dim=0)  # prune 90% of filters (channels)

# Optionally, you can remove pruning reparameterization to make it permanent
for layer in model.features:
    if isinstance(layer, nn.Conv2d):
        prune.remove(layer, 'weight')  # Permanently remove pruning

# Save the pruned model
torch.save(model.state_dict(), "structured_pruned_model_0_5.pth")