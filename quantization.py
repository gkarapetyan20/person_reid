import torch
from torch.quantization import quantize_dynamic
import torch.nn as nn
import torch.quantization
import torch.quantization.fuse_modules

class VGG_New(nn.Module):
    def __init__(self, num_classes=100):
        super(VGG_New, self).__init__()
        self.quant = torch.quantization.QuantStub()
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
        self.dequatn = torch.quantization.DeQuantStub()
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
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequatn(x)
        return x


def fuse_model(model):
    for module_name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for i in range(len(module) - 2):
                m1, m2, m3 = module[i], module[i+1], module[i+2]
                if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d) and isinstance(m3, nn.ReLU):
                    torch.quantization.fuse_modules(module, [str(i), str(i+1), str(i+2)], inplace=True)



model = VGG_New()
model.load_state_dict(torch.load('/home/aimaster/person_reid/CNN_with/vgg_epoch_30_good.pth', map_location='cpu'))
model.eval()

# Set quantization config
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # or 'x86' if you prefer

# Fuse
fuse_model(model)

# Prepare
model_prepared = torch.quantization.prepare(model)

# Calibrate with representative data (run some inference on data)
# Example: model_prepared(torch.randn(1, 3, 224, 224))

# Convert
model_quantized = torch.quantization.convert(model_prepared)

# Save quantized model
torch.save(model_quantized.state_dict(), "quant.pth")


