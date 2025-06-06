import torch
import torch.nn as nn
import timm

class SiameseTransformerNetwork(nn.Module):
    def __init__(self, backbone="vit_base_patch16_224"):
        '''
        Creates a siamese network with a Vision Transformer backbone using timm.

        Parameters:
            backbone (str): Transformer backbone from timm.
        '''
        super().__init__()

        # Load transformer backbone from timm (feature extractor only)
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)

        # Get output feature size from backbone
        self.out_features = self.backbone.num_features

        # Classification head for the absolute difference of two embeddings
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.out_features, 224),
            nn.LayerNorm(224),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(224, 64),
            nn.LayerNorm(64),
            nn.Sigmoid(),

            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        f1 = self.forward_once(x1)
        f2 = self.forward_once(x2)
        diff = torch.abs(f1 - f2)
        out = self.cls_head(diff)
        return out
