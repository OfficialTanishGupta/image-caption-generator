import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)

        # remove avgpool + fc
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)  
        # shape: (batch, 2048, 7, 7)

        batch_size, channels, height, width = features.size()

        # reshape to (batch, num_pixels, feature_dim)
        features = features.view(batch_size, channels, -1)
        features = features.permute(0, 2, 1)

        return features