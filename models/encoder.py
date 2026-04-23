import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.fc(features)
        return features