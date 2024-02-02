import torch
import torch.nn as nn
from torchvision import models

class PretrainedEfficientNet(nn.Module):
    def __init__(self, num_classes=10):  
        super(PretrainedEfficientNet, self).__init__()
        
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # Replace the classifier layer
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
