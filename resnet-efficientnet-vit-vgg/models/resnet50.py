from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50DeepFake(nn.Module):
    def __init__(self, hidden_dim=512, reduced_feature_dim=256):
        super().__init__()
        
        # Load pre-trained ResNet50
        pretrained_model = resnet50(ResNet50_Weights.DEFAULT)
        
        # Remove the classification head
        self.feature_extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
        
        # Feature reducer
        self.feature_reducer = nn.Sequential(
            nn.Linear(2048, reduced_feature_dim),
            nn.BatchNorm1d(reduced_feature_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(reduced_feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected a 4D tensor, but got {x.dim()}D tensor")
        
        # Feature extraction
        features = self.feature_extractor(x).squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        features = self.feature_reducer(features)
        
        # Classification
        output = self.classifier(features).squeeze(-1)
        return output

