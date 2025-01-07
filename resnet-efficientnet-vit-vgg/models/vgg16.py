from torchvision.models import resnet50, ResNet50_Weights
import torch
import torchvision.transforms.v2 as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from torchvision.models import  vgg16, VGG16_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

class Vgg16Deepfake(nn.Module):
    def __init__(self, feature_dim, hidden_dim, reduced_feature_dim=512):
        super().__init__()
        
        pre_trained_model = vgg16(weights=VGG16_Weights.DEFAULT)

        #Exraction de la feature Extractor (suppression de la dernière couche)
        self.feature_extractor = nn.Sequential(
            pre_trained_model.features,
            pre_trained_model.avgpool,
            nn.Flatten()
        )
        
        # Ajout d'une couche de réduction pour les LSTM
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, reduced_feature_dim),
            nn.BatchNorm1d(reduced_feature_dim),
            nn.ReLU()
        )
        
        self.rnn = nn.LSTM(
            input_size=reduced_feature_dim,
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.6
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
         #Freeze pour Transfer learning
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()

        if x.dim() == 5:
            print("good")
            batch_size, num_frames, c, h, w = x.size()
            x = x.view(-1, c, h, w)
        
        x = x.view(-1, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_frames, -1)
        
        # Reduce feature dimensions
        features = self.feature_reducer(features.reshape(-1, features.size(-1)))
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM processing
        h0 = torch.zeros(2, batch_size, self.rnn.hidden_size).to(features.device)
        c0 = torch.zeros(2, batch_size, self.rnn.hidden_size).to(features.device)
        out, _ = self.rnn(features, (h0, c0))
        
        # Classification on last frame
        output = self.classifier(out.mean(dim=1)).squeeze(-1)
        
        return output