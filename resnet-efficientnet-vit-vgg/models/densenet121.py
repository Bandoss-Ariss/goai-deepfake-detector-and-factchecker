from torchvision.models import densenet121, DenseNet121_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

class DenseNet121DeepFake(nn.Module):
    def __init__(self, hidden_dim, reduced_feature_dim=512, feature_dim=50176):
        super().__init__()
        
        pre_trained_model = densenet121(DenseNet121_Weights.DEFAULT)

        #Exraction de la feature Extractor (suppression de la dernière couche)
        self.feature_extractor = nn.Sequential(*[module for name, module in pre_trained_model.named_children()][:-1])
        
         # Ajout d'une couche de réduction pour les LSTM
        self.feature_reducer = nn.Sequential(
            nn.Linear(feature_dim, reduced_feature_dim),
            nn.BatchNorm1d(reduced_feature_dim),
            nn.ReLU()
        )
        
        self.rnn = nn.LSTM(
            input_size=reduced_feature_dim,
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.5
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
        
        x = x.view(-1, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_frames, -1)
        # Reduce feature dimensions
        features = self.feature_reducer(features.reshape(-1, features.size(-1)))
        features = features.view(batch_size, num_frames, -1)
        
        # LSTM processing
        h0 = torch.zeros(2*2, batch_size, self.rnn.hidden_size).to(features.device)
        c0 = torch.zeros(2*2, batch_size, self.rnn.hidden_size).to(features.device)
        out, _ = self.rnn(features, (h0, c0))
        
        # Classification on last frame
        output = self.classifier(out.mean(dim=1)).squeeze(-1)
        
        return output