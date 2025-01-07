from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResNet50DeepFakeFineTuned(nn.Module):
    def __init__(self, hidden_dim, reduced_feature_dim, feature_dim=2048):
        super().__init__()

        pretrained_model = resnet50(ResNet50_Weights.DEFAULT)

        self.feature_extractor = nn.Sequential(*[module for name, module in pretrained_model.named_children()][:-1])

      
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
            dropout=0.5
        )

        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        for name, param in self.feature_extractor.named_parameters():
            if 'layer4.2' not in name:
                param.requires_grad = False

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        
        # extract features
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
        
        output = self.classifier(out.mean(dim=1)).squeeze(-1)
        return output
        