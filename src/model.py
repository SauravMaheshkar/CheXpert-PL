from typing import List

import timm
import torch
from torch import nn
import torch.nn.functional as F


class CheXpertModel(nn.Module):
    def __init__(self, 
                 model_alias: str, 
                 feature_size: int, 
                 output_size: int,
                 hidden_layers: List[int] = [256, 32],
                 pretrained=True
                ) -> None:
        super(CheXpertModel, self).__init__()
        self.model = timm.create_model(model_alias, num_classes=0, pretrained=pretrained)
        self.hidden_layers = [
            nn.Linear(feature_size, hidden_layers[0]),
            nn.ReLU()
        ]

        for idx in range(1, len(hidden_layers)):
            self.hidden_layers.extend([
                nn.Linear(hidden_layers[idx - 1], hidden_layers[idx]),
                nn.ReLU()
            ])
        
        self.hidden_layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.fc = nn.Sequential(*self.hidden_layers)
    
    def forward(self, batch):
        batch = self.model(batch)
        batch = F.relu(batch)

        return self.fc(batch)
