""" 
Say Cheese!
"""

import torch
import torch.nn as nn

class CheeseNet(nn.Module):
    def __init__(self, num_classes=6, input_shape=(3, 224, 224)):
        super(CheeseNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 2),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )
        
        self._flattened_size = self._get_flattened_size(input_shape)
        
        self.classifier = nn.Sequential(
            nn.Linear(self._flattened_size, 32),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.Linear(64, num_classes)
        )

    def _get_flattened_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape) 
        with torch.no_grad():
            output = self.feature_extractor(dummy_input)
        return output.numel()  

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x