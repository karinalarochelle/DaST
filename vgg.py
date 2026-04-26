'''VGG11/13/16/19 in Pytorch.'''
# VGG networks are a classic CNN architecture with many small 3x3 convolutional layers
# M = MaxPool layer that reduces image size
# Numbers = number of filters/channels in each conv layer

import torch
import torch.nn as nn
import torch.nn.functional as F

# Architecture configurations for different VGG versions
# Larger numbers = more layers and more parameters
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        # Counter for tracking forward passes
        self.number = 0
        # Build all the convolutional layers from the configuration
        self.features = self._make_layers(cfg[vgg_name])
        # Final classification layers: reduce 512 features -> 10 classes
        self.classifier = nn.Sequential(
            nn.Dropout(),  # Randomly drop 50% of neurons (prevent overfitting)
            nn.Linear(512, 512),  # Reduce to 512
            nn.ReLU(True),  # Activation
            nn.Dropout(),  # Another dropout
            nn.Linear(512, 512),  # Another reduction
            nn.ReLU(True),  # Activation
            nn.Linear(512, 10),  # Final output: 10 classes (0-9)
        )

    def forward(self, x, sign=0):
        # Count forward passes (if sign==0)
        if sign == 0:
            self.number += 1
        # Pass through all convolutional layers
        out = self.features(x)
        # Flatten from 4D to 2D (batch_size, features)
        out = out.view(out.size(0), -1)
        # Pass through classification layers
        out = self.classifier(out)
        return out

    def get_number(self):
        # Return the counter value
        return self.number

    def _make_layers(self, cfg):
        # Build conv and pool layers from configuration
        layers = []
        in_channels = 3  # Start with 3 channels (RGB)
        count = 0
        
        for x in cfg:
            if x == 'M':
                # Add pooling layer (reduce size by 2x)
                count += 1
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # Add convolutional layer with batch norm and activation
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),  # Normalize before activation
                           nn.ReLU(inplace=True)]  # Activation function
                in_channels = x  # Next layer takes this layer's output as input
        
        # Add final average pooling
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

