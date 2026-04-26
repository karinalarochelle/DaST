'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# Small MNIST neural network (simple model)
class Net_s(nn.Module):
    def __init__(self):
        super(Net_s, self).__init__()
        # First convolutional layer: 1 input channel (grayscale) -> 20 channels
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # Second convolutional layer: 20 channels -> 50 channels
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # Fully connected layer 1: flatten to 4*4*50 neurons -> 500 neurons
        self.fc1 = nn.Linear(4*4*50, 500)
        # Fully connected layer 2 (output): 500 neurons -> 10 classes (digits 0-9)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Apply first conv layer and activation, then max pool (reduce size)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # Apply second conv layer and activation, then max pool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # Flatten the 4D tensor to 2D (batch_size, features)
        x = x.view(-1, 4*4*50)
        # Apply first fully connected layer with activation
        x = F.relu(self.fc1(x))
        # Apply output layer (no activation here, will use softmax in loss function)
        x = self.fc2(x)
        # Return log probabilities for each digit class
        return F.log_softmax(x, dim=1)

# Medium MNIST neural network
class Net_m(nn.Module):
    def __init__(self):
        self.number = 0  # Counter for tracking forward passes
        super(Net_m, self).__init__()
        # First conv layer: 1 -> 20 channels
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # Second conv layer: 20 -> 50 channels
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # Third conv layer: 50 -> 50 channels (same size, keeps detail)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        # Fully connected layer 1: 2*2*50 -> 500
        self.fc1 = nn.Linear(2*2*50, 500)
        # Output layer: 500 -> 10 classes
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, sign=0):
        # If sign==0, increment counter (used for tracking)
        if sign == 0:
            self.number += 1
        # Conv layers with pooling - progressively reduce image size
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        # Flatten and pass through fully connected layers
        x = x.view(-1, 2*2*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_number(self):
        # Return the counter value
        return self.number


# Large MNIST neural network
class Net_l(nn.Module):
    def __init__(self):
        super(Net_l, self).__init__()
        # First conv layer: 1 -> 20 channels
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # Second conv layer: 20 -> 50 channels
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # Third and fourth conv layers: 50 -> 50 (more processing)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        # Fully connected layers (50 neurons after aggressive pooling)
        self.fc1 = nn.Linear(50, 500)
        # Output layer: 500 -> 10 classes
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # Pass through all 4 conv layers with pooling between them
        # Each pooling reduces image size by half
        x = F.relu(self.conv1(x))  # First conv: detect basic features
        x = F.max_pool2d(x, 2, 2)  # Pool: reduce size, keep important features
        x = F.relu(self.conv2(x))  # Second conv: detect more complex patterns
        x = F.max_pool2d(x, 2, 2)  # Pool again
        x = F.relu(self.conv3(x))  # Third conv: refine features
        x = F.max_pool2d(x, 2, 2)  # Pool again
        x = F.relu(self.conv4(x))  # Fourth conv: final feature extraction
        x = F.max_pool2d(x, 2, 2)  # Final pool: image now very small
        # After 4 pooling operations, image is 1x1 with 50 channels
        x = x.view(-1, 50)  # Flatten to 1D vector
        # Fully connected layers: make final classification
        x = F.relu(self.fc1(x))  # Hidden layer with activation
        x = self.fc2(x)  # Output layer: raw scores for each digit
        return x  # Return raw scores (loss function will apply softmax)

