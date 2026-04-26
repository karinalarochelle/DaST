'''ResNet in PyTorch.
ResNet (Residual Networks) - Deep neural networks with "skip connections"
that allow gradients to flow better during training.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# BasicBlock: Basic building block for ResNet
# Contains 2 convolutional layers with a skip connection
class BasicBlock(nn.Module):
    # expansion: how much the output channels are multiplied compared to input
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # First conv layer: may reduce spatial size (stride), keep channels at 'planes'
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Second conv layer: keep size same (stride=1), keep channels at 'planes'
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Skip connection: pass input directly to output
        # If size or channels don't match, project them to match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                # Project input to match output channels/size
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # First conv + batch norm + activation
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv + batch norm (no activation yet)
        out = self.bn2(self.conv2(out))
        # Add skip connection (this is the key to ResNet!)
        out += self.shortcut(x)
        # Final activation
        out = F.relu(out)
        return out


# Bottleneck: More efficient building block (reduces then expands channels)
# Contains 3 convolutional layers with a skip connection
class Bottleneck(nn.Module):
    # expansion: output channels = input channels * 4
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 conv: reduce channels to 'planes' (less computation)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 conv: do main processing with fewer channels
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 conv: expand back to 'planes * expansion' channels
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # Skip connection: project input if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # Reduce, process, expand pattern
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # Add skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ResNet: Full ResNet network combining multiple blocks
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        # Track input channels for each layer
        self.in_planes = 64

        # Initial conv layer: 3 channels (RGB) -> 64 channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Build 4 layers of blocks with different channel sizes
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # for i, im in enumerate(x):
        #     for t, m, s in zip(x[i], [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]):
        #         t.sub_(m).div_(s)
        out = F.relu(self.bn1(self.conv1(x)))
        # print('mean:', self.bn1.running_mean)
        # print('var:', self.bn1.running_var)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()

