import torch
from torch import nn
from torch.nn import functional as F
from backbones.senet import SENet
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet50_se_net']

se_reduction = 16


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BottleNeckSE(nn.Module):
    expansion = 4

    def __init__(self, input_channel, num_channel, downsample=False, stride=1):
        super(BottleNeckSE, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(input_channel, num_channel, kernel_size=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(num_channel, num_channel, kernel_size=3, padding=1, bias=False, stride=1)
        self.conv3 = nn.Conv2d(num_channel, num_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        if downsample:
            self.conv_branch = nn.Conv2d(input_channel, num_channel*self.expansion, kernel_size=1, stride=stride)
            self.bn_branch = nn.BatchNorm2d(num_channel*self.expansion)
            self.se = SENet(channels=num_channel*self.expansion, reduction=se_reduction)
        self.bn1 = nn.BatchNorm2d(num_channel)
        self.bn2 = nn.BatchNorm2d(num_channel)
        self.bn3 = nn.BatchNorm2d(num_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.downsample:
            X = self.bn_branch(self.conv_branch(X))
            X = self.se(X)
            Y += X
        else:
            Y += X
        return F.relu(Y)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64
        ### stem layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        ### main layer
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # classifier
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(512*block.expansion, num_classes)
        # self.softmax = nn.Softmax(-1)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = stride != 1 or self.in_channel != channel*block.expansion
        layers=[]
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


def resnet50_se_net(pretrained=True):
    model = ResNet(BottleNeckSE, [3, 4, 6, 3])

    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50']), strict=False)

    return model
