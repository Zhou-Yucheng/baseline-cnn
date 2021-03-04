#!/usr/bin/python3.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def model_A(n_classes, pretrained=True):
    model_resnet = models.resnet18(pretrained=pretrained)
    num_features = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_features, n_classes)
    return model_resnet


def get_model(n_classes, model_name):
    if model_name == 'Model_A':
        return model_A(n_classes, True)
    elif model_name == 'Model_B':
        return model_A(n_classes, False)

    elif model_name == 'Model_C2':
        return NetC2(n_classes)
    elif model_name == 'Model_C-R3s':
        return ResNet3s(n_classes)
    elif model_name == 'Model_C-R3s_test':
        return ResNet3s_test(n_classes)


class NetC2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # input: 224x224
        self.conv_block1 = self.block(3, 32, 5)  # 216/2=108
        self.conv_block2 = self.block(32, 64, 3)  # 104/2=52
        self.conv_block3 = self.block(64, 128, 3)  # 48/2=24
        self.conv_block4 = self.block(128, 256, 3)  # 20/2=10
        self.conv_block5 = self.block(256, 256, 3)  # 6/2=3

        self.fc1 = nn.Linear(256 * 3 * 3, 200)
        self.fc2 = nn.Linear(200, n_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )


class ResBlock(nn.Module):
    # The basic building block in http://arxiv.org/abs/1512.03385
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False),  # BN handles bias!
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = self.layer(x)
        identity = x if out.shape[1] == x.shape[1] else torch.cat((x, torch.zeros_like(x)), 1)  # try (x,x)?
        out += identity
        return F.relu(out)


class ResNet3s(nn.Module):
    # ResNet-26, input size: 224x; use stride 2
    def __init__(self, n_classes=20):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # 112x (in_c, out_c, kernel_size, stride=1, padding=0, dilation=1,)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # 56x (kernel_size, stride=None, padding=0, dilation=1,)
        )

        self.layer1 = self._make_layer_s(64, 64, 3, False)  # 56x
        self.layer2 = self._make_layer_s(64, 128, 3, True)  # 28x
        self.layer3 = self._make_layer_s(128, 256, 3, True)  # 14x
        self.layer4 = self._make_layer_s(256, 512, 3, True)  # 7x

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)

    def _make_layer_s(self, in_channel, out_channel, n_block, down_sample=False):
        layers = []
        if down_sample:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=2),
                nn.BatchNorm2d(out_channel),
            ))
            layers.append(ResBlock(out_channel, out_channel))
        else:
            layers.append(ResBlock(in_channel, out_channel))

        for i in range(1, n_block):
            layers.append(ResBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet3s_test(nn.Module):
    # ResNet-26, input size: 224x; use stride 2
    def __init__(self, n_classes=20):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  # 112x (in_c, out_c, kernel_size, stride=1, padding=0, dilation=1,)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # 56x (kernel_size, stride=None, padding=0, dilation=1,)
        )

        # replace 2-stride down sample by max-pool
        self.layer1 = self._make_layer_s(64, 64, 3, False)  # 56x
        self.layer2 = self._make_layer_s(64, 128, 3, True)  # 28x
        self.layer3 = self._make_layer_s(128, 256, 3, True)  # 14x
        self.layer4 = self._make_layer_s(256, 512, 3, True)  # 7x

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, n_classes)

    def _make_layer_s(self, in_channel, out_channel, n_block, down_sample=False):
        layers = []
        if down_sample:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=2),
                nn.BatchNorm2d(out_channel),
            ))
            layers.append(ResBlock(out_channel, out_channel))
        else:
            layers.append(ResBlock(in_channel, out_channel))

        for i in range(1, n_block):
            layers.append(ResBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        self.x0 = x.detach()

        x = self.layer1(x)
        self.x1 = x.detach()
        x = self.layer2(x)
        self.x2 = x.detach()
        x = self.layer3(x)
        self.x3 = x.detach()
        x = self.layer4(x)
        self.x4 = x.detach()

        x = self.avgpool(x)
        self.feature_map = x.detach()
        x = torch.flatten(x, 1)

        return self.fc(x)
