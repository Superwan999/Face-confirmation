"""
    implement Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, _type=1
    ):
        super(MFM, self).__init__()
        self.out_channels = out_channels
        if _type == 1:
            self.filter = nn.Conv2d(
                in_channels,
                2 * out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class Group(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Group, self).__init__()
        self.conv_a = MFM(in_channels, in_channels, 1, 1, 0)
        self.conv = MFM(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class Resblock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Resblock, self).__init__()
        self.conv1 = MFM(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = MFM(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class Network9layers(nn.Module):

    def __init__(self, num_classes=79077):
        super(Network9layers, self).__init__()
        self.features = nn.Sequential(
            MFM(1, 48, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Group(48, 96, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Group(192, 128, 3, 1, 1),
            Group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        self.fc1 = MFM(8 * 8 * 128, 256, _type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out, x


class Network29layers(nn.Module):

    def __init__(self, block, layers, num_classes=79077):
        super(Network29layers, self).__init__()
        self.conv1 = MFM(1, 48, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = Group(48, 96, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = Group(96, 192, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = Group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = Group(128, 128, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc = MFM(8 * 8 * 128, 256, _type=0)
        self.fc2 = nn.Linear(256, num_classes)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.block1(x)
        x = self.group1(x)
        x = self.pool2(x)

        x = self.block2(x)
        x = self.group2(x)
        x = self.pool3(x)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        fc = F.dropout(fc, training=self.training)
        out = self.fc2(fc)
        return out, fc


class Network29layersV2(nn.Module):

    def __init__(self, block, layers, feat=True, num_classes=80013):
        super(Network29layersV2, self).__init__()
        self.conv1 = MFM(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = Group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = Group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = Group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = Group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(8 * 8 * 128, 256)
        if not feat:
            self.fc2 = nn.Linear(256, num_classes, bias=False)
        self.feat = feat

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        out = self.fc(x.view(x.size(0), -1))
        if not self.feat:
            x = F.dropout(out, training=self.training)
            out = self.fc2(x)
        return out, x


def lightCNN_9layers(**kwargs):
    model = Network9layers(**kwargs)
    return model


def lightCNN_29layers(**kwargs):
    model = Network29layers(Resblock, [1, 2, 3, 4], **kwargs)
    return model


def lightCNN_29layers_v2(**kwargs):
    model = Network29layersV2(Resblock, [1, 2, 3, 4], **kwargs)
    return model
