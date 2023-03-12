# borrowed from https://github.com/tonylins/pytorch-mobilenet-v2

import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(
                inp * expand_ratio, inp * expand_ratio,
                3, stride, 1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, fm_mult=1.0, dropout=0.5):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 16 == 0
        input_channel = int(32 * fm_mult)
        self.last_channel = int(1280 * fm_mult) if fm_mult > 1.0 else 1280
        # self.features = [conv_bn(3, input_channel, 2)]
        self.features = [conv_bn(3, input_channel, round(input_size / 224 * 2))]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * fm_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(
                        InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size // 16))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(self.last_channel, num_classes))

        self._initialize_weights()

    def forward(self, x, use_dropout=False):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        features = x
        if use_dropout:
            x = self.dropout(x)
        x = self.classifier(x)
        return x, features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
