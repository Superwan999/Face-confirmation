import os
from args import args
import torch
from torch import nn
import torch.nn.functional as F


init_func = {
    "kaiming": nn.init.kaiming_normal_,
    "xavier": nn.init.xavier_normal_}


def weight_initialize(model, conv_init, activation):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            func = init_func[conv_init]
            if conv_init == "kaiming" and activation == "LeakyReLU":
                func(m.weight, args.leaky_slope)
            else:
                func(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, conv_init, activation):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        # self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.reflection_pad = nn.ZeroPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.apply(weight_initialize(self, conv_init, activation))
        # weight_initialize(self, conv_init, activation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
        Upsamples the input and then does a convolution. This method gives better results
        compared to ConvTranspose2d.
        ref: http://distill.pub/2016/deconv-checkerboard/
        """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, upsample=None, upsample_mode='nearest'):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.upsample_mode = upsample_mode
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x_in):
        if self.upsample:
            x_in = F.interpolate(x_in, mode=self.upsample_mode,
                                 scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()  # python2 & 3 different keyoukewu
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def save(self, root=None, name=""):
        if root is None:
            root = os.path.join(".", "checkpoints")
            if not os.path.isdir(root):
                os.makedirs(root)
        torch.save(self.state_dict(), os.path.join(root, name))
        return name


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, channels, kernel_size=3, activation=args.activation_type):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn2 = nn.BatchNorm2d(channels)
        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(args.leaky_slope)
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        # out = self.activation(self.in1(self.conv1(x)))
        # out = self.in2(self.conv2(out))
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.activation(out + residual)
        return out
