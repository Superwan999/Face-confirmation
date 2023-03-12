from args import args
from torch import nn
from .modules import BasicModule, ConvLayer, ResidualBlock, weight_initialize


class Discriminator(BasicModule):

    def __init__(self):
        super(Discriminator, self).__init__()
        channel_output = args.discriminator_channels     # (3, 64, 128, 256, 512, 512)
        self.conv_init_type = args.conv_init_type

        # 128x128x3
        self.conv0 = ConvLayer(channel_output[0], channel_output[1],
                               kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        # 64x64x64
        self.conv1 = ConvLayer(channel_output[1], channel_output[2],
                               kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn1 = nn.BatchNorm2d(channel_output[2])
        # 32x32x128
        self.conv2 = ConvLayer(channel_output[2], channel_output[3],
                               kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn2 = nn.BatchNorm2d(channel_output[3])
        # 16x16x256
        self.conv3 = ConvLayer(channel_output[3], channel_output[4],
                               kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn3 = nn.BatchNorm2d(channel_output[4])
        # 8x8x512
        self.res4 = ResidualBlock(channel_output[4], kernel_size=3, activation=args.activation_type)
        self.conv4 = ConvLayer(channel_output[4], channel_output[5],
                               kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn4 = nn.BatchNorm2d(channel_output[5])
        # 4x4x512
        self.res5 = ResidualBlock(channel_output[5], kernel_size=3, activation=args.activation_type)
        self.conv5 = ConvLayer(channel_output[5], 1,
                               kernel_size=1, stride=1,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        if args.activation_type == "LeakReLU":
            self.activation = nn.LeakyReLU(args.leaky_slope)
        else:
            self.activation = nn.ReLU()
        weight_initialize(self, "kaiming", args.activation_type)

    def forward(self, x):
        y = self.activation(self.conv0(x))
        y = self.activation(self.bn1(self.conv1(y)))
        y = self.activation(self.bn2(self.conv2(y)))
        y = self.activation(self.bn3(self.conv3(y)))
        y = self.activation(self.bn4(self.conv4(self.res4(y))))
        y = self.conv5(self.res5(y))
        return y
