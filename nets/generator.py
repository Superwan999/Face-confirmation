from args import args
from .modules import BasicModule, ConvLayer, UpsampleConvLayer, ResidualBlock, weight_initialize
import torch
from torch import nn


class LocalPath(BasicModule):
    # HW 40x40, 32x40, 32x48
    def __init__(self, feature_layer_dim=64):
        super(LocalPath, self).__init__()
        encoder_channel = args.localpath_encoder_channels        # (3, 64, 128, 256, 512)
        decoder_channel = args.localpath_decoder_channels        # (256, 128, 64)
        self.lrelu = nn.LeakyReLU(args.leaky_slope)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoder
        # 0: HxWx64
        self.conv0 = ConvLayer(encoder_channel[0], encoder_channel[1], kernel_size=3, stride=1,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.res0 = ResidualBlock(encoder_channel[1])

        # 1: 20x20, 16x20, 16x24    x128
        self.conv1 = ConvLayer(encoder_channel[1], encoder_channel[2], kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.res1 = ResidualBlock(encoder_channel[2])
        self.bn1 = nn.BatchNorm2d(encoder_channel[2])

        # 2: 10x10, 8x10, 8x12      x256
        self.conv2 = ConvLayer(encoder_channel[2], encoder_channel[3], kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.res2 = ResidualBlock(encoder_channel[3])
        self.bn2 = nn.BatchNorm2d(encoder_channel[3])

        # 3: 5x5, 4x5, 4x6          x512
        self.conv3 = ConvLayer(encoder_channel[3], encoder_channel[4], kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn3 = nn.BatchNorm2d(encoder_channel[4])
        self.res3_1 = ResidualBlock(encoder_channel[4])
        self.res3_2 = ResidualBlock(encoder_channel[4])

        # decoder
        # 10x10, 8x10, 8x12         x256
        self.upsample3 = UpsampleConvLayer(
            encoder_channel[4], decoder_channel[0], kernel_size=3, stride=1, upsample=2)
        self.bn_upsample3 = nn.BatchNorm2d(decoder_channel[0])
        self.after_select3 = ConvLayer(
            decoder_channel[0] + encoder_channel[3], decoder_channel[0],  # 256 + 256
            kernel_size=3, stride=1,
            conv_init=args.conv_init_type, activation=args.activation_type)
        self.res_upsample3 = ResidualBlock(decoder_channel[0])
        # 20x20, 16x20, 16,24       x128
        self.upsample2 = UpsampleConvLayer(
            decoder_channel[0], decoder_channel[1], kernel_size=3, stride=1, upsample=2)
        self.bn_upsample2 = nn.BatchNorm2d(decoder_channel[1])
        self.after_select2 = ConvLayer(
            decoder_channel[1] + encoder_channel[2], decoder_channel[1],  # 128 + 128
            kernel_size=3, stride=1,
            conv_init=args.conv_init_type, activation=args.activation_type)
        self.res_upsample2 = ResidualBlock(decoder_channel[1])
        # 40x40, 32x40, 32x48       x64
        self.upsample1 = UpsampleConvLayer(
            decoder_channel[1], feature_layer_dim, kernel_size=3, stride=1, upsample=2)
        self.bn_upsample1 = nn.BatchNorm2d(feature_layer_dim)
        self.after_select1 = ConvLayer(
            feature_layer_dim + encoder_channel[1], feature_layer_dim,     # 128 + 128
            kernel_size=3, stride=1,
            conv_init=args.conv_init_type, activation=args.activation_type)
        self.res_upsample1 = ResidualBlock(feature_layer_dim)
        self.local_img = ConvLayer(
            feature_layer_dim, args.out_dim, kernel_size=3, stride=1,
            conv_init=args.conv_init_type, activation=args.activation_type)

    def forward(self, x):
        # encoder
        res0 = self.res0(self.lrelu(self.conv0(x)))
        res1 = self.res1(self.lrelu(self.bn1(self.conv1(res0))))
        res2 = self.res2(self.lrelu(self.bn2(self.conv2(res1))))
        res3_1 = self.res3_1(self.lrelu(self.bn3(self.conv3(res2))))
        res3_2 = self.res3_2(res3_1)

        # decoder
        upsample3 = self.lrelu(self.bn_upsample3(self.upsample3(res3_2)))
        res_upsample3 = self.res_upsample3(
            self.lrelu(self.bn_upsample3(self.after_select3(torch.cat([upsample3, res2], 1)))))

        upsample2 = self.lrelu(
            self.bn_upsample2(self.upsample2(res_upsample3)))
        res_upsample2 = self.res_upsample2(
            self.lrelu(self.bn_upsample2(self.after_select2(torch.cat([upsample2, res1], 1)))))

        upsample1 = self.lrelu(self.bn_upsample1(self.upsample1(res_upsample2)))
        res_upsample1 = self.res_upsample1(
            self.lrelu(self.bn_upsample1(self.after_select1(torch.cat([upsample1, res0], 1)))))

        local_img = self.tanh(self.local_img(res_upsample1))
        assert local_img.shape == x.shape, "{} {}".format(local_img.shape, x.shape)
        # 40x40, 32x40, 32x48       x3 x64
        return local_img, res_upsample1


class GlobalPath(BasicModule):

    def __init__(self, local_feature_layer_dim=64):
        super(GlobalPath, self).__init__()
        # (3, 64, 128, 256, 512)
        channel_output = args.globalpath_encoder_channels
        # (64, 32, 16, 8, 3)
        decoder_channel_initial = args.globalpath_decoder_initial_channels
        # (512, 256, 128, 64)
        decoder_channel_reconstruct = args.globalpath_decoder_reconstruct_channels
        decoder_upscale = args.decoder_upscales                                      # (4, 2, 2)

        ###############################
        # This part is Global Encoder #
        ###############################
        # 128x128x3
        self.conv0 = ConvLayer(channel_output[0], channel_output[1], kernel_size=7, stride=1,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.res0 = ResidualBlock(channel_output[1], kernel_size=7, activation=args.activation_type)
        # 128x128x64
        self.conv1 = ConvLayer(channel_output[1], channel_output[1], kernel_size=5, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn1 = nn.BatchNorm2d(channel_output[1])
        # 64x64x64

        self.res1 = ResidualBlock(channel_output[1], kernel_size=5, activation=args.activation_type)
        self.conv2 = ConvLayer(channel_output[1], channel_output[2], kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn2 = nn.BatchNorm2d(channel_output[2])
        # 32x32x128

        self.res2 = ResidualBlock(channel_output[2], kernel_size=3, activation=args.activation_type)
        self.conv3 = ConvLayer(channel_output[2], channel_output[3], kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn3 = nn.BatchNorm2d(channel_output[3])
        # 16x16x256

        self.res3 = ResidualBlock(channel_output[3], kernel_size=3, activation=args.activation_type)
        self.conv4 = ConvLayer(channel_output[3], channel_output[4], kernel_size=3, stride=2,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        self.bn4 = nn.BatchNorm2d(channel_output[4])
        # 8x8x512

        self.res4_1 = ResidualBlock(
            channel_output[4], kernel_size=3, activation=args.activation_type)
        self.res4_2 = ResidualBlock(
            channel_output[4], kernel_size=3, activation=args.activation_type)
        self.res4_3 = ResidualBlock(
            channel_output[4], kernel_size=3, activation=args.activation_type)
        self.res4_4 = ResidualBlock(
            channel_output[4], kernel_size=3, activation=args.activation_type)
        # 8x8x512
        # 128 / 16 = 8
        fc_border = args.input_img_size // pow(2, len(args.globalpath_encoder_channels) - 1)
        self.fc1 = nn.Linear(channel_output[4] * fc_border * fc_border, channel_output[4])
        self.vid = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # 1x1x256
        #####################################
        # This is the end of Global Encoder #
        #####################################

        ###############################
        # This part is Global Decoder #
        ###############################
        # 1x1x356
        self.fc2 = nn.Linear(channel_output[4] // 2 + args.noise_dim,
                             fc_border * fc_border * decoder_channel_initial[0])    # 8x8x64
        # 1x1x4096

        initial8_out_channels = decoder_channel_initial[0]              # 64
        initial16_out_channels = decoder_channel_initial[1]             # 32
        initial32_out_channels = decoder_channel_initial[2]             # 16
        initial64_out_channels = decoder_channel_initial[3]             # 8
        # UpsampleConvLayer: in_channels, out_channels, kernel_size, stride, upsample=None
        if "Interpolate" in args.upsample_mode:
            # due with interpolate mode
            parts = args.upsample_mode.split('|')
            if len(parts) == 2:
                mode = parts[1]
            elif len(parts) == 1:
                mode = 'nearest'
            else:
                mode = ''
                print("Sorry no such interpolation")
                exit()
        else:
            mode = ''
            print("Currently you have to use interpolation")
            exit()
        self.initial32 = UpsampleConvLayer(in_channels=decoder_channel_initial[0],  # 64
                                           out_channels=decoder_channel_initial[1],  # 32
                                           kernel_size=3, stride=1,
                                           upsample=decoder_upscale[0],  # 4
                                           upsample_mode=mode)
        self.initial64 = UpsampleConvLayer(in_channels=decoder_channel_initial[1],  # 32
                                           out_channels=decoder_channel_initial[2],  # 16
                                           kernel_size=3, stride=1,
                                           upsample=decoder_upscale[1],  # 2
                                           upsample_mode=mode)
        self.initial128 = UpsampleConvLayer(in_channels=decoder_channel_initial[2],  # 16
                                            out_channels=decoder_channel_initial[3],  # 8
                                            kernel_size=3, stride=1,
                                            upsample=decoder_upscale[2],  # 2
                                            upsample_mode=mode)

        # #################### 8 #####################
        dim8 = initial8_out_channels + channel_output[4]                    # 64 + 512
        self.before_select8 = ResidualBlock(
            dim8, kernel_size=3, activation=args.activation_type)
        self.before_reconstruct8 = ResidualBlock(
            dim8, kernel_size=3, activation=args.activation_type)
        self.reconstruct8 = ResidualBlock(
            dim8, kernel_size=3, activation=args.activation_type)
        # 8x8x(64+512)

        # #################### 16 #####################
        self.reconstruct_upsample16 = UpsampleConvLayer(
            in_channels=dim8,  # 64 + 512
            out_channels=decoder_channel_reconstruct[0],     # 512
            kernel_size=3, stride=1,
            upsample=decoder_upscale[1],                     # 2
            upsample_mode=mode)                              # nearest
        # 16x16x512
        dim16 = channel_output[3]           # 256
        self.before_select16 = ResidualBlock(dim16)
        reconstruct16_dim = dim16 + decoder_channel_reconstruct[0]                  # 256 + 512
        self.before_reconstruct16 = ResidualBlock(reconstruct16_dim)
        self.reconstruct16 = ResidualBlock(reconstruct16_dim)
        # 16x16x(256+512)

        # #################### 32 #####################
        self.reconstruct_upsample32 = UpsampleConvLayer(
            in_channels=reconstruct16_dim,                  # 256+512
            out_channels=decoder_channel_reconstruct[1],    # 256
            kernel_size=3, stride=1,
            upsample=decoder_upscale[1],                    # 2
            upsample_mode=mode)                             # nearest
        # 32x32x256
        dim32 = initial16_out_channels + channel_output[2] + args.in_dim * 2  # 32 + 128 + 3
        self.before_select32 = ResidualBlock(dim32)
        reconstruct32_dim = dim32 + decoder_channel_reconstruct[1]  # (32 + 128 + 3) + 256
        self.before_reconstruct32 = ResidualBlock(reconstruct32_dim)
        self.reconstruct32 = ResidualBlock(reconstruct32_dim)
        self.img32 = ConvLayer(reconstruct32_dim, args.out_dim, kernel_size=3, stride=1,
                               conv_init=args.conv_init_type, activation=args.activation_type)
        # 32x32x419

        # #################### 64 #####################
        self.reconstruct_upsample64 = UpsampleConvLayer(
            in_channels=reconstruct32_dim,                  # 163+256
            out_channels=decoder_channel_reconstruct[2],    # 128
            kernel_size=3, stride=1,
            upsample=decoder_upscale[1],                    # 2
            upsample_mode=mode)                             # nearest
        # 64x64x128
        dim64 = initial32_out_channels + channel_output[1] + args.in_dim * 2  # 16 + 64 + 3
        self.before_select64 = ResidualBlock(dim64, kernel_size=5)
        reconstruct64_dim = dim64 + decoder_channel_reconstruct[2] + args.out_dim  # 83 + 128 + 3
        self.before_reconstruct64 = ResidualBlock(reconstruct64_dim)
        self.reconstruct64 = ResidualBlock(reconstruct64_dim)
        # 64x64x(128+83+3)
        self.img64 = ConvLayer(reconstruct64_dim, args.out_dim, kernel_size=3, stride=1,
                               conv_init=args.conv_init_type, activation=args.activation_type)

        # #################### 128 #####################
        self.reconstruct_upsample128 = UpsampleConvLayer(
            in_channels=reconstruct64_dim,                  # 83 + 128
            out_channels=decoder_channel_reconstruct[3],    # 64
            kernel_size=3, stride=1,
            upsample=decoder_upscale[1],                    # 2
            upsample_mode=mode)                             # nearest
        dim128 = initial64_out_channels + channel_output[1] + args.in_dim * 2  # 8 + 64 + 3
        self.before_select128 = ResidualBlock(dim128, kernel_size=7)
        reconstruct128_dim = dim128 + decoder_channel_reconstruct[3] + \
            local_feature_layer_dim + args.out_dim*2  # 75 + 64 + 64 + 6
        self.reconstruct128 = ResidualBlock(reconstruct128_dim, kernel_size=5)
        # 128x128x209
        self.conv5 = ConvLayer(
            reconstruct128_dim, args.globalpath_decoder_channel_start, kernel_size=5, stride=1,
            activation=args.activation_type, conv_init=args.conv_init_type)
        self.bn5 = nn.BatchNorm2d(args.globalpath_decoder_channel_start)
        self.res5 = ResidualBlock(args.globalpath_decoder_channel_start)

        self.conv6 = ConvLayer(args.globalpath_decoder_channel_start,
                               args.globalpath_decoder_channel_start // 2,
                               kernel_size=3, stride=1,
                               activation=args.activation_type, conv_init=args.conv_init_type)
        self.bn6 = nn.BatchNorm2d(args.globalpath_decoder_channel_start // 2)

        self.conv7 = ConvLayer(args.globalpath_decoder_channel_start // 2,
                               3, kernel_size=3, stride=1,
                               activation=args.activation_type, conv_init=args.conv_init_type)

        if args.global_encoder_activation_type == "LeakyReLU":
            self.erelu = nn.LeakyReLU(args.leaky_slope)
        else:
            self.erelu = nn.ReLU()
        if args.global_decoder_activation_type == "LeakyReLU":
            self.drelu = nn.LeakyReLU(args.leaky_slope)
        else:
            self.drelu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(args.leaky_slope)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, I128, I64, I32, local_predict, local_feature, noise):
        # encoder:
        res0 = self.res0(self.lrelu(self.conv0(I128)))  # 128x128x64
        res1 = self.lrelu(self.bn1(self.conv1(res0)))  # 64x64x64
        res2 = self.lrelu(self.bn2(self.conv2(self.res1(res1))))  # 32x32x128
        res3 = self.lrelu(self.bn3(self.conv3(self.res2(res2))))  # 16x16x256
        res4 = self.lrelu(self.bn4(self.conv4(self.res3(res3))))  # 8x8x512
        fc1 = self.fc1(self.res4_4(
            self.res4_3(self.res4_2(self.res4_1(res4)))).view(res4.size(0), -1))  # 1x1x512
        # feature vector
        vid = self.vid(fc1.view(fc1.size(0), 1, -1)).squeeze_(1)  # 1x1x256
        # decoder
        fc2 = self.fc2(torch.cat([vid, noise], 1))  # 1x1x4096
        initial8 = fc2.view(fc2.size(0), -1, 8, 8)  # 8x8x64
        initial32 = self.initial32(initial8)  # 32x32x32
        initial64 = self.initial64(initial32)  # 64x64x16
        initial128 = self.initial128(initial64)  # 128x128x8

        tmp = torch.cat([initial8, res4], 1)
        before_select8 = self.before_select8(tmp)  # 8x8x(64+512)
        reconstruct8 = self.reconstruct8(before_select8)  # 8x8x(64+512)

        reconstruct_upsample16 = self.reconstruct_upsample16(reconstruct8)  # 16x16x512
        before_select16 = self.before_select16(res3)  # 16x16x256
        reconstruct16 = self.reconstruct16(
            torch.cat([reconstruct_upsample16, before_select16], 1))   # 16x16x(256+512)

        reconstruct_upsample32 = self.reconstruct_upsample32(reconstruct16)  # 32x32x256
        before_select32 = self.before_select32(
            torch.cat([res2, I32, initial32], 1))  # 32x32x(128+3+32)
        reconstruct32 = self.reconstruct32(
            torch.cat([reconstruct_upsample32, before_select32], 1))  # 32x32x(163+256)
        decoded_img32 = self.tanh(self.img32(reconstruct32))  # 32x32x3

        reconstruct_upsample64 = self.reconstruct_upsample64(reconstruct32)  # 64x64x128
        before_select64 = self.before_select64(
            torch.cat([res1, I64, initial64], 1))  # 64x64x(64+3+16)
        # 64x64x(128+83+3)
        reconstruct64 = self.reconstruct64(torch.cat(
            [reconstruct_upsample64, before_select64,
             torch.nn.functional.interpolate(
                 decoded_img32, (64, 64), mode='bilinear', align_corners=False)], 1))
        decoded_img64 = self.tanh(self.img64(reconstruct64))  # 64x64x3

        reconstruct_upsample128 = self.reconstruct_upsample128(reconstruct64)  # 128x128x64
        before_select128 = self.before_select128(
            torch.cat([res0, I128, initial128], 1))  # 128x128x(64+3+8)
        reconstruct128 = self.reconstruct128(  # 128x128x(64+75+3+64+3)
            torch.cat(
                [reconstruct_upsample128,
                 before_select128,
                 torch.nn.functional.interpolate(
                     decoded_img64, (128, 128), mode='bilinear', align_corners=False),
                 local_feature,
                 local_predict], 1))
        conv5 = self.lrelu(self.bn5(self.conv5(reconstruct128)))            # 128x128x64
        res5 = self.res5(conv5)                                             # 128x128x64
        conv6 = self.lrelu(self.bn6(self.conv6(res5)))                      # 128x128x32
        conv7 = self.conv7(conv6)                                           # 128x128x3
        decoded_img128 = self.tanh(conv7)
        return decoded_img128, decoded_img64, decoded_img32, vid


class PartCombiner(BasicModule):
    '''
    x         y
    39.4799 40.2799
    85.9613 38.7062
    63.6415 63.6473
    45.6705 89.9648
    83.9000 88.6898
    this is the mean locaiton of 5 landmarks
    '''
    def __init__(self):
        super(PartCombiner, self).__init__()

    def forward(self, f_left_eye, f_right_eye, f_nose, f_mouth):
        f_left_eye = torch.nn.functional.pad(
            f_left_eye,
            (args.leye_w - args.EYE_W // 2 - 1,
             args.input_img_size - (args.leye_w + args.EYE_W // 2 - 1),
             args.leye_h - args.EYE_H // 2 - 1,
             args.input_img_size - (args.leye_h + args.EYE_H // 2 - 1)),
            value=-1)
        f_right_eye = torch.nn.functional.pad(
            f_right_eye,
            (args.reye_w - args.EYE_W // 2 - 1,
             args.input_img_size - (args.reye_w + args.EYE_W // 2 - 1),
             args.reye_h - args.EYE_H // 2 - 1,
             args.input_img_size - (args.reye_h + args.EYE_H // 2 - 1)),
            value=-1)
        f_nose = torch.nn.functional.pad(
            f_nose,
            (args.nose_w - args.NOSE_W // 2 - 1,
             args.input_img_size - (args.nose_w + args.NOSE_W // 2 - 1),
             args.nose_h - args.NOSE_H // 2 - 1,
             args.input_img_size - (args.nose_h + args.NOSE_H // 2 - 1)),
            value=-1)
        f_mouth = torch.nn.functional.pad(
            f_mouth,
            (args.mouth_w - args.MOUTH_W // 2 - 1,
             args.input_img_size - (args.mouth_w + args.MOUTH_W // 2 - 1),
             args.mouth_h - args.MOUTH_H // 2 - 1,
             args.input_img_size - (args.mouth_h + args.MOUTH_H // 2 - 1)),
            value=-1)
        return torch.max(
                    torch.stack([f_left_eye, f_right_eye, f_nose, f_mouth], dim=0), 
                    dim=0
               )[0]


class Generator(BasicModule):

    def __init__(self, num_people=0):
        super(Generator, self).__init__()
        self.localpath_left_eye = LocalPath(args.localpath_feature_layer_dim)
        self.localpath_right_eye = LocalPath(args.localpath_feature_layer_dim)
        self.localpath_nose = LocalPath(args.localpath_feature_layer_dim)
        self.localpath_mouth = LocalPath(args.localpath_feature_layer_dim)
        self.globalpath = GlobalPath(args.localpath_feature_layer_dim)
        self.combiner = PartCombiner()
        # For optimizer
        # self.path_params = list(self.localpath_left_eye.parameters()) + \
        #     list(self.localpath_right_eye.parameters()) + \
        #     list(self.localpath_nose.parameters()) + \
        #     list(self.localpath_mouth.parameters()) + \
        #     list(self.globalpath.parameters())
        weight_initialize(self, "kaiming", "LeakyReLU")

    def forward(self, I128, I64, I32, left_eye, right_eye, nose, mouth, noise):
        # pass through local pathway
        # MxNx3           MxNx64
        left_eye_patch, left_eye_feature_patch_before_tanh = self.localpath_left_eye(left_eye)
        right_eye_patch, right_eye_feature_patch_before_tanh = self.localpath_right_eye(right_eye)
        nose_patch, nose_feature_patch_before_tanh = self.localpath_nose(nose)
        mouth_patch, mouth_feature_patch_before_tanh = self.localpath_mouth(mouth)

        # combine
        combine_patches = self.combiner(left_eye_patch, right_eye_patch, nose_patch, mouth_patch)
        combine_feature_patches = self.combiner(left_eye_feature_patch_before_tanh,
                                                right_eye_feature_patch_before_tanh,
                                                nose_feature_patch_before_tanh,
                                                mouth_feature_patch_before_tanh)
        combine_input_patches = self.combiner(left_eye, right_eye, nose, mouth)

        # pass through global pathway
        generated_I128, generated_I64, generated_I32, vid = self.globalpath(
            I128, I64, I32, combine_patches,
            combine_feature_patches, noise)

        return (generated_I128, generated_I64, generated_I32, combine_patches,
                left_eye_patch, right_eye_patch, nose_patch, mouth_patch, combine_input_patches)
