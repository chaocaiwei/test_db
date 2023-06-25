import torch
from torch import nn
from torch.nn import functional as F


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFFNetwork(nn.Module):

    # in_channels=[64, 128, 256, 512]
    # self.dim = [512, 256, 128]
    def __init__(self, level, dim=[512, 256, 128, 64], out_channnels=[1024, 512, 256, 128], rfb=False, vis=False):
        super(ASFFNetwork, self).__init__()
        self.level = level
        self.dim = dim
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(dim[1], self.inter_dim, 3, 2, leaky=False)
            self.stride_level_2 = add_conv(dim[2], self.inter_dim, 3, 2, leaky=False)
            self.stride_level_3 = add_conv(dim[3], self.inter_dim, 3, 2, leaky=False)
            self.expand = add_conv(self.inter_dim, out_channnels[0], 3, 1, leaky=False)
        elif level == 1:
            self.compress_level_0 = add_conv(dim[0], self.inter_dim, 1, 1, leaky=False)
            self.stride_level_2 = add_conv(dim[2], self.inter_dim, 3, 2, leaky=False)
            self.stride_level_3 = add_conv(dim[2], self.inter_dim, 3, 2, leaky=False)
            self.expand = add_conv(self.inter_dim, out_channnels[0], 3, 1, leaky=False)
        elif level == 2:
            self.compress_level_0 = add_conv(dim[0], self.inter_dim, 1, 1, leaky=False)
            self.compress_level_1 = add_conv(dim[1], self.inter_dim, 1, 1, leaky=False)
            self.stride_level_3 = add_conv(dim[2], self.inter_dim, 3, 2, leaky=False)
            self.expand = add_conv(self.inter_dim, out_channnels[0], 3, 1, leaky=False)
        elif level == 3:
            self.compress_level_0 = add_conv(dim[0], self.inter_dim, 1, 1, leaky=False)
            self.compress_level_1 = add_conv(dim[1], self.inter_dim, 1, 1, leaky=False)
            self.compress_level_2 = add_conv(dim[2], self.inter_dim, 1, 1, leaky=False)
            self.expand = add_conv(self.inter_dim, out_channnels[0], 3, 1, leaky=False)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)
        self.weight_level_3 = add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)

        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        if self.level == 0:
            level_0_resized = x_level_0

            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

            level_3_downsampled_inter = F.max_pool2d(x_level_3, 5, stride=4, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')

            level_1_resized = x_level_1

            level_2_resized = self.stride_level_2(x_level_2)

            level_3_downsampled_inter = F.max_pool2d(x_level_3, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(level_3_downsampled_inter)

        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')

            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')

            level_2_resized = x_level_2

            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 3:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=8, mode='nearest')

            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=4, mode='nearest')

            level_2_compressed = self.compress_level_2(x_level_2)
            level_2_resized = F.interpolate(level_2_compressed, scale_factor=2, mode='nearest')

            level_3_resized = x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :]+\
                            level_1_resized * levels_weight[:, 1:2, :, :]+\
                            level_2_resized * levels_weight[:, 2:3, :, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


