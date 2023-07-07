import torch
from torch import nn
from collections import OrderedDict
from backbones.asff import ASFFNetwork
from .seg_detector import SegDetector
from torch.nn import BatchNorm2d

class SegDetectorAsff(SegDetector):

    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 inner_channels=256, k=10,
                 bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        super(SegDetector, self).__init__()
        self.k = k
        self.serial = serial
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, bias=bias)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, bias=bias)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, bias=bias)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, bias=bias)

        dim = [inner_channels // 4, inner_channels // 4, inner_channels // 4, inner_channels // 4]
        out_channels = [inner_channels // 4, inner_channels // 4, inner_channels // 4, inner_channels // 4]
        self.asff0 = ASFFNetwork(level=0, dim=dim, out_channnels=out_channels)
        self.asff1 = ASFFNetwork(level=1, dim=dim, out_channnels=out_channels)
        self.asff2 = ASFFNetwork(level=2, dim=dim, out_channnels=out_channels)
        self.asff3 = ASFFNetwork(level=3, dim=dim, out_channnels=out_channels)

        self.conv5 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(inner_channels, inner_channels//4, 3, padding=1, bias=bias)

        self.out5 = nn.Upsample(scale_factor=8, mode='nearest')
        self.out4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.out3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.conv5.apply(self.weights_init)
        self.conv4.apply(self.weights_init)
        self.conv3.apply(self.weights_init)
        self.conv2.apply(self.weights_init)

        self.prob = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.prob.apply(self.weights_init)

        self.adaptive = adaptive
        self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
        self.thresh.apply(self.weights_init)



    # 像素不失2的倍数情况 up5 + in4 维度不统一
    def forward(self, X):
        c2, c3, c4, c5 = X
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        out4 = self.up5(in5) + in4  # 1/16
        out3 = self.up4(out4) + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4

        p5 = self.conv5(in5)
        p4 = self.conv4(out4)
        p3 = self.conv3(out3)
        p2 = self.conv2(out2)

        asff0 = self.asff0(p5, p4, p3, p2)
        asff1 = self.asff1(p5, p4, p3, p2)
        asff2 = self.asff2(p5, p4, p3, p2)
        asff3 = self.asff3(p5, p4, p3, p2)

        out0 = self.out5(asff0)
        out1 = self.out4(asff1)
        out2 = self.out3(asff2)
        out3 = asff3

        fuse = torch.cat((out0, out1, out2, out3), 1)
        prob = self.prob(fuse)
        if self.training:
            result = OrderedDict(prob=prob)
        else:
            return prob
        if self.training:
            if self.serial:
                fuse = torch.cat(
                        (fuse, nn.functional.interpolate(
                            prob, fuse.shape[2:])), 1)
            thresh = self.thresh(fuse)
            binary = self.step_function(prob, thresh)
            result.update(thresh=thresh, binary=binary)
        return result


    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels // 4, 3, padding=1, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            nn.BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))