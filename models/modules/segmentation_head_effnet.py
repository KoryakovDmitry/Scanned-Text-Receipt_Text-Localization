# xuatpham

import torch
from torch import nn
import torch.nn.functional as F


class FPEM_FFM_effnet(nn.Module):
    def __init__(self, backbone_out_channels, **kwargs):
        """
        PANnet
        :param backbone_out_channels
        """
        super().__init__()
        fpem_repeat = kwargs.get('fpem_repeat', 2)
        conv_out = 128

        # reduce layers 
        self.reduce_conv_bl0 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[0], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_bl1 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[1], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_bl2 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[2], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )
        self.reduce_conv_bl3 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[3], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )  
        self.reduce_conv_bl4 = nn.Sequential(
            nn.Conv2d(in_channels=backbone_out_channels[4], out_channels=conv_out, kernel_size=1),
            nn.BatchNorm2d(conv_out),
            nn.ReLU()
        )      


        self.fpems = nn.ModuleList()
        for i in range(fpem_repeat):
            self.fpems.append(FPEM(conv_out))

        self.out_conv = nn.Conv2d(in_channels=conv_out * len(backbone_out_channels), out_channels=6, kernel_size=1)

    def forward(self, x):
        bl0, bl1, bl2, bl3, bl4 = x
        # reduce channel
        bl0 = self.reduce_conv_bl0(bl0)
        bl1 = self.reduce_conv_bl1(bl1)
        bl2 = self.reduce_conv_bl2(bl2)
        bl3 = self.reduce_conv_bl3(bl3)
        bl4 = self.reduce_conv_bl4(bl4)

        # FPEM
        for i, fpem in enumerate(self.fpems):
            bl0, bl1, bl2, bl3, bl4 = fpem(bl0, bl1, bl2, bl3, bl4)
            if i == 0:
                bl0_ffm = bl0
                bl1_ffm = bl1
                bl2_ffm = bl2
                bl3_ffm = bl3
                bl4_ffm = bl4
            else:
                bl0_ffm += bl0
                bl1_ffm += bl1
                bl2_ffm += bl2
                bl3_ffm += bl3
                bl4_ffm += bl4
        
        # # get mean instead of addition
        # length_fpems = len(self.fpems)
        # if length_fpems != 0:
        #     bl0_ffm /= length_fpems
        #     bl1_ffm /= length_fpems
        #     bl2_ffm /= length_fpems
        #     bl3_ffm /= length_fpems
        # # end


        # FFM
        bl4 = F.interpolate(bl4_ffm, bl0_ffm.size()[-2:], mode='bilinear')
        bl3 = F.interpolate(bl3_ffm, bl0_ffm.size()[-2:], mode='bilinear')
        bl2 = F.interpolate(bl2_ffm, bl0_ffm.size()[-2:], mode='bilinear')
        bl1 = F.interpolate(bl1_ffm, bl0_ffm.size()[-2:], mode='bilinear')
        Fy = torch.cat([bl0_ffm, bl1, bl2, bl3, bl4], dim=1)
        y = self.out_conv(Fy)
        return y


class FPEM(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add4 = SeparableConv2d(in_channels, in_channels, 1)

        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add4 = SeparableConv2d(in_channels, in_channels, 2)

        
    def forward(self, bl0, bl1, bl2, bl3, bl4):
        # up
        bl3 = self.up_add1(self._upsample_add(bl4, bl3))
        bl2 = self.up_add2(self._upsample_add(bl3, bl2))
        bl1 = self.up_add3(self._upsample_add(bl2, bl1))
        bl0 = self.up_add4(self._upsample_add(bl1, bl0))

        # down
        bl1 = self.down_add1(self._upsample_add(bl1, bl0))
        bl2 = self.down_add2(self._upsample_add(bl2, bl1))
        bl3 = self.down_add3(self._upsample_add(bl3, bl2))
        bl4 = self.down_add4(self._upsample_add(bl4, bl3))
        return bl0, bl1, bl2, bl3, bl4


    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1,
                                        stride=stride, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
