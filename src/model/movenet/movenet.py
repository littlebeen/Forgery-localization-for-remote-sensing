
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 18:26:31 2020

@author: huangyu45
"""

import torch
import torch.nn as nn
import numpy as np




# class SEBottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
#         super(SEBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.se = SELayer(planes * 4, reduction)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.se(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Movenet(nn.Module):
    def __init__(self, args, data_size=512):
        super(Movenet, self).__init__()

        bh = data_size // 8
        bw = data_size // 8

        bn = 32  # base output number
        # -------------- conv -------------- #
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),  # 用卷积替代池化：stride=2
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv1c = nn.Conv2d(3, bn, kernel_size=1, padding=0, stride=1)
        self.conv11 = nn.Sequential(
            nn.Conv2d(3, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv11b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),  # 用卷积替代池化：stride=2
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv11ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * bn, 2 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * bn),
            nn.ReLU(True)
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(2 * bn, 2 * bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(2 * bn),
            nn.ReLU(True)
        )
        self.conv2c = nn.Conv2d(3, bn, kernel_size=2, padding=0, stride=2)
        self.conv22 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv22b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv22ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(3 * bn, 4 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * bn),
            nn.ReLU(True)
        )
        self.conv3b = nn.Sequential(
            nn.Conv2d(4 * bn, 4 * bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(4 * bn),
            nn.ReLU(True)
        )
        self.conv3c = nn.Conv2d(3, bn, kernel_size=4, padding=0, stride=4)
        self.conv33 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv33b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv33ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(5 * bn, 8 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * bn),
            nn.ReLU(True)
        )
        self.conv4b = nn.Sequential(
            nn.Conv2d(8 * bn, 8 * bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(8 * bn),
            nn.ReLU(True)
        )
        self.conv4c = nn.Conv2d(3, bn, kernel_size=8, padding=0, stride=8)
        self.conv44 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv44b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv44ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # -------------- middle -------------- #
        self.conv5 = nn.Sequential(
            nn.Conv2d(9 * bn, 8 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * bn),
            nn.ReLU(True)
        )
        # self.conv5c = nn.Conv2d(3, bn, kernel_size=16, padding=0, stride=16)
        self.conv55 = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.conv55ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )

        # -------------- deconv -------------- #
        self.deconv4 = nn.Sequential(
            nn.Upsample(size=(bw, bh), mode='bilinear'),
            nn.Conv2d(9 * bn, 8 * bn, kernel_size=1),
        )
        self.att4 = SELayer(channel=10*bn, reduction=16)

        self.deconv4b = nn.Sequential(
            nn.Conv2d(10 * bn, 8 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * bn),
            nn.ReLU(True)
        )

        self.deconv44 = nn.Sequential(
            nn.Upsample(size=(bw, bh), mode='bilinear'),
            nn.Conv2d(bn, bn, kernel_size=1),
        )
        self.deconv44b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv44ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(size=(2 * bw, 2 * bh), mode='bilinear'),
            nn.Conv2d(8 * bn, 4 * bn, kernel_size=1),
        )
        self.att3 = SELayer(channel=6 * bn, reduction=16)
        self.deconv3b = nn.Sequential(
            nn.Conv2d(6 * bn, 4 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * bn),
            nn.ReLU(True)
        )
        self.deconv33 = nn.Sequential(
            nn.Upsample(size=(2 * bw, 2 * bh), mode='bilinear'),
            nn.Conv2d(bn, bn, kernel_size=1),
        )
        self.deconv33b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv33ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.Upsample(size=(4 * bw, 4 * bh), mode='bilinear'),
            nn.Conv2d(4 * bn, 2 * bn, kernel_size=1),
        )
        self.att2 = SELayer(channel=4 * bn, reduction=16)
        self.deconv2b = nn.Sequential(
            nn.Conv2d(4 * bn, 2 * bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * bn),
            nn.ReLU(True)
        )
        self.deconv22 = nn.Sequential(
            nn.Upsample(size=(4 * bw, 4 * bh), mode='bilinear'),
            nn.Conv2d(bn, bn, kernel_size=1),
        )
        self.deconv22b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv22ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.Upsample(size=(8 * bw, 8 * bh), mode='bilinear'),
            nn.Conv2d(2 * bn, bn, kernel_size=1),
        )
        self.att1 = SELayer(channel=3 * bn, reduction=16)
        self.deconv1b = nn.Sequential(
            nn.Conv2d(3 * bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv11 = nn.Sequential(
            nn.Upsample(size=(8 * bw, 8 * bh), mode='bilinear'),
            nn.Conv2d(bn, bn, kernel_size=1),
        )
        self.deconv11b = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=3, padding=1),
            nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        self.deconv11ada = nn.Sequential(
            nn.Conv2d(bn, bn, kernel_size=1, padding=0, stride=1),
            # nn.BatchNorm2d(bn),
            nn.ReLU(True)
        )
        # -------------- end -------------- #
        self.conv_end = nn.Conv2d(bn, 2, kernel_size=3, padding=1, bias=False)
        self.conv_end_label_edge = nn.Conv2d(bn, 2, kernel_size=3, padding=1, bias=False)
        self.conv_end_img_edge = nn.Conv2d(bn, 1, kernel_size=3, padding=1, bias=False)

    # data: chw  ,  label: hwc  简单
    def forward(self, data, label=None):
        # data: [batch_size, c, h, w]

        L1 = self.conv1(data)
        L1b = self.conv1b(L1)
        L1c = self.conv1c(data)
        L11 = self.conv11(data)
        L11b = self.conv11b(L11)
        L11a = self.conv11ada(L11b)
        L11ada = L11a + L11b
        L1b2 = torch.cat([L1b, L11ada], 1)

        L2 = self.conv2(L1b2)
        L2b = self.conv2b(L2)
        L2c = self.conv2c(data)
        L22 = self.conv22(L11b)
        L22b = self.conv22b(L22)
        L22a = self.conv22ada(L22b)
        L22ada = L22a+L22b
        L2b2 = torch.cat([L2b, L22ada], 1)

        L3 = self.conv3(L2b2)
        L3b = self.conv3b(L3)
        L3c = self.conv3c(data)
        L33 = self.conv33(L22b)
        L33b = self.conv33b(L33)
        L33a = self.conv33ada(L33b)
        L33ada = L33a + L33b
        L3b2 = torch.cat([L3b, L33ada], 1)

        L4 = self.conv4(L3b2)
        L4b = self.conv4b(L4)
        L4c = self.conv4c(data)
        L44 = self.conv44(L33b)
        L44b = self.conv44b(L44)
        L44a = self.conv44ada(L44b)
        L44ada = L44a + L44b
        L4b2 = torch.cat([L4b, L44a], 1)

        L5 = self.conv5(L4b2)
        L55 = self.conv55(L44b)
        L55a = self.conv55ada(L55)
        L55ada = L55a + L55
        L52 = torch.cat([L5, L55ada], 1)

        DL4 = self.deconv4(L52)
        DL4_add = DL4 + L4
        DL44 = self.deconv44(L55)
        DL44_add = DL44 + L44
        DL44b = self.deconv44b(DL44_add)
        DL44bada = DL44b + DL44_add
        DL4_cat = torch.cat([DL4_add, L4c, DL44bada], 1)
        DL4_att = self.att4(DL4_cat)
        DL4b = self.deconv4b(DL4_att)

        DL3 = self.deconv3(DL4b)
        DL3_add = DL3 + L3
        DL33 = self.deconv33(DL44b)
        DL33_add = DL33 + L33
        DL33b = self.deconv33b(DL33_add)
        DL33ba = self.deconv33ada(DL33b)
        DL33bada = DL33ba + DL33b
        DL3_cat = torch.cat([DL3_add, L3c, DL33bada], 1)
        DL3_att = self.att3(DL3_cat)
        DL3b = self.deconv3b(DL3_att)

        DL2 = self.deconv2(DL3b)
        DL2_add = DL2 + L2
        DL22 = self.deconv22(DL33b)
        DL22_add = DL22 + L22
        DL22b = self.deconv22b(DL22_add)
        DL22ba = self.deconv22ada(DL22b)
        DL22bada = DL22ba + DL22_add
        DL2_cat = torch.cat([DL2_add, L2c, DL22bada], 1)
        DL2_att = self.att2(DL2_cat)
        DL2b = self.deconv2b(DL2_att)

        DL1 = self.deconv1(DL2b)
        DL1_add = DL1 + L1
        DL11 = self.deconv11(DL22b)
        DL11_add = DL11 + L11
        DL11b = self.deconv11b(DL11_add)
        DL11ba = self.deconv11ada(DL11b)
        DL11bada = DL11ba + DL11b
        DL1_cat = torch.cat([DL1_add, L1c, DL11bada], 1)
        DL1_att = self.att1(DL1_cat)
        DL1b = self.deconv1b(DL1_att)

        end = self.conv_end(DL1b)
        end_label_edge = self.conv_end_label_edge(DL1b)
        end_img_edge = self.conv_end_img_edge(DL11b)

        output = torch.argmax(end, dim=1)
        output2 = torch.argmax(end_label_edge, dim=1)
        output3 = end_img_edge  # torch.argmax(end_img_edge, dim=1)
        if not self.training: return output
        return end  #

