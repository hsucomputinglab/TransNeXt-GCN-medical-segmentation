import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.decoder import CASCADE_Add
from lib.transnext import transnext_base

logger = logging.getLogger(__name__)


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class TRANSNEXT_Cascaded(nn.Module):
    def __init__(self, n_class=1, img_size_s1=(224, 224), img_size_s2=(224, 224), model_scale='small',
                 decoder_aggregation='additive', interpolation='bilinear'):
        super(TRANSNEXT_Cascaded, self).__init__()

        self.n_class = n_class
        self.img_size_s1 = img_size_s1
        self.img_size_s2 = img_size_s2
        self.model_scale = model_scale
        self.decoder_aggregation = decoder_aggregation
        self.interpolation = interpolation

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

        self.backbone1 = transnext_base()
        self.backbone2 = transnext_base()

        print('Loading:', './pretrained_pth/transnext/transnext_base_224_1k.pth')
        state_dict1 = torch.load('./pretrained_pth/transnext/transnext_base_224_1k.pth')
        self.backbone1.load_state_dict(state_dict1, strict=False)

        print('Loading:', './pretrained_pth/transnext/transnext_base_224_1k.pth')
        state_dict2 = torch.load('./pretrained_pth/transnext/transnext_base_224_1k.pth')
        self.backbone2.load_state_dict(state_dict2, strict=False)

        self.channels = [768, 384, 192, 96]

        # decoder initialization
        if self.decoder_aggregation == 'additive':
            self.decoder = CASCADE_Add(channels=self.channels)
        else:
            sys.exit(
                "'" + self.decoder_aggregation + "' is not a valid decoder aggregation! Currently supported aggregations are 'additive' and 'concatenation'.")

        # Prediction heads initialization
        self.out_head1 = nn.Conv2d(self.channels[0], self.n_class, 1)
        self.out_head2 = nn.Conv2d(self.channels[1], self.n_class, 1)
        self.out_head3 = nn.Conv2d(self.channels[2], self.n_class, 1)
        self.out_head4 = nn.Conv2d(self.channels[3], self.n_class, 1)

        self.out_head4_in = nn.Conv2d(self.channels[3], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        # print(x.shape)

        # transformer backbone as encoder
        f1 = self.backbone1(F.interpolate(x, size=self.img_size_s1, mode=self.interpolation))

        # decoder
        x11_o, x12_o, x13_o, x14_o = self.decoder(f1[3], [f1[2], f1[1], f1[0]])

        # prediction heads
        p11 = self.out_head1(x11_o)
        p12 = self.out_head2(x12_o)
        p13 = self.out_head3(x13_o)
        p14 = self.out_head4(x14_o)
        print(1)
        print([p11.shape, p12.shape, p13.shape, p14.shape])
        # calculate feedback from 1st decoder
        p14_in = self.out_head4_in(x14_o)
        p14_in = self.sigmoid(p14_in)

        p11 = F.interpolate(p11, scale_factor=32, mode=self.interpolation)
        p12 = F.interpolate(p12, scale_factor=16, mode=self.interpolation)
        p13 = F.interpolate(p13, scale_factor=8, mode=self.interpolation)
        p14 = F.interpolate(p14, scale_factor=4, mode=self.interpolation)
        print(2)
        print([p11.shape, p12.shape, p13.shape, p14.shape])

        p14_in = F.interpolate(p14_in, scale_factor=4, mode=self.interpolation)

        # apply feedback from 1st decoder to input
        x_in = x * p14_in

        f2 = self.backbone2(F.interpolate(x_in, size=self.img_size_s2, mode=self.interpolation))

        skip1_0 = F.interpolate(f1[0], size=(f2[0].shape[-2:]), mode=self.interpolation)
        skip1_1 = F.interpolate(f1[1], size=(f2[1].shape[-2:]), mode=self.interpolation)
        skip1_2 = F.interpolate(f1[2], size=(f2[2].shape[-2:]), mode=self.interpolation)
        skip1_3 = F.interpolate(f1[3], size=(f2[3].shape[-2:]), mode=self.interpolation)

        x21_o, x22_o, x23_o, x24_o = self.decoder(f2[3] + skip1_3, [f2[2] + skip1_2, f2[1] + skip1_1, f2[0] + skip1_0])

        p21 = self.out_head1(x21_o)
        p22 = self.out_head2(x22_o)
        p23 = self.out_head3(x23_o)
        p24 = self.out_head4(x24_o)
        print(3)
        print([p21.shape, p22.shape, p23.shape, p24.shape])

        p21 = F.interpolate(p21, size=(p11.shape[-2:]), mode=self.interpolation)
        p22 = F.interpolate(p22, size=(p12.shape[-2:]), mode=self.interpolation)
        p23 = F.interpolate(p23, size=(p13.shape[-2:]), mode=self.interpolation)
        p24 = F.interpolate(p24, size=(p14.shape[-2:]), mode=self.interpolation)

        p1 = p11 + p21
        p2 = p12 + p22
        p3 = p13 + p23
        p4 = p14 + p24
        print(4)
        print([p1.shape, p2.shape, p3.shape, p4.shape])

        return p1, p2, p3, p4



