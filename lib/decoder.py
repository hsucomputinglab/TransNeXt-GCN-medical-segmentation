import torch
import torch.nn as nn
from lib.gcn_lib import Grapher as GCB


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, groups=1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_pool_out)))
        # print(x.shape)
        max_pool_out = self.max_pool(x)  # torch.topk(x,3, dim=1).values

        max_out = self.fc2(self.relu1(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CASCADE_Add(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], drop_path_rate=0.0, img_size=224, k=11, padding=5, conv='mr',
                 gcb_act='gelu'):
        super(CASCADE_Add, self).__init__()

        # 基本卷積及上採樣模組
        self.Conv_1x1 = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])

        self.Up3 = up_conv(ch_in=channels[0], ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1], F_l=channels[1], F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1], ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2], F_l=channels[2], F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=channels[2], ch_out=channels[2])

        self.Up1 = up_conv(ch_in=channels[2], ch_out=channels[3])
        self.AG1 = Attention_block(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2))
        self.ConvBlock1 = conv_block(ch_in=channels[3], ch_out=channels[3])

        # GCB 參數設定（與原先相同）
        self.padding = padding
        self.k = k  # neighbor num
        self.conv = conv  # graph conv layer 選項
        self.gcb_act = gcb_act  # 激活函數
        self.gcb_norm = 'batch'
        self.bias = True
        self.use_dilation = True
        self.epsilon = 0.2
        self.use_stochastic = False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [self.drop_path, self.drop_path, self.drop_path, self.drop_path]
        self.num_knn = [self.k, self.k, self.k, self.k]
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4

        # 定義各層的 GCB 模組
        self.gcb4 = nn.Sequential(
            GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm, self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0],
                n=self.HW // (4 * 4 * 4), drop_path=self.dpr[0], relative_pos=True, padding=self.padding),
        )
        self.gcb3 = nn.Sequential(
            GCB(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm, self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1],
                n=self.HW // (4 * 4), drop_path=self.dpr[1], relative_pos=True, padding=self.padding),
        )
        self.gcb2 = nn.Sequential(
            GCB(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm, self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2],
                n=self.HW // 4, drop_path=self.dpr[2], relative_pos=True, padding=self.padding),
        )
        self.gcb1 = nn.Sequential(
            GCB(channels[3], self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm, self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3],
                n=self.HW, drop_path=self.dpr[3], relative_pos=True, padding=self.padding),
        )

        # Channel Attention 與 Spatial Attention 模組
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(channels[1])
        self.CA2 = ChannelAttention(channels[2])
        self.CA1 = ChannelAttention(channels[3])
        self.SA = SpatialAttention()

        # 新增的學習門控參數，初始值設定為0，使網路初期傾向不使用 GCB 分支
        self.alpha4 = nn.Parameter(torch.zeros(1))
        self.alpha3 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.alpha1 = nn.Parameter(torch.zeros(1))

    def forward(self, x, skips):
        # 第一層處理：1x1 conv 與注意力機制
        d4 = self.Conv_1x1(x)
        d4_att = self.CA4(d4) * d4
        d4_att = self.SA(d4_att) * d4_att

        # 利用殘差連接將原始特徵與經 GCB 處理後的特徵融合
        gcb4_out = self.gcb4(d4_att)
        d4 = self.ConvBlock4(d4_att + self.alpha4 * gcb4_out)

        # 第三層
        d3 = self.Up3(d4)
        x3 = self.AG3(g=d3, x=skips[0])
        d3 = d3 + x3
        d3_att = self.CA3(d3) * d3
        d3_att = self.SA(d3_att) * d3_att
        gcb3_out = self.gcb3(d3_att)
        d3 = self.ConvBlock3(d3_att + self.alpha3 * gcb3_out)

        # 第二層
        d2 = self.Up2(d3)
        x2 = self.AG2(g=d2, x=skips[1])
        d2 = d2 + x2
        d2_att = self.CA2(d2) * d2
        d2_att = self.SA(d2_att) * d2_att
        gcb2_out = self.gcb2(d2_att)
        d2 = self.ConvBlock2(d2_att + self.alpha2 * gcb2_out)

        # 第一層
        d1 = self.Up1(d2)
        x1 = self.AG1(g=d1, x=skips[2])
        d1 = d1 + x1
        d1_att = self.CA1(d1) * d1
        d1_att = self.SA(d1_att) * d1_att
        gcb1_out = self.gcb1(d1_att)
        d1 = self.ConvBlock1(d1_att + self.alpha1 * gcb1_out)

        return d4, d3, d2, d1
