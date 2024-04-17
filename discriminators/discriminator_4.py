import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(int(x.size(1)) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]

# BasicModel使用了基本的preact残差模块，这样对深层的网络有更好的支持，主体为2个3x3卷积，输出通道数参考了densenet，在基本的输出大小上加上dense_depth
class BasicModel(nn.Module):
    def __init__(self, planes, dense_depth, kernel_size):
        super(BasicModel, self).__init__()
        self.preact = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.b1 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False),
        )
        # b2分支代表基本模块
        self.b2 = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
        )
        # b3分支代表dense_depth
        self.b3 = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes, dense_depth, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
        )

        self.se_1 = nn.Sequential(
            nn.Conv2d(planes, planes//4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(planes//4, planes, kernel_size=1),
            nn.Sigmoid(),
        )

        self.se_2 = nn.Sequential(
            nn.Conv2d(dense_depth, dense_depth // 4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dense_depth // 4, dense_depth, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 预处理
        out = self.preact(x)
        # 进行第一个3x3卷积
        out = self.b1(out)
        # 将第一个结果分别输入b2和b3分支
        out_b2 = self.b2(out)
        out_b3 = self.b3(out)
        # 使用se修改b2分支各通道权重
        w_1 = F.avg_pool2d(out_b2, int(out_b2.size(2)))
        w_1 = self.se_1(w_1)
        out_b2 = out_b2 * w_1
        # 使用se修改b3分支各通道权重
        w_2 = F.avg_pool2d(out_b3, int(out_b3.size(2)))
        w_2 = self.se_2(w_2)
        out_b3 = out_b3 * w_2
        # 进行残差连接
        out_b2 += x
        # 拼接上dense_depth的大小
        out = torch.cat([out_b2,out_b3],1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, last_planes, dense_depth, split_ratio=0.5):
        super(DenseBlock, self).__init__()
        self.last_planes = last_planes//4
        self.dense_depth = dense_depth
        self.split = SplitBlock(split_ratio)
        # 3x3
        self.b3x3 = BasicModel(self.last_planes, dense_depth, 3)
        # 5x5
        self.b5x5 = BasicModel(self.last_planes, dense_depth, 5)
        # 7x7
        self.b7x7 = BasicModel(self.last_planes, dense_depth, 7)
        # densecut
        self.densecut = nn.Sequential(
            nn.BatchNorm2d(last_planes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(last_planes, dense_depth, kernel_size=1, stride=1, bias=False),
        )
        self.shuffle = ShuffleBlock(groups=4)

    def forward(self, x):
        # 将输入通道数平均分为4块
        x1, x2 = self.split(x)
        x1_1, x1_2 = self.split(x1)
        x2_1, x2_2 = self.split(x2)

        # 第一块不处理，第二块进入3x3的BasicModel，第三块进入5x5的BasicModel，第四块进入7x7的BasicModel
        out_12 = self.b3x3(x1_2)
        out_21 = self.b5x5(x2_1)
        out_22 = self.b7x7(x2_2)

        # 对输入进行1x1降维，当作基本的dense_depth部分
        densecut = self.densecut(x)

        d = self.last_planes

        out = torch.cat([
                         out_12[:,:d,:,:],
                         out_21[:,:d,:,:],
                         out_22[:,:d,:,:],
                         x1_1,
                         densecut+out_12[:,d:,:,:]+out_21[:,d:,:,:]+out_22[:,d:,:,:]
                         ],1)

        # 把最后得到的5个块摇匀为4个块
        out = self.shuffle(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        # 先3x3再1x1
        self.b1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        )
        # 先1x1再3x3
        self.b2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.shuffle = ShuffleBlock(groups=2)

    def forward(self, x):
        # left
        out1 = self.b1(x)
        # right
        out2 = self.b2(x)
        # concat
        out = torch.cat([out1, out2], 1)
        out = self.shuffle(out)
        return out

class Block(nn.Module):
    def __init__(self, planes, out_planes, blocks, is_DownBlock = True):
        super(Block, self).__init__()
        self.is_DownBlock = is_DownBlock
        layers = []
        for i in range(blocks):
            lenth = math.sqrt(planes)
            layers.append(DenseBlock(int(planes), 16))
            planes = planes + 16
        self.layers = nn.Sequential(*layers)
        self.downblock = DownBlock(int(planes), out_planes)


    def forward(self, x):
        out = self.layers(x)
        if self.is_DownBlock:
            out = self.downblock(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, net_size = 1):
        super(Discriminator, self).__init__()

        # 对数据进行进入主体前的处理，使用了2个3x3的卷积
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 36, kernel_size=3, stride=1, padding=1, bias=False), #256
            nn.BatchNorm2d(36),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.main = nn.Sequential(
            Block(36, 40, 3),  # 64
            Block(40, 48, 3),  # 32
            Block(48, 64, 3),  # 16
            Block(64, 80, 3),  # 8 4
        )


        # 进行1x1卷积进行升维
        self.layer5 = nn.Sequential(
            nn.BatchNorm2d(80),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(80, 128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.linear = nn.Linear(128, 1)

    def forward(self, x):
        out = self.stem(x)
        # out = F.max_pool2d(out, 3, stride=2, padding=1)
        out = self.main(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.sigmoid(out)
        return out

def test():
    #net = DenseBlock(64,36)
    #net = Block(256, 484,3,False)
    net = Discriminator()
    print(net)
    x = torch.randn(4, 3, 64, 64)
    y = net(x)
    print(y)

    summary(net, input_size=(3, 256, 256))