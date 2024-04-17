import torch
import torch.nn as nn
import torch.nn.functional as F

# s-1
# k-p-1
# (In-1)*s-2*p+k

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
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
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]
    
class BasicBlock(nn.Module):
    negative_slope = 0.2

    def get_conv_group(self, planes, kernel_size):
        return nn.Sequential(
            nn.Conv2d(planes // 4, planes, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(planes, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, padding_mode='reflect', groups=planes),
            nn.InstanceNorm2d(planes, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(planes, planes // 4, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(planes // 4, affine=True, track_running_stats=True),
        )

    def get_se(self, planes):
        return nn.Sequential(
            nn.Conv2d(planes//4, planes, kernel_size=1),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(planes, planes//4, kernel_size=1),
            nn.Sigmoid()
        )

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()

        self.split = SplitBlock(0.5)
        self.shuffle = ShuffleBlock(groups=4)

        self.unify = nn.Sequential()
        if in_planes != planes:
            self.unify = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(planes, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)
            )

        self.model_3 = self.get_conv_group(planes, 3)
        self.se_3 = self.get_se(planes)
        self.model_5 = self.get_conv_group(planes, 5)
        self.se_5 = self.get_se(planes)
        self.model_7 = self.get_conv_group(planes, 7)
        self.se_7 = self.get_se(planes)

    def forward(self, x):
        x = self.unify(x)
        xl, xr = self.split(x)
        xn, x3 = self.split(xl)
        x5, x7 = self.split(xr)
        out3 = self.model_3(x3)
        out3 = out3 * self.se_3(F.avg_pool2d(out3, out3.size(2))) + x3
        out5 = self.model_5(x5)
        out5 = out5 * self.se_5(F.avg_pool2d(out5, out5.size(2))) + x5
        out7 = self.model_7(x7)
        out7 = out7 * self.se_7(F.avg_pool2d(out7, out7.size(2))) + x7
        out = torch.cat([out3, out5, out7, xn], 1)
        out = self.shuffle(out)
        return out

class Tree(nn.Module):
    def __init__(self, block, in_planes, planes, level=1, block_num=4, origin=True):
        super(Tree, self).__init__()
        self.level = level
        self.block_num = block_num
        if origin:
            self.prev_root = block(in_planes, planes)
        else:
            self.prev_root = block(planes, planes)
        if level == 1:
            self.root = block(block_num * planes, planes)
            self.__setattr__('block_%d' % 0, block(in_planes, planes))
        else:
            self.root = block((level + block_num) * planes, planes)
            for i in reversed(range(1, level)):
                if origin:
                    subtree = Tree(block, in_planes, planes, level=i, origin=origin)
                else:
                    subtree = Tree(block, planes, planes, level=i, origin=origin)
                self.__setattr__('level_%d' % i, subtree)
                origin = False
            self.__setattr__('block_%d' % 0, block(planes, planes))
        for i in range(1, block_num):
            self.__setattr__('block_%d' % i, block(planes, planes))

    def forward(self, x):
        #print("now level", self.level)
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            #print('finish level %d' %self.level,'-level %d' %i)
            xs.append(x)
        #print('finish level', self.level)
        for i in range(self.block_num):
            block_i = self.__getattr__('block_%d' % i)
            x = block_i(x)
            xs.append(x)
        xs = torch.cat(xs, 1)
        out = self.root(xs)
        return out

class Stem_block(nn.Module):
    negative_slope = 0.2
    def get_conv_group(self, planes, kernel_size):
        return nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, padding_mode = 'reflect', groups=planes),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(planes)
        )

    def get_se(self, planes):
        return nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(planes, planes, kernel_size=1),
            nn.Sigmoid()
        )

    def __init__(self, in_planes, planes, block_num=3, kernel_size=3):
        super(Stem_block, self).__init__()
        self.block_num = block_num
        self.unify = nn.Sequential()
        if in_planes != planes:
            self.unify = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(planes),
                nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            )

        for i in range(block_num):
            self.__setattr__('res_%d' % i, self.get_conv_group(planes, kernel_size))
            self.__setattr__('se_%d' % i, self.get_se(planes))

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(planes, planes, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
        )

    def forward(self, x):
        x = self.unify(x)

        for i in range(self.block_num):
            res_i = self.__getattr__('res_%d' % i)
            out = res_i(x)
            se_i = self.__getattr__('se_%d' % i)
            out = out*se_i(out)
            x = x + out

        out = self.upsample(x)
        return out

class Generator(nn.Module):
    negative_slope = 0.2

    def get_upsample(self, planes):
        return nn.Sequential(
            nn.ConvTranspose2d(planes, planes, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(planes, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)
        )

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        in_planes = 512
        model = []
        model += [nn.ConvTranspose2d(z_dim, in_planes, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(in_planes),
                  nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)]  #2x2

        model += [Stem_block(in_planes=in_planes, planes=in_planes//2, block_num=3, kernel_size=3)] # 4x4
        model += [Stem_block(in_planes=in_planes//2, planes=in_planes//2, block_num=4, kernel_size=3)] # 8x8

        model += [
            Tree(BasicBlock, in_planes=in_planes//2, planes=in_planes//4, level=2, block_num=3), # 16x16
            self.get_upsample(planes=in_planes//4)
        ]

        model += [
            Tree(BasicBlock, in_planes=in_planes//4, planes=in_planes//8, level=2, block_num=3), # 32x32
            self.get_upsample(planes=in_planes//8)
        ]

        model += [
            Tree(BasicBlock, in_planes=in_planes//8, planes=in_planes//16, level=2, block_num=3), # 64x64
            self.get_upsample(planes=in_planes//16),
            Tree(BasicBlock, in_planes=in_planes//16, planes=in_planes//16, level=2, block_num=3)
        ]

        model += [nn.Conv2d(in_planes//16, 3, 7, stride=1, padding=3, padding_mode = 'reflect'), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def test():
    net = Generator(z_dim=128)
    #net = Stem_block(in_planes=4,planes=16)
    #net = Tree(BasicBlock, 32, 16, 3, 3)
    print(net)
    from torchsummary import summary
    summary(net, input_size=(8, 128, 1, 1))

    x = torch.randn(8, 128, 1, 1)
    y = net(x)
    from torchviz import make_dot
    dot = make_dot(y, params=dict(net.named_parameters()))
    # dot.view()
    dot.render("G4_model_graph")
    print(y.size())
    print("successed")
