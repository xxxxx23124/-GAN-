import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    negative_slope = 0.2

    def get_conv_bottleneck(self, last_planes, in_planes, out_planes, dense_depth, kernel_size):
        return nn.Sequential(
            nn.Conv2d(last_planes, in_planes, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(in_planes, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      padding_mode='reflect', groups=in_planes),
            nn.InstanceNorm2d(in_planes, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(in_planes, out_planes + dense_depth, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_planes + dense_depth, affine=True, track_running_stats=True),
        )

    def get_se(self, in_planes, out_planes, dense_depth, feature_size):
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=feature_size, padding=0),
            nn.Conv2d(out_planes + dense_depth, in_planes, kernel_size=1),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(in_planes, out_planes + dense_depth, kernel_size=1),
            nn.Sigmoid()
        )

    def get_out_planes(self):
        if self.is_unify:
            return self.out_planes + 4 * self.dense_depth
        else:
            if self.root:
                return self.out_planes + 4 * self.dense_depth
            else:
                return self.last_planes + 3 * self.dense_depth

    def __init__(self, last_planes, in_planes, out_planes, dense_depth, root, feature_size, is_unify):
        super(BasicBlock, self).__init__()
        self.root = root
        self.last_planes = last_planes
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        self.is_unify = is_unify
        self.unify = nn.Sequential()
        if is_unify:
            self.unify = nn.Sequential(
                nn.Conv2d(last_planes, out_planes+dense_depth, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(out_planes+dense_depth, affine=True, track_running_stats=True)
            )
            self.model_3 = self.get_conv_bottleneck(out_planes+dense_depth, in_planes, out_planes, dense_depth, 3)
            self.model_5 = self.get_conv_bottleneck(out_planes+dense_depth, in_planes, out_planes, dense_depth, 5)
            self.model_7 = self.get_conv_bottleneck(out_planes+dense_depth, in_planes, out_planes, dense_depth, 7)
        else:
            self.model_3 = self.get_conv_bottleneck(last_planes, in_planes, out_planes, dense_depth, 3)
            self.model_5 = self.get_conv_bottleneck(last_planes, in_planes, out_planes, dense_depth, 5)
            self.model_7 = self.get_conv_bottleneck(last_planes, in_planes, out_planes, dense_depth, 7)

        self.shortcut = nn.Sequential()
        if root:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, out_planes + dense_depth, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(out_planes+dense_depth, affine=True, track_running_stats=True)
            )


        self.se_3 = self.get_se(in_planes, out_planes, dense_depth, feature_size)
        self.se_5 = self.get_se(in_planes, out_planes, dense_depth, feature_size)
        self.se_7 = self.get_se(in_planes, out_planes, dense_depth, feature_size)

    def forward(self, x):
        x = self.unify(x)
        out3 = self.model_3(x)
        se3 = self.se_3(out3)
        out3 = out3 * se3
        out5 = self.model_5(x)
        se5 = self.se_5(out5)
        out5 = out5 * se5
        out7 = self.model_7(x)
        se7 = self.se_7(out7)
        out7 = out7 * se7

        x = self.shortcut(x)

        d = self.out_planes
        out = torch.cat([x[:, :d, :, :] + out3[:, :d, :, :] + out5[:, :d, :, :] + out7[:, :d, :, :],
                         x[:, d:, :, :], out3[:, d:, :, :], out5[:, d:, :, :], out7[:, d:, :, :]], 1)
        return out

class Tree(nn.Module):
    def get_out_planes(self):
        return self.root.get_out_planes()
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, level, block_num, feature_size):
        super(Tree, self).__init__()
        self.level = level
        self.block_num = block_num
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        if level == 1:
            self.root_last_planes = out_planes * (block_num - 1)
            if last_planes < out_planes:
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, True)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % 0, sub_block)
            else:
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, False)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % 0, sub_block)
            for i in range(1, block_num):
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, False)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(self.root_last_planes, in_planes * block_num, out_planes, dense_depth, True, feature_size,
                                   False)

        else:
            self.root_last_planes = out_planes * (block_num - 1)
            if last_planes < out_planes:
                self.prev_root = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, True)
            else:
                self.prev_root = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, False)
            self.root_last_planes += self.prev_root.get_out_planes()

            for i in reversed(range(1, level)):
                subtree = Tree(last_planes, in_planes, out_planes, dense_depth, i, block_num, feature_size)
                last_planes = subtree.get_out_planes()
                self.root_last_planes += last_planes
                self.__setattr__('level_%d' % i, subtree)

            for i in range(block_num):
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, False)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(self.root_last_planes, in_planes * block_num, out_planes, dense_depth, True, feature_size,
                                   False)

    def forward(self, x):
        d = self.out_planes
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        for i in range(self.block_num):
            block_i = self.__getattr__('block_%d' % i)
            x = block_i(x)
            xs.append(x[:, :d, :, :])
        xs.append(x[:, d:, :, :])
        xs = torch.cat(xs, 1)
        out = self.root(xs)
        return out

class Stem_block(nn.Module):
    negative_slope = 0.2
    def get_conv_group(self, in_planes, planes, kernel_size, expansion=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, in_planes * expansion, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes * expansion),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(in_planes * expansion, in_planes * expansion, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, padding_mode='reflect', groups=in_planes * expansion),
            nn.BatchNorm2d(in_planes * expansion),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
            nn.Conv2d(in_planes * expansion, planes, kernel_size=1, stride=1, padding=0),
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

        self.shortcut = nn.Sequential()
        if in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(planes),
            )

        for i in range(block_num):
            self.__setattr__('res_%d' % i, self.get_conv_group(in_planes, planes, kernel_size))
            in_planes = planes
            self.__setattr__('se_%d' % i, self.get_se(planes))

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(planes, planes, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True),
        )



    def forward(self, x):
        res_i = self.__getattr__('res_%d' % 0)
        se_i = self.__getattr__('se_%d' % 0)
        out = res_i(x)
        out = out * se_i(out)
        x = self.shortcut(x) + out

        for i in range(1, self.block_num):
            res_i = self.__getattr__('res_%d' % i)
            se_i = self.__getattr__('se_%d' % i)
            out = res_i(x)
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
        last_planes = 512
        model = []
        model += [nn.ConvTranspose2d(z_dim, last_planes, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(last_planes),
                  nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)]  #2x2

        model += [Stem_block(in_planes=last_planes, planes=last_planes//2, block_num=3, kernel_size=3)] # 4x4
        model += [Stem_block(in_planes=last_planes//2, planes=last_planes//2, block_num=4, kernel_size=3)] # 8x8
        last_planes = last_planes//2

        t1 = Tree(last_planes, in_planes=64, out_planes=128, dense_depth=16, level=3, block_num=6,
                  feature_size=8)  # 8x8
        last_planes = t1.get_out_planes()
        model += [t1, self.get_upsample(planes=last_planes)]

        t2 = Tree(last_planes, in_planes=32, out_planes=64, dense_depth=12, level=3, block_num=6,
                  feature_size=16)  # 16x16
        last_planes = t2.get_out_planes()
        model += [t2, self.get_upsample(planes=last_planes)]

        t3 = Tree(last_planes, in_planes=16, out_planes=32, dense_depth=8, level=2, block_num=5,
                  feature_size=32)  # 32x32
        last_planes = t3.get_out_planes()
        t4 = Tree(last_planes, in_planes=8, out_planes=16, dense_depth=4, level=2, block_num=4,
                  feature_size=64)  # 64x64
        model += [t3, self.get_upsample(planes=last_planes), t4]
        last_planes = t4.get_out_planes()

        model += [nn.Conv2d(last_planes, 3, 9, stride=1, padding=4, padding_mode = 'reflect'), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)






def test():
    #net = BasicBlock(last_planes=16, in_planes=32, out_planes=12, dense_depth=8, root=False, feature_size=64)

    #net = Stem_block(in_planes=4,planes=16)
    #net = Tree(last_planes=16, in_planes=8, out_planes=8, dense_depth=8, level=5, block_num=6, feature_size=64)
    net = Generator(z_dim=128)
    #print(net)
    from torchsummary import summary
    summary(net, input_size=(4, 128, 1, 1))


    x = torch.randn(4, 128, 1, 1)
    y = net(x)
    from torchviz import make_dot
    #dot = make_dot(y, params=dict(net.named_parameters()))
    #dot.view()
    #dot.render("G5_model_graph")
    #print(y.size())
    #print(net.get_out_planes())
    print("successed")


