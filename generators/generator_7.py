import torch
import torch.nn as nn


class SqueezeExcitationBottleneck(nn.Module):
    negative_slope = 1e-2

    def get_conv_bottleneck(self, last_planes, in_planes, out_planes, dense_depth, kernel_size):
        return nn.Sequential(
            nn.Conv2d(last_planes, in_planes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_planes),
            nn.PReLU(),
            nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      padding_mode='reflect', groups=in_planes),
            nn.BatchNorm2d(in_planes),
            nn.PReLU(),
            nn.Conv2d(in_planes, out_planes + dense_depth, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_planes + dense_depth),
        )

    def get_se(self, in_planes, out_planes, dense_depth, feature_size):
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=feature_size, padding=0),
            nn.Conv2d(out_planes + dense_depth, in_planes, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(in_planes, out_planes + dense_depth, kernel_size=1),
            nn.Sigmoid()
        )

    def __init__(self, last_planes, in_planes, out_planes, dense_depth, kernel_size, feature_size):
        super(SqueezeExcitationBottleneck, self).__init__()
        self.res = self.get_conv_bottleneck(last_planes, in_planes, out_planes, dense_depth, kernel_size)
        self.se = self.get_se(in_planes, out_planes, dense_depth, feature_size)

    def forward(self, x):
        out = self.res(x)
        se = self.se(out)
        return out * se


class ResnetInit(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, kernel_size, feature_size):
        super(ResnetInit, self).__init__()
        self.residual = SqueezeExcitationBottleneck(last_planes, in_planes,
                                                    out_planes, dense_depth, kernel_size, feature_size)
        self.transient = SqueezeExcitationBottleneck(last_planes, in_planes,
                                                     out_planes, 0, kernel_size, feature_size)
        self.residual_across = SqueezeExcitationBottleneck(last_planes, in_planes,
                                                           out_planes, 0, kernel_size, feature_size)
        self.transient_across = SqueezeExcitationBottleneck(last_planes, in_planes,
                                                            out_planes, dense_depth, kernel_size, feature_size)

    def forward(self, x):
        x_residual, x_transient = x
        residual_r_r = self.residual(x_residual)
        residual_r_t = self.residual_across(x_residual)

        transient_t_t = self.transient(x_transient)
        transient_t_r = self.transient_across(x_transient)

        x_residual = residual_r_r + transient_t_r
        x_transient = residual_r_t + transient_t_t

        return x_residual, x_transient


class BasicBlock(nn.Module):

    def get_out_planes(self):
        if self.is_unify:
            return 2 * self.out_planes + 4 * self.dense_depth
        else:
            if self.root:
                return 2 * self.out_planes + 4 * self.dense_depth
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
                nn.Conv2d(last_planes, 2 * out_planes + dense_depth, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(2 * out_planes + dense_depth)
            )
            self.rir_3 = ResnetInit(out_planes + dense_depth, in_planes, out_planes, dense_depth, 3,
                                    feature_size)
            self.rir_5 = ResnetInit(out_planes + dense_depth, in_planes, out_planes, dense_depth, 5,
                                    feature_size)
            self.rir_7 = ResnetInit(out_planes + dense_depth, in_planes, out_planes, dense_depth, 7,
                                    feature_size)

        else:
            self.rir_3 = ResnetInit(last_planes - out_planes, in_planes, out_planes, dense_depth, 3, feature_size)
            self.rir_5 = ResnetInit(last_planes - out_planes, in_planes, out_planes, dense_depth, 5, feature_size)
            self.rir_7 = ResnetInit(last_planes - out_planes, in_planes, out_planes, dense_depth, 7, feature_size)

        self.shortcut = nn.Sequential()
        if root:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, 2 * out_planes + dense_depth, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(2 * out_planes + dense_depth)
            )

    def forward(self, x):
        d = self.out_planes
        x = self.unify(x)
        x_residual = torch.cat([x[:, :d, :, :], x[:, 2 * d:, :, :]], 1)
        x_transient = x[:, d:, :, :]
        x_residual_3, x_transient_3 = self.rir_3((x_residual, x_transient))
        x_residual_5, x_transient_5 = self.rir_5((x_residual, x_transient))
        x_residual_7, x_transient_7 = self.rir_7((x_residual, x_transient))

        x = self.shortcut(x)
        out = torch.cat(
            [x[:, :d, :, :] + x_residual_3[:, :d, :, :] + x_residual_5[:, :d, :, :] + x_residual_7[:, :d, :, :],
             x_transient_3 + x_transient_5 + x_transient_7,
             x[:, 2 * d:, :, :], x_residual_3[:, d:, :, :], x_residual_5[:, d:, :, :], x_residual_7[:, d:, :, :]], 1)
        return out


class Tree(nn.Module):
    def get_out_planes(self):
        return self.root.get_out_planes()

    def __init__(self, last_planes, in_planes, out_planes, dense_depth, level, block_num, feature_size):
        super(Tree, self).__init__()
        assert (block_num > 0)
        self.level = level
        self.block_num = block_num
        self.out_planes = out_planes
        self.dense_depth = dense_depth

        if level == 1:
            self.root_last_planes = 2 * out_planes * (block_num - 1)
            sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size,
                                   last_planes < 2 * out_planes)
            last_planes = sub_block.get_out_planes()
            self.__setattr__('block_%d' % 0, sub_block)
            for i in range(1, block_num):
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, False)
                last_planes = sub_block.get_out_planes()
                self.__setattr__('block_%d' % i, sub_block)
            self.root_last_planes += sub_block.get_out_planes()
            self.root = BasicBlock(self.root_last_planes, in_planes * block_num, out_planes, dense_depth, True,
                                   feature_size,
                                   False)

        else:
            self.root_last_planes = 2 * out_planes * (block_num - 1)
            self.prev_root = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size,
                                        last_planes < 2 * out_planes)
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
            self.root = BasicBlock(self.root_last_planes, in_planes * block_num, out_planes, dense_depth, True,
                                   feature_size,
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
            xs.append(x[:, :2 * d, :, :])
        xs.append(x[:, 2 * d:, :, :])
        xs = torch.cat(xs, 1)
        out = self.root(xs)
        return out


class Generator(nn.Module):
    negative_slope = 1e-2

    def get_upsample(self, planes, out_planes, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(),
        )

    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module("Upsample_1", self.get_upsample(planes=z_dim, out_planes=256 * 4, kernel_size=4, stride=1,
                                                              padding=0))  # 4
        self.model.add_module("Upsample_2",
                              self.get_upsample(planes=256 * 4, out_planes=256 * 2, kernel_size=4, stride=2,
                                                padding=1))  # 8
        self.model.add_module("Upsample_3",
                              self.get_upsample(planes=256 * 2, out_planes=256, kernel_size=4, stride=2,
                                                padding=1))  # 16
        t1 = Tree(256, in_planes=64, out_planes=64, dense_depth=8, level=2, block_num=4,
                  feature_size=16)
        last_planes = t1.get_out_planes()
        self.model.add_module("Tree_1", t1) 
        self.model.add_module("Upsample_4",
                              self.get_upsample(planes=last_planes, out_planes=last_planes, kernel_size=4, stride=2,
                                                padding=1))

        t2 = Tree(last_planes, in_planes=32, out_planes=32, dense_depth=4, level=2, block_num=3,
                  feature_size=32)
        last_planes = t2.get_out_planes()
        self.model.add_module("Tree_2", t2)
        self.model.add_module("Upsample_5",
                              self.get_upsample(planes=last_planes, out_planes=last_planes, kernel_size=4, stride=2,
                                                padding=1))

        t3 = Tree(last_planes, in_planes=16, out_planes=16, dense_depth=4, level=1, block_num=2,
                  feature_size=64)
        last_planes = t3.get_out_planes()
        self.model.add_module("Tree_3", t3)
        self.model.add_module("LastConv", nn.Conv2d(last_planes, 3, 9, stride=1, padding=4, padding_mode='reflect'))
        self.model.add_module("LastActivate", nn.Tanh())

    def forward(self, x):
        return self.model(x)
