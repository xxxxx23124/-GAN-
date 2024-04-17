import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class SelfAttention(nn.Module):
    def __init__(self, in_planes, embedding_channels):
        super(SelfAttention, self).__init__()
        self.key = nn.Conv2d(in_planes, embedding_channels, 1)
        self.query = nn.Conv2d(in_planes, embedding_channels, 1)
        self.value = nn.Conv2d(in_planes, embedding_channels, 1)
        self.self_att = nn.Conv2d(embedding_channels, in_planes, 1)
        self.gamma = nn.Parameter(torch.tensor(random.random()*0.2 + 0.03))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batchsize, C, H, W = x.size()
        N = H * W
        f_x = self.key(x).view(batchsize, -1, N)
        g_x = self.query(x).view(batchsize, -1, N)
        h_x = self.value(x).view(batchsize, -1, N)
        s = torch.bmm(f_x.permute(0, 2, 1), g_x)
        beta = self.softmax(s)
        v = torch.bmm(h_x, beta)
        v = v.view(batchsize, -1, H, W)
        o = self.self_att(v)
        y = self.gamma * o + x
        return y


class SEBottleneckSelfAttention(nn.Module):
    negative_slope = 1e-2

    def get_out_planes(self):
        return self.out_planes + self.dense_depth

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
        super(SEBottleneckSelfAttention, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.res = self.get_conv_bottleneck(last_planes, in_planes, out_planes, dense_depth, kernel_size)
        self.se = self.get_se(in_planes, out_planes, dense_depth, feature_size)

    def forward(self, x):
        out = self.res(x)
        se = self.se(out)
        return out * se


class ResnetInit(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, kernel_size, feature_size):
        super(ResnetInit, self).__init__()
        self.residual = SEBottleneckSelfAttention(last_planes, in_planes,
                                                  out_planes, dense_depth, kernel_size, feature_size)
        self.transient = SEBottleneckSelfAttention(last_planes, in_planes,
                                                   out_planes, 0, kernel_size, feature_size)
        self.residual_across = SEBottleneckSelfAttention(last_planes, in_planes,
                                                         out_planes, 0, kernel_size, feature_size)
        self.transient_across = SEBottleneckSelfAttention(last_planes, in_planes,
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
            return 2 * self.out_planes + 2 * self.dense_depth
        else:
            if self.root:
                return 2 * self.out_planes + 2 * self.dense_depth
            else:
                return self.last_planes + 1 * self.dense_depth

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

        else:
            self.rir_3 = ResnetInit(last_planes - out_planes, in_planes, out_planes, dense_depth, 3, feature_size)

        self.shortcut = nn.Sequential()
        if root:
            self.shortcut = nn.Sequential(
                nn.Conv2d(last_planes, 2 * out_planes + dense_depth, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(2 * out_planes + dense_depth)
            )
        self.attention = SelfAttention(self.get_out_planes(), self.get_out_planes())

    def forward(self, x):
        d = self.out_planes
        x = self.unify(x)
        x_residual = torch.cat([x[:, :d, :, :], x[:, 2 * d:, :, :]], 1)
        x_transient = x[:, d:, :, :]
        x_residual_3, x_transient_3 = self.rir_3((x_residual, x_transient))

        x = self.shortcut(x)
        out = torch.cat(
            [x[:, :d, :, :] + x_residual_3[:, :d, :, :],
             x_transient_3,
             x[:, 2 * d:, :, :], x_residual_3[:, d:, :, :]], 1)
        out = self.attention(out)
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
            sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size,
                                   last_planes < 2 * out_planes)
            last_planes = sub_block.get_out_planes()
            self.root_last_planes = last_planes
            self.__setattr__('block_%d' % 0, sub_block)
            for i in range(1, block_num):
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, False)
                last_planes = sub_block.get_out_planes()
                self.root_last_planes += last_planes
                self.__setattr__('block_%d' % i, sub_block)
            self.root = BasicBlock(self.root_last_planes, in_planes * block_num, out_planes, dense_depth, True,
                                   feature_size,
                                   False)

        else:
            self.prev_root = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size,
                                        last_planes < 2 * out_planes)
            self.root_last_planes = self.prev_root.get_out_planes()

            for i in reversed(range(1, level)):
                subtree = Tree(last_planes, in_planes, out_planes, dense_depth, i, block_num, feature_size)
                last_planes = subtree.get_out_planes()
                self.root_last_planes += last_planes
                self.__setattr__('level_%d' % i, subtree)

            for i in range(block_num):
                sub_block = BasicBlock(last_planes, in_planes, out_planes, dense_depth, False, feature_size, False)
                last_planes = sub_block.get_out_planes()
                self.root_last_planes += last_planes
                self.__setattr__('block_%d' % i, sub_block)
            self.root = BasicBlock(self.root_last_planes, in_planes * block_num, out_planes, dense_depth, True,
                                   feature_size,
                                   False)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = self.__getattr__('level_%d' % i)
            x = level_i(x)
            xs.append(x)
        for i in range(self.block_num):
            block_i = self.__getattr__('block_%d' % i)
            x = block_i(x)
            xs.append(x)
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
        planes = 64
        self.Upsample_1 = nn.Sequential(
            self.get_upsample(planes=z_dim, out_planes=planes * 16, kernel_size=4, stride=1, padding=0))

        self.Upsample_2 = nn.Sequential(
            self.get_upsample(planes=planes * 16, out_planes=planes * 8, kernel_size=4, stride=2, padding=1))
        t1 = Tree(planes * 8, in_planes=planes * 2, out_planes=planes * 2, dense_depth=planes // 4, level=1,
                  block_num=2,
                  feature_size=8)
        last_planes = t1.get_out_planes() + planes * 8
        self.Tree_1 = nn.Sequential(t1)
        self.Attention_1 = SelfAttention(last_planes, last_planes)

        self.Upsample_3 = nn.Sequential(
            self.get_upsample(planes=last_planes, out_planes=planes * 4, kernel_size=4, stride=2, padding=1))
        t2 = Tree(planes * 4, in_planes=planes, out_planes=planes, dense_depth=planes // 8, level=1, block_num=2,
                  feature_size=16)
        last_planes = t2.get_out_planes() + planes * 4
        self.Tree_2 = nn.Sequential(t2)
        self.Attention_2 = SelfAttention(last_planes, last_planes)

        self.Upsample_4 = nn.Sequential(
            self.get_upsample(planes=last_planes, out_planes=planes * 2, kernel_size=4, stride=2, padding=1))
        t3 = Tree(planes * 2, in_planes=planes // 2, out_planes=planes // 2, dense_depth=planes // 16, level=1,
                  block_num=2,
                  feature_size=32)
        last_planes = t3.get_out_planes() + planes * 2
        self.Tree_3 = nn.Sequential(t3)
        self.Attention_3 = SelfAttention(last_planes, last_planes)

        self.Upsample_5 = nn.Sequential(
            self.get_upsample(planes=last_planes, out_planes=planes, kernel_size=4, stride=2, padding=1))
        t4 = Tree(planes, in_planes=planes // 4, out_planes=planes // 4, dense_depth=planes // 32, level=1, block_num=2,
                  feature_size=64)
        last_planes = t4.get_out_planes() + planes
        self.Tree_4 = nn.Sequential(t4)
        self.Attention_4 = SelfAttention(last_planes, last_planes)

        self.LastConv = nn.Conv2d(last_planes, 3, 9, stride=1, padding=4, padding_mode='reflect')
        self.LastActivate = nn.Tanh()

    def forward(self, x):
        x = self.Upsample_1(x)

        x = self.Upsample_2(x)
        t = self.Tree_1(x)
        x = torch.cat([x, t], 1)
        x = self.Attention_1(x)

        x = self.Upsample_3(x)
        t = self.Tree_2(x)
        x = torch.cat([x, t], 1)
        x = self.Attention_2(x)

        x = self.Upsample_4(x)
        t = self.Tree_3(x)
        x = torch.cat([x, t], 1)
        x = self.Attention_3(x)

        x = self.Upsample_5(x)
        t = self.Tree_4(x)
        x = torch.cat([x, t], 1)
        x = self.Attention_4(x)

        x = self.LastConv(x)
        return self.LastActivate(x)


def test():
    # net = BasicBlock(last_planes=16, in_planes=32, out_planes=12, dense_depth=8, root=False, feature_size=64)

    # net = Stem_block(in_planes=4,planes=16)
    # net = Tree(last_planes=16, in_planes=8, out_planes=8, dense_depth=8, level=5, block_num=6, feature_size=64)
    net = Generator(z_dim=256)
    # print(net)
    from torchsummary import summary
    summary(net, input_size=(8, 256, 1, 1))
    from torchviz import make_dot

    x = torch.randn(8, 256, 1, 1)
    y = net(x)
    #dot = make_dot(y, params=dict(net.named_parameters()))
    # dot.view()
    #dot.render("G10_model_graph")
    print(y.size())
    # print(net.get_out_planes())
    print("successed")


def test_2():
    for last_planes in range(1, 8):
        for out_planes in range(1, 8):
            for dense_depth in range(6):
                x = torch.randn(2, last_planes, 4, 4)
                net = Tree(last_planes=last_planes, in_planes=4, out_planes=out_planes, dense_depth=dense_depth,
                           level=5, block_num=6,
                           feature_size=4)
                y = net(x)
                print(y.size())
                print(net.get_out_planes())
                print("successed")
    print("all_successed")
