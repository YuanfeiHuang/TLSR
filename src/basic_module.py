import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class carn_block(nn.Module):
    def __init__(self, n_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 act=nn.ReLU(True), transi_learn=False):
        super(carn_block, self).__init__()
        self.transi_learn = transi_learn
        self.b1 = nn.Sequential(
            res_block(n_channels, kernel_size, stride, padding, dilation, groups, bias, act, transi_learn))
        self.b2 = nn.Sequential(
            res_block(n_channels, kernel_size, stride, padding, dilation, groups, bias, act, transi_learn))
        self.b3 = nn.Sequential(
            res_block(n_channels, kernel_size, stride, padding, dilation, groups, bias, act, transi_learn))
        if transi_learn:
            self.c1 = nn.Sequential(
                conv_interp(2 * n_channels, n_channels, 1, 1, 0, 1, 1, bias, act=act))
            self.c2 = nn.Sequential(
                conv_interp(3 * n_channels, n_channels, 1, 1, 0, 1, 1, bias, act=act))
            self.c3 = nn.Sequential(
                conv_interp(4 * n_channels, n_channels, 1, 1, 0, 1, 1, bias, act=act))
        else:
            self.c1 = nn.Sequential(
                conv_layer(2 * n_channels, n_channels, 1, 1, 0, 1, 1, bias, act=act))
            self.c2 = nn.Sequential(
                conv_layer(3 * n_channels, n_channels, 1, 1, 0, 1, 1, bias, act=act))
            self.c3 = nn.Sequential(
                conv_layer(4 * n_channels, n_channels, 1, 1, 0, 1, 1, bias, act=act))

    def forward(self, x_plus):

        if self.transi_learn:
            b1 = self.b1(x_plus)
            b1['value'] = torch.cat([x_plus['value'], b1['value']], dim=1)
            c1 = self.c1(b1)

            b2 = self.b2(c1)
            b2['value'] = torch.cat([b1['value'], b2['value']], dim=1)
            c2 = self.c2(b2)

            b3 = self.b3(c2)
            b3['value'] = torch.cat([b2['value'], b3['value']], dim=1)
            c3 = self.c3(b3)
        else:
            b1 = self.b1(x_plus)
            b1 = torch.cat([x_plus, b1], dim=1)
            c1 = self.c1(b1)

            b2 = self.b2(c1)
            b2 = torch.cat([b1, b2], dim=1)
            c2 = self.c2(b2)

            b3 = self.b3(c2)
            b3 = torch.cat([b2, b3], dim=1)
            c3 = self.c3(b3)

        return c3


class carn(nn.Module):
    def __init__(self, n_blocks, n_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 act=nn.ReLU(True), transi_learn=False):
        super(carn, self).__init__()
        self.transi_learn = transi_learn
        self.carn_list_b = nn.ModuleList([carn_block(n_channels, kernel_size, stride, padding, dilation, groups, bias,
                                                     act, transi_learn) for _ in range(n_blocks)])
        if transi_learn:
            self.carn_list_c = nn.ModuleList(
                [conv_interp((i + 2) * n_channels, n_channels, 1, 1, 0, 1, 1, bias, act=act)
                 for i in range(n_blocks)])
        else:
            self.carn_list_c = nn.ModuleList([conv_layer((i + 2) * n_channels, n_channels, 1, 1, 0, 1, 1, bias, act=act)
                                              for i in range(n_blocks)])

    def forward(self, x_plus):
        n_blocks = self.carn_list_b.__len__()
        o = c = x_plus
        for i in range(n_blocks):
            if self.carn_list_b[i].transi_learn:
                b = self.carn_list_b[i](c)
                o['value'] = torch.cat([o['value'], b['value']], dim=1)
                c = self.carn_list_c[i](o)
            else:
                b = self.carn_list_b[i](c)
                o = torch.cat([o, b], dim=1)
                c = self.carn_list_c[i](o)

        return c


class res_block(nn.Module):
    def __init__(self, n_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 act=nn.ReLU(inplace=True), transi_learn=False):
        super(res_block, self).__init__()
        self.transi_learn = transi_learn
        if transi_learn:
            self.layer = nn.Sequential(
                *[conv_interp(n_channels, n_channels, kernel_size, stride, padding, dilation, groups, bias, act=act),
                  conv_interp(n_channels, n_channels, kernel_size, stride, padding, dilation, groups, bias, act=False)])
        else:
            self.layer = nn.Sequential(
                *[conv_layer(n_channels, n_channels, kernel_size, stride, padding, dilation, groups, bias, act=act),
                  conv_layer(n_channels, n_channels, kernel_size, stride, padding, dilation, groups, bias, act=False)])

    def forward(self, x_plus):
        if self.transi_learn:
            residue = self.layer(x_plus)
            residue['value'] = residue['value'] + x_plus['value']
        else:
            residue = self.layer(x_plus)
            residue = residue + x_plus

        return residue


class conv_interp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 act=False):
        super(conv_interp, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.act = act

    def forward(self, x_plus):
        x = x_plus["value"]
        DoT = x_plus["DoT"]
        transi_learn = x_plus["transi_learn"]
        if transi_learn:
            B, C_i, w, h = x.size()
            C_o, C_i, Kw, Kh = self.conv0.weight.data.size()
            # for only the case that "bias = False"
            # weight_data = self.conv0.weight.data
            # groups = self.conv0.groups
            kernel0 = self.conv0.weight.unsqueeze(0).repeat(B, 1, 1, 1, 1)
            kernel1 = self.conv1.weight.unsqueeze(0).repeat(B, 1, 1, 1, 1)
            kernel = (1 - DoT.view(B, 1, 1, 1, 1)) * kernel0 + DoT.view(B, 1, 1, 1, 1) * kernel1
            kernel = kernel.view(B * C_o, C_i, Kw, Kh)

            y = F.conv2d(x.view(1, B * C_i, w, h), kernel,
                         stride=self.conv0.stride, padding=self.conv0.padding,
                         dilation=self.conv0.dilation, groups=B)

            y = y.view(B, C_o, w, h)
        elif (DoT == 0).all():
            y = self.conv0(x)
        elif (DoT == 1).all():
            y = self.conv1(x)
        else:
            raise InterruptedError

        if self.act:
            y = self.act(y)

        return {'value': y, 'DoT': DoT, 'transi_learn': transi_learn}


class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 act=False):
        super(conv_layer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.act = act

    def forward(self, x):
        y = self.conv(x)

        if self.act:
            y = self.act(y)
        return y


class UpScale(nn.Sequential):
    def __init__(self, type, n_feats, scale, bn=False, act=nn.ReLU(inplace=True), bias=False):
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(np.log2(scale))):
                if type == 'DeConv':
                    layers.append(nn.ConvTranspose2d(n_feats, n_feats, 4, stride=2, padding=1, groups=1, bias=bias))
                elif type == 'SubPixel':
                    layers.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats, kernel_size=3, stride=1,
                                            padding=1, groups=1, bias=bias))
                    layers.append(nn.PixelShuffle(2))
                else:
                    raise InterruptedError
                if bn: layers.append(nn.BatchNorm2d(n_feats))
                if act: layers.append(act)
        elif scale == 3:
            layers.append(nn.Conv2d(in_channels=n_feats, out_channels=9 * n_feats, kernel_size=3, stride=1,
                                    padding=1, groups=1, bias=bias))
            layers.append(nn.PixelShuffle(3))
            if bn: layers.append(nn.BatchNorm2d(n_feats))
            if act: layers.append(act)
        else:
            raise NotImplementedError
        super(UpScale, self).__init__(*layers)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
