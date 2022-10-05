import torch
import torchvision.models as models
from collections import OrderedDict
from src.basic_module import *


class Generator(nn.Module):
    def __init__(self, n_colors, n_channels, n_homo_blocks, n_transi_layers, n_homo_width, n_transi_width, act=nn.ReLU(inplace=True), scale=4):
        super(Generator, self).__init__()
        self.input = nn.Conv2d(n_colors, n_channels, 3, 1, 1, 1, 1, True)

        self.feature_homo = nn.Sequential(
            *[
                net_group('ResNet', n_homo_blocks[1], n_channels, n_homo_width,
                          kernel_size=3, act=act, glo_res=True, transi_learn=False)
                for _ in range(n_homo_blocks[0])
            ],
            conv_layer(n_channels, n_channels, 3, 1, 1, 1, 1, False, act=False)
        )
        self.transi_learn = nn.Sequential(
            *[
                net_group('ResNet', n_transi_layers[1], n_channels, n_transi_width,
                          kernel_size=3, act=act, glo_res=False, transi_learn=True)
                for _ in range(n_transi_layers[0])
            ],
        )
        self.output = nn.Sequential(
            UpScale('SubPixel', n_channels, kernel=1, groups=1, scale=scale, bn=False, act=False, bias=False,
                    transi_learn=False),
            conv_layer(n_channels, n_colors, 3, 1, 1, 1, 1, False, act=False)
        )

    def forward(self, x):
        y = self.input(x['value'])
        y = self.feature_homo(y) + y

        y = self.transi_learn(
            {'value': y, 'DoT': x['DoT'], 'transi_learn': x['transi_learn']}
        )['value']

        y = self.output(y)

        return y

class PlainCNN(nn.Module):
    def __init__(self, n_colors, patch_size, n_channels, max_chn):
        super(PlainCNN, self).__init__()

        self.input = nn.Conv2d(n_colors, n_channels, kernel_size=3, padding=1)
        num_module = int(np.log2(patch_size))
        # num_layers = [(num_module-i) for i in range(num_module)]
        num_layers = [2 for _ in range(num_module)]
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ('block{:d}'.format(i),
                     nn.Sequential(
                         *[bottleneck_(min(n_channels * (2 ** (i)), max_chn), min(n_channels * (2 ** (i)) // 2, 64))
                           for _ in range(num_layers[i])],
                         nn.Conv2d(min(n_channels * (2 ** (i)), max_chn), min(n_channels * (2 ** (i + 1)), max_chn),
                                   kernel_size=1, stride=2, padding=0)
                     )) for i in range(num_module)
                ]
            )
        )
        self.out_channels = min(n_channels * (2 ** (num_module)), max_chn)

    def forward(self, x):
        fea = self.input(x)
        fea = self.features(fea)
        return fea


class DoTNet(nn.Module):
    def __init__(self, n_colors, patch_size):
        super(DoTNet, self).__init__()

        self.features = PlainCNN(n_colors, patch_size, 32, max_chn=256)

        self.classifier = nn.Sequential(
            nn.Linear(self.features.out_channels, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 1)
        )

        for key in self.classifier.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.kaiming_normal_(self.classifier.state_dict()[key])

        for key in self.features.state_dict():
            if key.split('.')[-1] == 'weight' and 'conv' in key:
                nn.init.kaiming_normal_(self.features.state_dict()[key])

    def forward(self, x):
        y = self.features(x)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        return y.squeeze(1)
