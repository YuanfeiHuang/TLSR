import torchvision.models as models
from collections import OrderedDict
from src.basic_module import *


class Generator(nn.Module):
    def __init__(self, n_colors, n_channels, n_homo_blocks, n_transi_layers, act=nn.ReLU(inplace=True), scale=4):
        super(Generator, self).__init__()
        self.input = nn.Conv2d(n_colors, n_channels, 3, 1, 1, 1, 1, True)

        self.feature_homo = nn.Sequential(
            carn(n_homo_blocks, n_channels, 3, 1, 1, 1, 1, bias=False, act=act, transi_learn=False))
        self.transi_learn = nn.Sequential(
            *[conv_interp(n_channels, n_channels, 3, 1, 1, 1, 1, False, act=act) for _ in range(n_transi_layers)])
        # self.transi_learn = nn.Sequential(
        #     *[res_block(n_channels, act=act, transi_learn=True) for _ in range(n_transi_layers)])

        self.upscale = nn.Sequential(UpScale('SubPixel', n_channels, scale, bn=False, act=False, bias=False))
        self.output = nn.Conv2d(n_channels, n_colors, 3, 1, 1, 1, 1, True)

    def forward(self, x_plus):
        body_value = self.input(x_plus['value'])
        body_value = self.feature_homo(body_value)

        body_plus = {'value': body_value, 'DoT': x_plus['DoT'], 'transi_learn': x_plus['transi_learn']}
        body_plus = self.transi_learn(body_plus)
        body_value = body_plus['value']

        body_value = self.upscale(body_value)
        output = self.output(body_value)

        return output


class DoTNet_ResNet50(nn.Module):
    def __init__(self):
        super(DoTNet_ResNet50, self).__init__()

        self.features = nn.Sequential(ResNet(Bottleneck, [3, 4, 6, 3], norm_layer=None))
        pretrained_resnet50 = models.resnet50(pretrained=True)
        premodel_dict = pretrained_resnet50.state_dict()

        model_dict = self.features[0].state_dict()
        for k, v in model_dict.items():
            model_dict[k] = premodel_dict[k]
        self.features[0].load_state_dict(model_dict)

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ('linear0', nn.Linear(2048, 128)),
                    ('relu', nn.ReLU(inplace=True)),
                    ('linear1', nn.Linear(128, 2))
                ]
            )
        )
        for key in self.classifier.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'linear' in key:
                    nn.init.kaiming_normal_(self.classifier.state_dict()[key])

    def forward(self, x):
        y = self.features(x)
        y = F.adaptive_avg_pool2d(y, 1)
        y = torch.flatten(y, 1)
        y = self.classifier(y)
        y = F.softmax(y, dim=1)[:, 0]
        return y
