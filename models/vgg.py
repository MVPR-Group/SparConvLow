'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from arg import args

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=args.number_class):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier_ = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.bn_fea = nn.BatchNorm1d(4096)
        # self.classifier = nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        # print(x.shape)(batchsize,3,size,size)
        out = self.features(x)
        # print(out.shape) #(batchsize,512,1,1)
        out = out.view(out.size(0), -1)
        out = self.classifier_(out)
        feature = F.normalize(self.bn_fea(out))
        # meanfeature = torch.mean(out_, dim=0)
        # feature = out_ - meanfeature
        # feastd = feature.std(dim=0)
        # one = torch.ones_like(feastd, dtype=torch.float32)
        # feastd_ = torch.where(feastd == 0, one, feastd)
        # feature = feature / feastd_
        out = self.classifier(out)
        return out, feature

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG13')
    x = torch.randn(2,3,224,224)
    y = net(x)
    # print(y.size())

test()
