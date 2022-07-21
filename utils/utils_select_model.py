from models import *
from models.resnet import resnet18, resnet34
from models.WideResNet import Wide_ResNet


def select_model(modelname):
    if modelname == 'VGG11':
        net = VGG('VGG11')
    elif modelname == 'VGG13':
        net = VGG('VGG13')
    elif modelname == 'VGG16':
        net = VGG('VGG16')
    elif modelname == 'VGG19':
        net = VGG('VGG19')
    elif modelname == 'ResNet18':
        net = resnet18
    elif modelname == 'ResNet34':
        net = resnet34()
    elif modelname == 'WideResNet':
        net = Wide_ResNet()
    elif modelname == 'EffNet':
        net = Eff_Net()
    return net
