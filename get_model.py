import torchvision.models as models


# Switch statement to call models
def get_model(arg):
    switcher = {
        'alexnet': models.alexnet(pretrained=True),
        'vgg11': models.vgg11(pretrained=True),
        'vgg13': models.vgg13(pretrained=True),
        'vgg16': models.vgg16(pretrained=True),
        'vgg19': models.vgg19(pretrained=True),
        'vgg11_bn': models.vgg11_bn(pretrained=True),
        'vgg13_bn': models.vgg13_bn(pretrained=True),
        'vgg16_bn': models.vgg16_bn(pretrained=True),
        'vgg19_bn': models.vgg19_bn(pretrained=True),
        'resnet18': models.resnet18(pretrained=True),
        'resnet34': models.resnet34(pretrained=True),
        'resnet50': models.resnet50(pretrained=True),
        'resnet101': models.resnet101(pretrained=True),
        'resnet152': models.resnet152(pretrained=True),
        'squeezenet1_0': models.squeezenet1_0(pretrained=True),
        'squeezenet1_1': models.squeezenet1_1(pretrained=True),
        'densenet121': models.densenet121(pretrained=True),
        'densenet169': models.densenet169(pretrained=True),
        'densenet201': models.densenet201(pretrained=True),
        'densenet161': models.densenet161(pretrained=True),
        'inception_v3': models.inception_v3(pretrained=True),
        'googlenet': models.googlenet(pretrained=True),
        'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0(pretrained=True),
        'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5(pretrained=True),
        'mobilenet_v2': models.mobilenet_v2(pretrained=True),
        'mobilenet_v3_large': models.mobilenet_v3_large(pretrained=True),
        'mobilenet_v3_small': models.mobilenet_v3_small(pretrained=True),
        'resnext50_32x4d': models.resnext50_32x4d(pretrained=True),
        'resnext101_32x8d': models.resnext101_32x8d(pretrained=True),
        'wide_resnet50_2': models.wide_resnet50_2(pretrained=True),
        'wide_resnet101_2': models.wide_resnet101_2(pretrained=True),
        'mnasnet1_0': models.mnasnet1_0(pretrained=True),
        'mnasnet0_5': models.mnasnet0_5(pretrained=True)
    }
    net = switcher.get(arg, -1)
    if net == -1:
        print('invalid key of: ' + arg + '. defaulting to mobilenet_v3_small')
        net = models.mobilenet_v3_small(pretrained=True)
    return net
