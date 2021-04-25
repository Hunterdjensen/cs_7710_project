import torchvision.models as models


# New implementation: Loading all of the models into a switch statement was
# causing a ~20s delay to run, this way we only call the function that we
# want, making it quicker :)
def alexnet():
    return models.alexnet(pretrained=True)


def vgg11():
    return models.vgg11(pretrained=True)


def vgg13():
    return models.vgg13(pretrained=True)


def vgg16():
    return models.vgg16(pretrained=True)


def vgg19():
    return models.vgg19(pretrained=True)


def vgg11_bn():
    return models.vgg11_bn(pretrained=True)


def vgg13_bn():
    return models.vgg13_bn(pretrained=True)


def vgg16_bn():
    return models.vgg16_bn(pretrained=True)


def vgg19_bn():
    return models.vgg19_bn(pretrained=True)


def resnet18():
    return models.resnet18(pretrained=True)


def resnet34():
    return models.resnet34(pretrained=True)


def resnet50():
    return models.resnet50(pretrained=True)


def resnet101():
    return models.resnet101(pretrained=True)


def resnet152():
    return models.resnet152(pretrained=True)


def squeezenet1_0():
    return models.squeezenet1_0(pretrained=True)


def squeezenet1_1():
    return models.squeezenet1_1(pretrained=True)


def densenet121():
    return models.densenet121(pretrained=True)


def densenet169():
    return models.densenet169(pretrained=True)


def densenet201():
    return models.densenet201(pretrained=True)


def densenet161():
    return models.densenet161(pretrained=True)


def inception_v3():
    return models.inception_v3(pretrained=True)


def googlenet():
    return models.googlenet(pretrained=True)


def shufflenet_v2_x1_0():
    return models.shufflenet_v2_x1_0(pretrained=True)


def shufflenet_v2_x0_5():
    return models.shufflenet_v2_x0_5(pretrained=True)


def mobilenet_v2():
    return models.mobilenet_v2(pretrained=True)


def mobilenet_v3_large():
    return models.mobilenet_v3_large(pretrained=True)


def mobilenet_v3_small():
    return models.mobilenet_v3_small(pretrained=True)


def resnext50_32x4d():
    return models.resnext50_32x4d(pretrained=True)


def resnext101_32x8d():
    return models.resnext101_32x8d(pretrained=True)


def wide_resnet50_2():
    return models.wide_resnet50_2(pretrained=True)


def wide_resnet101_2():
    return models.wide_resnet101_2(pretrained=True)


def mnasnet1_0():
    return models.mnasnet1_0(pretrained=True)


def mnasnet0_5():
    return models.mnasnet0_5(pretrained=True)


# Switch statement to call models
def get_model(arg):
    switcher = {
        'alexnet': alexnet,
        'vgg11': vgg11,
        'vgg13': vgg13,
        'vgg16': vgg16,
        'vgg19': vgg19,
        'vgg11_bn': vgg11_bn,
        'vgg13_bn': vgg13_bn,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'squeezenet1_0': squeezenet1_0,
        'squeezenet1_1': squeezenet1_1,
        'densenet121': densenet121,
        'densenet169': densenet169,
        'densenet201': densenet201,
        'densenet161': densenet161,
        'inception_v3': inception_v3,
        'googlenet': googlenet,
        'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
        'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
        'mobilenet_v2': mobilenet_v2,
        'mobilenet_v3_large': mobilenet_v3_large,
        'mobilenet_v3_small': mobilenet_v3_small,
        'resnext50_32x4d': resnext50_32x4d,
        'resnext101_32x8d': resnext101_32x8d,
        'wide_resnet50_2': wide_resnet50_2,
        'wide_resnet101_2': wide_resnet101_2,
        'mnasnet1_0': mnasnet1_0,
        'mnasnet0_5': mnasnet0_5,
    }
    func = switcher.get(arg, -1)
    if func == -1:
        print('invalid key of: ' + arg + '. defaulting to mobilenet_v3_small')
        return models.mobilenet_v3_small(pretrained=True)
    else:
        return func()


# Original implementation:
# Switch statement to call models
# def get_model(arg):
#     switcher = {
#         'alexnet': models.alexnet(pretrained=True),
#         'vgg11': models.vgg11(pretrained=True),
#         'vgg13': models.vgg13(pretrained=True),
#         'vgg16': models.vgg16(pretrained=True),
#         'vgg19': models.vgg19(pretrained=True),
#         'vgg11_bn': models.vgg11_bn(pretrained=True),
#         'vgg13_bn': models.vgg13_bn(pretrained=True),
#         'vgg16_bn': models.vgg16_bn(pretrained=True),
#         'vgg19_bn': models.vgg19_bn(pretrained=True),
#         'resnet18': models.resnet18(pretrained=True),
#         'resnet34': models.resnet34(pretrained=True),
#         'resnet50': models.resnet50(pretrained=True),
#         'resnet101': models.resnet101(pretrained=True),
#         'resnet152': models.resnet152(pretrained=True),
#         'squeezenet1_0': models.squeezenet1_0(pretrained=True),
#         'squeezenet1_1': models.squeezenet1_1(pretrained=True),
#         'densenet121': models.densenet121(pretrained=True),
#         'densenet169': models.densenet169(pretrained=True),
#         'densenet201': models.densenet201(pretrained=True),
#         'densenet161': models.densenet161(pretrained=True),
#         'inception_v3': models.inception_v3(pretrained=True),
#         'googlenet': models.googlenet(pretrained=True),
#         'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0(pretrained=True),
#         'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5(pretrained=True),
#         'mobilenet_v2': models.mobilenet_v2(pretrained=True),
#         'mobilenet_v3_large': models.mobilenet_v3_large(pretrained=True),
#         'mobilenet_v3_small': models.mobilenet_v3_small(pretrained=True),
#         'resnext50_32x4d': models.resnext50_32x4d(pretrained=True),
#         'resnext101_32x8d': models.resnext101_32x8d(pretrained=True),
#         'wide_resnet50_2': models.wide_resnet50_2(pretrained=True),
#         'wide_resnet101_2': models.wide_resnet101_2(pretrained=True),
#         'mnasnet1_0': models.mnasnet1_0(pretrained=True),
#         'mnasnet0_5': models.mnasnet0_5(pretrained=True)
#     }
#     net = switcher.get(arg, -1)
#     if net == -1:
#         print('invalid key of: ' + arg + '. defaulting to mobilenet_v3_small')
#         net = models.mobilenet_v3_small(pretrained=True)
#     return net
