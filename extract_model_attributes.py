import torchvision.models as models

net = models.googlenet(pretrained=True)     # models.mobilenet_v3_small(pretrained=True)
print(net)

for name, param in net.named_parameters():
    print(name, param.shape)