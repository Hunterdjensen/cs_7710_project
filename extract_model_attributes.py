import sys
import re
import torch
import torch.nn as nn
import torchvision.models as models
from bit_flipping import get_reference
from bit_flipping import get_layer
from get_model import get_model
from get_model import get_model2


# from torchsummary import summary


# Working solution: Add in network layers to monitor the IFMAP height/width, and print out other parameters
class Layer_Attr(nn.Module):
    def __init__(self, is_conv, info, name, file):
        super(Layer_Attr, self).__init__()
        self.is_conv = is_conv
        self.info = info
        self.name = name[:-7].replace('.', '_')  # Format the layer name ([:-7] to remove .weight)
        self.f = file  # For printing out results

    def forward(self, x):
        IFMAP_h = None
        IFMAP_w = None
        filter_h = None
        filter_w = None
        channels = None
        num_filter = None
        strides = None

        if self.is_conv:
            if len(list(x.shape)) != 4:  # Our conv IFMAP should always have 4 dimensions: [N, C, H, W]
                exit("Error: expected a shape of 4 for conv IFMAP, but got shape: " + str(x.shape))
            IFMAP_h = list(x.shape)[2]  # 3rd (index 2) dimension
            IFMAP_w = list(x.shape)[3]  # 4th (index 3) dimension
            kernel_regex = re.compile(r"kernel_size=\((\d+), (\d+)\)")
            match = kernel_regex.search(self.info)
            if match is not None:
                filter_h = match.group(1)
                filter_w = match.group(2)
            else:
                exit("Error: couldn't find kernel size match in string: " + str(self.info))

            channels_regex = re.compile(r"Conv2d\((\d+), (\d+)")
            match = channels_regex.search(self.info)
            if match is not None:
                channels = match.group(1)
                num_filter = match.group(2)
            else:
                exit("Error: couldn't find channels/num_filters in string: " + str(self.info))

            strides_regex = re.compile(r"stride=\((\d+)")
            match = strides_regex.search(self.info)
            if match is not None:
                strides = match.group(1)
            else:
                exit("Error: couldn't find strides in string: " + str(self.info))

        else:  # Is Fully Connected (Linear) layer
            IFMAP_h = 1  # Not sure if you can have a h/w >1 for a fully connected layer
            IFMAP_w = 1
            filter_h = 1
            filter_w = 1
            channels_regex = re.compile(r"in_features=(\d+)")
            match = channels_regex.search(self.info)
            if match is not None:
                channels = match.group(1)
            else:
                exit("Error: couldn't find channels in string: " + str(self.info))

            num_filter_regex = re.compile(r"out_features=(\d+)")
            match = num_filter_regex.search(self.info)
            if match is not None:
                num_filter = match.group(1)
            else:
                exit("Error: couldn't find num_filter in string: " + str(self.info))
            strides = 1

        # Final check:
        if IFMAP_h is None or IFMAP_w is None or filter_h is None or filter_w is None or channels is None or num_filter is None or strides is None:
            print("One of the following is None:", IFMAP_h, IFMAP_w, filter_h, filter_w, channels, num_filter, strides)
            exit("One of the .csv variables was none for layer " + str(self.name) + " with info: " + str(self.info))

        # print(self.name, self.info, " shape:", x.shape)
        print(self.name + ',', str(IFMAP_h) + ',', str(IFMAP_w) + ',', str(filter_h) + ',', str(filter_w) + ',',
              str(channels) + ',', str(num_filter) + ',', str(strides) + ',', file=self.f)
        return x


def create_csv(model_name):
    output_filename = 'model_configs/' + model_name + '.csv'
    f = open(output_filename, 'w')
    print("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, Channels, Num Filter, Strides,\n", file=f)
    net = get_model(model_name)
    net.eval()

    count = 0
    for name, param in net.named_parameters():
        if "bias" not in name:  # Only look at weights, so we don't get some layers twice
            layer = get_layer(name, net)
            if isinstance(layer, nn.modules.Conv2d) or isinstance(layer,
                                                                  nn.modules.Linear):  # Only convolutional and linear layers show up in output .csv
                count += 1
                is_conv = isinstance(layer, nn.modules.Conv2d)
                layer, prev, num = get_reference(name, net)
                if layer is not None:  # That means that this is a valid layer to add a BitFlipLayer behind (it's a layer with a weight)
                    if num:  # The final attribute is a number, we can index with []
                        layer[int(prev)] = nn.Sequential(Layer_Attr(is_conv, str(layer[int(prev)]), name, f),
                                                         layer[int(prev)])
                    else:  # The last attribute isn't an index, need to use set/get attr
                        # Don't worry about decoding the next line, it's just replacing the layer with itself and a
                        # Layer_Attr layer before it
                        setattr(layer, prev,
                                nn.Sequential(Layer_Attr(is_conv, str(getattr(layer, prev)), name, f),
                                              getattr(layer, prev)))
                else:
                    exit("Error, undefined layer: " + str(getattr(layer, prev)))
                # print("*")
                # print(getattr(layer, prev))
                # print(name, param.shape)
    print("There are", count, "network layers for csv. (model=" + str(model_name) + ")")
    _ = net(torch.zeros([1, 3, 224, 224]))  # Run network and output results
    f.close()


available_models = ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn',
                    'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 'resnet18', 'resnet34',
                    'resnet50', 'resnet101', 'resnet152', 'squeezenet1_0',
                    'squeezenet1_1', 'densenet121', 'densenet169', 'densenet201',
                    'densenet161', 'inception_v3', 'googlenet', 'shufflenet_v2_x1_0',
                    'shufflenet_v2_x0_5', 'mobilenet_v2', 'mobilenet_v3_large',
                    'mobilenet_v3_small', 'resnext50_32x4d', 'resnext101_32x8d',
                    'wide_resnet50_2', 'wide_resnet101_2', 'mnasnet1_0', 'mnasnet0_5']
for model in available_models:
    create_csv(model)


# Old attempts that didn't pan out:
# # Works but I can't figure out how to modify the network using .modules()
# for child in net.modules():
#     if isinstance(child, nn.modules.Conv2d):
#         print("***", child)
#         child = nn.Sequential(child, Layer_Attr(str(child)))
#     if isinstance(child, nn.modules.Linear):
#         print("**", child)
#         child = nn.Sequential(child, Layer_Attr(str(child)))

# Different method that almost works (shows output shape) but we need input feature map shape
# summary(net, (3, 224, 224))
