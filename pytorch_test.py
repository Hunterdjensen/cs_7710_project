import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image
import numpy as np
import struct

# Read in the labels from 'imagenet_classes.txt'
labels = []
with open('imagenet_classes.txt') as f:
    for line in f:
        line = line.split(',')
        class_name = line[1].strip()    # Name of class, line[0] contains line number
        labels.append(class_name)


# Define function for printing out the results
def print_top_5(output):
    images_per_batch, num_classes = output.shape
    percentages = torch.nn.functional.softmax(output, dim=1) * 100
    _, indices = torch.sort(output, descending=True)

    for image in range(images_per_batch):
        for i in range(5):
            idx = indices[image][i]     # Index of top class
            print(i+1, ": ", labels[idx], " ", percentages[image][idx].item())
        print()     # Newline


# From: https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex
# Converts a float type to a string
def float_to_hex(fl):
    return hex(struct.unpack('<I', struct.pack('<f', fl))[0])[2:]   # The [2:] removes the 0x prefix


# Convert hex string back to float type
def hex_to_float(hx):
    return struct.unpack('!f', bytes.fromhex(hx))[0]


# Convert hex string to binary string
def hex_to_bin(hx):
    return bin(int(hx, 16))[2:]     # [2:] removes 0b prefix


# Convert binary string to hex string
def bin_to_hex(bn):
    return hex(int(bn, 2))[2:]      # [2:] removes 0x prefix


# Uses the functions defined above to turn a float value into a binary string
def float_to_bin(fl):
    return hex_to_bin(float_to_hex(fl))


# Uses functions from above to convert a binary string into a float value
def bin_to_float(bn):
    return hex_to_float(bin_to_hex(bn))


# Verifies the functionality of the 6 functions above
def test_float_hex_bin(flt):
    print("Original float value: ", flt, "\t", type(flt))
    hex_fl = float_to_hex(flt)
    print("Converted to hex: ", hex_fl, "\t", type(hex_fl))
    bin_fl = hex_to_bin(hex_fl)
    print("Converted to bin: ", bin_fl, "\t", type(bin_fl))
    hex_fl2 = bin_to_hex(bin_fl)
    print("Converted back to hex: ", hex_fl2, "\t", type(hex_fl2))
    fl2 = hex_to_float(hex_fl2)
    print("And finally back in float: ", fl2, "\t", type(fl2))
    fl3 = bin_to_float(float_to_bin(flt))
    print("Verifying they all work together, your float is still: ", fl3)


normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    normalize
])

# image_net = datasets.ImageNet()

mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.eval()
# print(mobilenet)  # Prints architecture

img1 = Image.open("ILSVRC2017_test_00000004.JPEG")
img2 = Image.open("ILSVRC2017_test_00000017.JPEG")
img1_t = transform(img1)
img2_t = transform(img2)
# batch_t = torch.unsqueeze(img1_t, 0)  # Use for a single-image batch
batch_t = torch.stack((img1_t, img2_t), dim=0)  # Batch 2 images together
print("batch shape: ", batch_t.shape)

out = mobilenet(batch_t)    # out has shape [N, 1000] where N = batch size
print_top_5(out)

float_test = mobilenet.classifier[3].bias[1].item()
test_float_hex_bin(float_test)


# param_count = 0
# param_copy = None
# for param in mobilenet.parameters():
#     num = 0
#     numel = torch.numel(param)
#     if param.size() == torch.Size([1000]):
#         num = 1
#         print(param.size(), "   ", numel, "   ", num)
#         print(param[1])
#         # param[1] = 4.444
#         param_copy = param.clone()
#         param_copy[1] = 44.444
#         print(param[1])
#         print(param_copy[1])
#         param = param_copy
#         print(param[1])
#     param_count += numel
# print("Param count: ", param_count)
#
# # Print out all of the parameters and their names
# # for name, param in mobilenet.named_parameters():
# #     print(name, " ", param.shape)
#
# # print(param_copy.shape)
# mobilenet.classifier[3].bias = torch.nn.Parameter(param_copy)
#
# print("***")
# for param in mobilenet.parameters():
#     numel = torch.numel(param)
#     if param.size() == torch.Size([1000]):
#         print("Weight is now: ", param[1])
# print(mobilenet.parameters())
#
# print("\nNew results:")
# out = mobilenet(batch_t)
# print_top_5(out)