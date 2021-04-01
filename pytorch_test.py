import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image
import numpy as np
import dtype_conversions
from dtype_conversions import float_to_bin
from dtype_conversions import bin_to_float
from class_labels import print_top_5
import bit_flipping

#################################################################################################
#       This file is Hunter's sandbox, see file run_pytorch.py for more readable code           #
#################################################################################################


# Define the transform for all of our images - resizes/crops them to 256x256, normalizes (required for ImageNet)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# image_net = datasets.ImageNet()   # TODO: Figure out how to use datasets?

# Instantiate the model
mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.eval()    # Put the model in inference mode
# print(mobilenet)  # Prints architecture (num layers, etc.)

# Load two images, prepare in a batch
img1 = Image.open("ILSVRC2017_test_00000004.JPEG")
img2 = Image.open("ILSVRC2017_test_00000017.JPEG")
img1_t = transform(img1)
img2_t = transform(img2)
# batch_t = torch.unsqueeze(img1_t, 0)  # Use for a single-image batch
batch_t = torch.stack((img1_t, img2_t), dim=0)  # Batch 2 images together
print("batch shape: ", batch_t.shape)

# Run the network
out = mobilenet(batch_t)    # out has shape [N, 1000] where N = batch size
print_top_5(out)            # Print out the predictions

# Test out our dtype conversion functions:
# float_test = mobilenet.classifier[3].bias[1].item()
# dtype_conversions.test_float_hex_bin(float_test)

bit_flipping.bit_flip_init(mobilenet)

print(mobilenet.classifier[3].bias[1].item())
print(mobilenet.features[11].block[3][1].weight[95].item())
# print(mobilenet.state_dict())   # Does same thing as named_parameters()

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