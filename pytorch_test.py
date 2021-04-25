import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image
import numpy as np
import random
import os
import dtype_conversions
from dtype_conversions import float_to_bin
from dtype_conversions import bin_to_float
from class_labels import print_top_5
import bit_flipping
from bit_flipping import BitFlipLayer
from bit_flipping import get_reference
from bit_flipping import bit_flip_init
from class_labels import vote
from class_labels import get_label
from transformations import toSizeCenter, toTensor, transformImg
from bit_flipping import get_num_params
from bit_flipping import flip_n_bits_in_weights
from bit_flipping import add_activation_bit_flips
from bit_flipping import get_num_params
from bit_flipping import get_flips_in_activations
from get_model import get_model
from imagenet_c import corrupt

#################################################################################################
#       This file is Hunter's sandbox, see file run_pytorch.py for more readable code           #
#################################################################################################

MODELS = ['inception_v3', 'densenet161', 'alexnet']    # For an ensemble, put >1 network here

# imagenet-c:
CORRUPT_IMG = False
COR_NUM = 3
COR_SEVERITY = 1

# Bit-flipping corruptions:
num_weights_to_corrupt = 2  # Each batch, the network is reset and this many bits are randomly flipped in the weights
num_weights_permanently_stuck = 1  # This many bits will have "stuck-at faults" in the weights, permanently stuck at either 1 or 0
activation_success_odds = 100000000  # 1 in ~1000000000 activation bits will get flipped during each operation

# Model parameters:
num_batches = 2  # Number of loops performed, each with a new batch of images
batch_size = 3  # Number of images processed in a batch (in parallel)
val_image_dir = 'val/'  # The directory where validation images are stored
voting_heuristic = 'simple'     # Determines the algorithm used to predict between multiple models


#################################################################################################
#                                           Runtime:                                            #
#################################################################################################

# Instantiate the model(s)
networks = []
for i, m in enumerate(MODELS):
    net = get_model(m)
    net.name = str(i) + '_' + net.__class__.__name__    # Give the net a unique name (used by bit_flipping.py)
    net.eval()  # Put in evaluation mode (already pretrained)
    net = flip_n_bits_in_weights(num_weights_permanently_stuck, net)  # Introduce stuck-ats
    net = add_activation_bit_flips(net, activation_success_odds)  # Add layers to flip activation bits
    networks.append(net)

if CORRUPT_IMG:
    print('Corrupting with COR_NUM: ' + str(COR_NUM) + ' and COR_SEVERITY: ' + str(COR_SEVERITY))


# Run each batch
total_correct = 0
for batch_num in range(num_batches):
    # Load images and prepare them in a batch
    image_paths = random.sample(os.listdir(val_image_dir), batch_size)
    gt_labels = torch.tensor([get_label(image) for image in image_paths])  # Ground-truth label for each image

    batch_t = torch.empty((batch_size, 3, 224, 224))  # Shape of [N, C, H, W]
    for i in range(batch_size):
        img = Image.open(val_image_dir + '/' + image_paths[i]).convert("RGB")
        img = toSizeCenter(img)
        if CORRUPT_IMG:
            pic_np = np.array(img)  # numpy arr for corruption
            pic_np = corrupt(pic_np, severity=COR_SEVERITY, corruption_number=COR_NUM)  # See Readme for Calls
            img = Image.fromarray(np.uint8(pic_np))  # Back to PIL
        img_t = toTensor(img)
        batch_t[i,:,:,:] = img_t

    # Flip bits to corrupt the network, and run it
    out = torch.empty((len(MODELS), batch_size, 1000))    # Shape [M, N, 1000] where M = num models, and N = batch size
    for i, net in enumerate(networks):
        net_corrupt = flip_n_bits_in_weights(num_weights_to_corrupt, net)
        out[i,:,:] = net_corrupt(batch_t)

    predictions = vote(out, voting_heuristic)   # Returns predictions, with shape [N] (one prediction per image in batch)
    num_correct = torch.sum(predictions == gt_labels).item()    # Item() pulls the integer out of the tensor

    total_correct += num_correct
    print("Batch %d:  %d / %d" % (batch_num, num_correct, batch_size))


#################################################################################################
#                                         Print Results:                                        #
#################################################################################################

print("Percentage Correct: %.2f%%" % ((total_correct / (batch_size * num_batches)) * 100))
for i, net in enumerate(networks):
    print(MODELS[i] + str(':'))
    print("\t", num_weights_to_corrupt, "out of", (get_num_params(net) * 32),
          " weight bits temporarily corrupted, or %.8f%%"
          % ((num_weights_to_corrupt / (get_num_params(net) * 32)) * 100))
    print("\t", num_weights_permanently_stuck, "out of", (get_num_params(net) * 32),
          " weight bits permanently corrupted, or %.8f%%"
          % ((num_weights_permanently_stuck / (get_num_params(net) * 32)) * 100))
    print("\t", get_flips_in_activations(net),
          "activation bits were flipped during operation, approx: %.8f%%"
          % ((1 / (1 + activation_success_odds)) * 100))




# Testing simple voting with 2 networks:
# out = torch.tensor([[[5, 2, 0], [2, 4, 3], [2, 1, 0], [0, 7, 3]],       # Model 1 - 4 images, 3 classes
#                     [[4, 3, 0], [1, 0, 0], [1, 2, 3], [3, 8, 0]]])      # Model 2 -
# # Correct: [0, 1, 2, 0]
# # 1st model: T T F F    0, 1, 0, 1
# # 2nd model: T F T F    0, 0, 2, 2
# # Simple vo: T T F F
# # Max. conf: T T T F
# print(out.shape)
# vote(out, 'simple')

# Testing simple voting with 3 networks:
# out = torch.tensor([[[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 5]],      # Model 1 - 4 images, 3 classes
#                     [[5, 0, 0], [0, 0, 5], [0, 0, 5], [0, 5, 0]],      # Model 2 -
#                     [[5, 0, 0], [0, 0, 5], [5, 0, 0], [5, 0, 0]]])     # Model 3
# # Correct: [0, 2, 2, 2]
# # 1st model: 0, 1, 2, 2
# # 2nd model: 0, 2, 2, 1
# # 3rd model: 0, 2, 0, 0
# # Correct predictions should be [0, 2, 2, 2]
# print(out.shape)
# print(vote(out, 'simple'))



# # Instantiate the model
# net = models.densenet161(pretrained=True)
# bit_flip_init(net)
# print(net, '\n*****')
#
# layer = getattr(getattr(getattr(getattr(getattr(net, 'features'), 'denseblock2'), 'denselayer11'), 'conv2'), 'weight')
# print("success")
# print(layer.shape)


# mobilenet = models.mobilenet_v3_small(pretrained=True)
# print(mobilenet)
# bit_flip_init(mobilenet)
# layer = getattr(getattr(getattr(getattr(mobilenet, 'features')[11], 'block'), '3')[0], 'weight')
# print("**")
# print(layer.shape)



# setattr(getattr(getattr(mobilenet, 'features')[11], 'block')[2], 'fc1', nn.Sequential(layer, BitFlipLayer(1000)))
# name = 'features.11.block.2.fc1.weight'
# name = 'features.11.block.0.1.weight'
# name = 'features.12.0.weight'
# name = 'classifier.3.weight'
# layer, prev, num = get_reference(name, mobilenet)
# print("Prev and num are: ", prev, num)
# add_bit_flip_layers(name, mobilenet, 1000)



# Instantiate the model
# mobilenet = models.mobilenet_v3_small(pretrained=True)
# mobilenet.eval()  # Put the model in inference mode
# # print(mobilenet)  # Prints architecture (num layers, etc.)
#
# # Load two images, prepare in a batch
# img1 = Image.open("ILSVRC2017_test_00000004.JPEG")
# img2 = Image.open("ILSVRC2017_test_00000017.JPEG")
# img1_t = transform(img1)
# img2_t = transform(img2)
# # batch_t = torch.unsqueeze(img1_t, 0)  # Use for a single-image batch
# batch_t = torch.stack((img1_t, img2_t), dim=0)  # Batch 2 images together
# print("batch shape: ", batch_t.shape)
#
# # Run the network
# out = mobilenet(batch_t)  # out has shape [N, 1000] where N = batch size
# print_top_5(out)  # Print out the predictions
# # print(mobilenet)
#
# # for name, param in mobilenet.named_parameters():
# #     print(name, torch.numel(param))
#
#
# class Testme(nn.Module):  ## it is a subclass of module ##
#     def __init__(self, num_to_flip):
#         super(Testme, self).__init__()
#         self.k = num_to_flip
#
#     def forward(self, x):
#         num_elements = torch.numel(x)
#         view = x.view(num_elements)  # Create a 1D view
#         self.k = self.k % num_elements  # Use modulus in case k > num_elements
#         view[self.k] += 400
#         return x
#
#
# print("********* :) ********")
# bit_flipping.test_adding_BitFlipLayer_layers()


# mobilenet.classifier = nn.Sequential(Testme(0), mobilenet.classifier)
# mobilenet.classifier[0] = nn.Sequential(nn.ReLU(), mobilenet.classifier[0])
# mobilenet.classifier[0] = nn.Sequential(BitFlipLayer(5000), mobilenet.classifier[0], BitFlipLayer(5000))
# layer = getattr(mobilenet, 'classifier')[0]
# getattr(mobilenet, 'classifier')[0] = nn.Sequential(layer, BitFlipLayer(5000))


# layer = getattr(getattr(mobilenet, 'features')[11], 'block')[3]
# layer[1] = nn.Sequential(layer[1], BitFlipLayer(5000))
# layer = getattr(getattr(getattr(mobilenet, 'features')[11], 'block')[2], 'fc1')
# setattr(getattr(getattr(mobilenet, 'features')[11], 'block')[2], 'fc1', nn.Sequential(layer, BitFlipLayer(1000)))
# name = 'features.11.block.2.fc1.weight'
# name = 'features.11.block.0.1.weight'
# name = 'features.12.0.weight'
# name = 'classifier.3.weight'
# layer, prev, num = get_reference(name, mobilenet)
# print("Prev and num are: ", prev, num)
# add_bit_flip_layers(name, mobilenet, 1000)

# mobilenet = bit_flipping.add_activation_bit_flips(mobilenet, 100000000)
# # print(mobilenet)
#
# out = mobilenet(batch_t)  # out has shape [N, 1000] where N = batch size
# print_top_5(out)  # Print out the predictions
# print("Number of flips in activations: ", bit_flipping.get_flips_in_activations())

# Test out our dtype conversion functions:
# float_test = mobilenet.classifier[3].bias[1].item()
# dtype_conversions.test_float_hex_bin(float_test)

# bit_flipping.bit_flip_init(mobilenet)
# mobilenet = bit_flipping.flip_n_bits(200, mobilenet)

# Before:  01000000100000100011001110101110 4.068808555603027 toggle the 2th bit
# Traceback (most recent call last):
#   File "/Users/jenniferjensen/PycharmProjects/pytorch_eval_tests/pytorch_test.py", line 51, in <module>
#     bit_flipping.flip_n_bits(5, mobilenet, print_out=True)
#   File "/Users/jenniferjensen/PycharmProjects/pytorch_eval_tests/bit_flipping.py", line 50, in flip_n_bits
#     write_tensor(layer_tensor, rand_param, print_out=print_out)
#   File "/Users/jenniferjensen/PycharmProjects/pytorch_eval_tests/bit_flipping.py", line 85, in write_tensor
#     view[k] = bin_to_float(binary)
#   File "/Users/jenniferjensen/PycharmProjects/pytorch_eval_tests/dtype_conversions.py", line 37, in bin_to_float
#     return hex_to_float(bin_to_hex(bn))
#   File "/Users/jenniferjensen/PycharmProjects/pytorch_eval_tests/dtype_conversions.py", line 16, in hex_to_float
#     return struct.unpack('!f', bytes.fromhex(hx))[0]
# struct.error: unpack requires a buffer of 4 bytes

# print(mobilenet.classifier[3].bias[1].item())
# print(mobilenet.features[11].block[3][1].weight[95].item())
# print(getattr(mobilenet, 'features')[11].block[3][1].weight[95].item())
# print(getattr(mobilenet.features[11], 'block')[3][1].weight[95].item())
# print(getattr(mobilenet.features[11].block[3][1], 'weight')[95].item())
# print(getattr(getattr(getattr(mobilenet, 'features')[11], 'block')[3][1], 'weight')[95].item())
# l1 = getattr(mobilenet, 'features')
# l1 = l1[11]
# l2 = getattr(l1, 'block')
# l2 = l2[3][1]
# l3 = getattr(l2, 'weight')[95]
# print(l3.item())
# print(mobilenet.state_dict())   # Does same thing as named_parameters()

# EXEC test
# test_val = None
# str_test = "test_val = 'Halo Worl'"
# exec(str_test)
# print("test_val is: ", test_val)

# Test with torch views and indexing:
# tnsr = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
# print(tnsr)
# num_elements = torch.numel(tnsr)
# print(num_elements)
# view = tnsr.view(num_elements)
# print(view)
# view[3] = 999
# print(view)
# print(tnsr)

# keys = []
# for name, value in mobilenet.named_parameters():
#     keys.append(name)
#
# with torch.no_grad():
#     print("**")
#     print(keys[-1])
#     print("***")
#     lookup = keys[-1].split('.')
#     print(getattr(mobilenet, lookup[0])[int(lookup[1])].bias[1])

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
# # mobilenet.classifier[3].bias = torch.nn.Parameter(param_copy)
# # with torch.no_grad():
# # getattr(mobilenet, keys[-1].split('.')[0])[3].bias = torch.nn.Parameter(param_copy)
#
# with torch.no_grad():
#     tensor = bit_flipping.get_layer_tensor('classifier.3.bias', mobilenet)
#     bit_flipping.write_tensor(tensor, 1)
#     # tensor[1] = 44.44     # Equivalent to the line above

# print("***")
# for param in mobilenet.parameters():
#     numel = torch.numel(param)
#     if param.size() == torch.Size([1000]):
#         print("Weight is now: ", param[1])
# # print(mobilenet.parameters())
#
# print("\nNew results:")
# out = mobilenet(batch_t)
# print_top_5(out)
