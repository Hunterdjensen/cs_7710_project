import torch
from PIL import Image
import numpy as np
import random
import os  # For accessing files
from class_labels import get_label
from class_labels import vote
from bit_flipping import flip_n_bits_in_weights
from bit_flipping import add_activation_bit_flips
from bit_flipping import get_num_params
from bit_flipping import get_flips_in_activations
from bit_flipping import reset_flips_in_activations
from get_model import get_model
from transformations import toSizeCenter, toTensor
# Need to pip install pip install imagenet-c
# https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c
from imagenet_c import corrupt

#################################################################################################
#                                          Parameters:                                          #
#################################################################################################

MODELS = ['inception_v3', 'densenet161', 'alexnet']    # For an ensemble, put >1 network here

# imagenet-c:
CORRUPT_IMG = True
COR_NUM = 3
COR_SEVERITY = 1

# Bit-flipping corruptions:
num_weights_to_corrupt = 3  # Each batch, the network is reset and this many bits are randomly flipped in the weights
num_weights_permanently_stuck = 2  # This many bits will have "stuck-at faults" in the weights, permanently stuck at either 1 or 0
activation_success_odds = 1000000000  # 1 in ~1000000000 activation bits will get flipped during each operation

# Model parameters:
num_batches = 2  # Number of loops performed, each with a new batch of images
batch_size = 8  # Number of images processed in a batch (in parallel)
val_image_dir = 'val/'  # The directory where validation images are stored
voting_heuristic = 'simple'     # Determines the algorithm used to predict between multiple models


#################################################################################################
#                                           Runtime:                                            #
#################################################################################################

# Instantiate the model(s)
networks = []
for i, m in enumerate(MODELS):
    net = get_model(m)
    net.name = str(i) + '_' + net.__class__.__name__  # Give the net a unique name (used by bit_flipping.py)
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
