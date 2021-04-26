import torch
from PIL import Image
import numpy as np
import random
import os  # For accessing files
from class_labels import get_label
from class_labels import vote
from bit_flipping import flip_n_bits_in_weights, flip_stochastic_bits_in_weights
from bit_flipping import add_activation_bit_flips
from bit_flipping import get_num_params
from bit_flipping import get_flips_in_activations, get_flips_in_weights
from bit_flipping import get_num_activation_flips, get_num_weight_flips
from bit_flipping import reset_bit_flip_counters
from get_model import get_model
from transformations import toSizeCenter, toTensor
# Need to pip install pip install imagenet-c
# https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c
from imagenet_c import corrupt


def run(
    #################################################################################################
    #                                          Parameters:                                          #
    #################################################################################################
    MODELS = None,
    PRINT_OUT = True,    # Print out results at end

    # imagenet-c:
    CORRUPT_IMG = True,
    COR_NUM = 3,    # 8 is frost covering image
    COR_SEVERITY = 1,

    # Bit-flipping corruptions:
    stuck_at_faults = 2,  # This many bits will have "stuck-at faults" in the weights, permanently stuck at either 1 or 0
    weights_BER = 1e-10,  # Bit Error Rate for weights (applied each batch, assuming weights are reloaded for each batch)
    activation_BER = 1e-10,  # Bit Error Rate for activations, i.e. 1e-9 = ~(1 in 1000000000) errors in the activations

    # Model parameters:
    num_batches = 1,  # Number of loops performed, each with a new batch of images
    batch_size = 8,  # Number of images processed in a batch (in parallel)
    val_image_dir = 'val/',  # The directory where validation images are stored
    voting_heuristic = 'sum all'  # Determines the algorithm used to predict between multiple models
):
    if MODELS is None:
        MODELS = ['resnext101_32x8d', 'densenet161', 'inception_v3']  # For an ensemble, put >1 network here
    reset_bit_flip_counters()   # Do this at start in case calling run() multiple times

    #################################################################################################
    #                                           Runtime:                                            #
    #################################################################################################
    # Instantiate the model(s)
    networks = []
    for i, m in enumerate(MODELS):
        net = get_model(m)
        net.name = str(i) + '_' + net.__class__.__name__  # Give the net a unique name (used by bit_flipping.py)
        net.eval()  # Put in evaluation mode (already pretrained)
        if stuck_at_faults != 0:
            net = flip_n_bits_in_weights(stuck_at_faults, net)  # Introduce stuck-ats
        if activation_BER != 0:  # If nonzero chance of activation bit flips
            net = add_activation_bit_flips(net, activation_BER)  # Add layers to flip activation bits
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
            batch_t[i, :, :, :] = img_t

        # Run each network and store output in 'out'
        out = torch.empty(
            (len(MODELS), batch_size, 1000))  # Shape [M, N, 1000] where M = num models, and N = batch size
        for i, net in enumerate(networks):
            if weights_BER != 0:  # If nonzero chance of weight bit flips
                net = flip_stochastic_bits_in_weights(weights_BER, net)
            out[i, :, :] = net(batch_t)

        predictions = vote(out, voting_heuristic)  # Returns predictions, with shape [N] (one prediction per image)
        num_correct = torch.sum(predictions == gt_labels).item()  # Item() pulls the integer out of the tensor

        total_correct += num_correct
        print("Batch %d:  %d / %d" % (batch_num, num_correct, batch_size))

    #################################################################################################
    #                                         Print Results:                                        #
    #################################################################################################

    percentage_correct = (total_correct / (batch_size * num_batches)) * 100
    if PRINT_OUT:
        print("Percentage Correct: %.2f%%" % percentage_correct)
        for i, net in enumerate(networks):
            print(MODELS[i] + str(':'))
            print("\t Total bit flips in weights:", get_flips_in_weights(net), "or %.0f per minute of inference"
                  % (get_flips_in_weights(net) / (num_batches / (32 * 60))))  # 32 batches/second (32 fps) * 60 seconds
            print("\t Total bit flips in activations:", get_flips_in_activations(net), "or %.0f per minute of inference"
                  % (get_flips_in_activations(net) / (num_batches / (32 * 60))))  # 32 batches/second (32 fps) * 60 seconds
            print("\t", stuck_at_faults, "out of", (get_num_params(net) * 32),
                  " weight bits permanently corrupted, or %.8f%%"
                  % ((stuck_at_faults / (get_num_params(net) * 32)) * 100))

    return [percentage_correct, get_num_weight_flips(), get_num_activation_flips()]


def main():
    run()  # Run the module with default parameters if calling this file as main


if __name__ == "__main__":
    main()