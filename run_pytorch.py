import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image
import numpy as np
import random
import os   # For accessing files
from class_labels import print_top_5
from class_labels import get_label
from class_labels import get_num_correct
from bit_flipping import flip_n_bits_in_weights
from bit_flipping import add_activation_bit_flips
from bit_flipping import get_num_params
from bit_flipping import get_flips_in_activations
from bit_flipping import reset_flips_in_activations


# Define the transform for our images - resizes/crops them to 224x224, normalizes (required for ImageNet)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Instantiate the model
net = models.mobilenet_v3_small(pretrained=True)    # Feel free to experiment with different models here
net.eval()    # Put the model in inference mode
# print(net)  # Prints architecture (num layers, etc.)


# Run the network for a fixed number of batches and print the accuracy
num_batches = 4             # Number of loops performed, each with a new batch of images
batch_size = 16             # Number of images processed in a batch (in parallel)
num_weights_to_corrupt = 3  # Each batch, the network is reset and this many bits are randomly flipped in the weights
num_weights_permanently_stuck = 2   # This many bits will have "stuck-at faults" in the weights, permanently stuck at either 1 or 0
activation_success_odds = 1000000000    # 1 in ~1000000000 activation bits will get flipped during each operation

net = flip_n_bits_in_weights(num_weights_permanently_stuck, net)    # Introduce stuck-ats
net = add_activation_bit_flips(net, activation_success_odds)        # Add layers to flip activation bits (comment out to flip no activation bits)

total_correct = 0
for batch_num in range(num_batches):
    # Load images and prepare them in a batch
    image_dir = 'val/'
    random_files = random.sample(os.listdir(image_dir), batch_size)
    gt_labels = [get_label(file) for file in random_files]  # Ground-truth label for each img

    batch_t = torch.empty((batch_size, 3, 224, 224))    # Shape of [N, C, H, W]
    for i in range(batch_size):
        img = Image.open(image_dir + '/' + random_files[i]).convert("RGB")
        img_t = transform(img)
        batch_t[i,:,:,:] = img_t

    # Flip bits to corrupt the network, and run it
    net_corrupt = flip_n_bits_in_weights(num_weights_to_corrupt, net)
    out = net_corrupt(batch_t)    # out has shape [N, 1000] where N = batch size
    num_correct = get_num_correct(out, gt_labels)
    total_correct += num_correct
    print("Batch %d:  %d / %d" % (batch_num, num_correct, batch_size))
print("Percentage Correct: %.2f%%" % ((total_correct / (batch_size * num_batches)) * 100))
print(num_weights_to_corrupt, "out of", (get_num_params(net) * 32),
      " weight bits temporarily corrupted, or %.8f%%" % ((num_weights_to_corrupt / (get_num_params(net) * 32)) * 100))
print(num_weights_permanently_stuck, "out of", (get_num_params(net) * 32),
      " weight bits permanently corrupted, or %.8f%%" % ((num_weights_permanently_stuck / (get_num_params(net) * 32)) * 100))
print(get_flips_in_activations(), "activation bits were flipped during operation, approx: %.8f%%"
      % ((1/(1+activation_success_odds)) * 100))
