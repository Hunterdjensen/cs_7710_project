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
from bit_flipping import flip_n_bits
from bit_flipping import get_num_params


# Define the transform for our images - resizes/crops them to 224x224, normalizes (required for ImageNet)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Instantiate the model
net = models.mobilenet_v3_small(pretrained=True)
net.eval()    # Put the model in inference mode
# print(net)  # Prints architecture (num layers, etc.)


# Run the network for a fixed number of epochs and print the accuracy
num_epochs = 4              # Number of loops performed, each with a new batch of images
batch_size = 64             # Number of images processed in a batch (in parallel)
num_bits_to_corrupt = 5     # Each epoch, the network is reset and this many bits are randomly flipped

total_correct = 0
for epoch in range(num_epochs):
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
    net_corrupt = flip_n_bits(num_bits_to_corrupt, net)
    out = net_corrupt(batch_t)    # out has shape [N, 1000] where N = batch size
    num_correct = get_num_correct(out, gt_labels)
    total_correct += num_correct
    print("Epoch %d:  %d / %d" % (epoch, num_correct, batch_size))
print("Percentage Correct: %.2f%%" % ((total_correct/(batch_size*num_epochs))*100))
print(num_bits_to_corrupt, "out of", (get_num_params(net)*32),
      "bits corrupted, or %.8f%%" % ((num_bits_to_corrupt/(get_num_params(net)*32))*100))
