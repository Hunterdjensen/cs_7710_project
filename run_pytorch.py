import torch
import torchvision.models as models
from torchvision import datasets, transforms as T
from PIL import Image
import numpy as np
from class_labels import print_top_5


# Define the transform for our images - resizes/crops them to 256x256, normalizes (required for ImageNet)
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

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
