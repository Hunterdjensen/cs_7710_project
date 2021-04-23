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

##Need to pip install pip install imagenet-c
##https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c
from imagenet_c import corrupt

#########################
### Mode Arguments #####
#######################

#imagenet-c
CORRUPT_IMG = True
COR_NUM = 3
COR_SEVEREITY = 1

#Ensemble Mode
ENSEMBLE = True
MODEL_COUNT = 2
MODELS = []
MODELS.append('densenet') #densenet161
MODELS.append('inception') #v3
# MODELS.append('filler')

#Base Mode
MODEL = 'mobilenet_v3_small'

##Switch statement to call models
def model_to_call(arg):
    switcher = {
        'resnet18' : models.resnet18(pretrained=True),
        'alexnet' : models.alexnet(pretrained=True),
        'squeezenet' : models.squeezenet1_0(pretrained=True),
        'vgg16' : models.vgg16(pretrained=True),
        'densenet' : models.densenet161(pretrained=True),
        'inception' : models.inception_v3(pretrained=True),
        'googlenet' : models.googlenet(pretrained=True),
        'shufflenet' : models.shufflenet_v2_x1_0(pretrained=True),
        'mobilenet_v2' : models.mobilenet_v2(pretrained=True),
        'mobilenet_v3_large' : models.mobilenet_v3_large(pretrained=True),
        'mobilenet_v3_small' : models.mobilenet_v3_small(pretrained=True),
        'resnext50_32x4d' : models.resnext50_32x4d(pretrained=True),
        'wide_resnet50_2' : models.wide_resnet50_2(pretrained=True),
        'mnasnet' : models.mnasnet1_0(pretrained=True)
    }
    net = switcher.get(arg, -1)
    if(net == -1):
        print('invalid key of: ' + arg + '. defaulting to mobilenet_v3_small')
        net = models.mobilenet_v3_small(pretrained=True)
    return net

######################
## Transformations ###
######################
# Define the transform for our images - resizes/crops them to 224x224, normalizes (required for ImageNet)
toSizeCenter = T.Compose([
    T.Resize(256),
    T.CenterCrop(224)
    #,T.ToTensor()
    #,T.Normalize(mean=[0.485, 0.456, 0.406],
                #std=[0.229, 0.224, 0.225])
])

transformImg = T.Compose([
    T.Resize(256),
    T.CenterCrop(224)
    ,T.ToTensor()
    ,T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

toTensor = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    ])

#################################
# Instantiate the model(s)
networks = []

# network used shouldn't be hardcoded
#NOT SURE IF THIS FUNCTIONS
if (ENSEMBLE):
    for m in MODELS:
        net = model_to_call(m)
        net.eval()
        networks.append(net)
else:
    netw = models.mobilenet_v3_small(pretrained=True)    # Feel free to experiment with different models here
    netw.eval()    # Put the model in inference mode
    networks.append(netw)

# print(net)  # Prints architecture (num layers, etc.)

# Run the network for a fixed number of batches and print the accuracy
num_batches = 4             # Number of loops performed, each with a new batch of images
batch_size = 16             # Number of images processed in a batch (in parallel)
num_weights_to_corrupt = 3  # Each batch, the network is reset and this many bits are randomly flipped in the weights
num_weights_permanently_stuck = 2   # This many bits will have "stuck-at faults" in the weights, permanently stuck at either 1 or 0
activation_success_odds = 1000000000    # 1 in ~1000000000 activation bits will get flipped during each operation

for net in networks:
    net = flip_n_bits_in_weights(num_weights_permanently_stuck, net)    # Introduce stuck-ats
    net = add_activation_bit_flips(net, activation_success_odds)        # Add layers to flip activation bits (comment-out to turn off)

total_correct = 0

for batch_num in range(num_batches):
    # Load images and prepare them in a batch
    image_dir = 'val/'
    random_files = random.sample(os.listdir(image_dir), batch_size)
    gt_labels = [get_label(file) for file in random_files]  # Ground-truth label for each img

    batch_t = torch.empty((batch_size, 3, 224, 224))    # Shape of [N, C, H, W]
    print('Corrupting with cornum: ' + str(COR_NUM) +' and COR_SEVEREITY: ' + COR_SEVEREITY)

    for i in range(batch_size):
        img = Image.open(image_dir + '/' + random_files[i]).convert("RGB")
        ##Add Corruption. Comment out block for baseline
        if (CORRUPT_IMG):
            img_t = toSizeCenter(img)
            pic_np = np.array(img_t) #numpy arr for corruption
            pic_np = corrupt(pic_np, severity=COR_SEVEREITY, corruption_number=COR_NUM) #See Readme for Calls
            img = Image.fromarray(np.uint8(pic_np)) #Back to PIL
            img_t = toTensor(img)
        else:
            img_t = transformImg(img)

        # img.putdata(pic_np)
        batch_t[i,:,:,:] = img_t

    # Flip bits to corrupt the network, and run it
    net_corrupts = [] #Not sure if Needed
    outs = []
    predictions = []

    for net in networks:
        net_corrupt = flip_n_bits_in_weights(num_weights_to_corrupt, net)
        net_corrupts.append(net_corrupt)
        out = net_corrupt(batch_t)    # out has shape [N, 1000] where N = batch size
        outs.append(out)

######
# Compare predictions to correct labels, and return the number correct
    """
    def get_num_correct(output, correct_labels):
        images_per_batch, num_classes = output.shape
        _, indices = torch.sort(output, descending=True)
        num_correct = 0

        for image in range(images_per_batch):
            prediction = indices[image][0]  # [0] for the largest prediction
            if prediction == correct_labels[image]:
                num_correct += 1

        return num_correct
    """
    for output in outs:
        images_per_batch, num_classes = output.shape
        _, indices = torch.sort(output, descending=True)
        prediction = []
        for image in range(images_per_batch):
            prediction.append(indices[image][0])  # [0] for the largest prediction
        predicitons.append(prediction) #to be used in comparison loop

    finalout = []
    ##TO DO
    #handle difference
    for i in range(0, len(predictions[0])): #Loop through an output array to compare and create a final output arr
        if (predictions[0][i] == predictions[1][i]):
            finalout.append(predictions[1][0])
        else: ## How does CI work)
            print('diff hit. taking case 0')
            finalout.append(-1) ##Make a decider

    num_correct = get_num_correct(finalout, gt_labels)

    total_correct += num_correct
    print("Batch %d:  %d / %d" % (batch_num, num_correct, batch_size))

print("Percentage Correct: %.2f%%" % ((total_correct / (batch_size * num_batches)) * 100))
print(num_weights_to_corrupt, "out of", (get_num_params(net) * 32),
      " weight bits temporarily corrupted, or %.8f%%" % ((num_weights_to_corrupt / (get_num_params(net) * 32)) * 100))
print(num_weights_permanently_stuck, "out of", (get_num_params(net) * 32),
      " weight bits permanently corrupted, or %.8f%%" % ((num_weights_permanently_stuck / (get_num_params(net) * 32)) * 100))
print(get_flips_in_activations(), "activation bits were flipped during operation, approx: %.8f%%"
      % ((1/(1+activation_success_odds)) * 100))
