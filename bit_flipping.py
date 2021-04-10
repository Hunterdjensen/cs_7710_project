import torch
import torch.nn as nn
import numpy as np
from dtype_conversions import float_to_bin
from dtype_conversions import bin_to_float
import copy

#################################################################################################
#               Takes in models and randomly flips bits in their parameters                     #
#################################################################################################

# Global variables:
init_flag = False
total_param_count = 0           # Total number of parameters in the network
cumulative_param_count = None   # List of the param counts per layer, to use w/ Numpy.digitize()
names = None                    # List of layer "names" (i.e: 'features.12.block.2.fc1.weight')
num_activation_flips = 0


#################################################################################################
#                                 User-called functions:                                        #
#################################################################################################

# Flips n bits randomly inside the model - main function to call from this file
def flip_n_bits_in_weights(n, model, print_out=False):
    global init_flag, total_param_count, cumulative_param_count, names
    if init_flag is False:
        bit_flip_init(model)

    model_corrupted = copy.deepcopy(model)      # Make a copy of the network to corrupt

    # Then pick n random numbers which will correspond to which parameters to corrupt
    random_param_numbers = np.random.randint(low=0, high=total_param_count, size=(n,))

    # For each of those random parameters, get a reference to the layer the parameter belongs
    # to, grab its tensor, and then corrupt the parameter inside that tensor by flipping 1 bit
    for rand_param in random_param_numbers:
        layer_num = np.digitize(rand_param, cumulative_param_count)     # Find corresponding bin (layer) to corrupt
        if print_out is True:
            print("Flipping a bit in parameter #%d in layer %d (%s)" % (rand_param, layer_num, names[layer_num]))
        layer_tensor = get_layer_tensor(names[layer_num], model_corrupted)
        corrupt_weights(layer_tensor, k=rand_param, print_out=print_out)
    return model_corrupted


# Goes through all of the layers found in the model using model.named_parameters() and adds a
# custom BitFlipLayer after each of them, which stochastically flips bits in the activations as
# they're passing through.  The variable odds_of_no_flip determines the probability of a bit
# not flipping - if odd_of_no_flip is 2000, then there's a 2000:1 chance the bit will be unchanged,
# or conversely 1:2000 chance the bit will be flipped.
def add_activation_bit_flips(model, odds_of_no_flip):
    global init_flag, names
    if init_flag is False:
        bit_flip_init(model)

    model_corrupted = copy.deepcopy(model)  # Make a copy of the network to corrupt

    for name in names:
        layer, prev, num = get_reference(name, model_corrupted)     # Get the reference to a layer
        if layer is not None:  # That means that this is a valid layer to add a BitFlipLayer behind (it's a layer with a weight)
            if num:  # The final attribute is a number, we can index with []
                layer[int(prev)] = nn.Sequential(layer[int(prev)], BitFlipLayer(odds_of_no_flip))
            else:  # The last attribute isn't an index, need to use set/get attr
                # The following is messy but is essentially: layer = nn.Sequential(layer, BitFlipLayer())
                setattr(layer, prev, nn.Sequential(getattr(layer, prev), BitFlipLayer(odds_of_no_flip)))

    bit_flip_init(model_corrupted)  # IMPORTANT: Calls init again so that the global variable 'names' stays up-to-date
    return model_corrupted


# Return the total number of parameters (weights/biases) in the model
def get_num_params(model):
    global init_flag, total_param_count
    if init_flag is False:
        bit_flip_init(model)

    return total_param_count


# Get the number of bit flips that have occurred in the activations (BitFlipLayer layers)
# since the global variable was last reset
def get_flips_in_activations():
    global num_activation_flips
    return num_activation_flips


# Resets the global variable that's incremented in BitFlipLayer layers, which count activation bit flips
def reset_flips_in_activations():
    global num_activation_flips
    num_activation_flips = 0


#################################################################################################
#                                   Custom PyTorch Layer:                                       #
#################################################################################################

# PyTorch Sequential layer that stochastically flips bits in the activations
class BitFlipLayer(nn.Module):
    def __init__(self, odds_of_no_bit_flip):
        super(BitFlipLayer, self).__init__()
        odds = odds_of_no_bit_flip     # i.e. if this is 2000, then ~1 per 2000 bits will flip
        self.probability_of_flip = 1 / (1 + odds)   # Probability of a bit flip, per bit

    def forward(self, x):
        global num_activation_flips
        num_elements = torch.numel(x)
        bits_per_element = 32
        # Find probability per activation element, so we can use the corrupt_weights() function from above
        probability_of_flip_per_element = bits_per_element * self.probability_of_flip
        activations_flipped = np.random.choice([1, 0], size=num_elements, p=[probability_of_flip_per_element, 1-probability_of_flip_per_element])
        indices_of_flips = np.where(activations_flipped == 1)[0]    # Creates a 1-D Numpy array of indices to corrupt
        for idx in list(indices_of_flips):
            corrupt_weights(x, idx, print_out=False)
        num_activation_flips += indices_of_flips.shape[0]   # This global variable is to give us an idea of how many flips are occurring
        return x


#################################################################################################
#                            Helper Functions for bit flipping:                                 #
#################################################################################################


# Analyzes the model (i.e mobilenet) to get the number of parameters total and per layer
# Doesn't need to be called by user, flip_n_bits() will call it
def bit_flip_init(model):
    global init_flag, total_param_count, cumulative_param_count, names, num_activation_flips
    reset_flips_in_activations()    # Reset here just to be safe
    init_flag = True
    total_param_count = 0
    cumulative_param_count = []
    names = []

    for name, param in model.named_parameters():
        num_elements = torch.numel(param)
        total_param_count += num_elements
        cumulative_param_count.append(total_param_count)
        names.append(name)


# Get the tensor corresponding to the layer given by the 'name' argument.
# For example, the layer 'features.11.block.2.fc1.weight' contains the weight parameters
# for a layer of this network, which is a tensor of shape torch.Size([144, 576, 1, 1])
def get_layer_tensor(name, model):
    tensor = model
    split = name.split('.')  # 'features.11.block.2.fc1.weight' -> ['features', '11', 'block', '2', 'fc1', 'weight']
    for attribute in split:
        if attribute.isnumeric():   # Attribute is a number
            tensor = tensor[int(attribute)]
        else:                       # Attribute is a word
            tensor = getattr(tensor, attribute)
    return tensor


# Does the exact same thing as get_layer_tensor, but... messier.  If you must know:
# Since Python passes by object-reference, we can't assign a value to an object reference
# and have it affect the original object.  In this case, I'm trying to get a layer from our network
# (say features.11.block.2.fc1) and set it equal to a new torch.nn.Sequential() list of two layers,
# because we're adding additional layers into the network. BUT we can't assign an object reference.
# But note that if you have an index you can use [x] and get the reference to the actual object,
# otherwise we need to use setattr().
# So instead we have this gross function that will return MOST of the reference we need, along with
# the final attribute separately (called 'prev') and a flag saying whether it's a number or not, so
# that we can use the output of this function to assign using either [] or setattr().  There may be
# a better way to do this, but since we're trying to do something so obscure I couldn't find much
# about this online.
def get_reference(name, model):
    layer = model
    split = name.split('.')  # 'features.11.block.2.fc1.weight' -> ['features', '11', 'block', '2', 'fc1', 'weight']
    prev_is_num = False
    prev = None
    for attribute in split:
        if attribute == 'weight':   # done
            return layer, prev, prev_is_num
        else:
            if prev is None:    # First iteration
                prev = attribute
                prev_is_num = attribute.isnumeric()
            else:   # Second+ iteration
                if prev_is_num:
                    layer = layer[int(prev)]
                else:   # Previous was a word
                    layer = getattr(layer, prev)
                prev = attribute
                prev_is_num = attribute.isnumeric()
    return None, None, None


# Modifies the kth value of a layer_tensor.  It does this by creating a 1-D view
# of the tensor and indexing into its kth member to write to it, and flips one of
# its bits at random.
def corrupt_weights(layer_tensor, k, print_out=False, test_write_value=None):
    num_elements = torch.numel(layer_tensor)
    view = layer_tensor.view(num_elements)  # Create a 1D view
    k = k % num_elements    # Use modulus in case k > num_elements
    with torch.no_grad():   # Use no_grad so PyTorch doesn't try to do back-prop on this
        if test_write_value is not None:    # For testing purposes only
            view[k] = test_write_value
        else:   # Normal operation:
            assert(view[k].dtype == torch.float32)  # Haven't considered data-type conversions for non-floats yet :)
            binary = float_to_bin(view[k].item())               # Convert the desired parameter into binary
            bit_to_flip = np.random.randint(0, len(binary))     # Choose which of its bit to flip
            if print_out is True:
                print("**********")
                print("Before: ", binary, view[k].item(), "toggle the %dth bit" % (bit_to_flip+1))

            binary = binary[:bit_to_flip] + toggle_bit(binary[bit_to_flip]) + binary[bit_to_flip+1:]
            view[k] = bin_to_float(binary)                      # Put the corrupted data into the tensor
            if print_out is True:
                print("After:  ", binary, view[k].item())
                print("**********")


def toggle_bit(b):
    if b == '1':
        return '0'
    else:
        return '1'


# Verifies the functionality of corrupt_weights() to find and replace the kth element
def test_weight_corruption():
    dummy_tensor = torch.tensor([[[[1.0, 1.1], [1.2, 1.3]], [[2.0, 2.1], [2.2, 2.3]], [[3.0, 3.1], [3.2, 3.3]]]])
    print("Tensor shape: ", dummy_tensor.shape)
    print("Tensor before: \n", dummy_tensor)
    corrupt_weights(dummy_tensor, 0, test_write_value=0)
    corrupt_weights(dummy_tensor, 1, test_write_value=1)
    corrupt_weights(dummy_tensor, 2, test_write_value=2)
    corrupt_weights(dummy_tensor, 3, test_write_value=3)
    corrupt_weights(dummy_tensor, 4, test_write_value=4)
    corrupt_weights(dummy_tensor, 5, test_write_value=5)
    corrupt_weights(dummy_tensor, 6, test_write_value=6)
    corrupt_weights(dummy_tensor, 7, test_write_value=7)
    corrupt_weights(dummy_tensor, 8, test_write_value=8)
    corrupt_weights(dummy_tensor, 9, test_write_value=9)
    corrupt_weights(dummy_tensor, 10, test_write_value=10)
    corrupt_weights(dummy_tensor, 11, test_write_value=11)
    # write_tensor(dummy_tensor, 12, test_write_value=12)
    print("Tensor after: \n", dummy_tensor)
    print("Note how each of the 12 individual elements have been individually replaced.")
    print("If you uncomment the line to write spot 12, you'll see it overwrites spot 0.")


# This is code that runs through using the function add_activation_bit_flips() and should
# show results that are increasingly worse if you lower the odds (at default you should
# see ~20-30)
def test_adding_BitFlipLayer_layers(odds=10000000):
    import torchvision.models as models
    from torchvision import transforms as T
    from PIL import Image
    from class_labels import print_top_5

    mobilenet = models.mobilenet_v3_small(pretrained=True)
    mobilenet.eval()  # Put the model in inference mode

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    # Load two images, prepare in a batch
    img1 = Image.open("ILSVRC2017_test_00000004.JPEG")
    img2 = Image.open("ILSVRC2017_test_00000017.JPEG")
    img1_t = transform(img1)
    img2_t = transform(img2)
    batch_t = torch.stack((img1_t, img2_t), dim=0)  # Batch 2 images together

    bit_flip_init(mobilenet)    # Manually redo here just in case
    mobilenet = add_activation_bit_flips(mobilenet, odds)
    print(mobilenet)

    out = mobilenet(batch_t)  # out has shape [N, 1000] where N = batch size
    print_top_5(out)  # Print out the predictions
    print("Number of flips in activations: ", get_flips_in_activations())
    # You should know things are working by: viewing the output of print(mobilenet),
    # note that there should be BitFlipLayers after all of the conv2d, BatchNorm, Linear,
    # and other layers.  Finally, the print statement above should indicate there have
    # been >0 flips in the activations during the evaluation.
