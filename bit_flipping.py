import torch
import numpy as np
import re
import dtype_conversions
from dtype_conversions import float_to_bin
from dtype_conversions import bin_to_float

#################################################################################################
#               Takes in models and randomly flips bits in their parameters                     #
#################################################################################################

init_flag = False
total_param_count = 0           # Total number of parameters in the network
cumulative_param_count = None   # List of the param counts per layer, to use w/ Numpy.digitize()
names = None                    # List of layer "names" (i.e: 'features.12.block.2.fc1.weight')


# Analyzes the model (i.e mobilenet) to get the number of parameters total and per layer
# Doesn't need to be called by user, flip_n_bits() will call it
def bit_flip_init(model):
    global init_flag, total_param_count, cumulative_param_count, names
    init_flag = True
    total_param_count = 0
    cumulative_param_count = []
    names = []

    for name, param in model.named_parameters():
        num_elements = torch.numel(param)
        total_param_count += num_elements
        cumulative_param_count.append(total_param_count)
        names.append(name)
        # print(name, list(param.shape), cumulative_param_count[-1])
    print("Total params: ", total_param_count)
    # print("Total layers: ", len(names))
    # print(cumulative_param_count)
    # print("Digitize: ", np.digitize(444, cumulative_param_count))


# Get the tensor corresponding to the layer given by the 'name' parameter.
# For example, the layer 'features.11.block.2.fc1.weight' contains the weight
# parameters for this layer of the network, a tensor of shape torch.Size([144, 576, 1, 1])
def get_layer_tensor(name, model):
    tensor = model
    print(name)
    split = name.split('.')  # 'features.11.block.2.fc1.weight' -> ['features', '11', 'block', '2', 'fc1', 'weight']
    for attribute in split:
        if attribute.isnumeric():   # Attribute is a number
            tensor = tensor[int(attribute)]
        else:                       # Attribute is a word
            tensor = getattr(tensor, attribute)
    return tensor


# Writes to the kth value of a layer_tensor.  It does this by creating a 1-D view
# of the tensor and indexing into its kth member to write to it.
def write_tensor(layer_tensor, k, test_write_value=None):
    num_elements = torch.numel(layer_tensor)
    view = layer_tensor.view(num_elements)  # Create a 1D view
    k = k % num_elements    # Use modulus in case n > num_elements
    with torch.no_grad():
        if test_write_value is not None:
            view[k] = test_write_value
        else:
            view[k] = view[k] + 4


# Flips n bits randomly inside the model
def flip_n_bits(n, model, ):
    global init_flag, total_param_count, cumulative_param_count, names
    if init_flag is False:
        bit_flip_init(model)

    random_param_numbers = np.random.randint(low=0, high=total_param_count, size=(n,))
    print(random_param_numbers)
    for rand_param in random_param_numbers:
        layer_num = np.digitize(rand_param, cumulative_param_count)
        print("Flipping a bit in parameter #%d in layer %d" % (rand_param, layer_num))
        layer_tensor = get_layer_tensor(names[layer_num], model)
        print("shape: ", layer_tensor.shape)
        write_tensor(layer_tensor, rand_param)
