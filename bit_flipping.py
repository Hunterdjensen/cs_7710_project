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
total_param_count = 0
cumulative_param_count = None  # List of the param counts per layer, to use w/ Numpy.digitize()
names = None  # List of layer "names"
sizes = None  # List of sizes for each layer


# Analyzes the model to get the number of parameters total and per layer
# Pass in the model and the variable name of the model as a string (aka 'mobilenet')
def bit_flip_init(model):
    global init_flag, total_param_count, cumulative_param_count, names, sizes
    init_flag = True
    total_param_count = 0
    cumulative_param_count = []
    names = []
    sizes = []

    for name, param in model.named_parameters():
        num_elements = torch.numel(param)
        total_param_count += num_elements

        cumulative_param_count.append(total_param_count)
        names.append(name)
        sizes.append(list(param.shape))

        print(name, list(param.shape), cumulative_param_count[-1])
    print("Total params: ", total_param_count)
    print("Total layers: ", len(names))
    # print(cumulative_param_count)
    # print("Digitize: ", np.digitize(444, cumulative_param_count))


# FIXME: Sizes not needed now?
def get_tensor(name, model):
    print("Getting string for name: ", name)    # features.6.block.3.0.weight
    # split = name.split('.')     # Since names looks like: "classifier.3.bias"
    # out_str = 'layer_tensor = getattr(' + model_name + ', ' + split[0] + ')'
    # out_str = 'layer_tensor = ' + model_name + '.'
    # print("Model is called: ", model_name)
    # Do regex stuff:
    # search = re.findall('([a-zA-Z]+[.\d+]*)', name) # Split up into groups of names+numbers
    # num_groups = len(search)   # ['features.6.', 'block.3.0.', 'weight']
    # attr = model
    # for match in search:
    #     split = match.split('.')
    #     attr = getattr(attr, split)
    tensor = model
    split = name.split('.')
    print(split)
    for attribute in split:
        if attribute.isnumeric():   # Attribute is a number
            model = model[int(attribute)]
        else:   # Attribute is a word
            model = getattr(model, attribute)
    return model


# Flips n bits randomly inside the model
def flip_n_bits(n, model, ):
    global init_flag, total_param_count, cumulative_param_count, names, sizes
    if init_flag is False:
        bit_flip_init(model)    # FIXME - remove model_name
        # exit("Please call bit_flipping.bit_flip_init() before flip_n_bits()")

    random_param_numbers = np.random.randint(low=0, high=total_param_count, size=(n,))
    print(random_param_numbers)
    for rand_param in random_param_numbers:
        layer_num = np.digitize(rand_param, cumulative_param_count)
        print("Flipping a bit in parameter #%d in layer %d" % (rand_param, layer_num))
        param = get_tensor(names[layer_num], model)
        print("shape: ", param.shape)
        # with torch.no_grad():
        #     layer_name, idx, layer_type = names[layer_num].split('.')     # Since names looks like: "classifier.3.bias"
        #     if layer_type == 'weight':
        #         # value = getattr(model, layer_name)[int(idx)].weight[]
        #         pass
        #     elif layer_type == 'bias':
        #         pass
        #     else:
        #         exit("Error in bit_flipping.flip_n_bits(), unknown layer type: ", layer_type)
