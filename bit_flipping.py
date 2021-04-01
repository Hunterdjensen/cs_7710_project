import torch
import dtype_conversions
from dtype_conversions import float_to_bin
from dtype_conversions import bin_to_float

#################################################################################################
#               Takes in models and randomly flips bits in their parameters                     #
#################################################################################################

init_flag = False
total_param_count = 0
num_layers = 0


# Analyzes the model to get the number of parameters total and per layer
def bit_flip_init(model):
    global init_flag, total_param_count, num_layers
    init_flag = True
    total_param_count = 0
    num_layers = 0

    for name, param in model.named_parameters():
        num_elements = torch.numel(param)
        total_param_count += num_elements
        num_layers += 1
        print(name, " ", param.shape)
    print("Total params: ", total_param_count)
