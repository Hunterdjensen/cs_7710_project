from run_pytorch import run
from datetime import datetime
import torch

from numba import jit, cuda
import numpy as np


# output_dir = 'Results/'
# output_filename = 'ResNext_DenseNet_Inception_vs_Baseline.txt'
# outfile_path = output_dir + output_filename
# #outfile = open(outfile_path, 'w+')
#
# now = datetime.now()  # Get current time
#
# with open(outfile_path, 'w') as f:
#     print("Running simulation at time: " now.strftime("%m/%d/%Y %H:%M:%S") + "\n", file=outfile)
# #print("Running simulation at time: " + now.strftime("%m/%d/%Y %H:%M:%S") + "\n", file=outfile)
# #outfile.close()
#
#
# def print(string):
#     f = open(outfile_path, 'a')
#     print(string, file=f)
#     f.close()


#################################################################################################
#                                       Model Parameters:                                       #
#################################################################################################
models = [['resnext101_32x8d', 'densenet161', 'inception_v3'],  # Ensemble of 3 networks
          ['resnext101_32x8d', 'resnext101_32x8d']]  # Baseline

print_out = False   # Don't need to see print-out for every iteration
corrupt_img = False  # For this test we'll exclude it, in the future try a test with severity iterating 0-5

stuck_at_faults = 0  # Don't include for this test
BER_levels = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
heuristics = ['simple', 'sum all']

num_batches = 128  # The more the better, but 128 batches should cover most of our 1000 validation images

print("Test parameters:\nModels: " + str(models))
print("CORRUPT_IMG = " + str(corrupt_img))
print("Bit Error Rate values: " + str(BER_levels))
print("Voting Heuristics: " + str(heuristics))
print("For %d batches of size 8\n" % num_batches)

print("Cuda is available: ", torch.cuda.is_available())    # See if cuda is an option

#################################################################################################
#                                         Run through:                                          #
#################################################################################################
@jit(target='cuda')
def funcGPU():
    for voting_heuristic in heuristics:
        print(str(voting_heuristic) + " voting heuristic:")
        for model in models:
            print("\t" + str(model) + ":")
            for bit_error_rate in BER_levels:
                results = run(MODELS=model,
                              PRINT_OUT=print_out,
                              CORRUPT_IMG=corrupt_img,
                              stuck_at_faults=stuck_at_faults,
                              weights_BER=bit_error_rate,
                              activation_BER=bit_error_rate,
                              num_batches=num_batches,
                              voting_heuristic=voting_heuristic,
                              )
                accuracy, weight_flips, activation_flips = results
                print("\t\t" + str(bit_error_rate) + ": " + str(accuracy) + ", weight_flips: "
                      + str(list(weight_flips.values())) + " act_flips: " + str(list(activation_flips.values())))
                print(bit_error_rate, "acc:", accuracy, str(weight_flips))
    print("Completed successfully.")
