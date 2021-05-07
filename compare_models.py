import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from bit_flipping import get_num_params
from get_model import get_model

# # Accuracy values obtained from: https://pytorch.org/vision/stable/models.html
# size_dict = {'alexnet': (get_num_params(get_model('alexnet'), True), 56.522, 79.066),
#              'vgg11': (get_num_params(get_model('vgg11'), True), 69.02, 88.629),
#              'vgg13': (get_num_params(get_model('vgg13'), True), 69.928, 89.246),
#              'vgg16': (get_num_params(get_model('vgg16'), True), 71.592, 90.382),
#              'vgg19': (get_num_params(get_model('vgg19'), True), 72.376, 90.876),
#              'vgg11_bn': (get_num_params(get_model('vgg11_bn'), True), 70.37, 89.81),
#              'vgg13_bn': (get_num_params(get_model('vgg13_bn'), True), 71.586, 90.374),
#              'vgg16_bn': (get_num_params(get_model('vgg16_bn'), True), 73.36, 91.516),
#              'vgg19_bn': (get_num_params(get_model('vgg19_bn'), True), 74.218, 91.842),
#              'resnet18': (get_num_params(get_model('resnet18'), True), 69.758, 89.078),
#              'resnet34': (get_num_params(get_model('resnet34'), True), 73.314, 91.42),
#              'resnet50': (get_num_params(get_model('resnet50'), True), 76.13, 92.862),
#              'resnet101': (get_num_params(get_model('resnet101'), True), 77.374, 93.546),
#              'resnet152': (get_num_params(get_model('resnet152'), True), 78.312, 94.046),
#              'squeezenet1_0': (get_num_params(get_model('squeezenet1_0'), True), 58.092, 80.42),
#              'squeezenet1_1': (get_num_params(get_model('squeezenet1_1'), True), 58.178, 80.624),
#              'densenet121': (get_num_params(get_model('densenet121'), True), 74.434, 91.972),
#              'densenet169': (get_num_params(get_model('densenet169'), True), 75.6, 92.806),
#              'densenet201': (get_num_params(get_model('densenet201'), True), 76.896, 93.37),
#              'densenet161': (get_num_params(get_model('densenet161'), True), 77.138, 93.56),
#              'inception_v3': (get_num_params(get_model('inception_v3'), True), 77.294, 93.45),
#              'googlenet': (get_num_params(get_model('googlenet'), True), 69.778, 89.53),
#              'shufflenet_v2_x1_0': (get_num_params(get_model('shufflenet_v2_x1_0'), True), 69.362, 88.316),
#              'shufflenet_v2_x0_5': (get_num_params(get_model('shufflenet_v2_x0_5'), True), 60.552, 81.746),
#              'mobilenet_v2': (get_num_params(get_model('mobilenet_v2'), True), 71.878, 90.286),
#              'mobilenet_v3_large': (get_num_params(get_model('mobilenet_v3_large'), True), 74.042, 91.34),
#              'mobilenet_v3_small': (get_num_params(get_model('mobilenet_v3_small'), True), 67.668, 87.402),
#              'resnext50_32x4d': (get_num_params(get_model('resnext50_32x4d'), True), 77.618, 93.698),
#              'resnext101_32x8d': (get_num_params(get_model('resnext101_32x8d'), True), 79.312, 94.526),
#              'wide_resnet50_2': (get_num_params(get_model('wide_resnet50_2'), True), 78.468, 94.086),
#              'wide_resnet101_2': (get_num_params(get_model('wide_resnet101_2'), True), 78.848, 94.284),
#              'mnasnet1_0': (get_num_params(get_model('mnasnet1_0'), True), 73.456, 91.51),
#              'mnasnet0_5': (get_num_params(get_model('mnasnet0_5'), True), 67.734, 87.49)
#              }
#
# # Sort the dictionary. Using 'item[1][2]' sorts by accuracy, 'item[1]' sorts by size.
# size_dict = dict(sorted(size_dict.items(), key=lambda item: item[1][2]))
#
# print(size_dict)
# model_names = list(size_dict.keys())
# model_data = list(size_dict.values())
# model_sizes = [x[0] for x in model_data]
# model_acc_top1 = [x[1] for x in model_data]
# model_acc_top5 = [x[2] for x in model_data]
#
# x_pos = range(len(model_names))
#
# # To plot only the model size:
# # plt.style.use('ggplot')
# # x_pos = [i for i, _ in enumerate(model_names)]
# # plt.bar(x_pos, model_sizes, color='blue')
# # plt.title("Torch Model Comparison")
# # plt.ylabel("Num Parameters")
# # plt.xticks(x_pos, model_names, rotation='vertical')
# # plt.tight_layout(pad=2.0)
# # plt.show()
#
# # To plot both model size and accuracy:
# fig = plt.figure(figsize=(10, 6))
# ax1 = plt.subplot(1,1,1)
# w = 0.4
# plt.xticks(x_pos, model_names, rotation='vertical')
# pop = ax1.bar([x-w/2 for x in x_pos], model_sizes, width=w, color='b', align='center')
# plt.ylabel("Num Parameters")
# ax2 = ax1.twinx()
# gdp = ax2.bar([x+w/2 for x in x_pos], model_acc_top1, width=w, color='#55c5d4', align='center')   #77cdd9
# plt.ylabel("Top 1 Accuracy")
# plt.title("Torch Model Comparison")
# plt.tight_layout(pad=2.0)
# plt.show()



# BER_values = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
# # Numbers for 128 batches:
# # Simple (hard voting):
# # baseline_acc = [80.08, 79.297, 78.906, 74.707, 39.746, 0.0977]
# # ensemble_acc = [80.18, 79.1, 80.762, 76.66, 63.77, 0.293]
# # Sum all:
# # baseline_acc = [79.88, 79.59, 76.86, 68.36, 17.58, 0]
# # ensemble_acc = [78.81, 78.61, 78.61, 72.36, 24.805, 0]
#
# # Numbers for 1024 batches:
# # Simple (hard voting):
# baseline_acc = [79.541, 79.993, 79.199, 74.683, 42.236, 0.134]
# ensemble_acc = [80.042, 79.883, 80.078, 78.992, 61.975, 0.586]
# # Sum all (soft voting):
# baseline_acc_soft = [80.005, 79.919, 78.125, 69.751, 21.643, 0.000]
# ensemble_acc_soft = [80.139, 79.761, 79.321, 71.509, 26.367, 0.000]
# # Only 2 models:
# dense_inception = [76.367, 77.173, 76.489, 74.146, 55.908, 3.613]
# dense_inception_soft = [75.757, 75.806, 76.831, 73.193, 47.485, 0.439]
# x_pos = range(len(BER_values))
#
# # # New plot with three bars:
# # fig = plt.figure(figsize=(7.2, 5))
# # ax = plt.subplot(1,1,1)
# # w = 0.2
# # ax.bar([x-w for x in x_pos], baseline_acc, width=w, color='#1c4b99', align='center', label='baseline')
# # ax.bar([x for x in x_pos], ensemble_acc, width=w, color='#c80815', align='center', label='ensemble (hard voting)')
# # ax.bar([x+w for x in x_pos], ensemble_acc_soft, width=w, color='silver', align='center', label='ensemble (soft voting)')
# # ax.legend(loc="upper right")
# # plt.xticks(x_pos, BER_values)
# # plt.ylabel("Accuracy")
# # plt.xlabel("Bit Error Rate (weights and activations)")
# # plt.title("Baseline vs Ensemble (ResNext-101, DenseNet-161, Inception V3)")
# # plt.tight_layout(pad=2.0)
# # plt.show()
#
# # Original plot with just two bars:
# fig = plt.figure(figsize=(7.2, 5))
# ax = plt.subplot(1,1,1)
# w = 0.2
# ax.bar([x-w for x in x_pos], baseline_acc, width=w, color='#1c4b99', align='center', label='baseline')
# ax.bar([x for x in x_pos], dense_inception, width=w, color='green', align='center', label='ensemble (hard voting)')
# ax.bar([x+w for x in x_pos], dense_inception_soft, width=w, color='limegreen', align='center', label='ensemble (soft voting)')
# ax.legend(loc="upper right")
# plt.xticks(x_pos, BER_values)
# plt.ylabel("Accuracy")
# plt.xlabel("Bit Error Rate (weights and activations)")
# plt.title("Baseline (ResNext-101) vs 2-Model Ensemble (DenseNet-161, Inception V3)")
# plt.tight_layout(pad=2.0)
# plt.show()




# # Data using a different ensemble (WideResNet50, ResNext50, DenseNet161)
# BER_values = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
# x_pos = range(len(BER_values))
# # Numbers for 1024 batches:
# # Simple (hard voting):
# baseline_acc = [77.783, 78.906, 77.783, 75.562, 47.632, 0.513]
# ensemble_acc = [80.347, 79.297, 79.883, 79.395, 64.819, 0.806]
# # Sum all (soft voting):
# baseline_acc_soft = [78.296, 78.784, 77.979, 71.362, 29.468, 0.0]
# ensemble_acc_soft = [80.420, 81.982, 80.396, 73.145, 27.246, 0.0]
#
# # Figure with 3 bars:
# fig = plt.figure(figsize=(7.2, 5))
# ax = plt.subplot(1,1,1)
# w = 0.2
# ax.bar([x-w for x in x_pos], baseline_acc, width=w, color='#1c4b99', align='center', label='baseline')
# ax.bar([x for x in x_pos], ensemble_acc, width=w, color='#c80815', align='center', label='ensemble (hard voting)')
# ax.bar([x+w for x in x_pos], ensemble_acc_soft, width=w, color='silver', align='center', label='ensemble (soft voting)')
# ax.legend(loc="upper right")
# plt.xticks(x_pos, BER_values)
# plt.ylabel("Accuracy")
# plt.xlabel("Bit Error Rate (weights and activations)")
# plt.title("Baseline vs Ensemble (WideResNet-50, ResNext-50, DenseNet-161)")
# plt.tight_layout(pad=2.0)
# plt.show()
#
# # Original plot with just two bars:
# # ax = plt.subplot(1,1,1)
# # w = 0.4
# # ax.bar([x-w/2 for x in x_pos], baseline_acc, width=w, color='#1c4b99', align='center', label='baseline')
# # ax.bar([x+w/2 for x in x_pos], ensemble_acc_soft, width=w, color='#c80815', align='center', label='ensemble')
# # ax.legend(loc="upper right")
# # plt.xticks(x_pos, BER_values)
# # plt.ylabel("Accuracy")
# # plt.xlabel("Bit Error Rate (weights and activations)")
# # plt.title("Baseline vs 3-Model Ensemble (hard voting)")
# # plt.tight_layout(pad=2.0)
# # plt.show()




# distortion_severity = [0, 1, 2, 3, 4, 5]
# # Fog:
# # baseline_acc = [80.493, 80.029, 80.371, 81.006, 80.237, 80.444]
# # ensemble_acc = [80.151, 80.579, 80.481, 79.663, 80.017, 81.043]
# # Defocus blur:
# # baseline_acc = [80.164, 79.553, 80.42, 79.517, 80.322, 79.663]
# # ensemble_acc = [80.408, 80.09, 79.93, 79.858, 80.090, 79.26]
# # Frost: (8)
# frost_baseline_acc = [80.200, 68.140, 52.612, 40.723, 39.429, 33.887]
# frost_ensemble_acc = [80.200, 70.435, 57.129, 45.557, 43.555, 34.888]
# # Snow: (7)
# snow_baseline_acc = [80.200, 54.185, 42.163, 48.047, 35.913, 24.854]
# snow_ensemble_acc = [80.200, 65.161, 45.557, 50.464, 38.721, 29.663]
#
# baseline_acc = frost_baseline_acc + [0] + snow_baseline_acc
# ensemble_acc = frost_ensemble_acc + [0] + snow_ensemble_acc
# distortion_severity = distortion_severity + [''] + distortion_severity
# print(distortion_severity)
# x_pos = range(len(distortion_severity))
#
# fig = plt.figure(figsize=(7.2, 5))
# ax = plt.subplot(1,1,1)
# w = 0.4
# ax.bar([x-w/2 for x in x_pos], baseline_acc, width=w, color='#1c4b99', align='center', label='baseline')
# ax.bar([x+w/2 for x in x_pos], ensemble_acc, width=w, color='#c80815', align='center', label='ensemble')
# ax.legend(loc="upper right")
# plt.xticks(x_pos, distortion_severity)
# plt.ylabel("Accuracy")
# plt.xlabel("Distortion Severity (Frost)                               Distortion Severity (Snow)")
# plt.title("Baseline vs 3-Model Ensemble on Distorted Images")
# plt.tight_layout(pad=2.0)
# plt.show()





# Adversarial attacks:
adversarial_attack = ['ImageNet-A', 'Fast Gradient', 'Projected Gradient']
baseline_acc = [10.04, 23.682, 7.129]
ensemble_acc = [8.54, 40.723, 44.495]
ensemble_acc_soft = [6.04, 45.142, 7.153]
x_pos = range(len(adversarial_attack))

ax = plt.subplot(1,1,1)
w = 0.2
ax.bar([x-w for x in x_pos], baseline_acc, width=w, color='#1c4b99', align='center', label='baseline')
ax.bar([x for x in x_pos], ensemble_acc, width=w, color='#c80815', align='center', label='ensemble (hard voting)')
ax.bar([x+w for x in x_pos], ensemble_acc_soft, width=w, color='silver', align='center', label='ensemble (soft voting)')
ax.legend(loc="upper left")
plt.xticks(x_pos, adversarial_attack)
plt.ylabel("Accuracy")
plt.title("Adversarial Attacks on Baseline and 3-Model Ensemble")
plt.tight_layout(pad=2.0)
plt.show()
