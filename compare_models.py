import torch
import torchvision.models as models
import matplotlib.pyplot as plt
from bit_flipping import get_num_params
from get_model import get_model

# Accuracy values obtained from: https://pytorch.org/vision/stable/models.html
size_dict = {'alexnet': (get_num_params(get_model('alexnet'), True), 56.522, 79.066),
             'vgg11': (get_num_params(get_model('vgg11'), True), 69.02, 88.629),
             'vgg13': (get_num_params(get_model('vgg13'), True), 69.928, 89.246),
             'vgg16': (get_num_params(get_model('vgg16'), True), 71.592, 90.382),
             'vgg19': (get_num_params(get_model('vgg19'), True), 72.376, 90.876),
             'vgg11_bn': (get_num_params(get_model('vgg11_bn'), True), 70.37, 89.81),
             'vgg13_bn': (get_num_params(get_model('vgg13_bn'), True), 71.586, 90.374),
             'vgg16_bn': (get_num_params(get_model('vgg16_bn'), True), 73.36, 91.516),
             'vgg19_bn': (get_num_params(get_model('vgg19_bn'), True), 74.218, 91.842),
             'resnet18': (get_num_params(get_model('resnet18'), True), 69.758, 89.078),
             'resnet34': (get_num_params(get_model('resnet34'), True), 73.314, 91.42),
             'resnet50': (get_num_params(get_model('resnet50'), True), 76.13, 92.862),
             'resnet101': (get_num_params(get_model('resnet101'), True), 77.374, 93.546),
             'resnet152': (get_num_params(get_model('resnet152'), True), 78.312, 94.046),
             'squeezenet1_0': (get_num_params(get_model('squeezenet1_0'), True), 58.092, 80.42),
             'squeezenet1_1': (get_num_params(get_model('squeezenet1_1'), True), 58.178, 80.624),
             'densenet121': (get_num_params(get_model('densenet121'), True), 74.434, 91.972),
             'densenet169': (get_num_params(get_model('densenet169'), True), 75.6, 92.806),
             'densenet201': (get_num_params(get_model('densenet201'), True), 76.896, 93.37),
             'densenet161': (get_num_params(get_model('densenet161'), True), 77.138, 93.56),
             'inception_v3': (get_num_params(get_model('inception_v3'), True), 77.294, 93.45),
             'googlenet': (get_num_params(get_model('googlenet'), True), 69.778, 89.53),
             'shufflenet_v2_x1_0': (get_num_params(get_model('shufflenet_v2_x1_0'), True), 69.362, 88.316),
             'shufflenet_v2_x0_5': (get_num_params(get_model('shufflenet_v2_x0_5'), True), 60.552, 81.746),
             'mobilenet_v2': (get_num_params(get_model('mobilenet_v2'), True), 71.878, 90.286),
             'mobilenet_v3_large': (get_num_params(get_model('mobilenet_v3_large'), True), 74.042, 91.34),
             'mobilenet_v3_small': (get_num_params(get_model('mobilenet_v3_small'), True), 67.668, 87.402),
             'resnext50_32x4d': (get_num_params(get_model('resnext50_32x4d'), True), 77.618, 93.698),
             'resnext101_32x8d': (get_num_params(get_model('resnext101_32x8d'), True), 79.312, 94.526),
             'wide_resnet50_2': (get_num_params(get_model('wide_resnet50_2'), True), 78.468, 94.086),
             'wide_resnet101_2': (get_num_params(get_model('wide_resnet101_2'), True), 78.848, 94.284),
             'mnasnet1_0': (get_num_params(get_model('mnasnet1_0'), True), 73.456, 91.51),
             'mnasnet0_5': (get_num_params(get_model('mnasnet0_5'), True), 67.734, 87.49)
             }

# Sort the dictionary. Using 'item[1][2]' sorts by accuracy, 'item[1]' sorts by size.
size_dict = dict(sorted(size_dict.items(), key=lambda item: item[1][2]))

print(size_dict)
model_names = list(size_dict.keys())
model_data = list(size_dict.values())
model_sizes = [x[0] for x in model_data]
model_acc_top1 = [x[1] for x in model_data]
model_acc_top5 = [x[2] for x in model_data]

x_pos = range(len(model_names))

# To plot only the model size:
# plt.style.use('ggplot')
# x_pos = [i for i, _ in enumerate(model_names)]
# plt.bar(x_pos, model_sizes, color='blue')
# plt.title("Torch Model Comparison")
# plt.ylabel("Num Parameters")
# plt.xticks(x_pos, model_names, rotation='vertical')
# plt.tight_layout(pad=2.0)
# plt.show()

# To plot both model size and accuracy:
ax1 = plt.subplot(1,1,1)
w = 0.4
plt.xticks(x_pos, model_names, rotation='vertical')
pop = ax1.bar([x-w/2 for x in x_pos], model_sizes, width=w, color='b', align='center')
plt.ylabel("Num Parameters")
ax2 = ax1.twinx()
gdp = ax2.bar([x+w/2 for x in x_pos], model_acc_top1, width=w, color='#55c5d4', align='center')   #77cdd9
plt.ylabel("Top 1 Accuracy")
plt.title("Torch Model Comparison")
plt.tight_layout(pad=2.0)
plt.show()
