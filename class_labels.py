import torch


#################################################################################################
#     Reads in 'imagenet_classes.txt' and interfaces with it for viewing label predictions      #
#################################################################################################

# Read in the labels from 'imagenet_classes.txt'
labels = []
with open('imagenet_classes.txt') as f:
    for line in f:
        line = line.split(',')
        class_name = line[1].strip()    # Name of class, line[0] contains line number
        labels.append(class_name)
# print("imagenet_classes read in successfully.")


# Define function for printing out the results
def print_top_5(output):
    images_per_batch, num_classes = output.shape
    percentages = torch.nn.functional.softmax(output, dim=1) * 100
    _, indices = torch.sort(output, descending=True)

    for image in range(images_per_batch):
        for i in range(5):
            idx = indices[image][i]     # Index of top class
            print('{:<1d}:{:>8.2f}%  {:<8s}'.format(i+1, percentages[image][idx].item(), labels[idx]))
        print()     # Newline
