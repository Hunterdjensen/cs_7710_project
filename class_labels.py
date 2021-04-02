import torch


#################################################################################################
#     Reads in 'imagenet_classes.txt' and interfaces with it for viewing label predictions.     #
#     Also reads 'val.txt' and creates a dictionary so that image file names can map to         #
#     their ground-truth labels.                                                                #
#################################################################################################

# Read in the labels from 'imagenet_classes.txt'
labels = []
with open('imagenet_classes.txt') as f:
    for line in f:
        line = line.split(',')
        class_name = line[1].strip()    # Name of class, line[0] contains line number
        labels.append(class_name)

val_labels = {}
with open('val.txt') as f:
    for line in f:
        filename, label = line.split()
        val_labels[filename] = int(label)


# Define function for printing out the results
def print_top_5(output, correct_labels=None):
    images_per_batch, num_classes = output.shape
    percentages = torch.nn.functional.softmax(output, dim=1) * 100
    _, indices = torch.sort(output, descending=True)

    for image in range(images_per_batch):
        if correct_labels is not None:
            print("Correct label is: ", labels[correct_labels[image]])

        for i in range(5):
            idx = indices[image][i]     # Index of top class
            print('{:<1d}:{:>8.2f}%  {:<8s}'.format(i+1, percentages[image][idx].item(), labels[idx]))
        print()     # Newline


# Access val_labels
def get_label(file):
    return val_labels[file]


# Compare predictions to correct labels, and return the number correct
def get_num_correct(output, correct_labels):
    images_per_batch, num_classes = output.shape
    percentages = torch.nn.functional.softmax(output, dim=1) * 100
    _, indices = torch.sort(output, descending=True)
    num_correct = 0

    for image in range(images_per_batch):
        prediction = indices[image][0]  # [0] for the largest prediction
        if prediction == correct_labels[image]:
            num_correct += 1

    return num_correct
