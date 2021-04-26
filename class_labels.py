import torch
import os

#################################################################################################
#     Reads in 'imagenet_classes.txt' and interfaces with it for viewing label predictions.     #
#     Also reads 'val.txt' and creates a dictionary so that image file names can map to         #
#     their ground-truth labels.                                                                #
#################################################################################################
here = os.path.dirname(os.path.abspath(__file__))
#print(here)
filename = os.path.join(here, 'imagenet_classes.txt')

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
    _, indices = torch.sort(output, descending=True)
    num_correct = 0

    for image in range(images_per_batch):
        prediction = indices[image][0]  # [0] for the largest prediction
        if prediction == correct_labels[image]:
            num_correct += 1

    return num_correct


# Create predictions based on the votes from 1 or more models.
# Currently, the only mode supported is 'simple', which uses hard voting between the models
def vote(out, heuristic):
    if heuristic == 'simple':
        # Each model picks their Top 1, majority wins, tie-breakers go to the first model
        M, N, C = out.shape     # M = num models, N = batch size, C = num classes
        _, indices = torch.sort(out, descending=True)
        predictions = indices[:,:,0]    # Shape [M, N] (no longer a C because we only picked the top predictions)
        # print(predictions)
        if M <= 2:  # Voting between two or fewer models
            return predictions[0,:]     # Whether they agree or not, the first will always win
        if M == 3:
            # If the 2nd and 3rd don't agree, then the 1st model always wins
            bottom_two = torch.eq(predictions[1,:], predictions[2,:])
            correct_model = torch.where(bottom_two, 1, 0)    # Aka where 2nd and 3rd agree, use 2nd, else use 1st model
            return predictions.gather(dim=0, index=correct_model.unsqueeze(0))    # Index predictions with correct_model
    # TODO: Add more voting mechanisms here
    if heuristic == 'sum all':
        M, N, C = out.shape     # M = num models, N = batch size, C = num classes
        reduce = torch.sum(out, 0)
        _, indices = torch.sort(reduce, descending=True)
        predictions = indices[:,0]    # Shape [M, N] (could change the hardcoding)
        return predictions
    if heuristic == 'debug': ##playground heuristic
        M, N, C = out.shape     # M = num models, N = batch size, C = num classes
        print(out.shape)
        reduce = torch.sum(out, 0)
        print(reduce.shape)
        _, indices = torch.sort(reduce, descending=True)
        print('Reduced out: ', reduce)
        print('indices of red: ', indices)

        predictions = indices[:,0]    # Shape [M, N] (could change the hardcoding)
        print('pred print: ', predictions)

        return predictions
        ##Ideas:
        #Do all estimates that are > 0, sum them, take max
        #
    # elif heuristic == '':
