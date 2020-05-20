import torch
import random
import math

# Deep learning library developped in the scope of this project.
from dl_lib import *


torch.set_grad_enabled(False);

# set seeds
SEED = 1
torch.manual_seed(SEED)
random.seed(SEED)

# generate 2d points in [0,1] squared, targets are 1 if point inside the disk centered in (0.5,0.5) of squared radius 1/2pi and 0 outside.
# return coordinates and one-hot target tensors, both of size Nx2, plus classes tensor of size Nx1
def generate_points(nb):
    """Generates a dataset of size <nb> following the intruction in the miniproject proposal."""
    inputs = torch.empty((nb, 2))
    targets = torch.empty((nb, 2))
    classes = torch.empty((nb), dtype=torch.long)
    for i in range(nb):
        x = random.uniform(0,1) - 0.5
        y = random.uniform(0,1) - 0.5
        t = int(2 * math.pi * (pow(x, 2) + pow(y, 2)) < 1)
        inputs[i] = torch.tensor([x + 0.5, y + 0.5])
        targets[i,t] = 1
        targets[i,1-t] = 0
        classes[i] = t
    return inputs, targets, classes


def accuracy(model, inputs, classes):
    """Compute the accuray of the <model> prediction on the dataset <input>, given the corresponding true <classes>."""
    nb_samples = inputs.shape[0]
    pred = model.predict(inputs)
    _, pred_classes = pred.max(1)
    nb_errors = (pred_classes - classes).type(torch.BoolTensor).sum().item()
    return (nb_samples - nb_errors) / nb_samples


if __name__ == '__main__':
    
    # Global training parameters
    EPOCHS = 25
    BACTH_SIZE = 10
    LEARNING_RATE = 1e-1

    # model creation
    model = Sequential(
        Linear(2, 25), 
        Relu(),
        Linear(25, 25), 
        Relu(),
        Linear(25, 25), 
        Relu(),
        Linear(25, 2))
    
    print("Model:")
    print("---------------------")
    model.describe()
    print("---------------------")
    
    # generating points
    train_inputs, train_targets, train_classes = generate_points(1000)
    test_inputs, test_targets, test_classes = generate_points(1000)
    
    # Loss
    criterion = MSELoss()
    
    # Optimizer
    optimizer = SGD(model, LEARNING_RATE)
    
    model.training_mode(True)
    for e in range(EPOCHS):
            
        sum_loss = 0
        # iterate over each batch and update weights (replace train_targets with train_classes when using CrossEntropyLoss)
        for input_batch, target_batch in zip(train_inputs.split(BACTH_SIZE), train_targets.split(BACTH_SIZE)):
            
            # computing predicted values and loss
            predicted = model.forward(input_batch)
            loss = criterion.forward(predicted, target_batch)
            
            # averaging loss over the current epoch (for consistent logging independetly of BATCH_SIZE)
            sum_loss = sum_loss + (BACTH_SIZE / train_inputs.shape[0]) * loss

            # reinitializing gradients
            optimizer.zero_grad()
            
            # computing derivative of the loss between predicted and target
            gradwrtoutput = criterion.backward(predicted, target_batch)

            # backpropagate the loss through the net, adding the gradients in linear modules
            gradwrtinput = model.backward(gradwrtoutput)

            # update weights in linear modules
            optimizer.step()

      
        print("Epoch {} | Loss {:.3f}".format(e, sum_loss))
    
    # compute accuracy
    train_acc = accuracy(model, train_inputs, train_classes)
    test_acc = accuracy(model, test_inputs, test_classes)
    
    print("Train accuracy: {} | Test accuracy: {}".format(train_acc, test_acc))