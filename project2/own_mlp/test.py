import torch
import random
import math

from module import *
from loss import *
from linear import *
from activation import *
from sequential import *

# just to be sure
torch.set_grad_enabled(False);

# set seeds
torch.manual_seed(42)
random.seed(42)

# generate 2d points in [0,1] squared, targets are 0 if point inside the circle of squared radius 1/2pi and 1 outside.
# return coordinates and target tensors, both of size Nx2, plus classes tensor of size Nx1
def generate_points(nb):
    inputs = torch.empty((nb, 2))
    targets = torch.empty((nb, 2))
    classes = torch.empty((nb, 1))
    for i in range(nb):
        x = random.uniform(0,1) - 0.5
        y = random.uniform(0,1) - 0.5
        t = int(2 * math.pi * (pow(x, 2) + pow(y, 2)) < 1)
        inputs[i] = torch.tensor([x + 0.5, y + 0.5])
        targets[i,t] = 1
        targets[i,1-t] = 0
        classes[i] = t
    return inputs, targets, classes

# accuracy for classes prediction
def accuracy(model, inputs, classes):
    nb_samples = inputs.shape[0]
    pred = model.predict(inputs)
    _, pred_classes = pred.max(1)
    nb_errors = (pred_classes - classes[:,0]).type(torch.BoolTensor).sum().item()
    return (nb_samples - nb_errors) / nb_samples

# method for testing model on 2d radius classification
def test():
    
    # model creation
    model = Sequential(
        Linear(2, 25), 
        Relu(),
        Linear(25, 25), 
        Relu(),
        Linear(25, 25), 
        Relu(),
        Linear(25, 2),
        loss=MSELoss(), 
        out_size = 2)
    
    print("Model:")
    print("---------------------")
    model.describe()
    print("---------------------")
    
    # generating points
    train_inputs, train_targets, train_classes = generate_points(1000)
    test_inputs, test_targets, test_classes = generate_points(1000)
    
    # train model
    model.train(train_inputs, train_targets, epochs=25, batch_size=10, verbose=True)
    
    # compute accuracy
    train_acc = accuracy(model, train_inputs, train_classes)
    test_acc = accuracy(model, test_inputs, test_classes)
    
    print("Train accuracy: {} | Test accuracy: {}".format(train_acc, test_acc))
    
test()