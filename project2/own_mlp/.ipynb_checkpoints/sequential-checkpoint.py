import torch
import random
import math

from module import *
from loss import *
from linear import *
from activation import *


# A sequence of modules
class Sequential(Module):
    def __init__(self, *modules, loss, out_size):
        super(Sequential, self).__init__()
        self.layers = modules
        self.loss = loss
        self.out_size = out_size
        
    def forward(self, *input):
        x = input[0]
        for m in self.layers: # not forwarding through the loss
            x = m.forward(x)
        return x
    
    def train(self, inputs, targets, epochs, batch_size = 10, verbose=True):
        
        eta = 1e-1 / batch_size 
                
        for e in range(epochs):
            
            sum_loss = 0
            # iterate over each batch and update weights
            for input_batch, target_batch in zip(inputs.split(batch_size), targets.split(batch_size)):
                
                # computing predicted values and loss
                predicted = self.forward(input_batch) 
                sum_loss = sum_loss + self.loss.forward(predicted, target_batch) # LOG ???

                # computing derivative of the loss between predicted and target
                x = self.loss.backward(predicted, target_batch)

                # backpropagate the loss through the net, adding the gradients in linear modules
                for m in reversed(self.layers):
                    x = m.backward(x)
            
                # update weights in linear modules
                for m in self.layers:
                    if isinstance(m, Linear):
                        m.update_param(eta)
                        
            if verbose:        
                print("Epoch {} | Loss {:.2f}".format(e, sum_loss))
    
    def print_param(self):
        for m in self.layers:
            if isinstance(m, Linear):
                params = m.param()
                print("*** ", m.to_string(), " ***")
                print(params[0])
                print(params[1])
                print()
            
    def describe(self):
        for m in self.layers:
            print(m.to_string())
        print(self.loss.to_string())
            
    # predictions using current weights in the modules
    # input : tensor, samples aligned along first dimension
    def predict(self, inputs):
        return self.forward(inputs)