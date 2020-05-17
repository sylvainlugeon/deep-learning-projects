import torch
import random
import math

from module import *
from loss import *
from linear import *
from activation import *


# A sequence of modules
class Sequential(Updatable):
    def __init__(self, *modules, loss=None):
        super(Sequential, self).__init__()
        self.layers = modules
        self.loss = loss
        
    def forward(self, input):
        x = input
        for m in self.layers: # not forwarding through the loss
            x = m.forward(x)
        return x
    
    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        for m in reversed(self.layers): # not backwarding through the loss
            x = m.backward(x)
        return x
    
    def update_param(self, step_size):
        for m in self.layers:
            if isinstance(m, Updatable):
                m.update_param(step_size)
    
    
    def train(self, input, target, epochs, batch_size = 10, verbose=True):
        
        eta = 1e-1 
                
        for e in range(epochs):
            
            sum_loss = 0
            # iterate over each batch and update weights
            for input_batch, target_batch in zip(input.split(batch_size), target.split(batch_size)):
                
                # computing predicted values and loss
                predicted = self.forward(input_batch) 
                sum_loss = sum_loss + self.loss.loss(predicted, target_batch)

                # computing derivative of the loss between predicted and target
                x = self.loss.dloss(predicted, target_batch)

                # backpropagate the loss through the net, adding the gradients in linear modules
                self.backward(x)
            
                # update weights in linear modules
                self.update_param(eta)
                        
            if verbose:        
                print("Epoch {} | Loss {:.2f}".format(e, sum_loss))
    
    def param(self):
        for m in self.layers:
            if isinstance(m, Linear):
                params = m.param()
                print("*** ", m.to_string(), " ***")
                print(params[0])
                print(params[1])
                print()
            
    def describe(self):
        print(self.to_string())
        if self.loss is not None:
            print(self.loss.to_string())
        
    def to_string(self):
        s = ''
        for m in self.layers:
            s = s + m.to_string() + '\n'
        return s
            
    # predictions using current weights in the modules
    # input : tensor, samples aligned along first dimension
    def predict(self, input):
        return self.forward(input)