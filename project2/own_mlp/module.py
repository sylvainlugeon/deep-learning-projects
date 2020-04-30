import torch
import random
import math

# Mother class of all modules
class Module(object):
    
    # input : input to be passed into the layer
    def forward(self, *input): raise NotImplementedError
    
    # gradwrtoutput : gradient of the loss, with respect to the output of that layer (= input to next laxer)
    # return the gradient of the loss, with respect to the input of that layer (= output of previous layer)
    def backward(self, *gradwrtoutput): raise NotImplementedError
        
    def param(self): return []
    
    def to_string(self): raise NotImplementedError