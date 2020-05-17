import torch
import random
import math

# Mother class of all modules
class Module(object):
    
    # input : batch of samples aligned along first dimension
    # return the same samples after going through that module
    def forward(self, input): raise NotImplementedError
    
    # gradwrtoutput : batch of gradients of the loss, with respect to the output of that layer (= input to next laxer)
    # return batch of the gradient of the loss, with respect to the input of that layer (= output of previous layer)
    def backward(self, gradwrtoutput): raise NotImplementedError
        
    def param(self): return []
    
    def to_string(self): raise NotImplementedError