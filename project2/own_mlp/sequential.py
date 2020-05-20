import torch
import random
import math

from module import *
from loss import *
from linear import *
from activation import *


# A sequence of modules
class Sequential(Updatable):
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.layers = []
        for module in modules:
            if isinstance(module, Cell):
                self.layers.append(module)
            elif isinstance(module, Sequential):
                for layer in module.layers:
                    self.layers.append(layer)
            else:
                raise NotImplementedError("Invalid layer input") #not the right exception
                
    def add_layers(self,*modules):
        for module in modules:
            if isinstance(module,Cell):
                self.layers.append(module)
            elif isinstance(module,Sequential):
                for layer in module.layers:
                    self.layers.append(layer)
            else:
                raise NotImplementedError("Invalid layer input") #not the right exception
    
    def forward(self, input):
        x = input
        for m in self.layers: 
            x = m.forward(x)
        return x
    
    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        for m in reversed(self.layers): 
            x = m.backward(x)
        return x

    
    def param(self):
        p = []
        for m in self.layers:
            p.extend(m.param())
        return p
                
    def gradwrtparam(self):
        dLdp = []
        for m in self.layers:
            dLdp.extend(m.gradwrtparam())
        return dLdp
    
    def describe(self):
        print(self.to_string())
        
    def to_string(self):
        s = ''
        for m in self.layers:
            s = s + m.to_string() + '\n'
        return s
            
    # predictions using current weights in the modules
    # input : tensor, samples aligned along first dimension
    def predict(self, input):
        return self.forward(input)