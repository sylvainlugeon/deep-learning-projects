import torch
import random
import math

from .module import *
from .loss import *
from .linear import *
from .activation import *


class Sequential(Module):
    """Sequential layers module."""
    
    def __init__(self, *modules):
        """Constructor of the class."""
        super(Sequential, self).__init__()
        self.layers = []
        for module in modules:
            if isinstance(module, Cell):
                self.layers.append(module)
            elif isinstance(module, Sequential):
                for layer in module.layers:
                    self.layers.append(layer)
            else:
                raise Exception("Invalid layer input") 
                
    def add_layers(self,*modules):
        """Add a new layer to the module."""
        for module in modules:
            if isinstance(module,Cell):
                self.layers.append(module)
            elif isinstance(module,Sequential):
                for layer in module.layers:
                    self.layers.append(layer)
            else:
                raise Exception("Invalid layer input") 
    
    def forward(self, input):
        """Forward pass method for the squential net."""
        x = input
        for m in self.layers: 
            x = m.forward(x)
        return x
    
    def backward(self, gradwrtoutput):
        """Backward pass method for the squential net."""
        x = gradwrtoutput
        for m in reversed(self.layers): 
            x = m.backward(x)
        return x

    
    def param(self):
        """Return a list of all the parameters in the sequential module."""
        p = []
        for m in self.layers:
            p.extend(m.param())
        return p
                
    def gradwrtparam(self):
        """Return a list of all the gradients of parameters in the sequential module."""
        dLdp = []
        for m in self.layers:
            dLdp.extend(m.gradwrtparam())
        return dLdp
    
    def describe(self):
        """Print a description of the sequential module."""
        print(self.to_string())
        
    def to_string(self):
        """String describtion of the class."""
        s = ''
        for m in self.layers:
            s = s + m.to_string() + '\n'
        s = s.rstrip('\n')
        return s

    def training_mode(self, train):
        """Set training mode."""
        for m in self.layers:
            m.training_mode(train)

    def predict(self, input):
        """Predict the output without keeping track of the internal state during forward."""
        self.training_mode(False)
        
        return self.forward(input)