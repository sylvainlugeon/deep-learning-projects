import torch
import random
import math
from module import *

# Sigmoid activation 
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.in_value = None

    def forward(self, *input):
        x = 1 / (1 + (-input[0]).exp())
        self.in_value = x
        return x
    
    def derivative(self, v):
        return self.forward(v) * self.forward(-v) # fancy form for derivative
    
    def backward(self, *gradwrtoutput):
        return self.derivative(self.in_value) * gradwrtoutput[0] # elementwise multiplication
    
    def to_string(self): return "Sigmoid"
    
# ReLU activation
class Relu(Module):
    def __init__(self):
        super(Relu, self).__init__()
        self.in_value = None

    def forward(self, *input):
        x = input[0].clone()
        x[x < 0] = 0
        self.in_value = x
        return x
    
    def derivative(self, v):
        vv = v.clone()
        vv[vv <= 0] = 0
        vv[vv > 0] = 1
        return vv
    
    def backward(self, *gradwrtoutput):
        return self.derivative(self.in_value) * gradwrtoutput[0] # elementwise multiplication
    
    def to_string(self): return "ReLU"
    
# Tanh activation
class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.in_value = None
    
    def forward(self, *input):
        x = ( input[0].exp() - (-input[0]).exp() ) / (input[0].exp() + (-input[0]).exp() )
        self.in_value = x
        return x
    
    def derivative(self, v):
        return 1 - self.forward(v)*self.forward(v)
    
    def backward(self, *gradwrtoutput):
        return self.derivative(self.in_value) * gradwrtoutput[0]
    
    def to_string(self): return "Tanh"
