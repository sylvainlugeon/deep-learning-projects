import torch
import random
import math
from module import *
from cell import *

# Mother class of all activation m
class Activation(Cell):
    def __init__(self):
        super(Activation, self).__init__()
        
    def derivative(self, v): raise NotImplementedError
        
    def backward(self, gradwrtoutput):
        return self.derivative(self.in_value) * gradwrtoutput # elementwise multiplication
    

# Sigmoid activation 
class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        x = 1 / (1 + (-input).exp())
        self.in_value = x
        return x
    
    def derivative(self, v):
        return self.forward(v) * self.forward(-v) # fancy form for derivative
    
    def to_string(self): return "Sigmoid"
    
# ReLU activation
class Relu(Activation):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, input):
        x = input.clone()
        x[x < 0] = 0
        self.in_value = x
        return x
    
    def derivative(self, v):
        vv = v.clone()
        vv[vv <= 0] = 0
        vv[vv > 0] = 1
        return vv
    
    def to_string(self): return "ReLU"
    
# Tanh activation
class Tanh(Activation):
    def __init__(self):
        super(Tanh, self).__init__()
    
    def forward(self, input):
        x = (1 - (-2*input).exp() ) / (1 + (-2*input).exp())
        self.in_value = x
        return x
    
    def derivative(self, v):
        return 1 - self.forward(v)*self.forward(v)
    
    def to_string(self): return "Tanh"