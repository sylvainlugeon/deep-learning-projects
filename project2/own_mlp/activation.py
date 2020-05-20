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

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + (-x).exp())
    
    def forward(self, input):
        x = self.sigmoid(input)
        self.in_value = x
        return x
    
    def derivative(self, v):
        return self.sigmoid(v) * self.sigmoid(-v) 
    
    def to_string(self): return "Sigmoid"
    
# ReLU activation
class Relu(Activation):
    def __init__(self):
        super(Relu, self).__init__()

    @staticmethod
    def relu(x):
        x = x.clone()
        x[x < 0] = 0
        return x
    
    def forward(self, input):
        x = self.relu(input)
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
    
    @staticmethod
    def tanh(x):
        return (1 - (-2*x).exp() ) / (1 + (-2*x).exp())
        
    def forward(self, input):
        x = self.tanh(input)
        self.in_value = x
        return x
    
    def derivative(self, v):
        return 1 - self.tanh(v)*self.tanh(v)
    
    def to_string(self): return "Tanh"
    
class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()
    
    @staticmethod
    def softmax(x):
        return x.exp()/x.exp().sum(1).unsqueeze(1)
    
    def forward(self, input):
        x = self.softmax(input)
        self.in_value = x
        return x
    
    def derivative(self, v):
        return self.softmax(v)*(1 - self.softmax(v))
    
    def to_string(self): return "Softmax"