import torch
import random
import math
from .module import *
from .cell import *

class Activation(Cell):
    """Mother class (abstract) of all activations."""
    
    def __init__(self):
        """Constructor of the class."""
        super(Activation, self).__init__()
        
    def derivative(self, v): 
        """Virtual derivative method."""
        raise NotImplementedError
        
    def backward(self, gradwrtoutput):
        """Backward pass function."""
        return self.derivative(self.in_value) * gradwrtoutput # elementwise multiplication
    

class Sigmoid(Activation):
    """Sigmoid activation class."""
    
    def __init__(self):
        """Constructor of the class."""
        super(Sigmoid, self).__init__()

    @staticmethod
    def sigmoid(x):
        """Sigmoid function (static method)."""
        return 1 / (1 + (-x).exp())
    
    def forward(self, input):
        """Forward pass function for sigmoid activation."""
        x = self.sigmoid(input)
        self.update_in_value(x)
        return x
    
    def derivative(self, v):
        """Derivative of the sigmoid function"""
        return self.sigmoid(v) * self.sigmoid(-v) 
    
    def to_string(self): 
        """String describtion of the class."""
        return "Sigmoid"
    
# ReLU activation
class Relu(Activation):
    """Rectified linear unit (relu) activation class."""
    
    def __init__(self):
        """Constructor of the class."""
        super(Relu, self).__init__()

    @staticmethod
    def relu(x):
        """Rectified linear unit function (static method)."""
        x = x.clone()
        x[x < 0] = 0
        return x
    
    def forward(self, input):
        """Forward pass function for rectified linear unit activation."""
        x = self.relu(input)
        self.update_in_value(x)
        return x
    
    def derivative(self, v):
        """Derivative of the rectified linear unit function"""
        vv = v.clone()
        vv[vv <= 0] = 0
        vv[vv > 0] = 1
        return vv
    
    def to_string(self): 
        """String describtion of the class."""
        return "ReLU"
    
# Tanh activation
class Tanh(Activation):
    """Hypebolic tangent activation class."""
    
    def __init__(self):
        """Constructor of the class."""
        super(Tanh, self).__init__()
    
    @staticmethod
    def tanh(x):
        """Hyperbolic tangent function (static method)."""
        return (1 - (-2*x).exp() ) / (1 + (-2*x).exp())
        
    def forward(self, input):
        """Forward pass function for hyperbolic tangent activation."""
        x = self.tanh(input)
        self.update_in_value(x)
        return x
    
    def derivative(self, v):
        """Derivative of the hyperbolic tangent function"""
        return 1 - self.tanh(v)*self.tanh(v)
    
    def to_string(self):
        """String describtion of the class."""
        return "Tanh"
    
class Softmax(Activation):
    """Softmax activation class."""
    def __init__(self):
        """Constructor of the class."""
        super(Softmax, self).__init__()
    
    @staticmethod
    def softmax(x):
        """Softmax function (along dimension 1 of the input tensor, static method)."""
        return x.exp()/x.exp().sum(1).unsqueeze(1)
    
    def forward(self, input):
        """Forward pass function for softmax activation."""
        x = self.softmax(input)
        self.update_in_value(x)
        return x
    
    def derivative(self, v):
        """Derivative of the softmax function"""
        return self.softmax(v)*(1 - self.softmax(v))
    
    def to_string(self):
        """String describtion of the class."""
        return "Softmax"