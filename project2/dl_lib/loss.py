import torch
import random
import math
from .module import *
from .activation import Softmax

class MSELoss(Module):
    """Mean square error (MSE) loss module."""
    def __init__(self):
        """Constructor of the class."""
        super(MSELoss, self).__init__()
    
    def forward(self, input, target):
        """Forward pass function for the MSE loss."""
        return (input - target).pow(2).sum() / input.shape[0]
    
    def backward(self, input, target):
        """Backward pass function for the MSE loss."""
        return 2 * (input - target) / input.shape[0]
    
    def to_string(self): return "MSE Loss"
    
class CrossEntropyLoss(Module):
    """Cross-entropy loss module."""
    def __init__(self):
        """Constructor of the class."""
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, input, target_class):
        """Forward pass function for the cross-entropy loss."""
        return -(Softmax.softmax(input).gather(1,target_class.unsqueeze(1)).log().mean()).item()
    
    def backward(self, input, target_class):
        """Backward pass function for the cross-entropy loss."""
        x = Softmax.softmax(input.clone())
        x[range(input.shape[0]), target_class] -= 1
        return x / input.shape[0]
        
    def to_string(self):
        """String describtion of the class."""
        return "Cross-Entropy Loss"