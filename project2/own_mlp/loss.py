import torch
import random
import math
from module import *
from activation import Softmax


# MSE Loss
class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    def forward(self, input, target):
        return (input - target).pow(2).sum() / input.shape[0]
    
    def backward(self, input, target):
        return 2 * (input - target) / input.shape[0]
    
    def to_string(self): return "MSE Loss"
    
class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, input, target_class):
        return -(Softmax.softmax(input).gather(1,target_class.unsqueeze(1)).log().mean()).item()
    
    def backward(self, input, target_class):
        x = Softmax.softmax(input.clone())
        x[range(input.shape[0]), target_class] -= 1
        return x / input.shape[0]
        
    def to_string(self): return "Cross-Entropy Loss"