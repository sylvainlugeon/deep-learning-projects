import torch
import random
import math
from module import *

# Mother class of losses, defines methodes loss() and dloss()
class Loss(Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def loss(self, input, target): raise NotImplementedError
        
    def dloss(self, input, target): raise NotImplementedError

# MSE Loss
class MSELoss(Loss):
    def __init__(self):
        super(MSELoss, self).__init__()
    
    # input: predicted, target
    def loss(self, input, target):
        return (input - target).pow(2).sum() # dividing by nb of samples ? 
    
    # gradwrtoutput : predicted, target
    def dloss(self, input, target):
        return 2 * (input - target)
    
    def to_string(self): return "MSE Loss"