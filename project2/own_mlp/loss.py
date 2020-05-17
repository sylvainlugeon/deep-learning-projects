import torch
import random
import math
from module import *

# MSE Loss
class MSELoss(Module):
    
    # input: predicted, target
    def forward(self, *input):
        return (input[0] - input[1]).pow(2).sum() # dividing by nb of samples ? 
    
    # gradwrtoutput : predicted, target
    def backward(self, *gradwrtoutput):
        return 2 * (gradwrtoutput[0] - gradwrtoutput[1]) # dividing by nb of sample ? 
    
    def to_string(self): return "MSE Loss"