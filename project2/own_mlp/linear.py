import torch
import random
import math
from module import *
from cell import *
from updatable import *

class Linear(Cell, Updatable):
    def __init__(self, in_size, out_size):
        
        self.in_size = in_size
        self.out_size = out_size
        
        # initializing the weights
        stdv = 1. / math.sqrt(self.in_size)
        self.w = torch.empty(out_size, in_size).uniform_(-stdv, stdv)
        self.b = torch.empty(out_size).uniform_(-stdv, stdv)
        
        # initializing the gradients
        self.dw = torch.empty(out_size, in_size).zero_()
        self.db = torch.empty(out_size).zero_()
        
    
    def forward(self, input):
        self.in_value = input
        return self.w.mm(input.T).T + self.b
    
    def backward(self, gradwrtoutput):
        
        # gradient for all the samples in the batch
        dw_c = gradwrtoutput.T.mm(self.in_value)
        db_c = torch.sum(gradwrtoutput.T, dim=1) # right to use sum ??
        
        # updating gradients
        self.dw.add_(dw_c)
        self.db.add_(db_c)
        
        # gradient with respect to the input
        d_input = gradwrtoutput.mm(self.w)
        
        return d_input
    
    def param(self):
        return self.w, self.b 
    
    def gradwrtparam(self):
        return self.dw, self.db
        
    def to_string(self):
        return "Linear ({}, {})".format(self.in_size, self.out_size)