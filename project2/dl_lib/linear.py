import torch
import random
import math
from .cell import *

class Linear(Cell):
    """Linear layer module."""
    
    def __init__(self, in_size, out_size, bias=True):
        """Constructor of the class."""
        self.in_size = in_size
        self.out_size = out_size
        
        # initializing the weights
        stdv = 1. / math.sqrt(self.in_size)
        self.w = torch.empty(out_size, in_size).uniform_(-stdv, stdv)
        self.dw = torch.empty(out_size, in_size).zero_()
        
        # initializing the biase
        self.bias = bias
        if self.bias:
            self.b = torch.empty(out_size).uniform_(-stdv, stdv)
            self.db = torch.empty(out_size).zero_()
        elif self.bias is not False:
            raise Exception("Invalid input for bias")
        
    
    def forward(self, input):
        """Forward pass function for the linear layer."""
        self.update_in_value(input)
        if self.bias:
            return self.w.mm(input.T).T + self.b
        else:
            return self.w.mm(input.T).T
    
    def backward(self, gradwrtoutput):
        """Bacward pass function for the linear layer."""
        # gradient for all the samples in the batch
        dw_c = gradwrtoutput.T.mm(self.in_value)
        self.dw.add_(dw_c)
        
        if self.bias:
            db_c = torch.sum(gradwrtoutput.T, dim=1) # right to use sum ??
            self.db.add_(db_c)
        
        # gradient with respect to the input
        d_input = gradwrtoutput.mm(self.w)
        
        return d_input
    
    def param(self):
        """Return a list of the parameters of the module."""
        if self.bias:
            return self.w, self.b
        else:
            return self.w
    
    def gradwrtparam(self):
        """Return a list of the gradient of the loss w.r.t. the parameters of the module."""
        if self.bias:
            return self.dw, self.db
        else:
            return self.dw
        
    def to_string(self):
        """String describtion of the class."""
        if self.bias:
            return "Linear ({}, {})".format(self.in_size, self.out_size) + " with bias"
        else:
            return "Linear ({}, {})".format(self.in_size, self.out_size) + " without bias"