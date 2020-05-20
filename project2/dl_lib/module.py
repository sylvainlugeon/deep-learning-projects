import torch
import random
import math

class Module(object):
    """Mother class (abstract) of all modules"""

    def forward(self, input):
        """Virtual method for the forward pass. 
            -input : batch of samples aligned along first dimension 
        """
        raise NotImplementedError
    
    def backward(self, gradwrtoutput): 
        """Virtual method for the bacward pass. 
            -gradwrtoutput : batch of gradients of the loss, with respect to the output of that layer (= input to next laxer)
        
        """
        raise NotImplementedError
        
    def param(self): 
        """Return an empty list of parameters. Overriden in updatable layers."""
        return []
    
    def gradwrtparam(self):
        """Return an empty list of gradients of the loss w.r.t. the parameters. Overriden in updatable layers."""
        return []
    
    def to_string(self): 
        """Virtual method for the string describtion of the subclass."""
        raise NotImplementedError