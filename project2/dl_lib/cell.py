import torch
import random
import math

from .module import *

class Cell(Module):
    """Mother class of the modules that must keep track of their internal state during the forward pass."""
    
    def __init__(self, train=True):
        """Constructor of the class."""
        self.in_value = None
        self.train = train
    
    def training_mode(self, train):
        """Set training mode."""
        self.train = train
    
    def update_in_value(self, x):
        """Update the internal state."""
        if self.train:
            self.in_value = x