import torch
import random
import math

from module import *

# Mother class of modules that have to store their current value in memory
class Updatable(Module):
    def __init__(self):
        super(Updatable, self).__init__()
        
    def update_param(self, step_size): raise NotImplementedError
        