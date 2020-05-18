import torch
import random
import math

from module import *

# Mother class of modules that have to store their current value in memory
class Cell(Module):
    def __init__(self):
        self.in_value = None
        