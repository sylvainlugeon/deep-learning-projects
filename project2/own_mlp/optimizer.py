import torch
import random
import math

class Optimizer(object):
    
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    
    def __init__(self, model, lr=1e-3):
        super(SGD, self).__init__(model, lr)
    
    def step(self):
        for p, dLdp in zip(self.model.param(), self.model.gradwrtparam()):
            p.add_(- self.lr *  dLdp)
        
    def zero_grad(self):
        for dLdp in self.model.gradwrtparam():
            dLdp.zero_() 