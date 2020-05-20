import torch
import random
import math

class Optimizer(object):
    """Mother class (abstract) of all optimizers."""
    
    def __init__(self, model, lr):
        """Constructor of the class."""
        self.model = model
        self.lr = lr
    
    def step(self):
        """Virtual method for the optimization step."""
        raise NotImplementedError
    
    def zero_grad(self):
        """Set to zero all the gradients in <self.model>"""
        for dLdp in self.model.gradwrtparam():
            dLdp.zero_() 
    
class SGD(Optimizer):
    """Stochastic gradient descent (SGD) optimizer class."""
    
    def __init__(self, model, lr=1e-3):
        """Constructor of the class."""
        super(SGD, self).__init__(model, lr)
    
    def step(self):
        """SGD optimization step."""
        for p, dLdp in zip(self.model.param(), self.model.gradwrtparam()):
            p.add_(- self.lr * dLdp)
        
            
class Adam(Optimizer):
    """Adaptative moment estimation (Adam) optimizer class."""
    
    def __init__(self, model, lr=1e-3, betas=(0.9,0.999),eps=1e-08):
        """Constructor of the class."""
        super(Adam, self).__init__(model,lr)
        self.betas = betas
        self.eps = eps
        self.mt = []
        self.vt = []
        
    def step(self):
        """Adam optimization step."""
        i = 0
        for p, dLdp in zip(self.model.param(), self.model.gradwrtparam()):
            if len(self.mt) < i + 1 and len(self.vt) < i + 1:
                self.mt.append(torch.mul(dLdp, 1 - self.betas[0]))
                self.vt.append(torch.mul(dLdp * dLdp, 1 - self.betas[1]))
            else:
                self.mt[i] = torch.mul(self.mt[i],self.betas[0]) + torch.mul(dLdp, 1 - self.betas[0])
                self.vt[i] = torch.mul(self.vt[i],self.betas[1]) + torch.mul(dLdp * dLdp, 1 - self.betas[1])
            mt_hat = torch.mul(self.mt[i], 1 / (1 - self.betas[0]))
            vt_hat = torch.mul(self.vt[i], 1 / (1 - self.betas[1]))
            p.add_(torch.mul(mt_hat, -self.lr / (vt_hat.sqrt() + self.eps)))
            i = i + 1
