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
            p.add_(- self.lr * dLdp)
        
    def zero_grad(self):
        for dLdp in self.model.gradwrtparam():
            dLdp.zero_() 
            
class Adam(Optimizer):
    
    def __init__(self, model, lr=1e-3, betas=(0.9,0.999),eps=1e-08):
        super(Adam, self).__init__(model,lr)
        self.betas = betas
        self.eps = eps
        
        self.mt = []
        self.vt = []
        
    def step(self):
        i = 0
        for p, dLdp in zip(self.model.param(), self.model.gradwrtparam()):
            if len(self.mt)<i+1 and len(self.vt)<i+1:
                self.mt.append(torch.mul(dLdp,1-self.betas[0]))
                self.vt.append(torch.mul(dLdp*dLdp,1-self.betas[1]))
            else:
                self.mt[i] = torch.mul(self.mt[i],self.betas[0]) + torch.mul(dLdp,1-self.betas[0])
                self.vt[i] = torch.mul(self.vt[i],self.betas[1]) + torch.mul(dLdp*dLdp,1-self.betas[1])
            mt_hat = torch.mul(self.mt[i],1/(1-self.betas[0]))
            vt_hat = torch.mul(self.vt[i],1/(1-self.betas[1]))
            p.add_(torch.mul(mt_hat,-self.lr/(vt_hat.sqrt() + self.eps)))
            i = i+1
    def zero_grad(self):
        for dLdp in self.model.gradwrtparam():
            dLdp.zero_()
