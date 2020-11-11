import torch 
from .optimizer import Optimizer, required
import math


class pDCAe_exp(Optimizer):
    def __init__(self, params, lr = 1e-4, theta=10):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate : {}'.format(lr))
        if not 0.0 <= theta:
            raise ValueError('Invalid approximation coefficient theta :{}'.format(theta))
        defaults = dict(lr = lr,theta = theta)
        super(pDCAe_exp,self).__init__(params,defaults)

    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            theta = group['theta']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                #Initialize the k
                if len(state) == 0:
                    state['k_'] = state['k'] = 1
                    state['timestep'] = 0
                    state['one'] = torch.ones_like(p,memory_format=torch.preserve_format)
                    state['x_'] = torch.clone(p).detach()
                    # state['x'] = torch.clone(p).detach()
                    state['y'] = p
                    

                x_,y = state['x_'],state['y']
                k_,k = state['k_'],state['k']
                one = state['one']
                # calculate xi
                xi2 = one.add(torch.exp(torch.abs(p).mul(-1*theta)),alpha = -1)
                xi = theta*torch.sign(p).mul(xi2)
                # calculate beta
                beta = (k_-1)/k
                k_ = k
                k = (1+math.sqrt(1+4*k**2))/2
                state['x_'] = p.add(x_,alpha = beta).addcdiv(1+beta)



                '''
                next is about how to calculate new p which is from a segment function,
                but there is a problem about how to calculate gradient in y_t
                '''
                state['x'] = self.calculatext1(lr,p,grad,xi,theta)

                p = state['x'].mul(1+beta).add(state['x_'],alpha = -beta)

                state['timestep']+=1




    def calculatext1(self,lr,p,grad,xi,theta):
        med = grad.add(xi,alpha = -1)
        if p.data>med.add(theta,alpha = 1).mul(lr):
            m = med.add(theta)
            return p.data.add(m,alpha = -lr)
        
        elif p.data < med.add(theta,alpha = -1).mul(lr):
            m = med.add(theta,alpha = -1)
            return p.data.add(m,alpha = -lr)

        else:
            return 0













