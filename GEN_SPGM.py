import torch 
from .optimizer import Optimizer, required
import math


class GEN_SPGM_1(optimizer):
    def __init__(self,params,lr = 1e-4, lambda_ = 0.5,delta = 1e-10,rho_0 = 0.9,mu = 0.1,penalty = 'l_1'):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate lr : {}'.format(lr))
        if not 0.0 <= lambda_:
            raise ValueError('Invalid regularization parameter lambda_ :{}'.format(lambda_))
        if not 0.0<=delta:
            raise ValueError('Invalid small constant for positive delta :{}'.format(delta))
        if not 0.0<=mu<=1:
            raise ValueError('Invalid decaying parameter mu :{}'.format(mu))
        if not 0.0<=rho_0<=1:
            raise ValueError('Invalid momentum parameter rho_0 {}'.format(rho_0))
        if penalty not in ['l_1','l_0','l_0.5','l_0.6']:
            raise ValueError('Invalid penalty function {}'.format(penalty))

        defaults = dict(lr = lr,lambda_ = lambda_,delta = delta,mu = mu,rho_0 = rho_0,penalty = penalty)

        super(GEN_SPGM_1,self).__init__(params,defaults)


        @torch.no_grad()
        def step(self,closure = None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            
            for group in self.param_groups:
                lr = group['lr']
                lambda_ = group['lambda_']
                delta = group['delta']
                penalty = group['penalty']


                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data

                    state =  self.state[p]
                    if len(state) == 0:
                        state['timestep'] = 0
                        state['moment'] = torch.zeros_like(grad,memory_format = torch.preserve_format)

                    moment = state['moment']
                    rho = rho*mu**state['timestep']
                    state['tiemstep'] += 1
                    moment = moment.mul(rho).add(grad.mul(1-rho))

                    # C_t is preconditioner construction 
                    # C_t = torch.eye(p.size())
                    # whether it report a error

                    # C_t = torch.diag(C_t)
                    
                    C_t = torch.ones_like(grad)
                    C_t.add_(delta)
                    comparison = torch.sqrt(torch.reciprocal(C_t).mul(2*lr*lambda_))

                    phat = p - moment.addcdiv(C_t).mul(lr)

                    #####
                    # l_0 penalty function
                    if penalty == 'l_0':
                        p = self.calculatel0(phat,comparison)

                    if penalty == 'l_1':
                        p = self.calculatel1(phat,lr,lambda_,delta,C_t)

        

        def calculatel0(self,phat,comparison):
            absp = torch.abs(phat)
            index_p = torch.zeros_like(phat)
            pos = absp - comparison
            pos_greater = (pos > 0)
            pos_smaller = (pos < 0)
            # a equality circumstance need to be considered
            index_p[pos_smaller] = phat[pos_smaller]

            p = phat - index_p
            return p 

        def calculatel1(self,phat,lr,lambda_,delta,C_t):
            deno = torch.reciprocal(C_t.add(delta))
            value = torch.abs(phat).add(lr*lambda_*deno,alpha = -1)

            p = torch.sign(phat).addcmul(value)

            return p
            







