#
# coding : utf-8

'''
Implements a custom binary matrix factorisation 
NOTE: torch is required

TODO:
- Add GPU support
'''

import torch

def to_binary(z):
    return (z > .5).float()
    

def get_random_binary_matrix(size):
    return to_binary(torch.rand(size=size))  

    
class straight_through_gradient_estimator(torch.autograd.Function):
    ''' backprob through discrete variables '''
    @staticmethod
    def forward(ctx, theta):
        return to_binary(theta)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clip(-1.1, 1.1)

def binary_matrix_factorisation(matrix, rank=10, iterations=500, lr=1e-2):
    ''' Tries to approximate the provided `matrix` through 
        two other binary matrixes of lower rank via:
        Matrix =~= PQ^T  
        
        Storage reduction with be from n x m to n x rank + m x rank
    '''
    
    assert isinstance(matrix, torch.Tensor), 'matrix must be torch tensor!'
    
    st = straight_through_gradient_estimator.apply
    
    def objective(P,Q):
        return torch.linalg.norm(matrix - st(P@Q.T), ord='fro') / matrix.numel()
    
    P = get_random_binary_matrix([matrix.shape[0], rank])#, device=device, requires_grad=True)
    Q = get_random_binary_matrix([matrix.shape[1], rank])#, device=device, requires_grad=True)
    P.requires_grad=True
    Q.requires_grad=True
    
    optimizer = torch.optim.Adam([P, Q], lr=1e-12)
    
    best_pq = None
    
    obj_vals = []
    for i in range(iterations):
        if i == 5: # warmstart optimizer
            optimizer.param_groups[0]['lr'] = lr
        
        optimizer.zero_grad()
        p, q = st(P), st(Q)
        obj = objective(p, q)
        obj_vals.append(obj.item())
        
        if obj <= min(obj_vals):
            best_pq = (p.clone().cpu().detach(), q.clone().cpu().detach())
            
        obj.backward()
        optimizer.step()
        
    p, q = best_pq
    
    return p, q, obj_vals