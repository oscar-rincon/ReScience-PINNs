import os 
import sys
sys.path.insert(0, '../../Utilities/')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sp
import scipy.io
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KdVNN(nn.Module):
    def __init__(self):
        super(KdVNN, self).__init__()
        self.linear_in = nn.Linear(1, 50, dtype=torch.float64)
        self.linear_out = nn.Linear(50, 50, dtype=torch.float64)
        self.layers = nn.ModuleList([nn.Linear(50, 50, dtype=torch.float64) for _ in range(5)])
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)    

def fwd_gradients_0(dy: torch.Tensor, x: torch.Tensor):
    z = torch.ones(dy.shape).requires_grad_(True)
    g = torch.autograd.grad(dy, x, grad_outputs=z, create_graph=True)[0]
    return torch.autograd.grad(g, z, grad_outputs=torch.ones(g.shape), create_graph=True)[0]



def net_U0(model, x_pt, lambda_1, lambda_2, dt, IRK_alpha):
    lambda_2 = torch.exp(lambda_2)
    U = model(x_pt)
    U_x = fwd_gradients_0(U, x_pt)
    U_xx = fwd_gradients_0(U_x, x_pt)
    U_xxx = fwd_gradients_0(U_xx, x_pt)
    F = -lambda_1*U*U_x - lambda_2*U_xxx
    U0 = U - dt*torch.matmul(F, IRK_alpha.T)
    return U0
 
def net_U1(model, x_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta):
    lambda_2 = torch.exp(lambda_2)
    U = model(x_pt)
    U_x = fwd_gradients_0(U, x_pt)
    U_xx = fwd_gradients_0(U_x, x_pt)
    U_xxx = fwd_gradients_0(U_xx, x_pt)
    F = -lambda_1*U*U_x - lambda_2*U_xxx
    U1 = U + dt*torch.matmul(F, (IRK_beta-IRK_alpha).T)
    return U1 

def mse(model, x0_pt,x1_pt, lambda_1, lambda_2,dt, IRK_alpha, IRK_beta,u0_pt, u1_pt):
    U0=net_U0(model, x0_pt, lambda_1, lambda_2,dt, IRK_alpha)
    U1=net_U1(model, x1_pt, lambda_1, lambda_2,dt, IRK_alpha, IRK_beta)
    loss = torch.sum((u0_pt - U0) ** 2) + torch.sum((u1_pt - U1) ** 2)
    return loss

def train_adam(model, x0_pt, x1_pt, lambda_1, lambda_2,dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=50_000):
    optimizer = torch.optim.Adam(list(model.parameters())+[lambda_1,lambda_2], lr=1e-5)
    global iter
    for i in range(1,num_iter+1):
        iter += 1 
        optimizer.zero_grad()
        loss = mse(model, x0_pt,x1_pt, lambda_1, lambda_2,dt, IRK_alpha,IRK_beta,u0_pt,u1_pt)
        loss.backward(retain_graph=True)
        optimizer.step()
        lambda_1s.append(lambda_1.item())
        lambda_2s.append(torch.exp(lambda_2).item())
        error_lambda_1 = np.abs(lambda_1.detach().numpy() - 1.0)/1.0 *100
        error_lambda_2 = np.abs(torch.exp(lambda_2).detach().numpy() - 0.0025)/0.0025 * 100
        results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
        if i % 1000 == 0:
            #torch.save(model.state_dict(), f'models_iters/KdV_clean_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.detach().numpy().item()} - l2: {torch.exp(lambda_2).detach().numpy().item()}")
 

def train_lbfgs(model, x0_pt,x1_pt, lambda_1, lambda_2,dt, IRK_alpha,IRK_beta,u0_pt,u1_pt, num_iter=50_000):
    optimizer = torch.optim.LBFGS(list(model.parameters())+[lambda_1,lambda_2],
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  history_size=50,
                                  tolerance_grad=1e-5,
                                  line_search_fn='strong_wolfe',
                                  tolerance_change=1.0 * np.finfo(float).eps)
    closure_fn = partial(closure, model,optimizer, x0_pt,x1_pt, lambda_1, lambda_2,dt, IRK_alpha,IRK_beta,u0_pt,u1_pt)
    optimizer.step(closure_fn) 
    
def closure(model,optimizer,x0_pt,x1_pt, lambda_1, lambda_2,dt, IRK_alpha,IRK_beta,u0_pt,u1_pt):
    optimizer.zero_grad()
    loss = mse(model, x0_pt,x1_pt, lambda_1, lambda_2,dt, IRK_alpha,IRK_beta,u0_pt,u1_pt)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    lambda_1s.append(lambda_1.item())
    lambda_2s.append(torch.exp(lambda_2).item())
    error_lambda_1 = np.abs(lambda_1.detach().numpy() - 1.0)/1.0 *100
    error_lambda_2 = np.abs(torch.exp(lambda_2).detach().numpy() - 0.0025)/0.0025 * 100
    results.append([iter, loss.item(), error_lambda_1.item(),error_lambda_2.item()])
    if iter % 1000 == 0:
        torch.save(model.state_dict(), f'models_iters/KdV_clean_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.detach().numpy().item()} - l2: {torch.exp(lambda_2).detach().numpy().item()}")
    return loss    

if __name__ == "__main__":
    set_seed(42)  
    iter = 0      

    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')

    if not os.path.exists('training'):
        os.makedirs('training')
        
    q = 50
    skip = 120

    N0 = 199
    N1 = 201
    layers = [1, 50, 50, 50, 50, q]

    data = scipy.io.loadmat('../Data/KdV.mat')

    t_star = data['tt'].flatten()[:,None]
    x_star = data['x'].flatten()[:,None]
    Exact = np.real(data['uu'])

    idx_t = 40

    noise = 0.0    

    idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x,:]
    u0 = Exact[idx_x,idx_t][:,None] 
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
        
    idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x,:]
    u1 = Exact[idx_x,idx_t + skip][:,None]
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])

    dt = torch.tensor((t_star[idx_t+skip] - t_star[idx_t]).item())
        
    # Doman bounds
    lb = x_star.min(0)
    ub = x_star.max(0)    

    # Load IRK weights
    tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
    weights =  np.reshape(tmp[0:q**2+q], (q+1, q))    
    IRK_alpha = torch.from_numpy(weights[0:-1,:]).double()
    IRK_beta = torch.from_numpy(weights[-1:,:]).double()       
    IRK_times = tmp[q**2+q:]
        
    x0_pt = torch.from_numpy(x0) 
    x0_pt.requires_grad = True    
    x1_pt = torch.from_numpy(x1) 
    x1_pt.requires_grad = True
    u0_pt = torch.from_numpy(u0) 
    u1_pt = torch.from_numpy(u1) 
    
    lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    lambda_2 = torch.nn.Parameter(torch.tensor([-6.0], dtype=torch.float64, requires_grad=True))  
    lambda_1s = []
    lambda_2s = []
    results = []         
    
    model = KdVNN()
    model.apply(init_weights)  
    
    # Entrenamiento con Adam
    start_time_adam = time.time()
    train_adam(model, x0_pt,x1_pt, lambda_1, lambda_2,dt, IRK_alpha,IRK_beta,u0_pt,u1_pt, num_iter=50_000)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")    
    
    
    # Entrenamiento con LBFGS
    start_time_lbfgs = time.time()
    train_lbfgs(model, x0_pt,x1_pt, lambda_1, lambda_2,dt, IRK_alpha,IRK_beta,u0_pt,u1_pt, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")
    
    # Tiempo total de entrenamiento
    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Obtener el valor del loss L2 final
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6f}")

    # Obtener el valor del loss L2 final
    final_l2 = results[-1][2]
    print(f"Final L2: {final_l2:.6f}")

    # Guardar los tiempos en un archivo de texto junto con el loss L2 final
    with open('training/KdV_training_summary_clean.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
        file.write(f"Final Loss: {final_loss:.6f}\n")
        file.write(f"Final L2: {final_l2:.6f}\n")
             
    results = np.array(results)
    lambda_1s = np.array(lambda_1s)
    lambda_2s = np.array(lambda_2s)
    np.savetxt("training/KdV_training_data_clean.csv", results, delimiter=",", header="Iter,Loss,l1,l2", comments="")
    np.savetxt("training/lambda_1s_clean.csv", lambda_1s, delimiter=",", header="l1", comments="")    
    np.savetxt("training/lambda_2s_clean.csv", lambda_2s, delimiter=",", header="l2", comments="")    
    torch.save(model.state_dict(), f'KdV_clean.pt')    