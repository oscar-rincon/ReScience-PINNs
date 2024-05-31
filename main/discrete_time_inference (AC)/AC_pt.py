import torch 
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial
import os
import warnings
warnings.filterwarnings("ignore")  # Ignore warning messages
from torch.utils.data import DataLoader, TensorDataset

iter = 1

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class ACNN(nn.Module):
    def __init__(self,):
        super(ACNN, self).__init__()
        self.linear_in = nn.Linear(1, 200, dtype=torch.float64)
        self.linear_out = nn.Linear(200, 101, dtype=torch.float64)  # Ajustado para coincidir con IRK_weights
        self.layers = nn.ModuleList([nn.Linear(200, 200, dtype=torch.float64) for _ in range(5)])
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x 

def fwd_gradients_0(dy: torch.Tensor, x: torch.Tensor):
        z = torch.ones(dy.shape).requires_grad_(True)
        g = torch.autograd.grad(dy, x, grad_outputs=z, create_graph=True)[0]
        return torch.autograd.grad(g, z, grad_outputs=torch.ones(g.shape), create_graph=True)[0]
    
    
def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for _ in range(order):
        dy = torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(dy), create_graph=True, retain_graph=True)[0]
    return dy

def f(model, x_0, x_1, dt, IRK_weights):
    U1 = model(x_0)
    #print(f"U1.shape: {U1.shape}")  # Depuración
    U = U1[:, :-1]
    #print(f"U.shape: {U.shape}")  # Depuración
    #U_xx = derivative(U, x_0, order=2)
    U_x = fwd_gradients_0(U, x_0)
    U_xx = fwd_gradients_0(U_x, x_0)
    F = 5.0 * U - 5.0 * U**3 + 0.0001 * U_xx

    # Ajustar dimensiones de IRK_weights para coincidir con F
    #if IRK_weights.shape[1] != F.shape[1]:
    #    IRK_weights = IRK_weights[:F.shape[1], :F.shape[1]]

    #print(f"F.shape: {F.shape}, IRK_weights.shape: {IRK_weights.shape}")  # Depuración

    U0 = U1 - dt * torch.matmul(F, IRK_weights)
    #print(f"U0.shape: {U0.shape}")  # Depuración

    U1 = model(x_1)
    #U1_x = derivative(U1, x_1, order=1)
    U1_x = fwd_gradients_0(U1, x_1)
    return U0, U1, U1_x

def mse(model, x_0, x_1, dt, IRK_weights, U0_real):
    loss_fn = nn.MSELoss(reduction='sum')
    U0, U1, U1_x = f(model, x_0, x_1, dt, IRK_weights)
    #print(f"U0_real.shape: {U0_real.shape}, U0.shape: {U0.shape}")  # Depuración
    #U0_real_resized = U0_real[:U0.shape[0], :U0.shape[1]]
    loss = torch.sum((U0_real - U0) ** 2) + torch.sum((U1[0, :] - U1[1, :]) ** 2) + torch.sum((U1_x[0, :] - U1_x[1, :])**2)
    return loss

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def closure(model, optimizer, x_0, x_1, dt, IRK_weights, U0_real):
    optimizer.zero_grad()
    loss = mse(model, x_0, x_1, dt, IRK_weights, U0_real)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    print(f" iteration: {iter}  loss: {loss.item()}")
    if iter % 100 == 0:
        torch.save(model.state_dict(), f'models/model_LBFGS_{iter}.pt')
    return loss

def train_lbfgs(model, x_0, x_1, dt, IRK_weights, U0_real):
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=50000, max_eval=50000,tolerance_grad=1e-5, history_size=50, line_search_fn='strong_wolfe',tolerance_change=0.5 * np.finfo(float).eps)#, tolerance_grad=1e-5
    closure_fn = partial(closure, model, optimizer, x_0, x_1, dt, IRK_weights, U0_real)
    optimizer.step(closure_fn)

if __name__ == "__main__":
    set_seed(42)    
    q = 100
    lb = np.array([-1.0], dtype=np.float64)
    ub = np.array([1.0], dtype=np.float64)
    N = 200

    data = sp.loadmat('data/AC.mat')
    t = data['tt'].flatten()[:, None].astype(np.float64)
    x = data['x'].flatten()[:, None].astype(np.float64)
    Exact = np.real(data['uu']).T.astype(np.float64)
    idx_t0 = 20
    idx_t1 = 180
    dt = torch.from_numpy(t[idx_t1] - t[idx_t0]).to(torch.float64)

    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)
    x0 = x[idx_x,:]#x[:,:]#x[idx_x,:]
    x0 = torch.from_numpy(x0).to(torch.float64)
    x0.requires_grad = True
    u0 = Exact[idx_t0:idx_t0+1,idx_x].T#Exact[idx_t0:idx_t0+1,:].T
    u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    u0 = torch.from_numpy(u0).to(torch.float64)
    u0 = u0#.flatten()

    x1 = torch.from_numpy(np.vstack((lb, ub))).to(torch.float64)
    x1.requires_grad = True

    tmp = np.loadtxt('Butcher_IRK%d.txt' % (q), ndmin=2).astype(np.float64)
    IRK_weights = torch.from_numpy(np.reshape(tmp[0:q**2+q], (q+1, q))).to(torch.float64).T

    model = ACNN()
    model.apply(init_weights)
    model.to(torch.float64)
    model.train()
  
    train_lbfgs(model, x0, x1, dt, IRK_weights, u0)
