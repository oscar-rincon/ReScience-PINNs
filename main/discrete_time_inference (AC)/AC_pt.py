import torch
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial
import time
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ACNN(nn.Module):
    def __init__(self):
        super(ACNN, self).__init__()
        self.linear_in = nn.Linear(1, 200, dtype=torch.float64)
        self.linear_out = nn.Linear(200, 101, dtype=torch.float64)
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

def f(model, x_0, x_1, dt, IRK_weights):
    U1 = model(x_0)
    U = U1[:, :-1]
    U_x = fwd_gradients_0(U, x_0)
    U_xx = fwd_gradients_0(U_x, x_0)
    F = 5.0 * U - 5.0 * U**3 + 0.0001 * U_xx
    U0 = U1 - dt * torch.matmul(F, IRK_weights)
    U1 = model(x_1)
    U1_x = fwd_gradients_0(U1, x_1)
    return U0, U1, U1_x

def mse(model, x_0, x_1, dt, IRK_weights, U0_real):
    U0, U1, U1_x = f(model, x_0, x_1, dt, IRK_weights)
    loss = torch.sum((U0_real - U0) ** 2) + torch.sum((U1[0, :] - U1[1, :]) ** 2) + torch.sum((U1_x[0, :] - U1_x[1, :])**2)
    return loss

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def closure(model, optimizer, x_0, x_1, dt, IRK_weights, U0_real, Exact, idx_t1, results):
    optimizer.zero_grad()
    loss = mse(model, x_0, x_1, dt, IRK_weights, U0_real)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    U1_pred = model(x_star)
    pred = U1_pred[:, -1].detach().numpy()
    error = np.linalg.norm(pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
    results.append([iter, loss.item(), error])
    if iter % 100 == 0:
        torch.save(model.state_dict(), f'models_iters/pt_model_AC_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - L2: {error}")
    return loss

def train_adam(model, x_0, x_1, dt, IRK_weights, U0_real, Exact, idx_t1, results, num_iter=10_000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    global iter
    for i in range(1,num_iter+1):
        iter += 1 
        optimizer.zero_grad()
        loss = mse(model, x_0, x_1, dt, IRK_weights, U0_real)
        loss.backward(retain_graph=True)
        optimizer.step()
        U1_pred = model(x_star)
        pred = U1_pred[:, -1].detach().numpy()
        error = np.linalg.norm(pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)        
        results.append([iter, loss.item(), error])
        if i % 100 == 0:
            torch.save(model.state_dict(), f'models_iters/pt_model_AC_{iter}.pt')
            print(f"Adam - Iter: {i} - Loss: {loss.item()} - L2: {error}")
          

def train_lbfgs(model, x_0, x_1, dt, IRK_weights, U0_real, Exact, idx_t1, results, num_iter=50_000):
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  history_size=50,
                                  tolerance_grad=1e-5,
                                  line_search_fn='strong_wolfe',
                                  tolerance_change=1.0 * np.finfo(float).eps)
    closure_fn = partial(closure, model, optimizer, x_0, x_1, dt, IRK_weights, U0_real, Exact, idx_t1, results)
    optimizer.step(closure_fn)

if __name__ == "__main__":
    set_seed(42)
    q = 100
    lb = np.array([-1.0], dtype=np.float64)
    ub = np.array([1.0], dtype=np.float64)
    N = 200
    iter = 0
    data = sp.loadmat('../Data/AC.mat')
    t = data['tt'].flatten()[:, None].astype(np.float64)
    x = data['x'].flatten()[:, None].astype(np.float64)
    Exact = np.real(data['uu']).T.astype(np.float64)
    idx_t0 = 20
    idx_t1 = 180
    dt = torch.from_numpy(t[idx_t1] - t[idx_t0]).to(torch.float64)
    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)
    x0 = x[idx_x,:]
    x0 = torch.from_numpy(x0).to(torch.float64)
    x0.requires_grad = True
    u0 = Exact[idx_t0:idx_t0+1,idx_x].T
    u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    u0 = torch.from_numpy(u0).to(torch.float64)
    u0 = u0
    x1 = torch.from_numpy(np.vstack((lb, ub))).to(torch.float64)
    x1.requires_grad = True
    tmp = np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2).astype(np.float64)
    IRK_weights = torch.from_numpy(np.reshape(tmp[0:q**2+q], (q+1, q))).to(torch.float64).T
    x_star = torch.from_numpy(x).to(torch.float64)
    model = ACNN()
    model.apply(init_weights)
    model.to(torch.float64)
    model.train()
    results = []

    # Entrenamiento con Adam
    start_time_adam = time.time()
    train_adam(model, x0, x1, dt, IRK_weights, u0, Exact, idx_t1, results, num_iter=10_000)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")
    
    # Entrenamiento con LBFGS
    start_time_lbfgs = time.time()
    train_lbfgs(model, x0, x1, dt, IRK_weights, u0, Exact, idx_t1, results, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")
    
    # Tiempo total de entrenamiento
    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Guardar los tiempos en un archivo de texto
    with open('outputs/training_times.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
             
    results = np.array(results)
    np.savetxt("outputs/pt_training_AC.csv", results, delimiter=",", header="Iter,Loss,L2", comments="")
    torch.save(model.state_dict(), f'outputs/pt_model_AC.pt')
    
    
