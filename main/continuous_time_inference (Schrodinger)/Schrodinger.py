import torch 
import torch.nn as nn
import numpy as np
import scipy.io as sp
from functools import partial
from pyDOE import lhs
import time
import os 
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SchrodingerNN(nn.Module):
    def __init__(self):
        super(SchrodingerNN, self).__init__()
        self.linear_in = nn.Linear(2, 100)
        self.linear_out = nn.Linear(100, 2)
        self.layers = nn.ModuleList([nn.Linear(100, 100) for i in range(5)])
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x

def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs = torch.ones_like(dy), create_graph=True, retain_graph=True
        )[0]
    return dy

def f(model, x_f, t_f):
    h = model(torch.stack((x_f, t_f), axis = 1))
    u = h[:, 0]
    v = h[:, 1]
    u_t = derivative(u, t_f, order=1)
    v_t = derivative(v, t_f, order=1)
    u_xx = derivative(u, x_f, order=2)
    v_xx = derivative(v, x_f, order=2)
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u 
    return f_u, f_v

def mse_f(model, x_f, t_f):
    f_u, f_v = f(model, x_f, t_f)
    return (f_u**2 + f_v**2).mean()

def mse_0(model, x_0, u_0, v_0):
    t_0 = torch.zeros_like(x_0)
    h = model(torch.stack((x_0, t_0), axis = 1))
    h_u = h[:, 0]
    h_v = h[:, 1]
    return ((h_u-u_0)**2+(h_v-v_0)**2).mean()

def mse_b(model, t_b):
    x_b_left = torch.zeros_like(t_b)-5
    x_b_left.requires_grad = True
    h_b_left = model(torch.stack((x_b_left, t_b), axis = 1))
    h_u_b_left = h_b_left[:, 0]
    h_v_b_left = h_b_left[:, 1]
    h_u_b_left_x = derivative(h_u_b_left, x_b_left, 1)
    h_v_b_left_x = derivative(h_v_b_left, x_b_left, 1)
    
    x_b_right = torch.zeros_like(t_b)+5
    x_b_right.requires_grad = True
    h_b_right = model(torch.stack((x_b_right, t_b), axis = 1))
    h_u_b_right = h_b_right[:, 0]
    h_v_b_right = h_b_right[:, 1]
    h_u_b_right_x = derivative(h_u_b_right, x_b_right, 1)
    h_v_b_right_x = derivative(h_v_b_right, x_b_right, 1)

    mse_drichlet = (h_u_b_left-h_u_b_right)**2+(h_v_b_left-h_v_b_right)**2
    mse_newman = (h_u_b_left_x-h_u_b_right_x)**2+(h_v_b_left_x-h_v_b_right_x)**2
    mse_total = (mse_drichlet + mse_newman).mean()
    
    return mse_total

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)

def closure(model, optimizer, x_f, t_f, x_0, u_0, v_0, h_0, t):
    optimizer.zero_grad()
    loss = mse_f(model, x_f, t_f) + mse_0(model, x_0, u_0, v_0) + mse_b(model, t)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    pred = model(X_star)
    h_pred = (pred[:, 0]**2 + pred[:, 1]**2)**0.5
    error = np.linalg.norm(h_star-h_pred.detach().numpy(),2)/np.linalg.norm(h_star,2) 
    results.append([iter, loss.item(), error])    
    if iter % 100 == 0:
        torch.save(model.state_dict(), f'model/Schrodinger_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - L2: {error}")
    return loss

def train_adam(model, x_f, t_f, x_0, u_0, v_0, h_0, t, num_iter=50_000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    global iter
     
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        loss = mse_f(model, x_f, t_f) + mse_0(model, x_0, u_0, v_0) + mse_b(model, t)
        loss.backward(retain_graph=True)
        optimizer.step()
        pred = model(X_star)
        h_pred = (pred[:, 0]**2 + pred[:, 1]**2)**0.5
        error = np.linalg.norm(h_star-h_pred.detach().numpy(),2)/np.linalg.norm(h_star,2) 
        results.append([iter, loss.item(), error])
        iter += 1
        if iter % 100 == 0:
            torch.save(model.state_dict(), f'model/Schrodinger_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - L2: {error}")

def train_lbfgs(model,  x_f, t_f, x_0, u_0, v_0, h_0, t, num_iter=50_000):
    optimizer = torch.optim.LBFGS(model.parameters(),
                                    lr=1,
                                    max_iter=num_iter,
                                    max_eval=num_iter,
                                    tolerance_grad=1e-5,
                                    history_size=50,
                                    tolerance_change=1.0 * np.finfo(float).eps,
                                    line_search_fn="strong_wolfe")
 
    closure_fn = partial(closure, model, optimizer, x_f, t_f, x_0, u_0, v_0, h_0, t)
    optimizer.step(closure_fn)

if __name__== "__main__":
    set_seed(42)
    iter = 0
     
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])
     
    N0 = 50  
    N_b = 50  
    N_f = 20_000 

    data = sp.loadmat('../Data/NLS.mat')
    x_0 = torch.from_numpy(data['x'].astype(np.float32))
    x_0.requires_grad = True
    x_0 = x_0.flatten().T
    t = torch.from_numpy(data['tt'].astype(np.float32))
    t.requires_grad = True
    t = t.flatten().T

    h = torch.from_numpy(data['uu'])

    u_0 = torch.real(h)[:, 0]
    v_0 = torch.imag(h)[:, 0]
    h_0 = torch.stack((u_0, v_0), axis=1)

    c_f = lb + (ub-lb)*lhs(2, N_f)
    x_f = torch.from_numpy(c_f[:, 0].astype(np.float32))
    x_f.requires_grad = True
    t_f = torch.from_numpy(c_f[:, 1].astype(np.float32))
    t_f.requires_grad = True

    idx_0 = np.random.choice(x_0.shape[0], N0, replace=False)
    x_0 = x_0[idx_0]
    u_0 = u_0[idx_0]
    v_0 = v_0[idx_0]
    h_0 = h_0[idx_0]

    idx_b = np.random.choice(t.shape[0], N_b, replace=False)
    t_b = t[idx_b]
    
    X, T = torch.meshgrid(torch.tensor(data['x'].flatten()[:]), torch.tensor(data['tt'].flatten()[:]))
    xcol = X.reshape(-1, 1)
    tcol = T.reshape(-1, 1)
    X_star = torch.cat((xcol, tcol), 1).float()   
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2) 
    h_star = Exact_h.flatten()[:]

    model = SchrodingerNN()
    model.apply(init_weights)
    
    results = []
    
    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')

    if not os.path.exists('training'):
        os.makedirs('training')
        
    start_time_adam = time.time()
    train_adam(model, x_f, t_f, x_0, u_0, v_0, h_0, t_b, num_iter=5_000)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")

    start_time_lbfgs = time.time()
    train_lbfgs(model, x_f, t_f, x_0, u_0, v_0, h_0, t_b, num_iter=5_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")

    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Obtener el valor del loss L2 final
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6f}")

    # Obtener el valor del loss L2 final
    final_l2 = results[-1][2]
    print(f"Final L2: {final_l2:.6f}")

    # Guardar los tiempos en un archivo de texto junto con el loss L2 final
    with open('training/Schrodinger_training_summary.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
        file.write(f"Final Loss: {final_loss:.6f}\n")
        file.write(f"Final L2: {final_l2:.6f}\n")
        
    results = np.array(results)
    np.savetxt("training/Schrodinger_training_data.csv", results, delimiter=",", header="Iter,Loss,L2", comments="")
    torch.save(model.state_dict(), f'Schrodinger.pt')