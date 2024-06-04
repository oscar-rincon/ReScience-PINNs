import os 
import torch 
import torch.nn as nn
import numpy as np
import scipy.io
from functools import partial
from pyDOE import lhs
import time
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from itertools import product, combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.insert(0, '../../Utilities/')
#from utils_plots import * 
from plotting import *#newfig, savefig
sys.path.insert(0, '.')
import warnings

warnings.filterwarnings("ignore")

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class NSNN(nn.Module):
    def __init__(self):
        super(NSNN, self).__init__()
        self.linear_in = nn.Linear(3, 20, dtype=torch.float64)
        self.linear_out = nn.Linear(20, 2, dtype=torch.float64)
        self.layers = nn.ModuleList([nn.Linear(20, 20, dtype=torch.float64) for _ in range(9)])
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
        m.bias.data.fill_(0.0)
        
def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs = torch.ones_like(dy), create_graph=True, retain_graph=True
        )[0]
    return dy


def f(model, x_train_pt, y_train_pt, t_train_pt):
    psi_and_p = model(torch.stack((x_train_pt, y_train_pt, t_train_pt), axis = 1).view(-1, 3))
    psi = psi_and_p[:,0:1]
    p = psi_and_p[:,1:2]
    
    u = derivative(psi, y_train_pt, order=1)
    v = -derivative(psi, x_train_pt, order=1)
    
    u_t = derivative(u, t_train_pt, order=1)
    u_x = derivative(u, x_train_pt, order=1)
    u_y = derivative(u, y_train_pt, order=1)
    u_xx = derivative(u_x, x_train_pt, order=1)
    u_yy = derivative(u_y, y_train_pt, order=1)
    
    v_t = derivative(v, t_train_pt, order=1)
    v_x = derivative(v, x_train_pt, order=1)
    v_y = derivative(v, y_train_pt, order=1)
    v_xx = derivative(v_x, x_train_pt, order=1)
    v_yy = derivative(v_y, y_train_pt, order=1)    
    
    p_x = derivative(p, x_train_pt, order=1)
    p_y = derivative(p, y_train_pt, order=1)
    
    f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy)
    f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
    
    return u, v, p, f_u, f_v

def mse(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt):
    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = f(model, x_train_pt, y_train_pt, t_train_pt)    
    loss = torch.sum((u_train_pt - u_pred) ** 2) + torch.sum((v_train_pt - v_pred) ** 2) + torch.sum((f_u_pred) ** 2) + torch.sum((f_v_pred) ** 2)
    return loss

def train_adam(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000):
    optimizer = torch.optim.Adam(list(model.parameters())+[lambda_1,lambda_2], lr=1e-3)
    global iter
     
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        loss = mse(model,x_train_pt, y_train_pt, t_train_pt,u_train_pt,v_train_pt)
        loss.backward(retain_graph=True)
        optimizer.step()
        lambda_1s.append(lambda_1.item())
        lambda_2s.append(lambda_2.item())
        error_lambda_1 = np.abs(lambda_1.detach().numpy() - 1.0)*100
        error_lambda_2 = np.abs(lambda_2.detach().numpy() - 0.01)/0.01 * 100
        results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
        iter += 1
        if iter % 100 == 0:
            torch.save(model.state_dict(), f'models/pt_model_NS_clean_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.detach().numpy().item()} - l2: {lambda_2.detach().numpy().item()}")
            

def train_lbfgs(model,x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000):
    optimizer = torch.optim.LBFGS(list(model.parameters())+[lambda_1,lambda_2],
                                    lr=1,
                                    max_iter=num_iter,
                                    max_eval=num_iter,
                                    tolerance_grad=1e-5,
                                    history_size=50,
                                    tolerance_change=1.0 * np.finfo(float).eps,
                                    line_search_fn="strong_wolfe")
 
    closure_fn = partial(closure, model, optimizer, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000)
    optimizer.step(closure_fn)

def closure(model, optimizer, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000):
    optimizer.zero_grad()
    loss = mse(model,x_train_pt, y_train_pt, t_train_pt,u_train_pt,v_train_pt)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    lambda_1s.append(lambda_1.item())
    lambda_2s.append(lambda_2.item())
    error_lambda_1 = np.abs(lambda_1.detach().numpy() - 1.0)*100
    error_lambda_2 = np.abs(lambda_2.detach().numpy() - 0.01)/0.01 * 100
    results.append([iter, loss.item(), error_lambda_1.item(),error_lambda_2.item()])
    if iter % 100 == 0:
        torch.save(model.state_dict(), f'models/pt_model_NS_clean_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.detach().numpy().item()} - l2: {lambda_2.detach().numpy().item()}")
    return loss


def plot_solution(X_star, u_star, index):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
                     
if __name__== "__main__":
    set_seed(42)
    iter = 0
    
    N_train = 5000
    results = []
    
    # Load Data
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
            
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T

    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T

    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1

    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    x_train_pt = torch.from_numpy(x_train)
    x_train_pt.requires_grad = True
    y_train_pt = torch.from_numpy(y_train)
    y_train_pt.requires_grad = True
    t_train_pt = torch.from_numpy(t_train)
    t_train_pt.requires_grad = True
    noise = 0.00       
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])    
    u_train_pt = torch.from_numpy(u_train)
    v_train_pt = torch.from_numpy(v_train)
    
    lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    lambda_2 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))    
    lambda_1s = []
    lambda_2s = []

    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')

    if not os.path.exists('training'):
        os.makedirs('training')
    
    model = NSNN()
    model.apply(init_weights)  
    
    # Entrenamiento con Adam
    start_time_adam = time.time()
    train_adam(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=10_000)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")
    
    # Entrenamiento con LBFGS
    start_time_lbfgs = time.time()
    train_lbfgs(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=10_000)
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
    with open('training/NS_training_summary_clean.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
        file.write(f"Final Loss: {final_loss:.6f}\n")
        file.write(f"Final L2: {final_l2:.6f}\n")
             
    results = np.array(results)
    lambda_1s = np.array(lambda_1s)
    lambda_2s = np.array(lambda_2s)
    np.savetxt("training/NS_training_data_clean.csv", results, delimiter=",", header="Iter,Loss,l1,l2", comments="")
    np.savetxt("training/lambda_1s_clean.csv", lambda_1s, delimiter=",", header="l1", comments="")    
    np.savetxt("training/lambda_2s_clean.csv", lambda_2s, delimiter=",", header="l2", comments="")    
    torch.save(model.state_dict(), f'NS_clean.pt')

    
    # Test Data
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]   
    
    x_star_pt = torch.from_numpy(x_star)
    x_star_pt.requires_grad = True
    y_star_pt = torch.from_numpy(y_star)
    y_star_pt.requires_grad = True
    t_star_pt = torch.from_numpy(t_star)
    t_star_pt.requires_grad = True    
    
    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = f(model, x_star_pt, y_star_pt, t_star_pt) 
    u_pred = u_pred.detach().numpy()
    v_pred = v_pred.detach().numpy()
    p_pred = p_pred.detach().numpy()
    lambda_1_value = lambda_1.detach().numpy()   
    lambda_2_value = lambda_2.detach().numpy()   
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)    
    error_lambda_1 = np.abs(lambda_1.detach().numpy() - 1.0)*100
    error_lambda_2 = np.abs(lambda_2.detach().numpy() - 0.01)/0.01 * 100
         
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))     
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))    
        
