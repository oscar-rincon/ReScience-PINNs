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
from plotting import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import imageio
import math
import warnings

warnings.filterwarnings("ignore")


if not os.path.exists('figures'):
    os.makedirs('figures')
    
if not os.path.exists('figures_iters'):
    os.makedirs('figures_iters')    
    

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


# Load the data
lambda_1_values_clean = pd.read_csv('training/lambda_1s_clean.csv')
lambda_2_values_clean = pd.read_csv('training/lambda_2s_clean.csv')
lambda_1_values_noisy = pd.read_csv('training/lambda_1s_noisy.csv')
lambda_2_values_noisy = pd.read_csv('training/lambda_2s_noisy.csv')
KdV_training_data_clean = pd.read_csv('training/KdV_training_data_clean.csv')
KdV_training_data_noisy = pd.read_csv('training/KdV_training_data_noisy.csv')

# Create subplots
fig, axarr  = newfig(0.8, 0.8)

# Plot clean loss curve
axarr.semilogy(KdV_training_data_clean['Iter'], KdV_training_data_clean['Loss'], label='Clean', color='blue', linewidth=1)

# Plot noisy loss curve
axarr.semilogy(KdV_training_data_noisy['Iter'], KdV_training_data_noisy['Loss'], label='Noisy', color='red', linewidth=1)

axarr.set_xlabel('Iteration')
axarr.set_ylabel('Loss')
axarr.legend(frameon=False)

plt.tight_layout()
plt.savefig('figures/KdV_combined_loss_curve.pdf')

# Configuración de la figura
fig, axs = plt.subplots(1, 2, figsize=figsize(1.0, 0.3, nplots=2))

# Primer subplot
axs[0].plot(KdV_training_data_clean['Iter'], lambda_1_values_clean.values, label='Clean', color='blue', linewidth=1)
axs[0].plot(KdV_training_data_noisy['Iter'], lambda_1_values_noisy.values, label='Noisy', color='red', linewidth=1)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel(r'$\lambda_{1}$')

# Segundo subplot
axs[1].plot(KdV_training_data_clean['Iter'], lambda_2_values_clean.values, label=f'Clean', color='blue', linewidth=1)
axs[1].plot(KdV_training_data_noisy['Iter'], lambda_2_values_noisy.values, label=f'Noisy', color='red', linewidth=1)
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel(r'$\lambda_{2}$')
axs[1].legend(frameon=False)

plt.tight_layout()
plt.savefig('figures/KdV_lambda_curves.pdf')
    
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
 
lambda_1_value = lambda_1_values_clean['Lambda1'].iloc[-1] if isinstance(lambda_1_values_clean['Lambda1'], pd.Series) else lambda_1_values_clean['Lambda1'][-1]
lambda_2_value = lambda_2_values_clean['Lambda2'].iloc[-1] if isinstance(lambda_2_values_clean['Lambda2'], pd.Series) else lambda_2_values_clean['Lambda2'][-1]
lambda_1_value_noisy = lambda_1_values_noisy['Lambda1'].iloc[-1] if isinstance(lambda_1_values_noisy['Lambda1'], pd.Series) else lambda_1_values_noisy['Lambda1'][-1]
lambda_2_value_noisy = lambda_2_values_noisy['Lambda2'].iloc[-1] if isinstance(lambda_2_values_noisy['Lambda2'], pd.Series) else lambda_2_values_noisy['Lambda2'][-1]

images = []

model_path = f'KdV_clean.pt'
model = KdVNN()
model.load_state_dict(torch.load(model_path))
model.eval()

U0_pred = net_U0(model, x0_pt, torch.tensor(lambda_1_value), torch.tensor(lambda_2_value), dt, IRK_alpha) 
U0_pred = U0_pred.detach().numpy()
U1_pred = net_U1(model, x1_pt, torch.tensor(lambda_1_value), torch.tensor(lambda_2_value), dt, IRK_alpha, IRK_beta) 
U1_pred = U1_pred.detach().numpy() 

noise = 0.01    

idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
x0 = x_star[idx_x,:]
u0 = Exact[idx_x,idx_t][:,None] 
u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    
idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
x1 = x_star[idx_x,:]
u1 = Exact[idx_x,idx_t + skip][:,None]
u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])

model_path = f'KdV_noisy.pt'
model = KdVNN()
model.load_state_dict(torch.load(model_path))
model.eval()

U0_pred = net_U0(model, x0_pt, torch.tensor(lambda_1_value_noisy), torch.tensor(lambda_2_value_noisy), dt, IRK_alpha) 
U0_pred = U0_pred.detach().numpy()
U1_pred = net_U1(model, x1_pt, torch.tensor(lambda_1_value_noisy), torch.tensor(lambda_2_value_noisy), dt, IRK_alpha, IRK_beta) 
U1_pred = U1_pred.detach().numpy() 


######################################################################
############################# Plotting ###############################
######################################################################

fig, ax = newfig(1.0, 1.5)
ax.axis('off')

gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3+0.05, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])
    
h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow',
                extent=[t_star.min(),t_star.max(), lb[0], ub[0]],
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x_star.min(), x_star.max(), 2)[:,None]
ax.plot(t_star[idx_t]*np.ones((2,1)), line, 'w-', linewidth = 1.0)
ax.plot(t_star[idx_t + skip]*np.ones((2,1)), line, 'w-', linewidth = 1.0)    
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$u(t,x)$', fontsize = 10)

gs1 = gridspec.GridSpec(1, 2)
gs1.update(top=1-1/3-0.1, bottom=1-2/3, left=0.15, right=0.85, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x_star,Exact[:,idx_t][:,None], 'b', linewidth = 2, label = 'Exact')
ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t], u0.shape[0]), fontsize = 10)

ax = plt.subplot(gs1[0, 1])
ax.plot(x_star,Exact[:,idx_t + skip][:,None], 'b', linewidth = 2, label = 'Exact')
ax.plot(x1, u1, 'rx', linewidth = 2, label = 'Data')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t+skip], u1.shape[0]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

gs2 = gridspec.GridSpec(1, 2)
gs2.update(top=1-2/3-0.05, bottom=0, left=0.15, right=0.85, wspace=0.0)

ax = plt.subplot(gs2[0, 0])
ax.axis('off')
s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x + 0.0025 u_{xxx} = 0$ \\  \hline Identified PDE (clean data) & '
s2 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
s3 = r'Identified PDE (1\% noise) & '
s4 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
s5 = r'\end{tabular}$'
s = s1+s2+s3+s4+s5
ax.text(-0.24,0.42,s)

savefig('./figures/KdV.pdf') 


# Generate images for GIF
 
model_dir = 'models_iters/'
image_dir = 'figures_iters/'
gif_filename = 'figures/KdV.gif'


# Definir el límite
limite = 10_501
step = 100
# Cargar y graficar modelos
for iter_num in range(step, limite, step):
    # Obtener los índices correctos para lambda_1_values_clean
    if iter_num > lambda_1_values_clean['Lambda1'].index[-1]:
        iter_num_clean = math.floor(lambda_1_values_clean['Lambda1'].index[-1] / 1000) * 1000
    else:
        iter_num_clean = iter_num 

    # Obtener los índices correctos para lambda_1_values_noisy
    if iter_num > lambda_1_values_noisy['Lambda1'].index[-1]:
        iter_num_noisy = math.floor(lambda_1_values_noisy['Lambda1'].index[-1] / 1000) * 1000
    else:
        iter_num_noisy = iter_num 
    
    # Obtener los valores lambda
    lambda_1_value = lambda_1_values_clean['Lambda1'][iter_num_clean]
    lambda_2_value = lambda_2_values_clean['Lambda2'][iter_num_clean]
    lambda_1_value_noisy = lambda_1_values_noisy['Lambda1'][iter_num_noisy]
    lambda_2_value_noisy = lambda_2_values_noisy['Lambda2'][iter_num_noisy]
    
    fig, ax = newfig(1.0, 1.5)
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3+0.05, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
        
    h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow',
                    extent=[t_star.min(),t_star.max(), lb[0], ub[0]],
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(x_star.min(), x_star.max(), 2)[:,None]
    ax.plot(t_star[idx_t]*np.ones((2,1)), line, 'w-', linewidth = 1.0)
    ax.plot(t_star[idx_t + skip]*np.ones((2,1)), line, 'w-', linewidth = 1.0)    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize = 10)

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-1/3-0.1, bottom=1-2/3, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_star,Exact[:,idx_t][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t], u0.shape[0]), fontsize = 10)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_star,Exact[:,idx_t + skip][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x1, u1, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t+skip], u1.shape[0]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)

    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1-2/3-0.05, bottom=0, left=0.15, right=0.85, wspace=0.0)

    ax = plt.subplot(gs2[0, 0])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x + 0.0025 u_{xxx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1+s2+s3+s4+s5
    ax.text(-0.24,0.42,s)    
    
    image_filename = f'./figures_iters/KdV_{iter_num}.png'
    savefig(image_filename) 
     
# Create GIF
images = []
for i in range(step, limite, step):
    image_path = os.path.join(image_dir, f'KdV_{i}.png')
    images.append(imageio.imread(image_path))

imageio.mimsave(gif_filename, images, fps=5) 