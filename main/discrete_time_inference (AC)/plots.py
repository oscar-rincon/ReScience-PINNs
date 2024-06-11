import sys
sys.path.insert(0, '../../Utilities/')
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sp
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotting import * 
import imageio
import os 
 
warnings.filterwarnings("ignore")

if not os.path.exists('figures'):
    os.makedirs('figures')
    
if not os.path.exists('figures_iters'):
    os.makedirs('figures_iters')    

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

data = pd.read_csv('training/AC_training_data.csv')

fig, axarr = plt.subplots(1, 2, figsize=figsize(1.0, 0.3, nplots=2))  # Dos subplots en una fila

# Subplot 1: Loss
axarr[0].semilogy(data['Iter'], data['Loss'], label='Loss', color='gray', linewidth=1)
axarr[0].set_xlabel('Iteration')
axarr[0].set_ylabel('Loss')

# Subplot 2: L2 Error
axarr[1].semilogy(data['Iter'], data['L2'], label='L2 Error', color='gray', linewidth=1)
axarr[1].set_xlabel('Iteration')
axarr[1].set_ylabel(r'$\mathrm{L}_{2}$')
plt.tight_layout()  # Adjusting layout so subplots don't overlap

plt.savefig('figures/AC_training_curves.pdf')

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



model_path = f'AC.pt'
model = ACNN()
model.load_state_dict(torch.load(model_path))
model.eval()
U1_pred = model(x_star)

fig, ax = newfig(1.0, 1.2)
ax.axis('off')

gs0 = plt.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(Exact.T, interpolation='nearest', cmap='seismic',
                extent=[t.min(), t.max(), x_star.min(), x_star.max()],
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[idx_t0]*np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[idx_t1]*np.ones((2, 1)), line, 'w-', linewidth=1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$u(t,x)$', fontsize=10)

gs1 = plt.GridSpec(1, 2)
gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[idx_t0,:], 'b-', linewidth = 2) 
ax.plot(x0.detach().numpy(), u0.detach().numpy(), 'rx', linewidth = 2, label = 'Data')      
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
ax.set_xlim([lb-0.1, ub+0.1])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)


ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
ax.plot(x_star.detach().numpy(), U1_pred[:, -1].detach().numpy(), 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize=10)
ax.set_xlim([lb-0.1, ub+0.1])
ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)

image_filename = f'figures/AC.pdf'
savefig(image_filename)

images = []
limit = 9_801
step = 100
# Cargar y graficar modelos
for iter_num in range(step, limit, step):
    model_path = f'models_iters/AC_{iter_num}.pt'
    model = ACNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    U1_pred = model(x_star)
    
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')
    
    gs0 = plt.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(Exact.T, interpolation='nearest', cmap='seismic',
                  extent=[t.min(), t.max(), x_star.min(), x_star.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[idx_t0]*np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[idx_t1]*np.ones((2, 1)), line, 'w-', linewidth=1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize=10)
    
    gs1 = plt.GridSpec(1, 2)
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[idx_t0,:], 'b-', linewidth = 2) 
    ax.plot(x0.detach().numpy(), u0.detach().numpy(), 'rx', linewidth = 2, label = 'Data')      
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x_star.detach().numpy(), U1_pred[:, -1].detach().numpy(), 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize=10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
    
    image_filename = f'figures_iters/AC_{iter_num}.png'
    savefig(image_filename)
    plt.close(fig)

images = []

for i in range(step, limit, step):
    image_path = f'figures_iters/AC_{i}.png'
    images.append(imageio.imread(image_path))

imageio.mimsave('figures/AC.gif', images, fps=5)