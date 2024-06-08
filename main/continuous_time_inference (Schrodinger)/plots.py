import sys
import os
import numpy as np
import scipy.io as sp
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from plotting import *  # Assuming this module contains custom plotting functions
import imageio
from scipy.interpolate import griddata
from pyDOE import lhs
import torch
import torch.nn as nn
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add utilities path
sys.path.insert(0, '../../Utilities/')

# Check and create directories for saving figures
if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('figures_iters'):
    os.makedirs('figures_iters')

# Define the Schrodinger neural network model
class SchrodingerNN(nn.Module):
    def __init__(self):
        super(SchrodingerNN, self).__init__()
        # Define layers
        self.linear_in = nn.Linear(2, 100)
        self.linear_out = nn.Linear(100, 2)
        self.layers = nn.ModuleList([nn.Linear(100, 100) for i in range(5)])
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x

# Load training data
data = pd.read_csv('training/Schrodinger_training_data.csv')

# Plot training curves
plt.figure(figsize=(6.0, 2.2))

plt.subplot(1, 2, 1)
plt.semilogy(data['Iter'], data['Loss'], label='Loss', color='gray', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.semilogy(data['Iter'], data['L2'], label='L2 Error', color='gray', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('$L_{2}$')
plt.tight_layout()  # Add this line to make the layout tight

savefig('figures/Schrodinger_training_curves.pdf')

# Define problem domain
lb = np.array([-5.0, 0.0])
ub = np.array([5.0, np.pi/2])

# Define number of initial, boundary, and collocation points
N0 = 50  
N_b = 50  
N_f = 20_000 

# Load Schrodinger equation data
data = sp.loadmat('../Data/NLS.mat')
x_0 = torch.from_numpy(data['x']).flatten().T
t = torch.from_numpy(data['tt']).flatten().T
h = torch.from_numpy(data['uu'])
u_0 = torch.real(h)[:, 0]
v_0 = torch.imag(h)[:, 0]
h_0 = torch.stack((u_0, v_0), axis=1)

# Generate random points for initial and boundary conditions
idx_0 = np.random.choice(x_0.shape[0], N0, replace=False)
x_0 = x_0[idx_0]
u_0 = u_0[idx_0]
v_0 = v_0[idx_0]
h_0 = h_0[idx_0]
idx_b = np.random.choice(t.shape[0], N_b, replace=False)
t_b = t[idx_b]

# Generate collocation points
c_f = lb + (ub-lb)*lhs(2, N_f)
x_f = torch.from_numpy(c_f[:, 0]).requires_grad_(True)
t_f = torch.from_numpy(c_f[:, 1]).requires_grad_(True)

# Load pre-trained model
model = SchrodingerNN()
model.load_state_dict(torch.load(f'Schrodinger.pt'))
model.eval()

# Evaluate model
with torch.no_grad():
    usol = model(torch.cat((x_f.reshape(-1, 1), t_f.reshape(-1, 1)), 1))

    # Extract predictions
    h_pred = (usol[:, 0]**2 + usol[:, 1]**2)**0.5
    u_pred = usol[:, 0]
    v_pred = usol[:, 1]

    # Generate meshgrid for plotting
    X, T = torch.meshgrid(torch.tensor(data['x'].flatten()[:]), torch.tensor(data['tt'].flatten()[:]))
    xcol = X.reshape(-1, 1)
    tcol = T.reshape(-1, 1)
    X_star = torch.cat((xcol, tcol), 1).float()

    # Interpolate predictions for plotting
    U_pred = griddata(X_star, u_pred.detach().numpy().flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.detach().numpy().flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.detach().numpy().flatten(), (X, T), method='cubic')

    # Plotting
    fig, ax = newfig(1.2, 0.9)
    ax.axis('off')

    # Plot the heatmap
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    h = ax.imshow(H_pred, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    # Plot data points
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)

    # Add dashed lines
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'upper right', fontsize=8)
    ax.set_title('$|h(t,x)|$', fontsize = 10)

    # Plotting h(t,x) slices
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    # Plot h(t,x) at different times
    for idx, time_idx in enumerate([75, 100, 125]):
        ax = plt.subplot(gs1[0, idx])
        ax.plot(x,Exact_h[:,time_idx], 'b-', linewidth = 2, label = 'Exact')       
        ax.plot(x,Unp[:,time_idx], 'r--', linewidth = 2, label = 'Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')    
        ax.set_title('$t = %.2f$' % (t[time_idx]), fontsize = 10)
        ax.axis('square')
        ax.set_xlim([-5.1,5.1])
        ax.set_ylim([-0.1,5.1])

    # Save the figure
    plt.savefig('Schrodinger.pdf')
    plt.close()

# Generate images for GIF
mat_file = '../Data/NLS.mat'
model_dir = 'models_iters/'
image_dir = 'figures_iters/'
gif_filename = 'figures/Schrodinger.gif'

for i in range(1000, 57_001, 1000):
    model = SchrodingerNN()
    model_path = os.path.join(model_dir, f'Schrodinger_{i}.pt')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        usol = model(X_star)

        h_pred = (usol[:, 0]**2 + usol[:, 1]**2)**0.5
        u_pred = usol[:, 0]
        v_pred = usol[:, 1]

        # Interpolate predictions for plotting
        U_pred = griddata(X_star, u_pred.detach().numpy().flatten(), (X, T), method='cubic')
        V_pred = griddata(X_star, v_pred.detach().numpy().flatten(), (X, T), method='cubic')
        H_pred = griddata(X_star, h_pred.detach().numpy().flatten(), (X, T), method='cubic')

        # Plotting
        fig, ax = newfig(1.2, 0.9)
        ax.axis('off')

        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :])
        h = ax.imshow(H_pred, interpolation='nearest', cmap='YlGnBu', 
                      extent=[lb[1], ub[1], lb[0], ub[0]], 
                      origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)

        # Plot data points
        ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)

        # Add dashed lines
        line = np.linspace(x.min(), x.max(), 2)[:,None]
        ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
        ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
        ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    

        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        leg = ax.legend(frameon=False, loc = 'upper right', fontsize=8)
        ax.set_title('$|h(t,x)|$', fontsize = 10)

        gs1 = gridspec.GridSpec(1, 3)
        gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

        for idx, time_idx in enumerate([75, 100, 125]):
            ax = plt.subplot(gs1[0, idx])
            ax.plot(x,Exact_h[:,time_idx], 'b-', linewidth = 2, label = 'Exact')       
            ax.plot(x,Unp[:,time_idx], 'r--', linewidth = 2, label = 'Prediction')
            ax.set_xlabel('$x$')
            ax.set_ylabel('$|h(t,x)|$')    
            ax.set_title('$t = %.2f$' % (t[time_idx]), fontsize = 10)
            ax.axis('square')
            ax.set_xlim([-5.1,5.1])
            ax.set_ylim([-0.1,5.1])

        # Save the figure
        image_path = os.path.join(image_dir, f'Schrodinger_{i}.png')
        plt.savefig(image_path)
        plt.close()

# Create GIF
images = []
for i in range(1000, 57_001, 1000):
    image_path = os.path.join(image_dir, f'Schrodinger_{i}.png')
    images.append(imageio.imread(image_path))

imageio.mimsave(gif_filename, images, fps=10)
