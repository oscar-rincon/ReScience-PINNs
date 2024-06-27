# Extend system path to include the Utilities folder for additional modules
import sys
import os
import warnings
# Determine the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../Utilities')

# Change the working directory to the script's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

# Now import the pinns module
from pinns import *  # Importing Physics Informed Neural Networks utilities
from plotting import *  # Importing custom plotting utilities
 
# Import necessary libraries for deep learning, numerical computations, and data manipulation
import torch
import torch.nn as nn  # Neural network module
import numpy as np
import scipy.io as sp  # For loading MATLAB files
import pandas as pd  # For handling data structures
import matplotlib.pyplot as plt  # For plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable  # For advanced plot formatting

import imageio  # For reading and writing images
import os  # For handling file and directory paths

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

# Ensure the directories for saving figures exist, create them if they don't
if not os.path.exists('figures'):
    os.makedirs('figures')  # Directory for general figures

if not os.path.exists('figures_iters'):
    os.makedirs('figures_iters')  # Directory for figures from iterations

# Load training data from CSV
data = pd.read_csv('training/AC_training_data.csv')

# Create a figure with two subplots side by side
fig, axarr = plt.subplots(1, 2, figsize=figsize(1.0, 0.3, nplots=2))

# Plot Loss on the first subplot
axarr[0].semilogy(data['Iter'], data['Loss'], label='Loss', color='gray', linewidth=1)
axarr[0].set_xlabel('Iteration')  # X-axis label
axarr[0].set_ylabel('Loss')  # Y-axis label

# Plot L2 Error on the second subplot
axarr[1].semilogy(data['Iter'], data['L2'], label='L2 Error', color='gray', linewidth=1)
axarr[1].set_xlabel('Iteration')  # X-axis label
axarr[1].set_ylabel(r'$\mathrm{L}_{2}$')  # Y-axis label using LaTeX for L2

# Adjust layout to prevent overlap of subplots
plt.tight_layout()

# Save the figure to a PDF file
plt.savefig('figures/AC_training_curves.pdf')

# Define constants and load data
q = 100
lb = np.array([-1.0], dtype=np.float32)  # Lower bound
ub = np.array([1.0], dtype=np.float32)   # Upper bound
N = 200  # Number of points
iter = 0  # Initial iteration
data = sp.loadmat('../Data/AC.mat')  # Load dataset
t = data['tt'].flatten()[:, None].astype(np.float32)  # Time data
x = data['x'].flatten()[:, None].astype(np.float32)  # Spatial data
Exact = np.real(data['uu']).T.astype(np.float32)  # Exact solution
idx_t0 = 20  # Initial time index
idx_t1 = 180  # Final time index
dt = torch.from_numpy(t[idx_t1] - t[idx_t0]).to(torch.float32)  # Time step
noise_u0 = 0.0  # Initial noise level

# Select random points for initial condition
idx_x = np.random.choice(Exact.shape[1], N, replace=False)
x0 = x[idx_x,:]
x0 = torch.from_numpy(x0).to(torch.float32)
x0.requires_grad = True
u0 = Exact[idx_t0:idx_t0+1,idx_x].T
u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
u0 = torch.from_numpy(u0).to(torch.float32)

# Define boundary conditions
x1 = torch.from_numpy(np.vstack((lb, ub))).to(torch.float32)
x1.requires_grad = True

# Load IRK weights
tmp = np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2).astype(np.float32)
IRK_weights = torch.from_numpy(np.reshape(tmp[0:q**2+q], (q+1, q))).to(torch.float32).T

# Prepare data for model prediction
x_star = torch.from_numpy(x).to(torch.float32)

# Load and evaluate model
model_path = f'AC.pt'
model = MLP(input_size=1, output_size=101, hidden_layers=5, hidden_units=200, activation_function=nn.Tanh())
model.load_state_dict(torch.load(model_path))
model.eval()
U1_pred = model(x_star)

# Plotting
fig, ax = newfig(1.0, 1.2)
ax.axis('off')

# Setup grid for main plot
gs0 = plt.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

# Main plot
h = ax.imshow(Exact.T, interpolation='nearest', cmap='seismic',
              extent=[t.min(), t.max(), x_star.min(), x_star.max()],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

# Highlight specific times
line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[idx_t0]*np.ones((2, 1)), line, 'w-', linewidth=1)
ax.plot(t[idx_t1]*np.ones((2, 1)), line, 'w-', linewidth=1)

# Labels and title for main plot
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$u(t,x)$', fontsize=10)

# Setup grid for subplots
gs1 = plt.GridSpec(1, 2)
gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

# Subplot for initial time
ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[idx_t0,:], 'b-', linewidth = 2) 
ax.plot(x0.detach().numpy(), u0.detach().numpy(), 'rx', linewidth = 2, label = 'Data')      
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
ax.set_xlim([lb-0.1, ub+0.1])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

# Subplot for final time
ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
ax.plot(x_star.detach().numpy(), U1_pred[:, -1].detach().numpy(), 'r--', linewidth=2, label='Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize=10)
ax.set_xlim([lb-0.1, ub+0.1])
ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)

# Save figure
image_filename = f'figures/AC.pdf'
savefig(image_filename)

# Inicializar lista para almacenar imágenes
images = []

# Definir límite y paso para iteraciones
limit = 16_000
step = 1000

# Cargar y graficar modelos para cada iteración
for iter_num in range(step, limit, step):
    # Construir ruta del modelo y cargarlo
    model_path = f'models_iters/AC_{iter_num}.pt'
    model = MLP(input_size=1, output_size=101, hidden_layers=5, hidden_units=200, activation_function=nn.Tanh())
    model.load_state_dict(torch.load(model_path))
    model.eval()
    U1_pred = model(x_star)
    
    # Configurar figura y eje
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')
    
    # Configurar primera grilla de subplots
    gs0 = plt.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    # Mostrar imagen de datos exactos
    h = ax.imshow(Exact.T, interpolation='nearest', cmap='seismic',
                  extent=[t.min(), t.max(), x_star.min(), x_star.max()],
                  origin='lower', aspect='auto')
    # Configurar colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    # Dibujar líneas indicadoras de tiempo
    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[idx_t0]*np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[idx_t1]*np.ones((2, 1)), line, 'w-', linewidth=1)
    
    # Configurar etiquetas y título del eje
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize=10)
    
    # Configurar segunda grilla de subplots
    gs1 = plt.GridSpec(1, 2)
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)
    #print(t[idx_t1])
    # Subplot para tiempo t0
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[idx_t0,:], 'b-', linewidth=2) 
    ax.plot(x0.detach().numpy(), u0.detach().numpy(), 'rx', linewidth=2, label='Data')      
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title(f'$t = {t[idx_t0][0]:.2f}$', fontsize=10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False)

    # Subplot para tiempo t1
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[idx_t1, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x_star.detach().numpy(), U1_pred[:, -1].detach().numpy(), 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title(f'$t = {t[idx_t1][0]:.2f}$', fontsize=10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
    
    # Guardar figura y cerrar
    image_filename = f'figures_iters/AC_{iter_num}.png'
    savefig(image_filename)
    plt.close(fig)

# Reinicializar lista de imágenes
images = []

# Cargar imágenes guardadas y crear GIF
for i in range(step, limit, step):
    image_path = f'figures_iters/AC_{i}.png'
    images.append(imageio.imread(image_path))

# Guardar GIF
imageio.mimsave('figures/AC.gif', images, fps=5)