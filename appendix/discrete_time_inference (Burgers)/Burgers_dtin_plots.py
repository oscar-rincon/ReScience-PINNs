import sys
import os
import time
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
 
# Import third-party libraries for numerical and scientific computing
import torch  # PyTorch library for deep learning
import numpy as np  # NumPy library for numerical operations
import scipy.io  # SciPy module for MATLAB file I/O
import pandas as pd  # Pandas library for data manipulation and analysis
from scipy.interpolate import griddata  # Interpolation tool from SciPy

# Import third-party libraries for visualization
import matplotlib.pyplot as plt  # Matplotlib library for plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Tools for plot arrangement
import matplotlib.gridspec as gridspec  # Grid layout for subplots
import imageio  # Library for reading and writing a wide range of image data

# Import utilities for experimental design
from pyDOE import lhs  # Design of experiments, including Latin Hypercube Sampling
 
# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

# Create directories for storing figures if they do not already exist
if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('figures_iters'):
    os.makedirs('figures_iters')

# Load training data from CSV file
data = pd.read_csv('training/Burgers_dtin_training_data.csv')

# Create a figure with 2 subplots arranged horizontally
fig, axarr = plt.subplots(1, 2, figsize=figsize(1.0, 0.3, nplots=2))

# Plot 1: Training Loss over Iterations
# Plotting the loss on a semilogarithmic scale for better visualization
axarr[0].semilogy(data['Iter'], data['Loss'], label='Loss', color='gray', linewidth=1)
axarr[0].set_xlabel('Iteration')  # X-axis label
axarr[0].set_ylabel('Loss')  # Y-axis label

# Plot 2: L2 Error over Iterations
# Plotting the L2 error on a semilogarithmic scale as well
axarr[1].semilogy(data['Iter'], data['L2'], label='L2 Error', color='gray', linewidth=1)
axarr[1].set_xlabel('Iteration')  # X-axis label
axarr[1].set_ylabel(r'$\mathrm{L}_2$')  # Y-axis label using LaTeX for L2

# Adjust layout to prevent subplot overlap
plt.tight_layout()

# Save the figure to a PDF in the specified directory
plt.savefig('figures/Burgers_dtin_training_curves.pdf')   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

q = 500
noise = 0.0  # Noise level (unused)
N_u = 100  # Number of training points for u
N_f = 10_000  # Number of training points for f
N = 250
lb = np.array([-1.0])
ub = np.array([1.0])    
data = scipy.io.loadmat('../Data/burgers_shock.mat')
t = data['t'].flatten()[:,None] # T x 1
x = data['x'].flatten()[:,None] # N x 1
Exact = np.real(data['usol']).T.astype(np.float32) # T x N
idx_t0 = 10
idx_t1 = 90
dt = torch.from_numpy(t[idx_t1] - t[idx_t0]).to(torch.float32)  # Time step size

# Initial data
noise_u0 = 0.0
idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
x0 = x[idx_x,:]
u0 = Exact[idx_t0:idx_t0+1,idx_x].T
u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])

    
# Boudanry data
x1 = np.vstack((lb,ub))

# Test data
x_star = x

# Load IRK weights
tmp = np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2).astype(np.float32)
IRK_weights = torch.from_numpy(np.reshape(tmp[0:q**2+q], (q+1, q))).to(torch.float32).T
IRK_weights = IRK_weights.to(device)  # Move IRK weights for numerical solver

# Convert to tensors and set requires_grad for training with float precision
x0 = torch.from_numpy(x0).float().to(device)
x0.requires_grad = True
x1 = torch.from_numpy(x1).float().to(device)
x1.requires_grad = True
u0_real = torch.from_numpy(u0).float().to(device)
dt = dt.to(device)  # Move time step size
x_star = torch.from_numpy(x_star).float().to(device)
x_star.requires_grad = True    
 
# Initialize the model and apply initial weights
model = MLP(input_size=1, output_size=q+1, hidden_layers=4, hidden_units=50, activation_function=nn.Tanh()).float().to(device)
model_path = 'Burgers_dtin.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

U1_pred = model(x_star)
U1_pred = U1_pred.cpu().detach().numpy()

error = np.linalg.norm(U1_pred[:,-1] - Exact[idx_t1,:], 2)/np.linalg.norm(Exact[idx_t1,:], 2)
print('Error: %e' % (error))

######################################################################
############################# Plotting ###############################
######################################################################    

fig, ax = newfig(1.0, 1.2)
ax.axis('off')

####### Row 0: h(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)
    
line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[idx_t0]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[idx_t1]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
leg = ax.legend(frameon=False, loc = 'best')
ax.set_title('$u(t,x)$', fontsize = 10)


####### Row 1: h(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 2)
gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[idx_t0,:], 'b-', linewidth = 2) 
ax.plot(x0.cpu().detach().numpy(), u0, 'rx', linewidth = 2, label = 'Data')      
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
ax.set_xlim([lb-0.1, ub+0.1])
ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)


ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact[idx_t1,:], 'b-', linewidth = 2, label = 'Exact') 
ax.plot(x_star.cpu().detach().numpy(), U1_pred[:,-1], 'r--', linewidth = 2, label = 'Prediction')      
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize = 10)    
ax.set_xlim([lb-0.1, ub+0.1])

ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
    

plt.savefig('./figures/Burgers_dtin.pdf')  

# Generate images for GIF
 
model_dir = 'models_iters/'
image_dir = 'figures_iters/'
gif_filename = 'figures/Burgers_dtin.gif'
limit = 6_100
step = 1_00

for i in range(step, limit, step):
    # Initialize the model and apply initial weights
    model = MLP(input_size=1, output_size=q+1, hidden_layers=4, hidden_units=50, activation_function=nn.Tanh()).float().to(device)
    model_path = os.path.join(model_dir, f'Burgers_dtin_{i}.pt')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    U1_pred = model(x_star)
    U1_pred = U1_pred.cpu().detach().numpy()

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')

    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(Exact.T, interpolation='nearest', cmap='rainbow', 
                    extent=[t.min(), t.max(), x.min(), x.max()], 
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
        
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[idx_t0]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[idx_t1]*np.ones((2,1)), line, 'w-', linewidth = 1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)

    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[idx_t0,:], 'b-', linewidth = 2) 
    ax.plot(x0.cpu().detach().numpy(), u0, 'rx', linewidth = 2, label = 'Data')      
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[idx_t0]), fontsize = 10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)


    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[idx_t1,:], 'b-', linewidth = 2, label = 'Exact') 
    ax.plot(x_star.cpu().detach().numpy(), U1_pred[:,-1], 'r--', linewidth = 2, label = 'Prediction')      
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t[idx_t1]), fontsize = 10)    
    ax.set_xlim([lb-0.1, ub+0.1])

    ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)

    # Save the figure
    image_path = os.path.join(image_dir, f'Burgers_dtin_{i}.png')
    plt.savefig(image_path)
    plt.close()    

# Create GIF
images = []

for i in range(step, limit, step):
    image_path = os.path.join(image_dir, f'Burgers_dtin_{i}.png')
    images.append(imageio.imread(image_path))

imageio.mimsave(gif_filename, images, fps=7)  
