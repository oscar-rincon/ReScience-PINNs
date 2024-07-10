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
data = pd.read_csv('training/Burgers_ctin_training_data.csv')

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
plt.savefig('figures/Burgers_ctin_training_curves.pdf')   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

nu = 0.01 / np.pi  # Viscosity
noise = 0.0  # Noise level (unused)
N_u = 100  # Number of training points for u
N_f = 10_000  # Number of training points for f

# Load data
data = scipy.io.loadmat('../Data/burgers_shock.mat')
t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T

# Prepare training data
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None].T

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

# Boundary and initial conditions
xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
uu1 = Exact[0:1, :].T
xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
uu2 = Exact[:, 0:1]
xx3 = np.hstack((X[:, -1:], T[:, -1:]))
uu3 = Exact[:, -1:]

# Combine and select training points
X_u_train = np.vstack([xx1, xx2, xx3])
X_f_train = lb + (ub - lb) * lhs(2, N_f)
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([uu1, uu2, uu3])
idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
X_u_train = X_u_train[idx, :]
u_train = u_train[idx, :]

# Convert to tensors and set requires_grad for training
x_u = torch.from_numpy(X_u_train[:, 0:1].astype(np.float32)).to(device)
x_u.requires_grad = True
x_f = torch.from_numpy(X_f_train[:, 0:1].astype(np.float32)).to(device)
x_f.requires_grad = True
t_u = torch.from_numpy(X_u_train[:, 1:2].astype(np.float32)).to(device)
t_u.requires_grad = True
t_f = torch.from_numpy(X_f_train[:, 1:2].astype(np.float32)).to(device)
t_f.requires_grad = True
u_train_pt = torch.from_numpy(u_train).float().to(device)
nu = torch.tensor(nu).float().to(device)
x_star = torch.from_numpy(X_star[:, 0:1]).float().to(device)
x_star.requires_grad = True
t_star = torch.from_numpy(X_star[:, 1:2]).float().to(device)
t_star.requires_grad = True
u_star = torch.from_numpy(u_star).T.float().to(device)

# Initialize the model and apply initial weights
model = MLP(input_size=2, output_size=1, hidden_layers=8, hidden_units=20, activation_function=nn.Tanh()).to(device)
model_path = 'Burgers_ctin.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

u_pred = model(torch.cat((x_star, t_star), dim=1)) 

U_pred = griddata(torch.cat((x_star, t_star), dim=1).cpu().detach().numpy(), u_pred.flatten().cpu().detach().numpy(), (X, T), method='cubic')
Error = np.abs(Exact - U_pred)
 
min_value = np.min(U_pred)
max_value = np.max(U_pred)

######################################################################
############################# Plotting ###############################
######################################################################    

fig, ax = newfig(1.0, 1.1)
ax.axis('off')

####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(frameon=False, loc = 'best')
ax.set_title('$u(t,x)$', fontsize = 10)

####### Row 1: u(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')    
ax.set_title('$t = 0.25$', fontsize = 10)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 0.50$', fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])    
ax.set_title('$t = 0.75$', fontsize = 10)

image_path = f'figures/Burgers_ctin.pdf'
plt.savefig(image_path)



# Generate images for GIF
 
model_dir = 'models_iters/'
image_dir = 'figures_iters/'
gif_filename = 'figures/Burgers_ctin.gif'
limit = 4000
step = 100

for i in range(step, limit, step):
    model = MLP(input_size=2, output_size=1, hidden_layers=8, hidden_units=20, activation_function=nn.Tanh()).to(device)
    model_path = os.path.join(model_dir, f'Burgers_ctin_{i}.pt')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    u_pred = model(torch.cat((x_star, t_star), dim=1)) 

    U_pred = griddata(torch.cat((x_star, t_star), dim=1).cpu().detach().numpy(), u_pred.flatten().cpu().detach().numpy(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    


    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')

    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                    extent=[t.min(), t.max(), x.min(), x.max()], 
                    origin='lower', aspect='auto', vmin=min_value, vmax=max_value)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)

    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75$', fontsize = 10)


    # Save the figure
    image_path = os.path.join(image_dir, f'Burgers_ctin_{i}.png')
    plt.savefig(image_path)
    plt.close()

# Create GIF
images = []

for i in range(step, limit, step):
    image_path = os.path.join(image_dir, f'Burgers_ctin_{i}.png')
    images.append(imageio.imread(image_path))

imageio.mimsave(gif_filename, images, fps=3)    
    