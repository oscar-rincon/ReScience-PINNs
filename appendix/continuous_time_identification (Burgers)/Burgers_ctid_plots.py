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
 
# Suppress warnings to clean up output
warnings.filterwarnings("ignore")

# Third-party library imports for numerical and scientific computing
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import pandas as pd
import math

# PyTorch imports for deep learning
import torch
import torch.nn as nn

# Imports for design of experiments and statistical analysis
from pyDOE import lhs

# Matplotlib imports for plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import for creating GIFs from images
import imageio


# Create directories for saving figures if they do not already exist
if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('figures_iters'):
    os.makedirs('figures_iters')


# Load the data
lambda_1_values_clean = pd.read_csv('training/lambda_1s_clean.csv')
lambda_2_values_clean = pd.read_csv('training/lambda_2s_clean.csv')
lambda_1_values_noisy = pd.read_csv('training/lambda_1s_noisy.csv')
lambda_2_values_noisy = pd.read_csv('training/lambda_2s_noisy.csv')
NS_training_data_clean = pd.read_csv('training/Burgers_ctid_clean_training_data.csv')
NS_training_data_noisy = pd.read_csv('training/Burgers_ctid_noisy_training_data.csv')

# Create subplots for loss curves
fig, axarr  = newfig(0.8, 0.8)

# Plot clean loss curve
axarr.semilogy(NS_training_data_clean['Iter'], NS_training_data_clean['Loss'], label='Clean', color='blue', linewidth=1)

# Plot noisy loss curve
axarr.semilogy(NS_training_data_noisy['Iter'], NS_training_data_noisy['Loss'], label='Noisy', color='red', linewidth=1)

axarr.set_xlabel('Iteration')
axarr.set_ylabel('Loss')
axarr.legend(frameon=False)

plt.tight_layout()
plt.savefig('figures/Burgers_ctid_combined_loss_curve.pdf')

# Set up figure configuration
fig, axs = plt.subplots(1, 2, figsize=figsize(1.0, 0.3, nplots=2))

# First subplot for lambda_1 curves
axs[0].plot(NS_training_data_clean['Iter'], lambda_1_values_clean.values, label='Clean', color='blue', linewidth=1)
axs[0].plot(NS_training_data_noisy['Iter'], lambda_1_values_noisy.values, label='Noisy', color='red', linewidth=1)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel(r'$\lambda_{1}$')

# Second subplot for lambda_2 curves
axs[1].plot(NS_training_data_clean['Iter'], lambda_2_values_clean.values, label='Clean', color='blue', linewidth=1)
axs[1].plot(NS_training_data_noisy['Iter'], lambda_2_values_noisy.values, label='Noisy', color='red', linewidth=1)
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel(r'$\lambda_{2}$')

# Add legend to the subplots
axs[0].legend(frameon=False)
axs[1].legend(frameon=False)

plt.tight_layout()
plt.savefig('figures/Burgers_ctid_lambda_curves.pdf')    



# Set a fixed seed for reproducibility
set_seed(42)

# Check GPU availability and select device
device = torch.device('cuda')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create directories for storing models and training data if they don't exist
if not os.path.exists('models_iters'):
    os.makedirs('models_iters')
if not os.path.exists('training'):
    os.makedirs('training')

# Initialize variables
iter = 0  # Initialize iteration counter
nu = 0.01/np.pi

N_u = 2000
    
data = scipy.io.loadmat('../Data/burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]              

# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)    


noise = 0.01            
            
idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx,:]
u_train = u_star[idx,:]
u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])   

# Convert to tensors and set requires_grad for training with float precision
x_u = torch.from_numpy(X_u_train[:, 0:1]).float().to(device)
x_u.requires_grad = True
#x_f = torch.from_numpy(X_f_train[:, 0:1]).float().to(device)
#x_f.requires_grad = True
t_u = torch.from_numpy(X_u_train[:, 1:2]).float().to(device)
t_u.requires_grad = True
#t_f = torch.from_numpy(X_f_train[:, 1:2]).float().to(device)
#t_f.requires_grad = True
u_train_pt = torch.from_numpy(u_train).float().to(device)
nu = torch.tensor(nu).float().to(device)
x_star = torch.from_numpy(X_star[:, 0:1]).float().to(device)
x_star.requires_grad = True
t_star = torch.from_numpy(X_star[:, 1:2]).float().to(device)
t_star.requires_grad = True
u_star = torch.from_numpy(u_star).T.float().to(device)


# Initialize the model and apply initial weights
model = MLP(input_size=2, output_size=1, hidden_layers=8, hidden_units=20, activation_function=nn.Tanh()).to(device)
model_path = 'Burgers_ctid.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

U_pred = model(torch.cat((x_star, t_star), dim=1)) 

lambda_1_value = lambda_1_values_clean['l1'].iloc[-1] if isinstance(lambda_1_values_clean['l1'], pd.Series) else lambda_1_values_clean['l1'][-1]
lambda_2_value = lambda_2_values_clean['l2'].iloc[-1] if isinstance(lambda_2_values_clean['l2'], pd.Series) else lambda_2_values_clean['l2'][-1]
lambda_1_value_noisy = lambda_1_values_noisy['l1'].iloc[-1] if isinstance(lambda_1_values_noisy['l1'], pd.Series) else lambda_1_values_noisy['l1'][-1]
lambda_2_value_noisy = lambda_2_values_noisy['l2'].iloc[-1] if isinstance(lambda_2_values_noisy['l2'], pd.Series) else lambda_2_values_noisy['l2'][-1]

######################################################################
############################# Plotting ###############################
######################################################################    

fig, ax = newfig(1.0, 1.4)
ax.axis('off')

####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
ax.set_title('$u(t,x)$', fontsize = 10)

####### Row 1: u(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

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

####### Row 3: Identified PDE ##################    
gs2 = gridspec.GridSpec(1, 3)
gs2.update(top=1.0-2.0/3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)

ax = plt.subplot(gs2[:, :])
ax.axis('off')
s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
s3 = r'Identified PDE (1\% noise) & '
s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
s5 = r'\end{tabular}$'
s = s1+s2+s3+s4+s5
ax.text(0.1,0.1,s)
    
plt.savefig('./figures/Burgers_ctid.pdf')  