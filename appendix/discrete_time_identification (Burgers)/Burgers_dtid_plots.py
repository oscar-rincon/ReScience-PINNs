# Import standard libraries
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

# Check GPU availability and select device
device = torch.device('cuda')#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the data
lambda_1_values_clean = pd.read_csv('training/lambda_1s_clean.csv')
lambda_2_values_clean = pd.read_csv('training/lambda_2s_clean.csv')
lambda_1_values_noisy = pd.read_csv('training/lambda_1s_noisy.csv')
lambda_2_values_noisy = pd.read_csv('training/lambda_2s_noisy.csv')
NS_training_data_clean = pd.read_csv('training/Burgers_dtid_clean_training_data.csv')
NS_training_data_noisy = pd.read_csv('training/Burgers_dtid_noisy_training_data.csv')

# Create subplots for loss curves
fig, axarr  = newfig(0.8, 0.8)

# Plot clean loss curve
axarr.semilogy(NS_training_data_clean['Iter'], NS_training_data_clean['Loss'], label='Clean', color='blue', linewidth=1 )

# Plot noisy loss curve
axarr.semilogy(NS_training_data_noisy['Iter'], NS_training_data_noisy['Loss'], label='Noisy', color='red', linewidth=1, linestyle='--')

axarr.set_xlabel('Iteration')
axarr.set_ylabel('Loss')
axarr.legend(frameon=False)

plt.tight_layout()
plt.savefig('figures/Burgers_dtid_combined_loss_curve.pdf')

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
plt.savefig('figures/Burgers_dtid_lambda_curves.pdf')    


nu = 0.01/torch.pi 
nu = torch.tensor(nu).float().to(device)  # Viscosity coefficient
skip = 80
N0 = 199
N1 = 201
data = scipy.io.loadmat('../Data/burgers_shock.mat')
t_star = data['t'].flatten()[:,None]
x_star = data['x'].flatten()[:,None]
Exact = np.real(data['usol'])
idx_t = 10
noise = 0.0    
idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
x0 = x_star[idx_x,:]
u0 = Exact[idx_x,idx_t][:,None]
u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])  
idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
x1 = x_star[idx_x,:]
u1 = Exact[idx_x,idx_t + skip][:,None]
u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])
dt =  t_star[idx_t+skip] - t_star[idx_t]         
q = int(np.ceil(0.5*np.log(np.finfo(float).eps)/np.log(dt)))
dt = torch.from_numpy(dt).to(torch.float32).to(device)  # Time step size

# Load IRK weights for numerical integration
tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
weights = np.reshape(tmp[0:q**2+q], (q+1, q))    
IRK_alpha = torch.from_numpy(weights[0:-1,:]).float().to(device)
IRK_beta = torch.from_numpy(weights[-1:,:]).float().to(device)       
IRK_times = tmp[q**2+q:]

# Doman bounds
lb = x_star.min(0)
ub = x_star.max(0)

# Convert to tensors and set requires_grad for training with float precision
x0 = torch.from_numpy(x0).float().to(device)
x0.requires_grad = True
x1 = torch.from_numpy(x1).float().to(device)
x1.requires_grad = True
u0 = torch.from_numpy(u0).float().to(device)
u1 = torch.from_numpy(u1).float().to(device)
x_star = torch.from_numpy(x_star).float().to(device)
x_star.requires_grad = True   

# Initialize the model and apply initial weights
model = MLP(input_size=1, output_size=q, hidden_layers=5, hidden_units=50, activation_function=nn.Tanh()).float().to(device)
model_path = 'Burgers_dtid_clean.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

U1_pred = model(x_star)
U1_pred = U1_pred.cpu().detach().numpy()
min_value = np.min(-1)
max_value = np.max(1)
 
lambda_1_value = lambda_1_values_clean['l1'].iloc[-1] if isinstance(lambda_1_values_clean['l1'], pd.Series) else lambda_1_values_clean['l1'][-1]
lambda_2_value = lambda_2_values_clean['l2'].iloc[-1] if isinstance(lambda_2_values_clean['l2'], pd.Series) else lambda_2_values_clean['l2'][-1]
lambda_1_value_noisy = lambda_1_values_noisy['l1'].iloc[-1] if isinstance(lambda_1_values_noisy['l1'], pd.Series) else lambda_1_values_noisy['l1'][-1]
lambda_2_value_noisy = lambda_2_values_noisy['l2'].iloc[-1] if isinstance(lambda_2_values_noisy['l2'], pd.Series) else lambda_2_values_noisy['l2'][-1]

x_star = x_star.cpu().detach().numpy()
x0 = x0.cpu().detach().numpy()
x1 = x1.cpu().detach().numpy()
u0 = u0.cpu().detach().numpy()
u1 = u1.cpu().detach().numpy()

# Plotting

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
s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x + %.6f u_{xx} = 0$ \\  \hline Identified PDE (clean data) & ' % (nu)
s2 = r'$u_t + %.3f u u_x + %.6f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
s3 = r'Identified PDE (1\% noise) & '
s4 = r'$u_t + %.3f u u_x + %.6f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
s5 = r'\end{tabular}$'
s = s1+s2+s3+s4+s5
ax.text(-0.1,0.4,s)

plt.savefig('./figures/Burgers_dtid.pdf')  


# Generate images for GIF
 
model_dir = 'models_iters/'
image_dir = 'figures_iters/'
gif_filename = 'figures/Burgers_dtid.gif'
limit = 10_001
step = 1_00

for iter_num in range(step, limit, step):

    # Obtener los índices correctos para lambda_1_values_clean
    if iter_num > lambda_1_values_clean['l1'].index[-1]:
        iter_num_clean = math.floor(lambda_1_values_clean['l1'].index[-1] / 1000) * 1000
    else:
        iter_num_clean = iter_num 

    # Obtener los índices correctos para lambda_1_values_noisy
    if iter_num > lambda_1_values_noisy['l1'].index[-1]:
        iter_num_noisy = math.floor(lambda_1_values_noisy['l1'].index[-1] / 1000) * 1000
    else:
        iter_num_noisy = iter_num 
    
    # Obtener los valores lambda
    lambda_1_value = lambda_1_values_clean['l1'][iter_num_clean]
    lambda_2_value = lambda_2_values_clean['l2'][iter_num_clean]
    lambda_1_value_noisy = lambda_1_values_noisy['l1'][iter_num_noisy]
    lambda_2_value_noisy = lambda_2_values_noisy['l2'][iter_num_noisy]

    # Plotting

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
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x + %.6f u_{xx} = 0$ \\  \hline Identified PDE (clean data) & ' % (nu)
    s2 = r'$u_t + %.3f u u_x + %.6f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.3f u u_x + %.6f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1+s2+s3+s4+s5
    ax.text(-0.1,0.4,s)    

    image_filename = f'./figures_iters/Burgers_dtid_{iter_num}.png'
    savefig(image_filename) 
     
# Create GIF
images = []
for i in range(step, limit, step):
    image_path = os.path.join(image_dir, f'Burgers_dtid_{i}.png')
    images.append(imageio.imread(image_path))

imageio.mimsave(gif_filename, images, fps=5)      