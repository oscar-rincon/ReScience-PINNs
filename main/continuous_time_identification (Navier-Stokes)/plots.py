import os
import torch 
import torch.nn as nn
import numpy as np
import scipy.io
from functools import partial
from pyDOE import lhs
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product, combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable
import imageio
import pandas as pd
import sys
sys.path.insert(0, '../../Utilities/') 
from plotting import *
sys.path.insert(0, '.')
import warnings

warnings.filterwarnings("ignore")

if not os.path.exists('figures'):
    os.makedirs('figures')
    
if not os.path.exists('figures_iters'):
    os.makedirs('figures_iters')    

data = pd.read_csv('training/NS_clean_training_data.csv')


fig, ax  = newfig(1.0)

ax.semilogy(data['Iter'], data['Loss'], label='Loss', color='gray', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')

savefig('figures/NS_clean_loss_curve.pdf')


# Configuración de la figura
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

# Primer subplot
axs[0].semilogy(data['Iter'], data['l1'], label='L1', color='gray', linewidth=2)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel(r'$\lambda_{1}$ Error ($\%$)')
 
# Segundo subplot
axs[1].semilogy(data['Iter'], data['l2'], label='L2', color='gray', linewidth=2)
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel(r'$\lambda_{2}$ Error ($\%$)')
 

# Ajustar el espaciado entre los subplots
plt.tight_layout()

# Guardar la figura en diferentes formatos
plt.savefig('figures/NS_clean_error_curves.pdf')


data = pd.read_csv('training/NS_noisy_training_data.csv')

fig, ax  = newfig(1.0)

ax.semilogy(data['Iter'], data['Loss'], label='Loss', color='gray', linewidth=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')

savefig('figures/NS_noisy_loss_curve.pdf')

# Configuración de la figura
fig, axs = plt.subplots(1, 2, figsize=(8, 3))

# Primer subplot
axs[0].plot(data['Iter'], data['l1'], label='L1', color='gray', linewidth=2)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel(r'$\lambda_{1}$ Error ($\%$)')

# Segundo subplot
axs[1].plot(data['Iter'], data['l2'], label='L2', color='gray', linewidth=2)
axs[1].set_xlabel('Iteration')
axs[1].set_ylabel(r'$\lambda_{2}$ Error ($\%$)')


# Ajustar el espaciado entre los subplots
plt.tight_layout()

# Guardar la figura en diferentes formatos
plt.savefig('figures/NS_noisy_error_curves.pdf')


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

def derivative(dy: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for i in range(order):
        dy = torch.autograd.grad(
            dy, x, grad_outputs = torch.ones_like(dy), create_graph=True, retain_graph=True
        )[0]
    return dy

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
        
def f(model, x_train_pt, y_train_pt, t_train_pt,lambda_1,lambda_2):
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
u_train_pt = torch.from_numpy(u_train)
v_train_pt = torch.from_numpy(v_train)

#lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
#lambda_2 = torch.nn.Parameter(torch.zeros(1, requires_grad=True))    
#lambda_1s = []
lambda_1_values_clean = pd.read_csv('training/lambda_1s_clean.csv')
lambda_2_values_clean = pd.read_csv('training/lambda_2s_clean.csv')
lambda_1_values_noisy = pd.read_csv('training/lambda_1s_noisy.csv')
lambda_2_values_noisy = pd.read_csv('training/lambda_2s_noisy.csv')

lambda_2s = []
images = []


# lambda_1_value=lambda_1_values_clean['l1'][-1]
# lambda_2_value=lambda_2_values_clean['l2'][-1]
# lambda_1_value_noisy=lambda_1_values_noisy['l1'][-1]
# lambda_2_value_noisy=lambda_2_values_noisy['l2'][-1]
lambda_1_value = lambda_1_values_clean['l1'].iloc[-1] if isinstance(lambda_1_values_clean['l1'], pd.Series) else lambda_1_values_clean['l1'][-1]
lambda_2_value = lambda_2_values_clean['l2'].iloc[-1] if isinstance(lambda_2_values_clean['l2'], pd.Series) else lambda_2_values_clean['l2'][-1]
lambda_1_value_noisy = lambda_1_values_noisy['l1'].iloc[-1] if isinstance(lambda_1_values_noisy['l1'], pd.Series) else lambda_1_values_noisy['l1'][-1]
lambda_2_value_noisy = lambda_2_values_noisy['l2'].iloc[-1] if isinstance(lambda_2_values_noisy['l2'], pd.Series) else lambda_2_values_noisy['l2'][-1]



model_path = f'NS_clean.pt'
model = NSNN()
model.load_state_dict(torch.load(model_path))
model.eval()

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

u_pred, v_pred, p_pred, f_u_pred, f_v_pred = f(model, x_star_pt, y_star_pt, t_star_pt,lambda_1_value,lambda_2_value) 
u_pred = u_pred.detach().numpy()
v_pred = v_pred.detach().numpy()
p_pred = p_pred.detach().numpy()
#lambda_1_value = lambda_1.detach().numpy()   
#lambda_2_value = lambda_2.detach().numpy()   
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)    
error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
        
#print('Error u: %e' % (error_u))    
#print('Error v: %e' % (error_v))    
#print('Error p: %e' % (error_p))     
#print('Error l1: %.5f%%' % (error_lambda_1))                             
#print('Error l2: %.5f%%' % (error_lambda_2))  

# Predict for plotting
lb = X_star.min(0)
ub = X_star.max(0)
nn_ = 200
x = np.linspace(lb[0], ub[0], nn_)
y = np.linspace(lb[1], ub[1], nn_)
X, Y = np.meshgrid(x,y)

UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')   

######################################################################
############################# Plotting ###############################
######################################################################    
    # Load Data
data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')
        
x_vort = data_vort['x'] 
y_vort = data_vort['y'] 
w_vort = data_vort['w'] 
#modes = np.asscalar(data_vort['modes'])
#nel = np.asscalar(data_vort['nel'])    
modes = data_vort['modes'].item()
nel = data_vort['nel'].item()    

xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')

box_lb = np.array([1.0, -2.0])
box_ub = np.array([8.0, 2.0])

fig, ax = newfig(1.015, 0.8)
ax.axis('off')

######## Row 2: Pressure #######################
########      Predicted p(t,x,y)     ########### 
gs2 = gridspec.GridSpec(1, 2)
gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.50)
ax = plt.subplot(gs2[:, 0])
h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
                extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal', 'box')
ax.set_title('Predicted pressure', fontsize = 10)

########     Exact p(t,x,y)     ########### 
ax = plt.subplot(gs2[:, 1])
h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
                extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(h, cax=cax)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal', 'box')
ax.set_title('Exact pressure', fontsize = 10)


######## Row 3: Table #######################
gs3 = gridspec.GridSpec(1, 2)
gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
ax = plt.subplot(gs3[:, :])
ax.axis('off')

s = r'$\begin{tabular}{|c|c|}';
s = s + r' \hline'
s = s + r' Correct PDE & $\begin{array}{c}'
s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
s = s + r' \end{array}$ \\ '
s = s + r' \hline'
s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
s = s + r' \\'
s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
s = s + r' \end{array}$ \\ '
s = s + r' \hline'
s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
s = s + r' \\'
s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
s = s + r' \end{array}$ \\ '
s = s + r' \hline'
s = s + r' \end{tabular}$'

ax.text(0.015,0.4,s)

#savefig('./figures/NavierStokes_prediction_{iter_num}')       
savefig('figures/NS.png')
savefig('figures/NS.pdf')
 


# Cargar y graficar modelos
for iter_num in range(100, 20_000, 100):

    lambda_1_value=lambda_1_values_clean['l1'][iter_num-1]
    lambda_2_value=lambda_2_values_clean['l2'][iter_num-1]
    lambda_1_value_noisy=lambda_1_values_noisy['l1'][iter_num-1]
    lambda_2_value_noisy=lambda_2_values_noisy['l2'][iter_num-1]

    model_path = f'model/NS_clean_{iter_num}.pt'
    model = NSNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

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

    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = f(model, x_star_pt, y_star_pt, t_star_pt,lambda_1_value,lambda_2_value) 
    u_pred = u_pred.detach().numpy()
    v_pred = v_pred.detach().numpy()
    p_pred = p_pred.detach().numpy()
    #lambda_1_value = lambda_1.detach().numpy()   
    #lambda_2_value = lambda_2.detach().numpy()   
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)    
    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
            
    #print('Error u: %e' % (error_u))    
    #print('Error v: %e' % (error_v))    
    #print('Error p: %e' % (error_p))     
    #print('Error l1: %.5f%%' % (error_lambda_1))                             
    #print('Error l2: %.5f%%' % (error_lambda_2))  

    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn_ = 200
    x = np.linspace(lb[0], ub[0], nn_)
    y = np.linspace(lb[1], ub[1], nn_)
    X, Y = np.meshgrid(x,y)

    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')   

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
        # Load Data
    data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')
            
    x_vort = data_vort['x'] 
    y_vort = data_vort['y'] 
    w_vort = data_vort['w'] 
    #modes = np.asscalar(data_vort['modes'])
    #nel = np.asscalar(data_vort['nel'])    
    modes = data_vort['modes'].item()
    nel = data_vort['nel'].item()    

    xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
    yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
    ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')

    box_lb = np.array([1.0, -2.0])
    box_ub = np.array([8.0, 2.0])

    fig, ax = newfig(1.015, 0.8)
    ax.axis('off')

    ######## Row 2: Pressure #######################
    ########      Predicted p(t,x,y)     ########### 
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.50)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
                    extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Predicted pressure', fontsize = 10)

    ########     Exact p(t,x,y)     ########### 
    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
                    extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                    origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Exact pressure', fontsize = 10)


    ######## Row 3: Table #######################
    gs3 = gridspec.GridSpec(1, 2)
    gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs3[:, :])
    ax.axis('off')

    s = r'$\begin{tabular}{|c|c|}';
    s = s + r' \hline'
    s = s + r' Correct PDE & $\begin{array}{c}'
    s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
    s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' \end{tabular}$'

    ax.text(0.015,0.4,s)

    #savefig('./figures/NavierStokes_prediction_{iter_num}')       
    image_filename = f'figures_iters/NS_{iter_num}.png'
    savefig(image_filename)
    images.append(image_filename)    
# Crear el GIF
with imageio.get_writer('figures/NS.gif', mode='I', duration=0.5) as writer:
    for filename in images:
        image = imageio.imread(filename)
        writer.append_data(image)

# Eliminar las imágenes temporales si no se necesitan
#for filename in images:
#    os.remove(filename)      