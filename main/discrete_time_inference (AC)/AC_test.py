
import sys
sys.path.insert(0, '../../Utilities/')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
import time
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
# Definir la red neuronal
class NeuralNet(nn.Module):
    def __init__(self, layers):
        super(NeuralNet, self).__init__()
        modules = []
        for l in range(len(layers) - 1):
            modules.append(nn.Linear(layers[l], layers[l+1]))
            modules.append(nn.Tanh())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

# Función de pérdida personalizada
def custom_loss(U0_pred, U1_pred, U1_x_pred, u0_true, u1_true, u1_x_true):
    loss_U0 = torch.sum(torch.square(u0_true - U0_pred))
    loss_U1 = torch.sum(torch.square(u1_true[0,:] - u1_true[1,:]))
    loss_U1_x = torch.sum(torch.square(u1_x_true[0,:] - u1_x_true[1,:]))
    return loss_U0 + loss_U1 + loss_U1_x

def net_U0(model, x, dt, IRK_weights):
    U1 = model(x)
    U = U1[:, :-1]
    U_x = torch.autograd.grad(U, x, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    U_xx = torch.autograd.grad(U_x, x, grad_outputs=torch.ones_like(U_x), create_graph=True)[0]
    F = 5.0 * U - 5.0 * U**3 + 0.0001 * U_xx
    U0 = U1 - dt * torch.matmul(F, torch.tensor(IRK_weights.T, dtype=torch.float32))
    return U0

if __name__ == "__main__":
    q = 100
    layers = [1, 200, 200, 200, 200, q+1]
    lb = np.array([-1.0])
    ub = np.array([1.0])
    N = 200

    data = scipy.io.loadmat('../Data/AC.mat')

    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['uu']).T

    idx_t0 = 20
    idx_t1 = 180
    dt =torch.tensor(t[idx_t1] - t[idx_t0], dtype=torch.float32) 

    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
    x0 = x[idx_x,:]
    u0 = Exact[idx_t0:idx_t0+1,idx_x].T
    u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])

    x1 = np.vstack((lb,ub))
    x_star = x

    # Cargar los pesos y tiempos IRK
    tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
    IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
    IRK_times = tmp[q**2+q:]

    # Convertir los datos a tensores de PyTorch
    x0_tensor = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    u0_tensor = torch.tensor(u0, dtype=torch.float32)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, requires_grad=True)

    # Definir el modelo
    model = NeuralNet(layers)

    # Definir el optimizador
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Entrenamiento
    start_time = time.time()
    for it in range(100):
        optimizer.zero_grad()

        # Forward pass
        U0_pred = net_U0(model, x0_tensor, dt, IRK_weights)
        U1_pred = model(x1_tensor)
        U1_x_pred = torch.autograd.grad(U1_pred, x1_tensor, grad_outputs=torch.ones_like(U1_pred), create_graph=True)[0]

        # Calcular la pérdida
        loss = custom_loss(U0_pred, U1_pred, U1_x_pred, u0_tensor, U1_pred, U1_x_pred)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print progress
        if it % 10 == 0:
            elapsed = time.time() - start_time
            print(f'Iteration {it}: Loss = {loss.item():.4f}, Time = {elapsed:.2f} seconds')
            start_time = time.time()

    # Predicción
    U1_pred = model(torch.tensor(x_star, dtype=torch.float32))

    error = np.linalg.norm(U1_pred[:,-1].detach().numpy() - Exact[idx_t1,:], 2) / np.linalg.norm(Exact[idx_t1,:], 2)
    print('Error: %e' % (error))

    # Define la función newfig si aún no está definida en tu script.
    #def newfig(width, ratio):
    #   fig = plt.figure(figsize=(width, width * ratio))
    #    ax = fig.add_subplot(111)
    #    return fig, ax

    # Define los datos
    idx_t0 = 20
    idx_t1 = 180
    t_idx_t0 = t[idx_t0][0]
    t_idx_t1 = t[idx_t1][0]
    print(t_idx_t0)
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')
    
    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(Exact.T, interpolation='nearest', cmap='seismic', 
                  extent=[t.min(), t.max(), x_star.min(), x_star.max()], 
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
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')      
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t_idx_t0), fontsize = 10)
    ax.set_xlim([lb-0.1, ub+0.1])
    ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)


    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[idx_t1,:], 'b-', linewidth = 2, label = 'Exact') 
    ax.plot(x_star, U1_pred[:,-1].detach().numpy(), 'r--', linewidth = 2, label = 'Prediction')      
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = %.2f$' % (t_idx_t0), fontsize = 10)    
    ax.set_xlim([lb-0.1, ub+0.1])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
    
    savefig('./figures/AC_t')  