# Import standard libraries
import sys  # System-specific parameters and functions
import os   # Miscellaneous operating system interfaces
import time  # Time access and conversions
import warnings  # Warning control

# Modify the module search path, so we can import utilities from a specific folder
sys.path.insert(0, '../../Utilities/')

# Import third-party libraries
import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network module in PyTorch
import numpy as np  # NumPy library for numerical operations
import scipy.io    # SciPy module for MATLAB file I/O

# Import additional utilities
from functools import partial  # Higher-order functions and operations on callable objects
from pyDOE import lhs  # Design of experiments for Python, including Latin Hypercube Sampling

# Import custom modules
from pinns import *  # Physics Informed Neural Networks utilities

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")
 
def net_U0(model, x, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta):
    """
    Simulates one step of a dynamical system using a neural network model and IRK integration.

    Args:
        model (torch.nn.Module): The neural network model that represents the dynamical system.
        x (torch.Tensor): The input state of the system.
        lambda_1 (float): The first parameter influencing the system's dynamics.
        lambda_2 (float): The second parameter influencing the system's dynamics, exponentiated within the function.
        dt (float): The time step for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients of the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients of the IRK integration method, not used in this function but included for consistency with related functions.

    Returns:
        torch.Tensor: The predicted state of the system at the next time step.
    """
    lambda_2 = torch.exp(lambda_2)
    U = model(x)
    U_x = fwd_gradients_0(U, x,device=device)
    U_xx = fwd_gradients_0(U_x, x,device=device)
    F = - lambda_1*U*U_x + lambda_2*U_xx 
    U0 = U - dt * torch.matmul(F, IRK_alpha.T)
    return U0 


def net_U1(model, x, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta):
    """
    Simulates one step of a dynamical system using a neural network model and IRK integration.

    Args:
        model (torch.nn.Module): The neural network model that represents the dynamical system.
        x (torch.Tensor): The input state of the system.
        lambda_1 (float): The first parameter influencing the system's dynamics.
        lambda_2 (float): The second parameter influencing the system's dynamics, exponentiated within the function.
        dt (float): The time step for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients of the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients of the IRK integration method, not used in this function but included for consistency with related functions.

    Returns:
        torch.Tensor: The predicted state of the system at the next time step.
    """
    lambda_2 = torch.exp(lambda_2)
    U = model(x)
    U_x = fwd_gradients_0(U, x,device=device)
    U_xx = fwd_gradients_0(U_x, x,device=device)
    F = - lambda_1*U*U_x + lambda_2*U_xx 
    U1 = U + dt * torch.matmul(F, (IRK_beta-IRK_alpha).T)
    return U1 

def mse(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta):
    """
    Calculates the mean squared error (MSE) loss for a neural network model solving a PDE.

    This function computes the MSE loss by comparing the predicted solutions of the PDE at two different
    points (or conditions) against their true solutions. It uses the implicit Runge-Kutta (IRK) method
    parameters for temporal integration.

    Args:
        model (torch.nn.Module): The neural network model used for prediction.
        x0 (torch.Tensor): The input tensor for the initial condition.
        x1 (torch.Tensor): The input tensor for the boundary condition.
        u0 (torch.Tensor): The true solution tensor for the initial condition.
        u1 (torch.Tensor): The true solution tensor for the boundary condition.
        lambda_1 (float): The first regularization parameter.
        lambda_2 (float): The second regularization parameter.
        dt (float): The time step size.
        IRK_alpha (torch.Tensor): The alpha coefficients of the IRK method.
        IRK_beta (torch.Tensor): The beta coefficients of the IRK method.

    Returns:
        torch.Tensor: The computed MSE loss.
    """    
    U0 = net_U0(model, x0, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    U1 = net_U1(model, x1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    return torch.sum((u0-U0)**2)  + torch.sum((u1-U1)**2)  

def train_adam(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, num_iter=50_000):
    """
    Trains a neural network model using the Adam optimizer over a specified number of iterations to solve a PDE problem.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x0 (torch.Tensor): The spatial input tensor for the initial condition.
        x1 (torch.Tensor): The spatial input tensor for the boundary condition.
        u0 (torch.Tensor): The observed data values for the initial condition.
        u1 (torch.Tensor): The observed data values for the boundary condition.
        lambda_1 (torch.Tensor): A learnable parameter related to the PDE solution.
        lambda_2 (torch.Tensor): A learnable parameter related to the PDE solution.
        dt (float): The time step size for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients of the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients of the IRK integration method.
        num_iter (int, optional): The number of iterations to train the model. Defaults to 50,000.

    Note:
        The function uses a global variable `iter` to track the iteration count. `iter` is incremented with each iteration.
        Training progress can be optionally tracked by storing tuples of (iteration number, loss) in a global list `results`.
        Ensure these are properly initialized before calling this function if used.
        
        The function also saves the model state every 1000 iterations to a file named 'Burgers_{iter}.pt' in the 'models_iters' directory.
    """
    optimizer = torch.optim.Adam(list(model.parameters()) + [lambda_1, lambda_2], lr=1e-3)
    global iter
     
    for i in range(1, num_iter + 1):
        iter += 1 
        optimizer.zero_grad()
        loss = mse(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
        loss.backward(retain_graph=True)
        optimizer.step()
        #lambda_1s.append(lambda_1.item())
        #lambda_2s.append(torch.exp(lambda_2).item())
        #error_lambda_1 = np.abs(lambda_1.cpu().detach().numpy() - 1.0) / 1.0 * 100
        #error_lambda_2 = np.abs(torch.exp(lambda_2).cpu().detach().numpy() - nu.cpu().detach().numpy()) / nu.cpu().detach().numpy() * 100
        #results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
        #if i % 100 == 0:
        #    torch.save(model.state_dict(), f'models_iters/Burgers_clean_dtid_{iter}.pt')
        #    print(f"Adam - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {torch.exp(lambda_2).cpu().detach().numpy().item()}")

def closure(model, optimizer, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta):
    """
    Performs a single optimization step using the provided model and optimizer, and calculates the loss.

    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer to use for the optimization step.
        x0 (torch.Tensor): The spatial input tensor for the initial condition.
        x1 (torch.Tensor): The spatial input tensor for the boundary condition.
        u0 (torch.Tensor): The observed data values for the initial condition.
        u1 (torch.Tensor): The observed data values for the boundary condition.
        lambda_1 (torch.Tensor): A learnable parameter related to the PDE solution.
        lambda_2 (torch.Tensor): A learnable parameter related to the PDE solution.
        dt (float): The time step size for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients of the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients of the IRK integration method.

    Returns:
        torch.Tensor: The calculated loss for the current optimization step.
    """
    optimizer.zero_grad()
    loss = mse(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    loss.backward(retain_graph=True)
    global iter
    iter += 1    
    #lambda_1s.append(lambda_1.item())
    #lambda_2s.append(torch.exp(lambda_2).item())
    #error_lambda_1 = np.abs(lambda_1.cpu().detach().numpy() - 1.0) / 1.0 * 100
    #error_lambda_2 = np.abs(torch.exp(lambda_2).cpu().detach().numpy() -nu.cpu().detach().numpy()) / nu.cpu().detach().numpy() * 100
    #results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
    #if iter % 100 == 0:
    #    torch.save(model.state_dict(), f'models_iters/Burgers_dtid_clean_{iter}.pt')
    #    print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {torch.exp(lambda_2).cpu().detach().numpy().item()}")
    return loss 

def train_lbfgs(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, num_iter=50_000):
    """
    Trains a neural network model using the LBFGS optimizer over a specified number of iterations to solve a PDE problem.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x0 (torch.Tensor): The spatial input tensor for the initial condition.
        x1 (torch.Tensor): The spatial input tensor for the boundary condition.
        u0 (torch.Tensor): The observed data values for the initial condition.
        u1 (torch.Tensor): The observed data values for the boundary condition.
        lambda_1 (torch.Tensor): A learnable parameter related to the PDE solution.
        lambda_2 (torch.Tensor): A learnable parameter related to the PDE solution.
        dt (float): The time step size for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients of the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients of the IRK integration method.
        num_iter (int, optional): The maximum number of iterations for the LBFGS optimizer. Defaults to 50,000.

    Note:
        The `closure` function required by the LBFGS optimizer is defined externally and must be available in the
        scope where this function is called. It should accept the model, optimizer, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta as arguments,
        and return the computed loss.
    """
    optimizer = torch.optim.LBFGS(list(model.parameters()) + [lambda_1, lambda_2],
                                  lr=1,
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  tolerance_grad=1e-7,
                                  tolerance_change=1.0 * np.finfo(float).eps,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')    
    closure_fn = partial(closure, model, optimizer, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    optimizer.step(closure_fn)    

def main_loop(skip, noise, num_layers, num_neurons): 

    nu = 0.01/torch.pi 
    nu = torch.tensor(nu).float().to(device)  # Viscosity coefficient
    skip = skip
    N0 = 199
    N1 = 201
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    t_star = data['t'].flatten()[:,None]
    x_star = data['x'].flatten()[:,None]
    Exact = np.real(data['usol'])
    idx_t = 10 
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

    # Initialize model parameters for regularization
    lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device)).float()
    lambda_2 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device)-6).float()
    lambda_1s = []  # List to track lambda_1 values
    lambda_2s = []  # List to track lambda_2 values
     
    # Initialize the model and apply initial weights
    model = MLP(input_size=1, output_size=q, hidden_layers=num_layers, hidden_units=num_neurons, activation_function=nn.Tanh()).float().to(device)
    model.apply(init_weights)

    # Training phase

    # Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, num_iter=0)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    #print(f"Adam training time: {adam_training_time:.2f} seconds")

    # L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")

    # Total training time
    total_training_time = adam_training_time + lbfgs_training_time
    #print(f"Total training time: {total_training_time:.2f} seconds")

    error_lambda_1 = (np.abs(lambda_1.cpu().detach().numpy() - 1.0) / 1.0 * 100).item()
    error_lambda_2 = (np.abs(torch.exp(lambda_2).cpu().detach().numpy() -nu.cpu().detach().numpy()) / nu.cpu().detach().numpy() * 100).item()


    print(f"Percentage Error Lambda 1: {error_lambda_1:.6f}%")
    print(f"Percentage Error Lambda 2: {error_lambda_2:.6f}%")

    return error_lambda_1, error_lambda_2

if __name__ == "__main__": 

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
    results = []
    # Initialize variables
    iter = 0  # Initialize iteration counter
    nu = 0.01/torch.pi 
    nu = torch.tensor(nu).float().to(device)  # Viscosity coefficient    
    N0 = 199
    N1 = 201  
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    t_star = data['t'].flatten()[:,None]
    x_star = data['x'].flatten()[:,None]
    Exact = np.real(data['usol'])
    idx_t = 10
    noise = 0.0   
    skip = 80
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

    # Initialize model parameters for regularization
    lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device)).float()
    lambda_2 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device)-6).float()
    lambda_1s = []  # List to track lambda_1 values
    lambda_2s = []  # List to track lambda_2 values

    skip = [20, 40, 60, 80]
    noise = [0.0, 0.01, 0.05, 0.1]
    num_layers = [1,2,3,4]
    num_neurons = [10,25,50]
    
    error_lambda_1_table_1 = np.zeros((len(skip), len(noise)))
    error_lambda_2_table_1 = np.zeros((len(skip), len(noise)))
    error_lambda_1_table_2 = np.zeros((len(num_layers), len(num_neurons)))
    error_lambda_2_table_2 = np.zeros((len(num_layers), len(num_neurons)))
    
    for i in range(len(skip)):
        for j in range(len(noise)):
            error_lambda_1_table_1[i,j], error_lambda_2_table_1[i,j] = main_loop(skip[i], noise[j], num_layers[-1], num_neurons[-1])
             
    for i in range(len(num_layers)):
        for j in range(len(num_neurons)):
            error_lambda_1_table_2[i,j], error_lambda_2_table_2[i,j] = main_loop(skip[-1], noise[0], num_layers[i], num_neurons[j])
                  
    np.savetxt('./tables/error_lambda_1_table_1.csv', error_lambda_1_table_1, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')
    np.savetxt('./tables/error_lambda_2_table_1.csv', error_lambda_2_table_1, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')
    np.savetxt('./tables/error_lambda_1_table_2.csv', error_lambda_1_table_2, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')
    np.savetxt('./tables/error_lambda_2_table_2.csv', error_lambda_2_table_2, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')
