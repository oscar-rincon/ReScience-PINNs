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


def fwd_gradients_0(dy: torch.Tensor, x: torch.Tensor, device=torch.device('cpu')):
    """
    Computes the second-order gradient of `dy` with respect to `x`.

    Args:
        dy (torch.Tensor): The tensor whose gradient will be computed.
        x (torch.Tensor): The tensor with respect to which the gradient of `dy` will be computed.
        device (torch.device, optional): The device on which the tensors will be allocated. Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: The second-order gradient of `dy` with respect to `x`.
    """
    z = torch.ones(dy.shape, dtype=torch.float32, requires_grad=True, device=device)
    g = torch.autograd.grad(dy, x, grad_outputs=z, create_graph=True)[0]
    return torch.autograd.grad(g, z, grad_outputs=torch.ones(g.shape, dtype=torch.float32, device=device), create_graph=True)[0]

def f(model, x, x_1, dt, IRK_weights):
    """
    Simulates one step of a dynamical system using a neural network model and IRK integration.

    Args:
        model (torch.nn.Module): The neural network model that represents the dynamical system.
        x_0 (torch.Tensor): The initial state of the system.
        x_1 (torch.Tensor): The final state of the system for which we want to predict the derivatives.
        dt (float): The time step for the simulation.
        IRK_weights (torch.Tensor): The weights for the IRK integration method.

    Returns:
        tuple: A tuple containing:
            - U0 (torch.Tensor): The predicted state of the system at the next time step.
            - U1 (torch.Tensor): The state of the system at x_1 as predicted by the model.
            - U1_x (torch.Tensor): The spatial derivative of the system's state at x_1.
    """
    nu = 0.01/torch.pi
    U1 = model(x)
    U = U1[:, :-1]
    U_x = fwd_gradients_0(U, x,device=device)
    U_xx = fwd_gradients_0(U_x, x,device=device)
    F = - U*U_x + nu*U_xx 
    U0 = U1 - dt * torch.matmul(F, IRK_weights)
    U1 = model(x_1)
    return U0, U1 

def mse(model, x, x_1, dt, IRK_weights, U0_real):
    """
    Calculates the mean squared error (MSE) loss for a dynamical system simulation.

    Args:
        model (torch.nn.Module): The neural network model used for the simulation.
        x_0 (torch.Tensor): The initial state of the system.
        x_1 (torch.Tensor): The final state of the system for which we want to predict the derivatives.
        dt (float): The time step for the simulation.
        IRK_weights (torch.Tensor): The weights for the IRK integration method.
        U0_real (torch.Tensor): The real or expected state of the system at the next time step.

    Returns:
        torch.Tensor: The calculated MSE loss.
    """
    U0, U1 = f(model, x, x_1, dt, IRK_weights)
    loss = torch.sum((U0_real - U0) ** 2) + torch.sum((U1) ** 2)  
    return loss
 
def closure(model, optimizer, x, x_1, x_star, dt, IRK_weights, U0_real):
    """
    Performs a single optimization step and updates the training results.

    Args:
        model (torch.nn.Module): The neural network model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        x_0 (torch.Tensor): The initial condition input to the model.
        x_1 (torch.Tensor): The boundary condition input to the model.
        dt (float): The time step size.
        IRK_weights (torch.Tensor): The weights for the implicit Runge-Kutta method.
        U0_real (torch.Tensor): The real values of the initial condition for loss computation.
        Exact (numpy.ndarray): The exact solution of the system for error calculation.
        idx_t1 (int): The index of the time step at which the error is calculated.
        results (list): A list to store the iteration number, loss, and L2 error for logging.

    Returns:
        torch.Tensor: The computed loss for the current optimization step.
    """
    optimizer.zero_grad()
    loss = mse(model, x, x_1, dt, IRK_weights, U0_real)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    U1_pred = model(x_star)
    pred = U1_pred[:, -1].detach().cpu().numpy()
    error = np.linalg.norm(pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
    results.append([iter, loss.item(), error])
    if iter % 100 == 0:
        torch.save(model.state_dict(), f'models_iters/Burgers_dtin_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - L2: {error}")
    return loss 

def train_adam(model, x, x_1, x_star, dt, IRK_weights, U0_real, num_iter=50_000):
    """
    Trains the given model using the Adam optimizer over a specified number of iterations.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x_0 (torch.Tensor): The initial condition input to the model.
        x_1 (torch.Tensor): The boundary condition input to the model.
        dt (float): The time step size.
        IRK_weights (torch.Tensor): The weights for the implicit Runge-Kutta method.
        U0_real (torch.Tensor): The real values of the initial condition for loss computation.
        Exact (numpy.ndarray): The exact solution of the system for error calculation.
        idx_t1 (int): The index of the time step at which the error is calculated.
        results (list): A list to store the iteration number, loss, and L2 error for logging.
        num_iter (int, optional): The number of iterations for training. Defaults to 50,000.

    Note:
        The function assumes the presence of a global variable `iter` used for tracking the
        iteration count across different training sessions.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    global iter
    for i in range(1, num_iter + 1):
        iter += 1
        optimizer.zero_grad()
        loss = mse(model, x, x_1, dt, IRK_weights, U0_real)
        loss.backward(retain_graph=True)
        optimizer.step()
        U1_pred = model(x_star)
        pred = U1_pred[:, -1].detach().cpu().numpy()
        error = np.linalg.norm(pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
        results.append([iter, loss.item(), error])
        if i % 100 == 0:
            torch.save(model.state_dict(), f'models_iters/Burgers_dtin_{iter}.pt')
            print(f"Adam - Iter: {i} - Loss: {loss.item()} - L2: {error}")    

def train_lbfgs(model, x, x_1, x_star, dt, IRK_weights, U0_real, num_iter=50_000):
    """
    Trains a neural network model using the LBFGS optimizer to solve a PDE problem.
 
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x_u (torch.Tensor): The spatial input tensor for the observed data.
        x_f (torch.Tensor): The spatial input tensor for the PDE residual calculation.
        t_u (torch.Tensor): The temporal input tensor for the observed data.
        t_f (torch.Tensor): The temporal input tensor for the PDE residual calculation.
        nu (float): The viscosity parameter for the PDE.
        u_train_pt (torch.Tensor): The observed data values corresponding to x_u and t_u.
        num_iter (int, optional): The maximum number of iterations for the LBFGS optimizer. Defaults to 50,000.

    Note:
        The `closure` function required by the LBFGS optimizer is defined externally and must be available in the
        scope where this function is called. It should accept the model, optimizer, and all data tensors as arguments,
        and return the computed loss.
    """
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=1,
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  tolerance_grad=1e-7,
                                  tolerance_change=1.0 * np.finfo(float).eps,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')    
    closure_fn = partial(closure, model, optimizer, x, x_1, x_star, dt, IRK_weights, U0_real)
    optimizer.step(closure_fn)

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
    if not os.path.exists('tables'):
        os.makedirs('tables')
        
    # Initialize variables
    results = []
    iter = 0  # Initialize iteration counter
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
    model.apply(init_weights)

    # Training phase

    # Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x0, x1, x_star, dt, IRK_weights, u0_real, num_iter=0)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")

    # L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x0, x1, x_star, dt, IRK_weights, u0_real, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")

    # Total training time
    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Final loss and L2 error
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6e}")
    final_l2 = results[-1][2]
    print(f"Final L2: {final_l2:.6e}")

    # Save times in a text file along with the final L2 loss
    with open('training/Burgers_dtin_training_summary.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2e} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2e} seconds\n")
        file.write(f"Total training time: {total_training_time:.2e} seconds\n")
        file.write(f"Total iterations: {iter:.6e}\n") 
        file.write(f"Final Loss: {final_loss:.6e}\n")
        file.write(f"Final L2: {final_l2:.6e}\n")
             
    # Convert results to NumPy array and save to CSV
    results = np.array(results)
    np.savetxt("training/Burgers_dtin_training_data.csv", results, delimiter=",", header="Iter,Loss,L2", comments="")

    # Save model state
    torch.save(model.state_dict(), 'Burgers_dtin.pt')