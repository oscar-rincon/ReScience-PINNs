# Standard library imports
import os
import time
import warnings

# Extend system path to include the Utilities folder for additional modules
import sys
# Determine the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
utilities_dir = os.path.join(current_dir, '../../Utilities')

# Change the working directory to the script's directory
os.chdir(current_dir)

# Modify the module search path to include utilities directory
sys.path.insert(0, utilities_dir)

# Now import the pinns module
from pinns import *  # Adjust this import based on your actual module structure

# Third-party imports
import numpy as np
import scipy.io as sp
import torch
import torch.nn as nn
from functools import partial
 
# Disable all warnings (not recommended for production code)
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

def f(model, x_0, x_1, dt, IRK_weights):
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
    U1 = model(x_0)
    U = U1[:, :-1]
    U_x = fwd_gradients_0(U, x_0,device=device)
    U_xx = fwd_gradients_0(U_x, x_0,device=device)
    F = 5.0 * U - 5.0 * U**3 + 0.0001 * U_xx
    U0 = U1 - dt * torch.matmul(F, IRK_weights)
    U1 = model(x_1)
    U1_x = fwd_gradients_0(U1, x_1,device=device)
    return U0, U1, U1_x

def mse(model, x_0, x_1, dt, IRK_weights, U0_real):
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
    U0, U1, U1_x = f(model, x_0, x_1, dt, IRK_weights)
    loss = torch.sum((U0_real - U0) ** 2) + torch.sum((U1[0, :] - U1[1, :]) ** 2) + torch.sum((U1_x[0, :] - U1_x[1, :])**2)
    return loss

def closure(model, optimizer, x_0, x_1, dt, IRK_weights, U0_real, Exact, idx_t1, results):
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
    loss = mse(model, x_0, x_1, dt, IRK_weights, U0_real)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    U1_pred = model(x_star)
    pred = U1_pred[:, -1].detach().cpu().numpy()
    error = np.linalg.norm(pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
    results.append([iter, loss.item(), error])
    if iter % 1000 == 0:
        torch.save(model.state_dict(), f'models_iters/AC_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - L2: {error}")
    return loss

def train_adam(model, x_0, x_1, dt, IRK_weights, U0_real, Exact, idx_t1, results, num_iter=50_000):
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
        loss = mse(model, x_0, x_1, dt, IRK_weights, U0_real)
        loss.backward(retain_graph=True)
        optimizer.step()
        U1_pred = model(x_star)
        pred = U1_pred[:, -1].detach().cpu().numpy()
        error = np.linalg.norm(pred - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
        results.append([iter, loss.item(), error])
        if i % 1000 == 0:
            torch.save(model.state_dict(), f'models_iters/AC_{iter}.pt')
            print(f"Adam - Iter: {i} - Loss: {loss.item()} - L2: {error}")
          
def train_lbfgs(model, x_0, x_1, dt, IRK_weights, U0_real, Exact, idx_t1, results, num_iter=50_000):
    """
    Trains the given model using the LBFGS optimizer for a specified number of iterations.

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
        num_iter (int, optional): The maximum number of iterations for the LBFGS optimizer. Defaults to 50,000.

    Note:
        The `closure` function, which is required by the LBFGS optimizer, is assumed to be defined elsewhere.
        It should compute the loss, perform backpropagation, and optionally log training progress.
    """
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  history_size=100,
                                  tolerance_grad=1e-7,
                                  line_search_fn='strong_wolfe',
                                  tolerance_change=1.0 * np.finfo(float).eps)
    closure_fn = partial(closure, model, optimizer, x_0, x_1, dt, IRK_weights, U0_real, Exact, idx_t1, results)
    optimizer.step(closure_fn)

if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(42)

    # Set the device to CPU (change to GPU if available)
    device = torch.device('cpu')  # Use 'cuda' if GPU is available
    print(f'Using device: {device}')

    # Initialize parameters
    q = 100  # IRK parameter
    lb = np.array([-1.0], dtype=np.float32)  # Lower bound of the domain
    ub = np.array([1.0], dtype=np.float32)  # Upper bound of the domain
    N = 200  # Number of points
    iter = 0  # Iteration counter

    # Load the dataset
    data = sp.loadmat('../Data/AC.mat')
    t = data['tt'].flatten()[:, None].astype(np.float32)  # Time points
    x = data['x'].flatten()[:, None].astype(np.float32)  # Spatial points
    Exact = np.real(data['uu']).T.astype(np.float32)  # Exact solution

    # Select specific time indices for training
    idx_t0 = 20  # Initial time index
    idx_t1 = 180  # Final time index
    dt = torch.from_numpy(t[idx_t1] - t[idx_t0]).to(torch.float32)  # Time step size

    # Initialize noise level for initial condition
    noise_u0 = 0.0

    # Randomly select spatial points for training
    idx_x = np.random.choice(Exact.shape[1], N, replace=False)
    x0 = x[idx_x,:]
    x0 = torch.from_numpy(x0).to(torch.float32)
    x0.requires_grad = True

    # Prepare initial condition with optional noise
    u0 = Exact[idx_t0:idx_t0+1,idx_x].T
    u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])
    u0 = torch.from_numpy(u0).to(torch.float32)

    # Prepare boundary condition
    x1 = torch.from_numpy(np.vstack((lb, ub))).to(torch.float32)
    x1.requires_grad = True

    # Load IRK weights
    tmp = np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin=2).astype(np.float32)
    IRK_weights = torch.from_numpy(np.reshape(tmp[0:q**2+q], (q+1, q))).to(torch.float32).T

    # Prepare full spatial domain for prediction
    x_star = torch.from_numpy(x).to(torch.float32)

    # Initialize the model
    model = MLP(input_size=1, output_size=101, hidden_layers=5, hidden_units=200, activation_function=nn.Tanh())
    model.apply(init_weights)  # Initialize model weights
    model.to(device)  # Move model to the specified device
    model.train()  # Set model to training mode

    # Initialize results list for logging
    results = []

    # Ensure necessary directories exist for saving models and training logs
    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')  # Create directory for model iterations if it doesn't exist

    if not os.path.exists('training'):
        os.makedirs('training')  # Create directory for training logs if it doesn't exist

    # Move tensors to the specified device (GPU or CPU)
    x0 = x0.to(device)  # Move initial condition points
    x1 = x1.to(device)  # Move boundary condition points
    dt = dt.to(device)  # Move time step size
    IRK_weights = IRK_weights.to(device)  # Move IRK weights for numerical solver
    u0 = u0.to(device)  # Move initial condition values
    x_star = x_star.to(device)  # Move full spatial domain points for prediction
       
    # Training with Adam
    start_time_adam = time.time()
    train_adam(model, x0, x1, dt, IRK_weights, u0, Exact, idx_t1, results, num_iter=10_000)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")
    
    # Training with LBFGS
    start_time_lbfgs = time.time()
    train_lbfgs(model, x0, x1, dt, IRK_weights, u0, Exact, idx_t1, results, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")
    
    # Total training time
    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Get final L2 loss value
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6f}")

    # Get final L2 loss value
    final_l2 = results[-1][2]
    print(f"Final L2: {final_l2:.6f}")

    # Save times in a text file along with the final L2 loss
    with open('training/AC_training_summary.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
        file.write(f"Total iterations: {iter}\n") 
        file.write(f"Final Loss: {final_loss:.6f}\n")
        file.write(f"Final L2: {final_l2:.6f}\n")
             
    # Convert results to NumPy array and save to CSV
    results = np.array(results)
    np.savetxt("training/AC_training_data.csv", results, delimiter=",", header="Iter,Loss,L2", comments="")

    # Save model state
    torch.save(model.state_dict(), 'AC.pt')
