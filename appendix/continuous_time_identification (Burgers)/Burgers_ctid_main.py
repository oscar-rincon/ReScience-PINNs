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

def f(model, x, t, nu):
    lambda_2 = torch.exp(lambda_2)
    u = model(torch.cat((x, t), dim=1))
    u_t = derivative(u, t, order=1)
    u_x = derivative(u, x, order=1)
    u_xx = derivative(u, x, order=2)
    f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
    return f

def mse_f(model, x, t, nu):
    f_pred = f(model, x, t, nu)
    return (f_pred**2).mean()

def mse_u(model, x, t, u_train_pt):
    u = model(torch.cat((x, t), dim=1))
    return ((u_train_pt - u) ** 2).mean()

def train_adam(model, x_u, x_f, t_u, t_f, nu, u_train_pt, num_iter=50_000):
    """
    Trains a neural network model using the Adam optimizer over a specified number of iterations to solve a PDE problem.

    Args:
        model: The neural network model to be trained.
        x_u: The spatial input tensor for the observed data.
        x_f: The spatial input tensor for the PDE residual calculation.
        t_u: The temporal input tensor for the observed data.
        t_f: The temporal input tensor for the PDE residual calculation.
        nu: The viscosity parameter for the PDE.
        u_train_pt: The observed data values corresponding to x_u and t_u.
        num_iter (int, optional): The number of iterations to train the model. Defaults to 50,000.

    Note:
        The function uses global variables `iter` and `results` to track the iteration count and to store
        the training progress, respectively. `iter` is incremented with each iteration, and `results` stores
        tuples of (iteration number, loss, L2 error). Ensure these are properly initialized before calling this function.
        
        The function also saves the model state every 1000 iterations to a file named 'Burgers_{iter}.pt' in the 'models_iters' directory.
    """
    optimizer = torch.optim.Adam(list(model.parameters()) + [lambda_1, lambda_2], lr=1e-3)
    global iter
     
    for i in range(1, num_iter + 1):
        iter += 1 
        optimizer.zero_grad()
        loss = mse_f(model, x_f, t_f, nu) + mse_u(model, x_u, t_u, u_train_pt)
        loss.backward(retain_graph=True)
        optimizer.step()
        lambda_1s.append(lambda_1.item())
        lambda_2s.append(torch.exp(lambda_2).item())
        error_lambda_1 = np.abs(lambda_1.cpu().detach().numpy() - 1.0) / 1.0 * 100
        error_lambda_2 = np.abs(torch.exp(lambda_2).cpu().detach().numpy() - nu) / nu * 100
        results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
        if i % 100 == 0:
            torch.save(model.state_dict(), f'models_iters/Burgers_ctid_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {torch.exp(lambda_2).cpu().detach().numpy().item()}")

def closure(model, optimizer, x_u, x_f, t_u, t_f, nu, u_train_pt):
    """
    Performs a single optimization step using the provided model and optimizer, and calculates the loss.
 
    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer to use for the optimization step.
        x_u (torch.Tensor): The spatial input tensor for the observed data.
        x_f (torch.Tensor): The spatial input tensor for the PDE residual calculation.
        t_u (torch.Tensor): The temporal input tensor for the observed data.
        t_f (torch.Tensor): The temporal input tensor for the PDE residual calculation.
        nu (float): The viscosity parameter for the PDE.
        u_train_pt (torch.Tensor): The observed data values corresponding to x_u and t_u.

    Returns:
        torch.Tensor: The calculated loss for the current optimization step.
    """
    optimizer.zero_grad()
    loss = mse_f(model, x_f, t_f, nu) + mse_u(model, x_u, t_u, u_train_pt)
    loss.backward(retain_graph=True)
    global iter
    iter += 1    
    lambda_1s.append(lambda_1.item())
    lambda_2s.append(torch.exp(lambda_2).item())
    error_lambda_1 = np.abs(lambda_1.detach().numpy() - 1.0) / 1.0 * 100
    error_lambda_2 = np.abs(torch.exp(lambda_2).detach().numpy() -nu) / nu * 100
    results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
    if iter % 1000 == 0:
        torch.save(model.state_dict(), f'models_iters/KdV_clean_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.detach().numpy().item()} - l2: {torch.exp(lambda_2).detach().numpy().item()}")
    return loss 

def train_lbfgs(model, x_u, x_f, t_u, t_f, nu, u_train_pt, num_iter=50_000):
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
    optimizer = torch.optim.LBFGS(list(model.parameters()) + [lambda_1, lambda_2],
                                  lr=1,
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  tolerance_grad=1e-7,
                                  tolerance_change=1.0 * np.finfo(float).eps,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')    
    closure_fn = partial(closure, model, optimizer, x_u, x_f, t_u, t_f, nu, u_train_pt)
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

    # Initialize variables
    iter = 0  # Initialize iteration counter
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

    # Convert to tensors and set requires_grad for training with float precision
    x_u = torch.from_numpy(X_u_train[:, 0:1]).float().to(device)
    x_u.requires_grad = True
    x_f = torch.from_numpy(X_f_train[:, 0:1]).float().to(device)
    x_f.requires_grad = True
    t_u = torch.from_numpy(X_u_train[:, 1:2]).float().to(device)
    t_u.requires_grad = True
    t_f = torch.from_numpy(X_f_train[:, 1:2]).float().to(device)
    t_f.requires_grad = True
    u_train_pt = torch.from_numpy(u_train).float().to(device)
    nu = torch.tensor(nu).float().to(device)
    x_star = torch.from_numpy(X_star[:, 0:1]).float().to(device)
    x_star.requires_grad = True
    t_star = torch.from_numpy(X_star[:, 1:2]).float().to(device)
    t_star.requires_grad = True
    u_star = torch.from_numpy(u_star).T.float().to(device)

    # Initialize model parameters for regularization
    lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device)).float()
    lambda_2 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device)-6).float()
    lambda_1s = []  # List to track lambda_1 values
    lambda_2s = []  # List to track lambda_2 values
    results = []


    # Initialize the model and apply initial weights
    model = MLP(input_size=2, output_size=1, hidden_layers=8, hidden_units=20, activation_function=nn.Tanh()).float().to(device)
    model.apply(init_weights)

    # Training phase

    # Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x_u, x_f, t_u, t_f, nu, u_train_pt, num_iter=0)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")

    # L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x_u, x_f, t_u, t_f, nu, u_train_pt, num_iter=1)
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

    # Save training summary
    with open('training/Burgers_ctin_training_summary.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.6e} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.6e} seconds\n")
        file.write(f"Total training time: {total_training_time:.6e} seconds\n")
        file.write(f"Total iterations: {iter:.6e}\n")
        file.write(f"Final Loss: {final_loss:.6e}\n")
        file.write(f"Final L2: {final_l2:.6e}\n")

    # Save training data and model state
    results = np.array(results)
    np.savetxt("training/Burgers_ctin_training_data.csv", results, delimiter=",", header="Iter,Loss,L2", comments="")
    torch.save(model.state_dict(), 'Burgers_ctin.pt')

    # Calculate percentage error for lambda_1 and lambda_2
    error_lambda_1 = np.abs(lambda_1s[-1] - 1.0) / 1.0 * 100
    error_lambda_2 = np.abs(lambda_2s[-1] - nu) / nu * 100
    print(f"Percentage Error Lambda 1: {error_lambda_1:.6f}%")
    print(f"Percentage Error Lambda 2: {error_lambda_2:.6f}%")

    # Save training summary to a text file
    with open('training/Burgers_ctin_training_summary.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
        file.write(f"Total iterations: {iter}\n") 
        file.write(f"Final Loss: {final_loss:.6f}\n")
        file.write(f"Percentage Error Lambda 1: {error_lambda_1:.6f}%\n")
        file.write(f"Percentage Error Lambda 2: {error_lambda_2:.6f}%\n")

    # Convert results to numpy array for processing
    results = np.array(results)
    # Calculate percentage errors for lambda_1 and lambda_2
    error_lambda_1s = np.abs(np.array(lambda_1s) - 1.0) / 1.0 * 100
    error_lambda_2s = np.abs(np.array(lambda_2s) - nu) / nu * 100
    # Save results and errors to CSV files
    np.savetxt("training/Burgers_ctin_training_data.csv", np.column_stack([results[:,0], results[:,1], error_lambda_1s, error_lambda_2s]), delimiter=",", header="Iter,Loss,ErrorLambda1,ErrorLambda2", comments="")
    np.savetxt("training/lambda_1s.csv", lambda_1s, delimiter=",", header="Lambda1", comments="")    
    np.savetxt("training/lambda_2s.csv", lambda_2s, delimiter=",", header="Lambda2", comments="")
    # Save model state
    torch.save(model.state_dict(), 'Burgers_ctin.pt')     