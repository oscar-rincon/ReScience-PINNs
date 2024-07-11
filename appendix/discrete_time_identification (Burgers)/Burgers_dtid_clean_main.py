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
        x (torch.Tensor): The spatial input tensor for the system.
        lambda_1 (torch.Tensor): The first learnable parameter for the PDE solution.
        lambda_2 (torch.Tensor): The second learnable parameter for the PDE solution, before applying the exponential function.
        dt (float): The time step for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients for the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients for the IRK integration method.

    Returns:
        torch.Tensor: The predicted state of the system at the next time step after applying the IRK integration method.
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
        x (torch.Tensor): The spatial input tensor for the system.
        lambda_1 (torch.Tensor): The first learnable parameter for the PDE solution.
        lambda_2 (torch.Tensor): The second learnable parameter for the PDE solution, before applying the exponential function.
        dt (float): The time step for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients for the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients for the IRK integration method.

    Returns:
        torch.Tensor: The predicted state of the system at the next time step after applying the IRK integration method.
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
    Calculates the mean squared error (MSE) between the predicted and actual values for two time steps.

    This function computes the MSE for the predictions of a neural network model at two different time steps. It uses
    the model to predict the system's state at these time steps and compares these predictions to the actual values.

    Args:
        model (torch.nn.Module): The neural network model used for prediction.
        x0 (torch.Tensor): The spatial input tensor for the first time step.
        x1 (torch.Tensor): The spatial input tensor for the second time step.
        u0 (torch.Tensor): The actual values of the system's state at the first time step.
        u1 (torch.Tensor): The actual values of the system's state at the second time step.
        lambda_1 (torch.Tensor): The first learnable parameter for the PDE solution.
        lambda_2 (torch.Tensor): The second learnable parameter for the PDE solution.
        dt (float): The time difference between the two time steps.
        IRK_alpha (torch.Tensor): The IRK (Implicit Runge-Kutta) alpha coefficients.
        IRK_beta (torch.Tensor): The IRK beta coefficients.

    Returns:
        torch.Tensor: The calculated mean squared error between the predicted and actual values for the two time steps.
    """     
    U0 = net_U0(model, x0, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    U1 = net_U1(model, x1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    return torch.sum((u0-U0)**2)  + torch.sum((u1-U1)**2)  

def train_adam(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, num_iter=50_000):
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
        loss = mse(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
        loss.backward(retain_graph=True)
        optimizer.step()
        lambda_1s.append(lambda_1.item())
        lambda_2s.append(torch.exp(lambda_2).item())
        error_lambda_1 = np.abs(lambda_1.cpu().detach().numpy() - 1.0) / 1.0 * 100
        error_lambda_2 = np.abs(torch.exp(lambda_2).cpu().detach().numpy() - nu.cpu().detach().numpy()) / nu.cpu().detach().numpy() * 100
        results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
        if i % 100 == 0:
            torch.save(model.state_dict(), f'models_iters/Burgers_clean_dtid_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {torch.exp(lambda_2).cpu().detach().numpy().item()}")

def closure(model, optimizer, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta):
    """
    Performs a single optimization step using the provided model and optimizer, and calculates the loss based on the mean squared error between the model's predictions and the observed data, as well as the PDE residual.

    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        optimizer (torch.optim.Optimizer): The optimizer to use for the optimization step.
        x0 (torch.Tensor): The spatial input tensor for the initial state of the system.
        x1 (torch.Tensor): The spatial input tensor for the final state of the system.
        u0 (torch.Tensor): The observed initial state values of the system.
        u1 (torch.Tensor): The observed final state values of the system.
        lambda_1 (torch.Tensor): The first learnable parameter for the PDE solution.
        lambda_2 (torch.Tensor): The second learnable parameter for the PDE solution, before applying the exponential function.
        dt (float): The time step for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients for the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients for the IRK integration method.

    Returns:
        torch.Tensor: The calculated loss for the current optimization step.
    """
    optimizer.zero_grad()
    loss = mse(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    loss.backward(retain_graph=True)
    global iter
    iter += 1    
    lambda_1s.append(lambda_1.item())
    lambda_2s.append(torch.exp(lambda_2).item())
    error_lambda_1 = np.abs(lambda_1.cpu().detach().numpy() - 1.0) / 1.0 * 100
    error_lambda_2 = np.abs(torch.exp(lambda_2).cpu().detach().numpy() -nu.cpu().detach().numpy()) / nu.cpu().detach().numpy() * 100
    results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
    if iter % 100 == 0:
        torch.save(model.state_dict(), f'models_iters/Burgers_dtid_clean_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {torch.exp(lambda_2).cpu().detach().numpy().item()}")
    return loss 

def train_lbfgs(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, num_iter=50_000):
    """
    Trains a neural network model using the LBFGS optimizer to solve a PDE problem by minimizing the difference between observed data and model predictions, as well as ensuring the PDE residuals are minimized.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x0 (torch.Tensor): The spatial input tensor for the initial state of the system.
        x1 (torch.Tensor): The spatial input tensor for the final state of the system.
        u0 (torch.Tensor): The observed initial state values of the system.
        u1 (torch.Tensor): The observed final state values of the system.
        lambda_1 (torch.Tensor): The first learnable parameter for the PDE solution.
        lambda_2 (torch.Tensor): The second learnable parameter for the PDE solution, before applying the exponential function.
        dt (float): The time step for the simulation.
        IRK_alpha (torch.Tensor): The alpha coefficients for the IRK integration method.
        IRK_beta (torch.Tensor): The beta coefficients for the IRK integration method.
        num_iter (int, optional): The maximum number of iterations for the LBFGS optimizer. Defaults to 50,000.

    Note:
        The `closure` function required by the LBFGS optimizer is defined externally and must be available in the
        scope where this function is called. It should accept the model, optimizer, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta as arguments, and return the computed loss.
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

    iter = 0  # Initialize iteration counter
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

    # Initialize model parameters for regularization
    lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device)).float()
    lambda_2 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device)-6).float()
    lambda_1s = []  # List to track lambda_1 values
    lambda_2s = []  # List to track lambda_2 values
    results = []
 
    # Initialize the model and apply initial weights
    model = MLP(input_size=1, output_size=q, hidden_layers=5, hidden_units=50, activation_function=nn.Tanh()).float().to(device)
    model.apply(init_weights)

    # Training phase

    # Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, num_iter=0)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")

    # L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x0, x1, u0, u1, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")

    # Total training time
    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Final loss and L2 error
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6e}")

    # Calculate percentage error for lambda_1 and lambda_2
    error_lambda_1 = np.abs(lambda_1s[-1] - 1.0) / 1.0 * 100
    error_lambda_2 = np.abs(lambda_2s[-1] - nu.cpu().detach().numpy()) / nu.cpu().detach().numpy() * 100
    print(f"Percentage Error Lambda 1: {error_lambda_1:.6f}%")
    print(f"Percentage Error Lambda 2: {error_lambda_2:.6f}%")

    # Save training summary to a text file
    with open('training/Burgers_dtid_clean_training_summary.txt', 'w') as file:
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
    error_lambda_2s = np.abs(np.array(lambda_2s) - nu.cpu().detach().numpy()) / nu.cpu().detach().numpy() * 100
    # Save results and errors to CSV files
    np.savetxt("training/Burgers_dtid_clean_training_data.csv", np.column_stack([results[:,0], results[:,1], error_lambda_1s, error_lambda_2s]), delimiter=",", header="Iter,Loss,ErrorLambda1,ErrorLambda2", comments="")
    np.savetxt("training/lambda_1s_clean.csv", lambda_1s, delimiter=",", header="l1", comments="")    
    np.savetxt("training/lambda_2s_clean.csv", lambda_2s, delimiter=",", header="l2", comments="")
    # Save model state
    torch.save(model.state_dict(), 'Burgers_dtid_clean.pt')        
    
