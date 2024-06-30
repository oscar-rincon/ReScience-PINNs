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

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from functools import partial


def fwd_gradients_0(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Computes the first-order forward gradient of a given tensor with respect to another tensor.

    Args:
        dy (torch.Tensor): The tensor whose gradient will be computed.
        x (torch.Tensor): The tensor with respect to which the gradient of `dy` will be computed.

    Returns:
        torch.Tensor: The computed first-order forward gradient.
    """
    z = torch.ones(dy.shape, device=dy.device).requires_grad_()
    g = torch.autograd.grad(dy, x, grad_outputs=z, create_graph=True)[0]
    ones = torch.ones(g.shape, device=g.device)
    return torch.autograd.grad(g, z, grad_outputs=ones, create_graph=True)[0]
 
def net_U0(model, x_pt, lambda_1, lambda_2, dt, IRK_alpha):
    """
    Computes the prediction of U0 using the given neural network model and parameters.
 
    Args:
        model (torch.nn.Module): The neural network model used for prediction.
        x_pt (torch.Tensor): The input tensor for which the prediction is made.
        lambda_1 (float): The coefficient for the non-linear term in the KdV equation.
        lambda_2 (float): The log-transformed coefficient for the third spatial derivative term in the KdV equation.
        dt (float): The time step size used in the integration.
        IRK_alpha (torch.Tensor): The IRK weights used in the integration.

    Returns:
        torch.Tensor: The predicted value of U0 after one time step.
    """
    lambda_2 = torch.exp(lambda_2)
    U = model(x_pt)
    U_x = fwd_gradients_0(U, x_pt)
    U_xx = fwd_gradients_0(U_x, x_pt)
    U_xxx = fwd_gradients_0(U_xx, x_pt)
    F = -lambda_1 * U * U_x - lambda_2 * U_xxx
    U0 = U - dt * torch.matmul(F, IRK_alpha.T)
    return U0
 
def net_U1(model, x_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta):
    """
    Computes the prediction of U1 using the given neural network model and parameters.

    Args:
        model (torch.nn.Module): The neural network model used for prediction.
        x_pt (torch.Tensor): The input tensor for which the prediction is made.
        lambda_1 (float): The coefficient for the non-linear term in the KdV equation.
        lambda_2 (float): The log-transformed coefficient for the third spatial derivative term in the KdV equation.
        dt (float): The time step size used in the integration.
        IRK_alpha (torch.Tensor): The IRK weights used in the integration for alpha coefficients.
        IRK_beta (torch.Tensor): The IRK weights used in the integration for beta coefficients.

    Returns:
        torch.Tensor: The predicted value of U1 after one time step, adjusted by IRK weights.
    """
    lambda_2 = torch.exp(lambda_2)
    U = model(x_pt)
    U_x = fwd_gradients_0(U, x_pt)
    U_xx = fwd_gradients_0(U_x, x_pt)
    U_xxx = fwd_gradients_0(U_xx, x_pt)
    F = -lambda_1 * U * U_x - lambda_2 * U_xxx
    U1 = U + dt * torch.matmul(F, (IRK_beta - IRK_alpha).T)
    return U1 
 
def mse(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt):
    """
    Computes the mean squared error loss for predictions of U0 and U1 against true values.

    Args:
        model (torch.nn.Module): The neural network model used for prediction.
        x0_pt (torch.Tensor): The input tensor for predicting U0.
        x1_pt (torch.Tensor): The input tensor for predicting U1.
        lambda_1 (float): The coefficient for the non-linear term in the KdV equation.
        lambda_2 (float): The log-transformed coefficient for the third spatial derivative term in the KdV equation.
        dt (float): The time step size used in the integration.
        IRK_alpha (torch.Tensor): The IRK weights used in the integration for alpha coefficients.
        IRK_beta (torch.Tensor): The IRK weights used in the integration for beta coefficients.
        u0_pt (torch.Tensor): The true values of U0.
        u1_pt (torch.Tensor): The true values of U1.

    Returns:
        torch.Tensor: The computed mean squared error loss.
    """
    U0 = net_U0(model, x0_pt, lambda_1, lambda_2, dt, IRK_alpha)
    U1 = net_U1(model, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    loss = torch.sum((u0_pt - U0) ** 2) + torch.sum((u1_pt - U1) ** 2)
    return loss

# Function to train the model using Adam optimizer
def train_adam(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=50_000):
    """
    Trains the neural network model using the Adam optimizer.
 
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x0_pt (torch.Tensor): The input tensor for the initial condition.
        x1_pt (torch.Tensor): The input tensor for the boundary condition.
        lambda_1 (torch.Tensor): The trainable parameter for the non-linear term in the KdV equation.
        lambda_2 (torch.Tensor): The trainable parameter for the third spatial derivative term in the KdV equation.
        dt (float): The time step size used in the integration.
        IRK_alpha (torch.Tensor): The IRK weights for alpha coefficients.
        IRK_beta (torch.Tensor): The IRK weights for beta coefficients.
        u0_pt (torch.Tensor): The true solution tensor for the initial condition.
        u1_pt (torch.Tensor): The true solution tensor for the boundary condition.
        num_iter (int, optional): The number of iterations for training. Defaults to 50,000.

    Note:
        The function uses global variables to store the iteration count and training results for logging purposes.
    """    
    optimizer = torch.optim.Adam(list(model.parameters()) + [lambda_1, lambda_2], lr=1e-3)
    global iter
    for i in range(1, num_iter+1):
        iter += 1 
        optimizer.zero_grad()
        loss = mse(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt)
        loss.backward(retain_graph=True)
        optimizer.step()
        lambda_1s.append(lambda_1.item())
        lambda_2s.append(torch.exp(lambda_2).item())
        error_lambda_1 = np.abs(lambda_1.cpu().detach().numpy() - 1.0) / 1.0 * 100
        error_lambda_2 = np.abs(torch.exp(lambda_2).cpu().detach().numpy() - 0.0025) / 0.0025 * 100
        results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
        if i % 1000 == 0:
            torch.save(model.state_dict(), f'models_iters/KdV_noisy_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {torch.exp(lambda_2).cpu().detach().numpy().item()}")

# Function to train the model using L-BFGS optimizer
def train_lbfgs(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=50_000):
    """
    Trains the neural network model using the L-BFGS optimizer.
 
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x0_pt (torch.Tensor): The input tensor for the initial condition.
        x1_pt (torch.Tensor): The input tensor for the boundary condition.
        lambda_1 (torch.Tensor): The trainable parameter for the non-linear term in the KdV equation.
        lambda_2 (torch.Tensor): The trainable parameter for the third spatial derivative term in the KdV equation.
        dt (float): The time step size used in the integration.
        IRK_alpha (torch.Tensor): The IRK weights for alpha coefficients.
        IRK_beta (torch.Tensor): The IRK weights for beta coefficients.
        u0_pt (torch.Tensor): The true solution tensor for the initial condition.
        u1_pt (torch.Tensor): The true solution tensor for the boundary condition.
        num_iter (int, optional): The number of iterations for the L-BFGS optimization. Defaults to 50,000.

    Note:
        The closure function, which is required by the L-BFGS optimizer to compute the loss and gradients,
        is not defined within this docstring. It should be defined externally and passed to this function.
    """    
    optimizer = torch.optim.LBFGS(list(model.parameters()) + [lambda_1, lambda_2],
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  history_size=100,
                                  tolerance_grad=1e-7,
                                  line_search_fn='strong_wolfe',
                                  tolerance_change=1.0 * np.finfo(float).eps)
    closure_fn = partial(closure, model, optimizer, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt)
    optimizer.step(closure_fn) 
    
# Closure function for L-BFGS optimization
def closure(model, optimizer, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt):
    """
    Performs a single optimization step using the L-BFGS algorithm.
 
    Args:
        model (torch.nn.Module): The neural network model being optimized.
        optimizer (torch.optim.Optimizer): The optimizer instance, expected to be an L-BFGS optimizer.
        x0_pt (torch.Tensor): Tensor representing initial condition inputs.
        x1_pt (torch.Tensor): Tensor representing boundary condition inputs.
        lambda_1 (torch.Tensor): Trainable parameter representing a coefficient in the differential equation.
        lambda_2 (torch.Tensor): Trainable parameter representing another coefficient in the differential equation.
        dt (float): Time step size used in numerical integration.
        IRK_alpha (torch.Tensor): Tensor of alpha coefficients for IRK integration.
        IRK_beta (torch.Tensor): Tensor of beta coefficients for IRK integration.
        u0_pt (torch.Tensor): Tensor representing the true solution for the initial condition.
        u1_pt (torch.Tensor): Tensor representing the true solution for the boundary condition.

    Returns:
        torch.Tensor: The loss computed for the current set of model parameters.
    """    
    optimizer.zero_grad()
    loss = mse(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    lambda_1s.append(lambda_1.item())
    lambda_2s.append(torch.exp(lambda_2).item())
    error_lambda_1 = np.abs(lambda_1.detach().numpy() - 1.0) / 1.0 * 100
    error_lambda_2 = np.abs(torch.exp(lambda_2).detach().numpy() - 0.0025) / 0.0025 * 100
    results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
    if iter % 100 == 0:
        torch.save(model.state_dict(), f'models_iters/KdV_noisy_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.detach().numpy().item()} - l2: {torch.exp(lambda_2).detach().numpy().item()}")
    return loss    

# Main function
if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(42)  
    iter = 0      

    # Create directories if they do not exist
    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')
    if not os.path.exists('training'):
        os.makedirs('training')

    # Check for GPU availability and set the device accordingly
    device = torch.device('cpu') # Uncomment the next part to enable GPU if available: torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f'Using device: {device}')

    # Define simulation parameters
    q = 50  # Number of IRK weights
    skip = 120  # Time steps to skip for the second set of initial conditions
    N0 = 199  # Number of initial condition points for the first set
    N1 = 201  # Number of initial condition points for the second set
    layers = [1, 50, 50, 50, 50, q]  # Neural network layers configuration

    # Load the dataset
    data = scipy.io.loadmat('../Data/KdV.mat')
    t_star = data['tt'].flatten()[:,None]  # Time data
    x_star = data['x'].flatten()[:,None]  # Spatial data
    Exact = np.real(data['uu'])  # Solution matrix
    idx_t = 40  # Time index for extracting initial conditions
    noise = 0.01  # Noise level

    # Sample initial conditions with noise for training
    idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x,:].float()
    u0 = Exact[idx_x,idx_t][:,None] 
    u0 = u0 + noise * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])

    idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x,:].float()
    u1 = Exact[idx_x,idx_t + skip][:,None] 
    u1= u1 + noise * np.std(u1) * np.random.randn(u1.shape[0], u1.shape[1])

    # Calculate the time difference for the second set of initial conditions
    dt = torch.tensor((t_star[idx_t+skip] - t_star[idx_t]).item()).to(device)

    # Determine the domain bounds
    lb = x_star.min(0)
    ub = x_star.max(0)    

    # Load IRK weights for time integration
    tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
    weights = np.reshape(tmp[0:q**2+q], (q+1, q))    
    IRK_alpha = torch.from_numpy(weights[0:-1,:]).float().to(device)
    IRK_beta = torch.from_numpy(weights[-1:,:]).float().to(device)       
    IRK_times = tmp[q**2+q:]

    # Convert sampled data to PyTorch tensors and move to the specified device
    x0_pt = torch.from_numpy(x0).float().to(device)
    x0_pt.requires_grad = True
    x1_pt = torch.from_numpy(x1).float().to(device)
    x1_pt.requires_grad = True
    u0_pt = torch.from_numpy(u0).to(device)
    u1_pt = torch.from_numpy(u1).to(device)

    # Initialize lambda parameters as trainable
    lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
    lambda_2 = torch.nn.Parameter(torch.tensor([-6.0], dtype=torch.float32, requires_grad=True, device=device))  
    lambda_1s = []  # List to store lambda_1 values during training
    lambda_2s = []  # List to store lambda_2 values during training
    results = []  # List to store training results
    
    # Initialize the model and apply initial weights
    model = MLP(input_size=1, output_size=50, hidden_layers=5, hidden_units=50, activation_function=nn.Tanh()).to(device)
    model.apply(init_weights)
    
    # Training with Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=50_000)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")    
    
    # Training with L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")
    
    # Total training time
    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Final loss value
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6f}")

    # Calculate percentage error for lambda_1 and lambda_2
    error_lambda_1 = np.abs(lambda_1s[-1] - 1.0) / 1.0 * 100
    error_lambda_2 = np.abs(lambda_2s[-1] - 0.0025) / 0.0025 * 100
    print(f"Percentage Error Lambda 1: {error_lambda_1:.6f}%")
    print(f"Percentage Error Lambda 2: {error_lambda_2:.6f}%")

    # Save training summary to a text file
    with open('training/KdV_training_summary_noisy.txt', 'w') as file:
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
    error_lambda_2s = np.abs(np.array(lambda_2s) - 0.0025) / 0.0025 * 100
    # Save results and errors to CSV files
    np.savetxt("training/KdV_training_data_noisy.csv", np.column_stack([results[:,0], results[:,1], error_lambda_1s, error_lambda_2s]), delimiter=",", header="Iter,Loss,ErrorLambda1,ErrorLambda2", comments="")
    np.savetxt("training/lambda_1s_noisy.csv", lambda_1s, delimiter=",", header="Lambda1", comments="")    
    np.savetxt("training/lambda_2s_noisy.csv", lambda_2s, delimiter=",", header="Lambda2", comments="")
    # Save model state
    torch.save(model.state_dict(), f'KdV_noisy.pt')   
