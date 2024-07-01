# Standard library imports
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
 

# Third-party imports
import torch
import torch.nn as nn
import numpy as np
import scipy.io
from functools import partial  # Higher-order functions and operations on callable objects

# Disable all warnings (not recommended for production code)
import warnings
warnings.filterwarnings("ignore")
 
def f(model, x_train_pt, y_train_pt, t_train_pt):
    """
    Calculates the velocity components (u, v), pressure (p), and residuals of the Navier-Stokes equations (f_u, f_v)
    for a given set of input points using the provided model.

    Args:
        model (torch.nn.Module): The neural network model used to predict the solution.
        x_train_pt (torch.Tensor): The x-coordinates of the training points.
        y_train_pt (torch.Tensor): The y-coordinates of the training points.
        t_train_pt (torch.Tensor): The time instances of the training points.

    Returns:
        tuple: A tuple containing:
            - u (torch.Tensor): The x-component of velocity at the training points.
            - v (torch.Tensor): The y-component of velocity at the training points.
            - p (torch.Tensor): The pressure at the training points.
            - f_u (torch.Tensor): The residual of the u-momentum equation of the Navier-Stokes equations.
            - f_v (torch.Tensor): The residual of the v-momentum equation of the Navier-Stokes equations.
    """
    psi_and_p = model(torch.stack((x_train_pt, y_train_pt, t_train_pt), axis=1).view(-1, 3))
    psi = psi_and_p[:, 0:1]
    p = psi_and_p[:, 1:2]
    
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
    
    f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
    f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)
    
    return u, v, p, f_u, f_v

def mse(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt):
    """
    Computes the mean squared error loss for the Navier-Stokes problem. This includes the error in the predicted
    velocity components (u, v) against their true values, and the residuals of the Navier-Stokes equations (f_u, f_v).

    Args:
        model (torch.nn.Module): The neural network model used for predictions.
        x_train_pt (torch.Tensor): The x-coordinates of the training points.
        y_train_pt (torch.Tensor): The y-coordinates of the training points.
        t_train_pt (torch.Tensor): The time instances of the training points.
        u_train_pt (torch.Tensor): The true x-component of velocity at the training points.
        v_train_pt (torch.Tensor): The true y-component of velocity at the training points.

    Returns:
        torch.Tensor: The computed mean squared error loss as a float tensor.
    """
    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = f(model, x_train_pt, y_train_pt, t_train_pt)    
    loss = torch.sum((u_train_pt - u_pred) ** 2) + torch.sum((v_train_pt - v_pred) ** 2) + torch.sum((f_u_pred) ** 2) + torch.sum((f_v_pred) ** 2)
    return loss.float()

def train_adam(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000):
    """
    Trains the given model using the Adam optimizer to solve the Navier-Stokes equations.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x_train_pt (torch.Tensor): The x-coordinates of the training data points.
        y_train_pt (torch.Tensor): The y-coordinates of the training data points.
        t_train_pt (torch.Tensor): The time instances of the training data points.
        u_train_pt (torch.Tensor): The true x-component of velocity at the training data points.
        v_train_pt (torch.Tensor): The true y-component of velocity at the training data points.
        num_iter (int, optional): The number of iterations for the training process. Defaults to 50,000.

    Note:
        The function uses global variables `iter`, `lambda_1s`, `lambda_2s`, and `results` to track the iteration count,
        the history of lambda_1 and lambda_2 values, and the training results, respectively.
    """    
    optimizer = torch.optim.Adam(list(model.parameters()) + [lambda_1, lambda_2], lr=1e-3)
    global iter
     
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        loss = mse(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt)
        loss.backward(retain_graph=True)
        optimizer.step()
        lambda_1s.append(lambda_1.item())
        lambda_2s.append(lambda_2.item())
        error_lambda_1 = np.abs(lambda_1.cpu().detach().numpy() - 1.0) * 100
        error_lambda_2 = np.abs(lambda_2.cpu().detach().numpy() - 0.01) / 0.01 * 100
        results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
        iter += 1
        if iter % 1000 == 0:
            torch.save(model.state_dict(), f'models_iters/NS_noisy_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {lambda_2.cpu().detach().numpy().item()}")
 
def train_lbfgs(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000):
    """
    Trains the given model using the L-BFGS optimizer, specifically tailored for solving the Navier-Stokes equations.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x_train_pt (torch.Tensor): The x-coordinates of the training data points.
        y_train_pt (torch.Tensor): The y-coordinates of the training data points.
        t_train_pt (torch.Tensor): The time instances of the training data points.
        u_train_pt (torch.Tensor): The true x-component of velocity at the training data points.
        v_train_pt (torch.Tensor): The true y-component of velocity at the training data points.
        num_iter (int, optional): The maximum number of iterations for the L-BFGS optimizer. Defaults to 50,000.

    Note:
        The function uses a closure function to compute the loss and perform the backward pass, which is a requirement
        for the L-BFGS optimizer in PyTorch. The optimizer's `step` method is called with this closure function.
    """    
    optimizer = torch.optim.LBFGS(list(model.parameters()) + [lambda_1, lambda_2],
                                  lr=1,
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  tolerance_grad=1e-7,
                                  history_size=100,
                                  tolerance_change=1.0 * np.finfo(float).eps,
                                  line_search_fn="strong_wolfe")
 
    closure_fn = partial(closure, model, optimizer, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000)
    optimizer.step(closure_fn)
 
def closure(model, optimizer, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000):
    """
    Performs a single optimization step using the L-BFGS algorithm.

    Args:
        model (torch.nn.Module): The neural network model being optimized.
        optimizer (torch.optim.Optimizer): The L-BFGS optimizer instance.
        x_train_pt (torch.Tensor): The tensor containing the x-coordinates of the training data points.
        y_train_pt (torch.Tensor): The tensor containing the y-coordinates of the training data points.
        t_train_pt (torch.Tensor): The tensor containing the time instances of the training data points.
        u_train_pt (torch.Tensor): The tensor containing the true x-component of velocity at the training data points.
        v_train_pt (torch.Tensor): The tensor containing the true y-component of velocity at the training data points.
        num_iter (int, optional): The maximum number of iterations for the optimization. Defaults to 50,000.

    Returns:
        torch.Tensor: The loss computed for the current set of model parameters.

    Note:
        This function updates global variables `iter`, `lambda_1s`, `lambda_2s`, and `results` to track the optimization
        progress, including the iteration count, the values of the regularization parameters `lambda_1` and `lambda_2`,
        and the training loss. It prints the progress every 1000 iterations.
    """    
    optimizer.zero_grad()
    loss = mse(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    lambda_1s.append(lambda_1.item())
    lambda_2s.append(lambda_2.item())
    error_lambda_1 = np.abs(lambda_1.cpu().detach().numpy() - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2.cpu().detach().numpy() - 0.01) / 0.01 * 100
    results.append([iter, loss.item(), error_lambda_1.item(), error_lambda_2.item()])
    if iter % 1000 == 0:
        torch.save(model.state_dict(), f'models_iters/NS_noisy_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {lambda_2.cpu().detach().numpy().item()}")
    return loss

# Main function
if __name__== "__main__":
    # Set the seed for reproducibility
    set_seed(42)
    iter = 0

    # Check and print the GPU or CPU device being used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create directories if they do not exist
    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')
    if not os.path.exists('training'):
        os.makedirs('training')

    # Initialize training parameters
    N_train = 5000
    results = []

    # Load dataset
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
    U_star = data['U_star']  # Velocity data: N x 2 x T
    P_star = data['p_star']  # Pressure data: N x T
    t_star = data['t']       # Time steps: T x 1
    X_star = data['X_star']  # Spatial coordinates: N x 2

    # Determine the number of spatial points and time steps
    N = X_star.shape[0]  # Number of spatial points
    T = t_star.shape[0]  # Number of time steps

    # Rearrange data for training
    XX = np.tile(X_star[:, 0:1], (1, T))  # Repeat X for each time step
    YY = np.tile(X_star[:, 1:2], (1, T))  # Repeat Y for each time step
    TT = np.tile(t_star, (1, N)).T        # Repeat time for each spatial point

    UU = U_star[:, 0, :]  # U velocity component
    VV = U_star[:, 1, :]  # V velocity component
    PP = P_star           # Pressure

    # Flatten data for training
    x = XX.flatten()[:, None]  # Flatten X
    y = YY.flatten()[:, None]  # Flatten Y
    t = TT.flatten()[:, None]  # Flatten time
    u = UU.flatten()[:, None]  # Flatten U velocity
    v = VV.flatten()[:, None]  # Flatten V velocity
    p = PP.flatten()[:, None]  # Flatten pressure

    # Select random indices for training data
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]
    t_train = t[idx, :]
    u_train = u[idx, :]
    v_train = v[idx, :]

    # Convert training data to PyTorch tensors and enable gradients
    x_train_pt = torch.from_numpy(x_train).float().requires_grad_(True)
    y_train_pt = torch.from_numpy(y_train).float().requires_grad_(True)
    t_train_pt = torch.from_numpy(t_train).float().requires_grad_(True)

    # Add noise to velocity data for training
    noise = 0.01
    u_train += noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])
    v_train += noise * np.std(v_train) * np.random.randn(v_train.shape[0], v_train.shape[1])
    u_train_pt = torch.from_numpy(u_train).float()
    v_train_pt = torch.from_numpy(v_train).float()

    # Initialize model parameters
    lambda_1 = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
    lambda_2 = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=True)
    lambda_1s = []
    lambda_2s = []

    # Move tensors to the specified device
    x_train_pt = x_train_pt.to(device)
    y_train_pt = y_train_pt.to(device)
    t_train_pt = t_train_pt.to(device)
    u_train_pt = u_train_pt.to(device)
    v_train_pt = v_train_pt.to(device)

    # Initialize and configure the neural network model
    model = MLP(input_size=3, output_size=2, hidden_layers=9, hidden_units=20, activation_function=nn.Tanh()).to(device)
    model.apply(init_weights)
    
    # Training with Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=200_000)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")
    
    # Training with L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x_train_pt, y_train_pt, t_train_pt, u_train_pt, v_train_pt, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")
    
    # Total training time
    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Final loss value
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6f}")

    # Final L2 loss value
    final_l2 = results[-1][2]
    print(f"Final L2: {final_l2:.6f}")

    # model_path = f'NS_noisy.pt'
    # model = NSNN().to(device)
    # model.load_state_dict(torch.load(model_path))
    
    # Define the snapshot index for test data
    snap = np.array([100])

    # Extract test data for x, y, and t coordinates
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]
    t_star = TT[:, snap]

    # Extract velocity (u, v) and pressure (p) from test data
    u_star = U_star[:, 0, snap]
    v_star = U_star[:, 1, snap]
    p_star = P_star[:, snap]

    # Convert numpy arrays to torch tensors and set to require gradients
    x_star_pt = torch.from_numpy(x_star).float().to(device)
    x_star_pt.requires_grad = True
    y_star_pt = torch.from_numpy(y_star).float().to(device)
    y_star_pt.requires_grad = True
    t_star_pt = torch.from_numpy(t_star).float().to(device)
    t_star_pt.requires_grad = True

    # Predict velocity and pressure using the model
    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = f(model, x_star_pt, y_star_pt, t_star_pt)

    # Convert predictions back to numpy arrays
    u_pred = u_pred.cpu().detach().numpy()
    v_pred = v_pred.cpu().detach().numpy()
    p_pred = p_pred.cpu().detach().numpy()

    # Calculate the values of lambda parameters
    lambda_1_value = lambda_1.cpu().detach().numpy().item()
    lambda_2_value = lambda_2.cpu().detach().numpy().item()

    # Calculate errors between predicted and actual values
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - 0.01) / 0.01 * 100

    # Print error metrics
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))

    # Save training summary to a text file
    with open('training/NS_training_summary_noisy.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
        file.write(f"Total iterations: {iter}\n")
        file.write(f"Final Loss: {final_loss:.6f}\n")
        file.write(f"Final L2: {final_l2:.6f}\n")
        file.write(f"Percentage Error Lambda 1: {error_lambda_1:.6f}%\n")
        file.write(f"Percentage Error Lambda 2: {error_lambda_2:.6f}%\n")

    # Save training data and lambda values to CSV files
    np.savetxt("training/NS_training_data_noisy.csv", results, delimiter=",", header="Iter,Loss,l1,l2", comments="")
    np.savetxt("training/lambda_1s_noisy.csv", lambda_1s, delimiter=",", header="l1", comments="")
    np.savetxt("training/lambda_2s_noisy.csv", lambda_2s, delimiter=",", header="l2", comments="")

    # Save the model's state
    torch.save(model.state_dict(), 'NS_noisy.pt')
