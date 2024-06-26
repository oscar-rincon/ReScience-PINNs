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
import scipy.io as sp  # SciPy module for MATLAB file I/O

# Import additional utilities
from functools import partial  # Higher-order functions and operations on callable objects
from pyDOE import lhs  # Design of experiments for Python, including Latin Hypercube Sampling

# Import custom modules
from pinns import *  # Physics Informed Neural Networks utilities

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

def f(model, x_f, t_f):
    """
    Computes the components of a PDE for a given model, spatial, and temporal inputs.

    This function evaluates the partial differential equation (PDE) components 'f_u' and 'f_v'
    based on the model's predictions. It is used to calculate the derivatives of the model's
    output with respect to spatial and temporal inputs, which are essential for solving PDEs
    using neural networks.

    Args:
        model: The neural network model used to predict the PDE solution.
        x_f: The spatial input tensor.
        t_f: The temporal input tensor.

    Returns:
        Tuple[Tensor, Tensor]: The computed PDE components 'f_u' and 'f_v'.
    """
    h = model(torch.stack((x_f, t_f), axis = 1))
    u = h[:, 0]
    v = h[:, 1]
    u_t = derivative(u, t_f, order=1)
    v_t = derivative(v, t_f, order=1)
    u_xx = derivative(u, x_f, order=2)
    v_xx = derivative(v, x_f, order=2)
    f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
    f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u 
    return f_u, f_v

def mse_f(model, x_f, t_f):
    """
    Calculates the mean squared error (MSE) of the residuals from a partial differential equation (PDE).

    Args:
        model: The neural network model used for solving the PDE.
        x_f: The spatial input tensor.
        t_f: The temporal input tensor.

    Returns:
        float: The mean squared error of the PDE residuals.
    """
    f_u, f_v = f(model, x_f, t_f)
    return (f_u**2 + f_v**2).mean()

def mse_0(model, x_0, u_0, v_0):
    """
    Calculates the mean squared error (MSE) between the model's predictions and the initial conditions.

    Args:
        model: The neural network model used for predicting the initial conditions.
        x_0: The spatial input tensor for the initial condition.
        u_0: The actual initial condition values for the first component.
        v_0: The actual initial condition values for the second component.

    Returns:
        float: The mean squared error between the predicted and actual initial conditions.
    """
    t_0 = torch.zeros_like(x_0)
    h = model(torch.stack((x_0, t_0), axis = 1))
    h_u = h[:, 0]
    h_v = h[:, 1]
    return ((h_u-u_0)**2+(h_v-v_0)**2).mean()

def mse_b(model, t_b):
    """
    Calculates the mean squared error (MSE) for boundary conditions in a PDE model.

    Args:
        model: The neural network model used for predicting boundary conditions.
        t_b: The temporal input tensor for the boundary conditions.

    Returns:
        float: The total mean squared error for both Dirichlet and Neumann boundary conditions.
    """
    # Left boundary
    x_b_left = torch.zeros_like(t_b)-5
    x_b_left.requires_grad = True
    h_b_left = model(torch.stack((x_b_left, t_b), axis = 1))
    h_u_b_left = h_b_left[:, 0]
    h_v_b_left = h_b_left[:, 1]
    h_u_b_left_x = derivative(h_u_b_left, x_b_left, 1)
    h_v_b_left_x = derivative(h_v_b_left, x_b_left, 1)
    
    # Right boundary
    x_b_right = torch.zeros_like(t_b)+5
    x_b_right.requires_grad = True
    h_b_right = model(torch.stack((x_b_right, t_b), axis = 1))
    h_u_b_right = h_b_right[:, 0]
    h_v_b_right = h_b_right[:, 1]
    h_u_b_right_x = derivative(h_u_b_right, x_b_right, 1)
    h_v_b_right_x = derivative(h_v_b_right, x_b_right, 1)

    # Compute MSE for Dirichlet and Neumann boundary conditions
    mse_drichlet = (h_u_b_left-h_u_b_right)**2+(h_v_b_left-h_v_b_right)**2
    mse_newman = (h_u_b_left_x-h_u_b_right_x)**2+(h_v_b_left_x-h_v_b_right_x)**2
    mse_total = (mse_drichlet + mse_newman).mean()
    
    return mse_total

def closure(model, optimizer, x_f, t_f, x_0, u_0, v_0, h_0, t):
    """
    Performs a single optimization step using LBFGS optimizer for a neural network model.

    Args:
        model: The neural network model being optimized.
        optimizer: The LBFGS optimizer instance.
        x_f: The spatial input tensor for the PDE residual calculation.
        t_f: The temporal input tensor for the PDE residual calculation.
        x_0: The spatial input tensor for the initial condition.
        u_0: The actual initial condition values for the first component.
        v_0: The actual initial condition values for the second component.
        h_0: Not used in the function but typically represents initial condition values for comparison.
        t: The temporal input tensor for the boundary condition.

    Returns:
        torch.Tensor: The total loss computed as the sum of the PDE residual, initial condition, and boundary condition losses.
    """
    optimizer.zero_grad()
    loss = mse_f(model, x_f, t_f) + mse_0(model, x_0, u_0, v_0) + mse_b(model, t)
    loss.backward(retain_graph=True)
    global iter
    iter += 1
    pred = model(X_star)
    h_pred = (pred[:, 0]**2 + pred[:, 1]**2)**0.5
    error = np.linalg.norm(h_star-h_pred.cpu().detach().numpy(),2)/np.linalg.norm(h_star,2) 
    results.append([iter, loss.item(), error])    
    if iter % 100 == 0:
        torch.save(model.state_dict(), f'models_iters/Schrodinger_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - L2: {error}")
    return loss

def train_adam(model, x_f, t_f, x_0, u_0, v_0, h_0, t, num_iter=50_000):
    """
    Trains a neural network model using the Adam optimizer over a specified number of iterations.

    Args:
        model: The neural network model to be trained.
        x_f: The spatial input tensor for the PDE residual calculation.
        t_f: The temporal input tensor for the PDE residual calculation.
        x_0: The spatial input tensor for the initial condition.
        u_0: The actual initial condition values for the first component.
        v_0: The actual initial condition values for the second component.
        h_0: Not used in the function but typically represents initial condition values for comparison.
        t: The temporal input tensor for the boundary condition.
        num_iter (int, optional): The number of iterations to train the model. Defaults to 50,000.

    Note:
        The function uses global variables `iter` and `results` to track the iteration count and to store
        the training progress, respectively. Ensure these are properly initialized before calling this function.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    global iter
     
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        loss = mse_f(model, x_f, t_f) + mse_0(model, x_0, u_0, v_0) + mse_b(model, t)
        loss.backward(retain_graph=True)
        optimizer.step()
        pred = model(X_star)
        h_pred = (pred[:, 0]**2 + pred[:, 1]**2)**0.5
        error = np.linalg.norm(h_star-h_pred.cpu().detach().numpy(),2)/np.linalg.norm(h_star,2) 
        results.append([iter, loss.item(), error])
        iter += 1
        if iter % 100 == 0:
            torch.save(model.state_dict(), f'models_iters/Schrodinger_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - L2: {error}")

# Function for L-BFGS training
def train_lbfgs(model,  x_f, t_f, x_0, u_0, v_0, h_0, t, num_iter=50_000):
    """
    Trains a neural network model using the L-BFGS optimizer to solve differential equations.

    Args:
        model: The neural network model to be trained.
        x_f: The spatial input tensor for the PDE residual calculation.
        t_f: The temporal input tensor for the PDE residual calculation.
        x_0: The spatial input tensor for the initial condition.
        u_0: The actual initial condition values for the first component.
        v_0: The actual initial condition values for the second component.
        h_0: Not used in the function but typically represents initial condition values for comparison.
        t: The temporal input tensor for the boundary condition.
        num_iter (int, optional): The maximum number of iterations for the L-BFGS optimizer. Defaults to 50,000.

    Note:
        The function uses a partial function `closure_fn` as a closure required by the L-BFGS optimizer.
        This closure computes the loss and gradients at each optimization step.
    """
    optimizer = torch.optim.LBFGS(model.parameters(),
                                    lr=0.001,
                                    max_iter=num_iter,
                                    max_eval=num_iter,
                                    tolerance_grad=1e-5,
                                    history_size=100,
                                    tolerance_change=1.0 * np.finfo(float).eps,
                                    line_search_fn="strong_wolfe")
 
    closure_fn = partial(closure, model, optimizer, x_f, t_f, x_0, u_0, v_0, h_0, t)
    optimizer.step(closure_fn)

if __name__== "__main__":
    set_seed(42)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize iteration counter
    iter = 0    

    # Define lower and upper bounds for the domain
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])     

    # Set the number of initial, boundary, and collocation points
    N0 = 50  
    N_b = 50  
    N_f = 20_000 

    # Load data from MATLAB file
    data = sp.loadmat('../Data/NLS.mat')

    # Prepare initial condition data
    x_0 = torch.from_numpy(data['x'].astype(np.float32)).flatten().T
    x_0.requires_grad = True
    t = torch.from_numpy(data['tt'].astype(np.float32)).flatten().T
    t.requires_grad = True
    h = torch.from_numpy(data['uu'])
    u_0 = torch.real(h)[:, 0]
    v_0 = torch.imag(h)[:, 0]
    h_0 = torch.stack((u_0, v_0), axis=1)

    # Generate collocation points using Latin Hypercube Sampling
    c_f = lb + (ub-lb)*lhs(2, N_f)
    x_f = torch.from_numpy(c_f[:, 0].astype(np.float32))
    x_f.requires_grad = True
    t_f = torch.from_numpy(c_f[:, 1].astype(np.float32))
    t_f.requires_grad = True

    # Select random samples for initial and boundary conditions
    idx_0 = np.random.choice(x_0.shape[0], N0, replace=False)
    x_0 = x_0[idx_0]
    u_0 = u_0[idx_0]
    v_0 = v_0[idx_0]
    h_0 = h_0[idx_0]

    idx_b = np.random.choice(t.shape[0], N_b, replace=False)
    t_b = t[idx_b].to(device)  # Ensure boundary condition points are on the correct device

    # Create meshgrid for plotting
    X, T = torch.meshgrid(torch.tensor(data['x'].flatten()), torch.tensor(data['tt'].flatten()))
    xcol = X.reshape(-1, 1)
    tcol = T.reshape(-1, 1)
    X_star = torch.cat((xcol, tcol), 1).float()   

    # Extract exact solution for comparison
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2) 
    h_star = Exact_h.flatten()[:]

    # Initialize the model with specified parameters and apply weights
    model = MLP(input_size=2, output_size=2, hidden_layers=5, hidden_units=100, activation_function=nn.Tanh())
    model.apply(init_weights)
    
    # Move tensors to device
    x_f=x_f.to(device)
    t_f=t_f.to(device)
    x_0=x_0.to(device)
    u_0=u_0.to(device)
    v_0=v_0.to(device)
    h_0=h_0.to(device)
    t_b=t_b.to(device)
    X_star=X_star.to(device)
    model.to(device)
    
    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')

    if not os.path.exists('training'):
        os.makedirs('training')
    results = []
        
    # Training with Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x_f, t_f, x_0, u_0, v_0, h_0, t_b, num_iter=0)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")

    # Training with L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x_f, t_f, x_0, u_0, v_0, h_0, t_b, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")

    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Obtain the final loss L2
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6f}")

    # Obtain the final L2 error
    final_l2 = results[-1][2]
    print(f"Final L2: {final_l2:.6f}")

    # Save training summary to a text file
    with open('training/Schrodinger_training_summary.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
        file.write(f"Total iterations: {iter}\n")   
        file.write(f"Final Loss: {final_loss:.6f}\n")
        file.write(f"Final L2: {final_l2:.6f}\n")

    # Convert results to numpy array, save training data to CSV, and save model state
    results = np.array(results)
    np.savetxt("training/Schrodinger_training_data.csv", results, delimiter=",", header="Iter,Loss,L2", comments="")
    torch.save(model.state_dict(), f'Schrodinger.pt')
