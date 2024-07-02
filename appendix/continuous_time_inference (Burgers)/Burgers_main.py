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
    u = model(torch.cat((x, t), dim=1))
    u_t = derivative(u, t, order=1)
    u_x = derivative(u, x, order=1)
    u_xx = derivative(u, x, order=2)
    f = u_t + u * u_x - nu * u_xx
    return f

def mse_f(model, x, t, nu):
    f_pred = f(model, x, t, nu)
    return (f_pred**2).mean()

def mse_u(model, x, t, u_train_pt):
    u = model(torch.cat((x, t), dim=1))
    return ((u_train_pt - u) ** 2).mean()

def train_adam(model, x_u, x_f, t_u, t_f, nu, u_train_pt, num_iter=50_000):
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
        loss = mse_f(model, x_f, t_f, nu) + mse_u(model, x_u, t_u, u_train_pt)
        loss.backward(retain_graph=True)
        optimizer.step()
        u_pred = model(torch.cat((x_star, t_star), dim=1))
        error = np.linalg.norm(u_star.cpu().detach().numpy()-u_pred.cpu().detach().numpy(),2)/np.linalg.norm(u_star.cpu().detach().numpy(),2)
        results.append([iter, loss.item(), error])
        iter += 1
        if iter % 1000 == 0:
            torch.save(model.state_dict(), f'models_iters/Burgers_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - L2: {error}")

def closure(model, optimizer, x_u, x_f, t_u, t_f, nu, u_train_pt):
    optimizer.zero_grad()
    loss = mse_f(model, x_f, t_f, nu) + mse_u(model, x_u, t_u, u_train_pt)
    loss.backward(retain_graph=True)
    global iter
    iter += 1    
    u_pred = model(torch.cat((x_star, t_star), dim=1)) 
    error = np.linalg.norm(u_star.cpu().detach().numpy()-u_pred.cpu().detach().numpy(),2)/np.linalg.norm(u_star.cpu().detach().numpy(),2)
    results.append([iter, loss.item(), error]) 
    if iter % 1000 == 0:
        torch.save(model.state_dict(), f'models_iters/Burgers_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - L2: {error}")
    
    return loss

def train_lbfgs(model, x_u, x_f, t_u, t_f, nu, u_train_pt, num_iter=50_000):
    optimizer = torch.optim.LBFGS(model.parameters(),
                                  lr=1,
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  tolerance_grad=1e-7,
                                  history_size=100,
                                  tolerance_change=1.0 * np.finfo(float).eps,
                                  line_search_fn="strong_wolfe")
    closure_fn = partial(closure, model, optimizer, x_u, x_f, t_u, t_f, nu, u_train_pt)
    optimizer.step(closure_fn)

if __name__ == "__main__":
    set_seed(42)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')

    if not os.path.exists('training'):
        os.makedirs('training')
    results = []

    # Initialize iteration counter
    iter = 0  

    nu = 0.01 / np.pi

    noise = 0.0

    N_u = 100
    N_f = 10_000

    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x, t)
    
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None].T              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    x_u = X_u_train[:, 0:1]
    t_u = X_u_train[:, 1:2]
    x_f = X_f_train[:, 0:1]
    t_f = X_f_train[:, 1:2]

    x_star = X_star[:, 0:1]
    t_star = X_star[:, 1:2]
 

    # Initialize the model with specified parameters and apply weights
    model = MLP(input_size=2, output_size=1, hidden_layers=8, hidden_units=20, activation_function=nn.Tanh()).to(device)
    model.apply(init_weights)

    x_u = torch.from_numpy(x_u.astype(np.float32)).to(device)   
    x_u.requires_grad = True 
    x_f = torch.from_numpy(x_f.astype(np.float32)).to(device)   
    x_f.requires_grad = True
    t_u = torch.from_numpy(t_u.astype(np.float32)).to(device)   
    t_u.requires_grad = True 
    t_f = torch.from_numpy(t_f.astype(np.float32)).to(device)   
    t_f.requires_grad = True
    u_train_pt = torch.from_numpy(u_train).float().to(device)    
    nu = torch.tensor(nu).float().to(device)
    x_star = torch.from_numpy(x_star).float().to(device)
    x_star.requires_grad = True 
    t_star = torch.from_numpy(t_star).float().to(device)
    t_star.requires_grad = True 
    u_star = torch.from_numpy(u_star).T.float().to(device)  

    # Training with Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x_u, x_f, t_u, t_f, nu, u_train_pt, num_iter=10_000)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")

    # Training with L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x_u, x_f, t_u, t_f, nu, u_train_pt, num_iter=50_000)
    end_time_lbfgs = time.time()
    lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
    print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")

    total_training_time = adam_training_time + lbfgs_training_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    # Obtain the final loss L2
    final_loss = results[-1][1]
    print(f"Final Loss: {final_loss:.6e}")

    # Obtain the final L2 error
    final_l2 = results[-1][2]
    print(f"Final L2: {final_l2:.6e}")

    # Save training summary to a text file
    with open('training/Schrodinger_training_summary.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.6e} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.6e} seconds\n")
        file.write(f"Total training time: {total_training_time:.6e} seconds\n")
        file.write(f"Total iterations: {iter}\n")   
        file.write(f"Final Loss: {final_loss:.6e}\n")
        file.write(f"Final L2: {final_l2:.6e}\n")  


    # Convert results to numpy array, save training data to CSV, and save model state
    results = np.array(results)
    np.savetxt("training/Burgers_training_data.csv", results, delimiter=",", header="Iter,Loss,L2", comments="")
    torch.save(model.state_dict(), f'Burgers.pt')