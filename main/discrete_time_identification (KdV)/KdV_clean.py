import os 
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io  
from functools import partial
import warnings

# Add utility folder to system path
sys.path.insert(0, '../../Utilities/')
warnings.filterwarnings("ignore")

# Function to set random seed for reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the neural network model for KdV equation
class KdVNN(nn.Module):
    def __init__(self):
        super(KdVNN, self).__init__()
        self.linear_in = nn.Linear(1, 50)
        self.linear_out = nn.Linear(50, 50)
        self.layers = nn.ModuleList([nn.Linear(50, 50) for _ in range(5)])
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.linear_out(x)
        return x
    
# Initialize weights of the neural network
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)    

# Function to compute the first-order forward gradient
def fwd_gradients_0(dy: torch.Tensor, x: torch.Tensor):
    z = torch.ones(dy.shape, device=dy.device).requires_grad_()
    g = torch.autograd.grad(dy, x, grad_outputs=z, create_graph=True)[0]
    ones = torch.ones(g.shape, device=g.device)
    return torch.autograd.grad(g, z, grad_outputs=ones, create_graph=True)[0]

# Function to compute U0 prediction
def net_U0(model, x_pt, lambda_1, lambda_2, dt, IRK_alpha):
    lambda_2 = torch.exp(lambda_2)
    U = model(x_pt)
    U_x = fwd_gradients_0(U, x_pt)
    U_xx = fwd_gradients_0(U_x, x_pt)
    U_xxx = fwd_gradients_0(U_xx, x_pt)
    F = -lambda_1 * U * U_x - lambda_2 * U_xxx
    U0 = U - dt * torch.matmul(F, IRK_alpha.T)
    return U0
 
# Function to compute U1 prediction
def net_U1(model, x_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta):
    lambda_2 = torch.exp(lambda_2)
    U = model(x_pt)
    U_x = fwd_gradients_0(U, x_pt)
    U_xx = fwd_gradients_0(U_x, x_pt)
    U_xxx = fwd_gradients_0(U_xx, x_pt)
    F = -lambda_1 * U * U_x - lambda_2 * U_xxx
    U1 = U + dt * torch.matmul(F, (IRK_beta - IRK_alpha).T)
    return U1 

# Function to compute mean squared error loss
def mse(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt):
    U0 = net_U0(model, x0_pt, lambda_1, lambda_2, dt, IRK_alpha)
    U1 = net_U1(model, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta)
    loss = torch.sum((u0_pt - U0) ** 2) + torch.sum((u1_pt - U1) ** 2)
    return loss

# Function to train the model using Adam optimizer
def train_adam(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=50_000):
    optimizer = torch.optim.Adam(list(model.parameters()) + [lambda_1, lambda_2], lr=1e-5)
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
        if i % 100 == 0:
            torch.save(model.state_dict(), f'models_iters/KdV_clean_{iter}.pt')
            print(f"Adam - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.cpu().detach().numpy().item()} - l2: {torch.exp(lambda_2).cpu().detach().numpy().item()}")

# Function to train the model using L-BFGS optimizer
def train_lbfgs(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=50_000):
    optimizer = torch.optim.LBFGS(list(model.parameters()) + [lambda_1, lambda_2],
                                  max_iter=num_iter,
                                  max_eval=num_iter,
                                  history_size=100,
                                  tolerance_grad=1e-5,
                                  line_search_fn='strong_wolfe',
                                  tolerance_change=1.0 * np.finfo(float).eps)
    closure_fn = partial(closure, model, optimizer, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt)
    optimizer.step(closure_fn) 
    
# Closure function for L-BFGS optimization
def closure(model, optimizer, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt):
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
        torch.save(model.state_dict(), f'models_iters/KdV_clean_{iter}.pt')
        print(f"LBFGS - Iter: {iter} - Loss: {loss.item()} - l1: {lambda_1.detach().numpy().item()} - l2: {torch.exp(lambda_2).detach().numpy().item()}")
    return loss    

# Main function
if __name__ == "__main__":
    set_seed(42)  
    iter = 0      

    if not os.path.exists('models_iters'):
        os.makedirs('models_iters')

    if not os.path.exists('training'):
        os.makedirs('training')

    # Check for GPU availability
    device = torch.device('cpu') # torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print(f'Using device: {device}')

    # Define parameters
    q = 50
    skip = 120
    N0 = 199
    N1 = 201
    layers = [1, 50, 50, 50, 50, q]

    # Load data
    data = scipy.io.loadmat('../Data/KdV.mat')
    t_star = data['tt'].flatten()[:,None]
    x_star = data['x'].flatten()[:,None]
    Exact = np.real(data['uu'])
    idx_t = 40
    noise = 0.0    

    # Sample data points
    idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x,:]
    u0 = Exact[idx_x,idx_t][:,None] 
    u0 = u0 + noise * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])        
    idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x,:]
    u1 = Exact[idx_x,idx_t + skip][:,None]
    u1 = u1 + noise * np.std(u1) * np.random.randn(u1.shape[0], u1.shape[1])
    dt = torch.tensor((t_star[idx_t+skip] - t_star[idx_t]).item()).to(device)
        
    # Domain bounds
    lb = x_star.min(0)
    ub = x_star.max(0)    

    # Load IRK weights
    tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
    weights =  np.reshape(tmp[0:q**2+q], (q+1, q))    
    IRK_alpha = torch.from_numpy(weights[0:-1,:]).float().to(device)
    IRK_beta = torch.from_numpy(weights[-1:,:]).float().to(device)       
    IRK_times = tmp[q**2+q:]

    # Convert data to PyTorch tensors and move to GPU
    x0_pt = torch.from_numpy(x0).float().to(device)
    x0_pt.requires_grad = True
    x1_pt = torch.from_numpy(x1).float().to(device)
    x1_pt.requires_grad = True
    u0_pt = torch.from_numpy(u0).to(device)
    u1_pt = torch.from_numpy(u1).to(device)
    
    # Define lambda_1 and lambda_2 as trainable parameters
    lambda_1 = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=device))
    lambda_2 = torch.nn.Parameter(torch.tensor([-6.0], dtype=torch.float32, requires_grad=True, device=device))  
    lambda_1s = []
    lambda_2s = []
    results = []         
    
    model = KdVNN().to(device)
    model.apply(init_weights)  
    
    # Training with Adam optimizer
    start_time_adam = time.time()
    train_adam(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=0)
    end_time_adam = time.time()
    adam_training_time = end_time_adam - start_time_adam
    print(f"Adam training time: {adam_training_time:.2f} seconds")    
    
    # Training with L-BFGS optimizer
    start_time_lbfgs = time.time()
    train_lbfgs(model, x0_pt, x1_pt, lambda_1, lambda_2, dt, IRK_alpha, IRK_beta, u0_pt, u1_pt, num_iter=100_000)
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
    with open('training/KdV_training_summary_clean.txt', 'w') as file:
        file.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
        file.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
        file.write(f"Total training time: {total_training_time:.2f} seconds\n")
        file.write(f"Total iterations: {iter}\n") 
        file.write(f"Final Loss: {final_loss:.6f}\n")
        file.write(f"Percentage Error Lambda 1: {error_lambda_1:.6f}%\n")
        file.write(f"Percentage Error Lambda 2: {error_lambda_2:.6f}%\n")

    results = np.array(results)
    error_lambda_1s = np.abs(np.array(lambda_1s) - 1.0) / 1.0 * 100
    error_lambda_2s = np.abs(np.array(lambda_2s) - 0.0025) / 0.0025 * 100

    np.savetxt("training/KdV_training_data_clean.csv", np.column_stack([results[:,0], results[:,1], error_lambda_1s, error_lambda_2s]), delimiter=",", header="Iter,Loss,ErrorLambda1,ErrorLambda2", comments="")
    np.savetxt("training/lambda_1s_clean.csv", lambda_1s, delimiter=",", header="Lambda1", comments="")    
    np.savetxt("training/lambda_2s_clean.csv", lambda_2s, delimiter=",", header="Lambda2", comments="")
    torch.save(model.state_dict(), f'KdV_clean.pt')   
