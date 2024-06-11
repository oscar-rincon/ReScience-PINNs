# [Re] Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations 

This project is a replication of ''Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations'' by M. Raissi, P. Perdikaris, and G.E. Karniadakis from 2019.

Full reference to the original article :

> Raissi, M., P. Perdikaris, and G. E. Karniadakis. “Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.” Journal of Computational Physics 378 (February 1, 2019): 686–707. https://doi.org/10.1016/j.jcp.2018.10.045.


The aim of this project was to:

>- Reproduce the figures from the main manuscript of Raissi et al. (2019), originally obtained with Tensorflow 1x, using the Python library PyTorch. 
> - Save the models obtained from the training.
> - Record the training information such as computing times and the accuracies achieved.

The replication of the original article was successful and submitted to Rescience C.

## Repository Organisation

`main/`:

- `Data/`: Contains .mat files with the required inputs for the models.
- `continuous_time_inference (Schrodinger)/`: Results in Figure 1, corresponding to the the 3.1.1. Example (Schrodinger equation).
- `discrete_time_inference (AC)/`: Results in Figure 2, corresponding the the 3.2.1. Example (Allen–Cahn equation).
- `continuous_time_identification (Navier-Stokes)/`: Results in Figure 4, corresponding the the 4.1.1. Example (Navier–Stokes equation).
- `discrete_time_identification (KdV)/`: Results in Figure 5, corresponding the the 4.2.1. Example (Korteweg–de Vries equation).

Each example contains the main and plotting codes, figures (`figures/`), model (`.pt`) and summary information about the training process (`training/`).  

## Installation

We recommend setting up a new Python environment with conda. You can do this by running the following commands:

```
conda create --name ReScience-PINNs-env
conda activate ReScience-PINNs-env
```

Next, clone this repository by using the command:

 ```
git clone https://github.com/oscar-rincon/ReScience-PINNs.git
 ```

Go to the `ReScience-PINNs/` folder and run the following command to install the necessary dependencies:

 ```
conda env update --name ReScience-PINNs-env --file ReScience-PINNs.yaml
 ```

To verify the packages installed in your `ReScience-PINNs-env` conda environment, you can use the following command:

 ```
conda list -n ReScience-PINNs-env
 ```

 
## Hardware configuration

The models were trained with a NVIDIA GeForce RTX 4060 GPU card. The summary of the training information such as the computing times is included in the the folder of each simulation.

