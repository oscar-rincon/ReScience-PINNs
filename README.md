# ReScience-PINNs

This project is a replication of ''Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations'' by M. Raissi, P. Perdikaris, and G.E. Karniadakis from 2019.

The aim of this project was to:

- Reproduce the figures from the main manuscript of Raissi et al. (2019) using the Python library PyTorch. 
- Save the models obtained from the training.
- Record the training information such as computing times and the accuracies achieved.

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

 

