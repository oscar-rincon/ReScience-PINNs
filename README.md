# ReScience-PINNs

This project is a replication of ''Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations'' by M. Raissi, P. Perdikaris, and G.E. Karniadakis.

The aim of this project was to:

- Reproduce the main results from Raissi et al. (2019) using the Python libraries Tensorflow 1x (used in the original paper) and Tensorflow 2x. 
- Save the Tensorflow models obtained from the training of the models presented in the article.
- Create Jupyter notebooks from the original codes, including the stored models, to facilitate exploration of the program's variables and functions. 

## Repository Organisation

`main/`:

- `Data/`: Contains .mat files with the required inputs for the models.
- `continuous_time_inference (Schrodinger)/`: Results corresponding the the 3.1.1. Example (Schrodinger equation).
- `discrete_time_inference (AC)/`: Results corresponding the the 3.2.1. Example (Allen–Cahn equation).
- `continuous_time_identification (Navier-Stokes)/`: Results corresponding the the 4.1.1. Example (Navier–Stokes equation).
- `discrete_time_identification (KdV)/`: Results corresponding the the 4.2.1. Example (Korteweg–de Vries equation).

Each example contains the complete codes (`.py`), the notebooks (`.ipynb`), figures (`figures/`) and the models (models/).  

## Installation

We recommend setting up a new Python environment with conda. You can do this by running the following commands:

 ```
 conda create --name ReScience-PINNs-2-env
 conda activate ReScience-PINNs-2-env
 ```

Next, clone this repository by using the command:

 ```
git clone https://github.com/oscar-rincon/ReScience-PINNs.git
 ```

Finally, go to the `ReScience-PINNs/` folder and run the following command to install the necessary dependencies:

In case of Tensorflow 1x:

 ```
 conda env update --name ReScience-PINNs-env --file ReScience-PINNs-1.yaml
 ```

 In case of Tensorflow 2x:

 ```
 conda env update --name ReScience-PINNs-2-env --file ReScience-PINNs-2.yaml
 ```

To verify the packages installed in your `ReScience-PINNs-env` conda environment, you can use the following command:

 ```
 conda list -n ReScience-PINNs-env

 ## Computation times