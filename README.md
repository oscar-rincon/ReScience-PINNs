# Replication - Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/oscar-rincon/ReScience-PINNs/HEAD)

---

This project is a replication of ''Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations'' by M. Raissi, P. Perdikaris, and G.E. Karniadakis from 2019.

Full reference to the original article :

> Raissi, M., P. Perdikaris, and G. E. Karniadakis. “Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.” Journal of Computational Physics 378 (February 1, 2019): 686–707. https://doi.org/10.1016/j.jcp.2018.10.045.

GitHub Repository of original work: 

> https://github.com/maziarraissi/PINNs

The aim of this repository was to:

>- Reproduce the figures from the main manuscript of Raissi et al. (2019), originally obtained with Tensorflow 1x, using the Python library PyTorch. 
> - Save the models obtained from the training.
> - Record the training information such as computing times and the accuracies achieved.

---

## Repository Organisation

`main/`:

- `Data/`: Contains .mat files with the required inputs for the models.

- `continuous_time_inference (Schrodinger)/`: Results in Figure 1, corresponding to the the 3.1.1. Example (Schrodinger equation).

- `discrete_time_inference (AC)/`: Results in Figure 2, corresponding the the 3.2.1. Example (Allen–Cahn equation).

- `continuous_time_identification (Navier-Stokes)/`: Results in Figure 4, corresponding the the 4.1.1. Example (Navier–Stokes equation).

- `discrete_time_identification (KdV)/`: Results in Figure 5, corresponding the the 4.2.1. Example (Korteweg–de Vries equation).

`appendix/`:

- `Data/`: Contains .mat files with the required inputs for the models.

- `continuous_time_inference (Burgers)/`: Results in Figure A.6, corresponding to the the A.1. Continuous time models.

- `discrete_time_inference (Burgers)/`: Results in Figure A.7, corresponding to the the A.7. Discrete time models.

- `continuous_time_identification (Burgers)/`: Results in Figure B.8, corresponding to the the B.2. Discrete time models.

- `discrete_time_identification (Burgers)/`: Results in Figure B.9, corresponding to the the B.3. Discrete time models.

Each example contains the main and plotting codes, figures (`figures/`), model (`.pt`) and summary information about the training process (`training/`).  

---

## Installation

We recommend setting up a new Python environment with conda. You can do this by running the following commands:

```
conda env create -f environment.yml
conda activate ReScience-PINNs-env
```

To verify the packages installed in your `ReScience-PINNs-env` conda environment, you can use the following command:

 ```
conda list -n ReScience-PINNs-env
 ```

---

## Running Scripts


**Run the scripts individually:**


### Main scripts

**Schrodinger Equation - Continuous time inference**

```bash
make run_Schrodinger_main
```

```bash
make run_Schrodinger_plots
```

**AC Equation - Discrete time inference**

```bash
make run_AC_main
```

```bash
make run_AC_plots
```

**NS equation - clean and noisy data - Continuous time identification**

```bash
make run_NS_clean_main
```

```bash
make run_NS_noisy_main
```

```bash
make run_NS_plots
```

**kdV equation - clean and noisy data - Discrete time identification**

```bash
make run_kdV_clean_main
```

```bash
make run_kdV_noisy_main
```

 
```bash
make run_kdV_plots
```
 
### Appendix scripts

**Burgers equation -  Continuous time inference**

```bash
make run_Burgers_ctin_main
```

```bash
make run_Burgers_ctin_plots
```

```bash
make run_Burgers_ctin_main_systematic
```

**Burgers equation -  Discrete time inference**

```bash
make run_Burgers_dtin_main
```

```bash
make run_Burgers_dtin_plots
```

```bash
make run_Burgers_dtin_main_systematic
```

**Burgers equation -  Continuous time identification**

```bash
make run_Burgers_ctid_main
```

```bash
make run_Burgers_ctid_plots
```

```bash
make run_Burgers_ctid_main_systematic
```

**Burgers equation -  Discrete time identification**

```bash
make run_Burgers_dtid_main
```

```bash
make run_Burgers_dtid_plots
```

```bash
make run_Burgers_dtid_main_systematic
```

**Run all scripts in sequence:**

 
   ```bash
   make all
   ```

---

## Hardware configuration

The models were trained with a NVIDIA GeForce RTX A2000 GPU card. The summary of the training information such as the computing times is included in the the folder of each simulation.

---

## List of scripts

> - Schrodinger_main.py
> - Schrodinger_plots.py
> - AC_main.py
> - AC_plots.py
> - NS_clean_main.py
> - NS_noisy_main.py
> - NS_plots.py
> - kdV_clean_main.py
> - kdV_noisy_main.py
> - kdV_plots.py  
> - Burgers_ctin_main.py 
> - Burgers_ctin_plots.py 
> - Burgers_ctin_main_systematic.py  
> - Burgers_dtin_main.py  
> - Burgers_dtin_plots.py  
> - Burgers_dtin_main_systematic.py 
> - Burgers_ctid_clean_main.py 
> - Burgers_ctid_noisy_main.py 
> - Burgers_ctid_plots.py
> - Burgers_ctid_main_systematic.py  
> - Burgers_dtid_clean_main.py
> - Burgers_dtid_noisy_main.py   
> - Burgers_dtid_plots.py  
> - Burgers_dtid_main_systematic.py  

---    