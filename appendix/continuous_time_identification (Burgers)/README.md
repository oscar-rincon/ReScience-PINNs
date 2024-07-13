# Burgers equation - Continuous time inference

![Burgers_ctid](figures/Burgers_ctid.gif)

## Summary

### Clean data

- Total training time: $5.035211 \times 10^{2}$ seconds
- Total number of iterations: $14,083$
- Error in estimating $\lambda_{1}$: $1.776218$ $\times$ $10^{-3}$ %
- Error in estimating $\lambda_{2}$: $1.386773$ $\times$ $10^{-1}$ %

### Noisy data

- Total training time: $1.016188 \times 10^{2}$ seconds
- Total number of iterations: $3,220$  
- Error in estimating $\lambda_{1}$: $6.062984$ $\times$ $10^{-2}$ %
- Error in estimating $\lambda_{2}$: $3.716909$ $\times$ $10^{-1}$ % 


## Running Scripts


**Run the scripts individually:**

```bash
make run_Burgers_ctid_main
```


```bash
make run_Burgers_ctid_plots
```

```bash
make run_Burgers_ctid_main_systematic
```

**Run all scripts in sequence:**

```bash
make all
```