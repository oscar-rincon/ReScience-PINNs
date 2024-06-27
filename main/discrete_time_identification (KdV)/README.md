# Korteweg–de Vries equation

 ![Korteweg–de Vries](figures/KdV.gif)

## Summary

### Clean data

- Total training time: $1.8996 \times 10^2$ seconds
- Total number of iterations: $1.0541 \times 10^3$
- Error in estimating $\lambda_{1}$: 0.002%
- Error in estimating $\lambda_{2}$: 0.013%

### Noisy data

- Total training time: $1.3847 \times 10^2$ seconds
- Total number of iterations: $8.239 \times 10^3$
- Error in estimating $\lambda_{1}$: 0.194%
- Error in estimating $\lambda_{2}$: 0.180%

## Running Korteweg–de Vries equation Scripts


```bash
make all
```

or

```bash
make run_KdV_clean_main
```

```bash
make run_KdV_noisy_main
```

and then 


```bash
make run_KdV_plots
```

to clean generated files:

```bash
make clean
```