# Navier–Stokes equation

![Navier–Stokes](figures/NS.gif)

## Summary

### Clean data

- Total training time: $3.03296 \times 10^3$ seconds
- Total number of iterations: $2.7017 \times 10^4$
- Error in estimating $\lambda_{1}$: $0.27$%
- Error in estimating $\lambda_{2}$: $6.75$%

### Noisy data

- Total training time: $6.68903 \times 10^3$ seconds
- Total number of iterations: $5.9526 \times 10^4$
- Error in estimating $\lambda_{1}$: $0.34$%
- Error in estimating $\lambda_{2}$: $7.67$%

## Running Navier–Stokes equation Scripts


```bash
make all
```

or

```bash
make run_NS_clean_main
```

```bash
make run_NS_noisy_main
```

and then 


```bash
make run_NS_plots
```

to clean generated files:

```bash
make clean
```