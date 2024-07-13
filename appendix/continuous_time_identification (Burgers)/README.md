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

## Systematic study

**Table A.3** 
 
| $N_{u}$ / Noise |      0%    |     1%     |      5%     |      10%     |
|---|---------------------|---------------------|---------------------|---------------------|
| 500 | $1.622$             | $1.222$             | $0.489$             | $2.590$             |
| 1000 | $0.014$             | $0.048$             | $0.133$             | $2.749$             |
| 1500 | $0.020$             | $0.068$             | $0.020$             | $0.689$             |
| 2000 | $0.020$             | $0.038$             | $0.996$             | $0.072$             |

| Layers / Neurons| 10 | 20 | 40 |
|--|----|-----|---|
|2| $13.286$ | $0.081$ | $0.659$ |
|4| $0.658$  | $0.056$ | $0.008$ |
|6| $0.191$  | $0.022$ | $0.010$ |
|8| $0.150$  | $0.028$ | $0.002$ |

**Table B.7**

| $N_{u}$ / Noise |      0%    |     1%     |      5%     |      10%     |
|---|---|---------------------|---------------------|---------------------|
| 500| $32.633$ | $18.556$ | $13.076$ | $11.928$ |
| 1000| $0.404$  | $1.116$  | $9.106$  | $7.530$  |
| 1500| $0.391$  | $0.815$  | $3.662$  | $17.065$ |
| 2000| $0.528$  | $0.062$  | $4.117$  | $5.613$  |


| Layers / Neurons| 10 | 20 | 40 |
|--|----|-----|---|
|2| $75.909$ | $0.468$  | $5.324$ |
|4| $8.258$  | $1.821$  | $0.847$  |
|6| $1.454$  | $0.651$  | $2.034$  |
|8| $0.110$  | $0.503$  | $0.255$ |

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