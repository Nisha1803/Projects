# Bayesian ABC-SMC calibration for the Phragmites / RCS cellular automaton

This replaces the history-matching calibration (RCS Watch manuscript, Sec. 2.6)
with Approximate Bayesian Computation using Sequential Monte Carlo (ABC-SMC).
History matching only rules out implausible parameter sets. ABC-SMC returns a
**posterior distribution**, so you get credible intervals and correlations for
each of the 9 parameters instead of a single "best fit".

Base R only, so there are no packages to install. Parallel runs use `parallel::mclapply`
on Linux/macOS. On Windows it runs serially.

## Files

| File | What it does |
|---|---|
| `R/rcs_ca.R` | Reference CA written from the paper's equations (`simulate_rcs()`), summary statistics, and a synthetic test landscape |
| `R/abc_smc.R` | Generic ABC-SMC: uniform/integer priors, MAD-scaled distance, adaptive epsilon, Gaussian + discrete kernels, importance weights, checkpointing |
| `scripts/run_synthetic_test.R` | Parameter-recovery test with known "true" parameters |
| `scripts/run_real_data_template.R` | Driver for `Southeast_Data_Clip` and `RCS_Phragmites_10_17_26`. **Column names are guesses, so edit them first** |
| `results/` | Logs and posterior tables from the synthetic tests |

## Using your own simulator instead of `rcs_ca.R`

`abc_smc()` only needs a function `theta -> numeric vector of summaries`:

```r
simulate <- function(theta) {
  out <- your_phrag_ode_model(PHRAG_INTERCEPT = theta["b_ph0"], ...,
                              NEI_RADIUS = theta["r_storm"])
  c(out$area_by_year, out$west_by_year, out$east_by_year)
}
fit <- abc_smc(simulate, prior, obs, n_particles = 1000, ncores = 8)
```

`rcs_ca.R` records assumptions A1–A6 at the top of the file. Those are the points where
the manuscript is ambiguous (for example, whether local spread needs at least one
colonized neighbour). If your code makes a different choice, the posterior
will be different too.

## Algorithm

1. Draw N parameter sets from the prior and simulate each one. The MAD of each
   summary statistic across these runs sets the distance scale, then the best N are kept.
2. For each generation g: set ε_g to the `alpha` quantile (default 0.5) of the
   previous distances. Resample particles by weight, perturb them with
   N(0, 2·Σ_w), and use a discretized normal for `r_storm`. Accept a candidate when
   distance ≤ ε_g, then reweight by prior / Σ_j w_j K(θ | θ_j).
3. Stop when the acceptance rate falls below `min_acc_rate`, which means the
   posterior has stopped tightening, or when `max_gen` is reached.

Each particle uses one stochastic simulation. History matching used 100
replicates per parameter set. ABC handles the stochasticity through the
acceptance step, so per-parameter replicates are not needed.

## Synthetic recovery test: the result you need to see

Truth: the paper's best fit (with b_ph2 = 1, r_storm = 12 on a 100 x 220 grid). Prior: wide
uniforms. 300 particles. `shrink` is how much narrower the 90% credible interval is than
the prior: 0 means you learned nothing, 1 means the parameter is pinned down.

| param | truth | area only: 90% CI | shrink | area + E/W fronts + RCS cells: 90% CI | shrink |
|---|---|---|---|---|---|
| b_ph0 | -3.25 | [-5.72, -1.41] | 0.04 | [-5.64, -1.62] | 0.11 |
| b_ph1 | 5.50 | [2.68, 9.51] | 0.05 | [2.35, 9.24] | 0.04 |
| b_ph2 | 1.00 | [0.27, 2.80] | 0.06 | [0.22, 2.89] | 0.01 |
| b_rcs0 | -7.00 | [-8.61, -3.23] | 0.00 | [-8.71, -3.65] | 0.06 |
| b_rcs1 | 8.25 | [4.78, 11.74] | 0.03 | [5.01, 11.59] | 0.09 |
| b_rcs2 | 1.75 | [0.20, 3.83] | -0.01 | [0.29, 3.84] | 0.01 |
| b_st0 | 4.25 | [-1.42, 5.78] | 0.00 | [-1.59, 5.80] | -0.03 |
| b_st1 | 0.75 | [0.13, 3.86] | -0.04 | [0.09, 3.78] | -0.02 |
| r_storm | 12 | [4, 24] | -0.01 | [8, 19] | **0.44** |

The truth was inside every interval, so the sampler works. With one RCS area time series,
though, the posterior for nearly every parameter is essentially the prior. Adding the front
positions constrains only `r_storm`. The posterior correlation between b_rcs0 and b_rcs1 is -0.64,
which means the data only fixes the RCS logistic curve along one direction, not both coefficients.

This means the paper's single best-fit set (Table 3) is one point on a large flat region,
and it should not be reported as *the* parameter values. To actually identify the betas you need
more information, not a better algorithm:
- **Phragmites data.** Without observed Phragmites cover, b_ph0, b_ph1 and b_ph2 are
  only weakly informed through RCS. If `RCS_Phragmites_10_17_26` has Phragmites presence, add it
  as a summary statistic.
- **Spatial survey match.** For each survey cell and year, record whether the simulation agrees with the
  observed presence or absence. That uses the 140+ survey points, not just their hull.
- **Fewer free parameters.** Fix or tie parameters the data can't separate. For example, b_st0 = 4.25
  puts q_nri at about 0.99 everywhere, so NRI barely matters and b_st0/b_st1 are close to unidentifiable by design.
