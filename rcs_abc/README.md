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
