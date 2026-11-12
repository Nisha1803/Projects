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
positions constrains only `r_storm`. The weighted posterior correlation between b_rcs0 and b_rcs1 is -0.55 (see `06_rcs_ridge.png`),
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

## Figures

`Rscript scripts/plot_results.R` writes these to `results/figures/`:

| File | Shows |
|---|---|
| `01_posterior_marginals.png` | Prior vs posterior density for each parameter, with the true value |
| `02_shrinkage.png` | How much each 90% interval narrowed compared with the prior (the key diagnostic) |
| `03_credible_intervals.png` | 90% credible intervals and medians vs the true value |
| `04_tolerance_schedule.png` | Tolerance ε against cumulative simulations (convergence) |
| `05_posterior_predictive.png` | Area and west/east fronts re-simulated from the posterior vs observed |
| `06_rcs_ridge.png` | Joint b_rcs0/b_rcs1 posterior: the data constrains a combination, not each coefficient |
| `07_ca_maps.png` | CA state maps at ticks 0, 5 and 10 |

## Running ABC on Ethan's actual model (real data)

`R/ethan_adapter.R` calls Ethan's `run_ca_two_layer()` from `Functions/CellularAutomata.R`
**unchanged**, so ABC uses the real model rather than the stand-in in `R/rcs_ca.R`.
Inputs are built with Ethan's own `Initialize.R` + `PreProcessing.R`, and the observed
summaries are computed on that same grid, so simulated and observed values line up cell for cell.

```bash
Rscript tests/test_fast_diamond.R ../Functions        # check: the speed-up is exact
sbatch scripts/run_abc_ethan.slurm /path/to/ethan_project   # project with Functions/ and Data/
```

- **Speed.** One simulation (2000-2025, r = 37, 484,272 cells) takes about 40 s with the
  original code. 85% of that is `.manhattan_diamond_sum_grid`. `R/fast_diamond.R` computes the same
  sums with a rotated summed-area table: identical output (200 random grids, plus a full
  same-seed simulation), and about 13 s per simulation. A 15,000-simulation ABC run is about
  55 core-hours: roughly 2 h on 32 cores.
- **Observed data** (default): cumulative RCS hull area plus west/east fronts for 2021-2023.
  2017 is the introduction itself, so the model matches it by construction. 2025 is excluded
  because it has 2 records (`ABC_YEARS` to change).
- **Prior**: the union of history-matching ranges R7-R10 (Table 4), truncated with
  `constraint_reasonable_logits` from `HMParameters.R`.

### Storm schedule bug in CellularAutomata.R

`resolve_storms_this_tick()` indexes `STORMS_PER_TICK` by tick, and tick 1 = 2001
(`START_YEAR = 2000`). So `c(4, 7, 4, 5, 8, 4, 3, 2, 5, 1)` is applied to **2001-2010**, and the
last value (**1 storm**) is reused for **every year from 2011 on**. That includes the whole RCS
period, 2017-2025. The manuscript says these counts describe 2016-2025.
`STORM_MODE=as_coded` reproduces the model as it runs now. `STORM_MODE=calendar` puts the counts
on their stated years and needs `STORMS_PRE_2016` (15 values for 2001-2015).

### Real-data result (preliminary: 200 particles, stopped after generation 5)

Settings: START_YEAR = 2016 (Phragmites seeded from records), storm counts on their stated
years, fitted to hull area + west front for 2021-2023. 9,800 simulations, final tolerance 2.73,
acceptance 4.6%. Stopped on purpose: acceptance was falling fast and the posterior had stopped
moving. Reproduce the table and figures with `Rscript scripts/plot_real_abc.R /path/to/ethan_project`.

Shrinkage is measured against the **truncated** prior (the logit constraint applied), not the
bounding box. Against the box, the constraint alone makes b_ph1 look about 30% "learned".

| param | median | 90% CI | prior 90% | shrink |
|---|---|---|---|---|
| b_ph0 | -4.82 | [-6.07, -3.27] | [-6.15, -3.23] | 0.04 |
| b_ph1 | 5.75 | [4.45, 8.17] | [4.36, 7.77] | -0.09 |
| b_ph2 | 1.08 | [0.13, 2.55] | [0.09, 2.83] | 0.12 |
| b_rcs0 | -5.35 | [-6.63, -3.36] | [-6.88, -3.49] | 0.03 |
| b_rcs1 | 7.02 | [5.16, 9.82] | [5.14, 9.34] | -0.11 |
| b_rcs2 | 1.08 | [0.09, 3.48] | [0.12, 3.69] | 0.05 |
| b_st0 | 3.27 | [0.70, 5.69] | [0.54, 5.72] | 0.04 |
| b_st1 | 1.92 | [0.14, 3.07] | [0.17, 3.09] | 0.00 |
| r_storm | 31 | [22, 38] | [21, 39] | 0.11 |

**The real data do not identify any of the 9 parameters.** Almost every parameter set
the history-matching ranges allow fits three years of hull area and west front equally well. The
posterior-predictive check (`results/ethan/figures/04_posterior_predictive.png`) shows two
structural problems that more ABC generations cannot fix:
- **East front:** the model stays at about 840 km (90% band reaches 876 km by 2025). The surveys record 906 km
  from 2022, so the model cannot produce the east extent at all (the east front was not fitted).
- **RCS-occupied cells:** the model predicts thousands of cells (posterior median about 1,900 in 2023).
  The observed data have 273 detected cells. The model fills the hull, while the survey data are
  presence-only detections, so the two are not the same quantity. Comparing them needs a
  detection/observation model, not a different parameter set.

Figures in `results/ethan/figures/`: `01_posterior_marginals`, `02_shrinkage`,
`03_tolerance_schedule`, `04_posterior_predictive` (2017-2025, fitted vs. unfitted points),
`05_parameter_ridges`.
