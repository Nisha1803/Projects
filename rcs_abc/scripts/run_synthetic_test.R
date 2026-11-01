# Parameter-recovery test: simulate "observed" data from known parameters on a
# synthetic landscape, run ABC-SMC, and check whether the posterior recovers
# them. Run this BEFORE trusting ABC on the real data: if a parameter is not
# recovered here, the real data cannot identify it either.
#
#   Rscript scripts/run_synthetic_test.R area   # area-only summaries (as in the paper)
#   Rscript scripts/run_synthetic_test.R rich   # area + front positions + RCS cell count

args <- commandArgs(trailingOnly = TRUE)
summary_set <- if (length(args)) args[1] else "rich"
n_particles <- if (length(args) > 1) as.integer(args[2]) else 300
ncores <- max(1, parallel::detectCores())

source("R/rcs_ca.R")
source("R/abc_smc.R")
set.seed(20261101)

land <- make_synthetic_landscape()
storms <- c(4, 7, 4, 5, 8, 4, 3, 2, 5, 1)   # STORMS_PER_TICK from the notes

# Paper best-fit values, except b_ph2 moved off the prior boundary and
# r_storm scaled down for the smaller synthetic grid.
truth <- c(b_ph0 = -3.25, b_ph1 = 5.50, b_ph2 = 1.00,
           b_rcs0 = -7.00, b_rcs1 = 8.25, b_rcs2 = 1.75,
           b_st0 = 4.25, b_st1 = 0.75, r_storm = 12)

prior <- make_uniform_prior(
  lower = c(b_ph0 = -6, b_ph1 = 2,  b_ph2 = 0, b_rcs0 = -9, b_rcs1 = 4,
            b_rcs2 = 0, b_st0 = -2, b_st1 = 0, r_storm = 3),
  upper = c(b_ph0 = -1, b_ph1 = 10, b_ph2 = 3, b_rcs0 = -3, b_rcs1 = 12,
            b_rcs2 = 4, b_st0 = 6,  b_st1 = 4, r_storm = 25),
  integer = "r_storm")

pick <- switch(summary_set,
  area = "area",
  rich = c("area", "n_rcs", "west", "east"),
  stop("summary set must be 'area' or 'rich'"))

# summaries for ticks 1..T (tick 0 is the fixed initial condition)
summarise <- function(sim_stats) as.vector(sim_stats[-1, pick, drop = FALSE])
simulate <- function(theta) summarise(simulate_rcs(theta, land, storms)$summaries)

obs <- simulate(truth)

dir.create("results", showWarnings = FALSE)
fit <- abc_smc(simulate, prior, obs, n_particles = n_particles, alpha = 0.5,
               max_gen = as.integer(Sys.getenv("MAX_GEN", "12")), min_acc_rate = 0.02, ncores = ncores,
               checkpoint = sprintf("results/synthetic_%s_checkpoint.rds", summary_set))
saveRDS(fit, sprintf("results/synthetic_%s_fit.rds", summary_set))

tab <- posterior_table(fit, truth)
print(round(tab[, c("truth", "q05", "median", "q95", "shrink")], 2))
cat(sprintf("\n%d / %d true values inside the 90%% credible interval\n",
            sum(tab$covered), nrow(tab)))
cat(sprintf("total simulations: %d\n", fit$final$n_sim))
write.csv(tab, sprintf("results/synthetic_%s_posterior.csv", summary_set))
