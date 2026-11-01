# ABC-SMC on the real data. TEMPLATE: the column names below are guesses,
# because Southeast_Data_Clip and RCS_Phragmites_10_17_26 were not available
# when this was written. Edit the three sections marked EDIT.
#
#   Rscript scripts/run_real_data_template.R

source("R/rcs_ca.R")
source("R/abc_smc.R")
set.seed(1)
ncores <- max(1, parallel::detectCores())

# ---------------------------------------------------------------------------
# EDIT 1: landscape (Southeast_Data_Clip). One row per 1 km cell.
# ---------------------------------------------------------------------------
grid <- read.csv("data/Southeast_Data_Clip.csv")
col_x      <- "x_min"          # cell X bound  -> column index
col_y      <- "y_min"          # cell Y bound  -> row index
col_mph    <- "maxent_phrag"   # Phragmites MaxEnt (0-1)
col_mrcs   <- "maxent_rcs_cold" # RCS MaxEnt, cold model (0-1)
col_nri    <- "nri_comp"       # composite NRI, normalized 0-1
col_water  <- "pct_water"      # % of cell covered by water
col_rcs0   <- "rcs_2017"       # survey count at tick 0: >=1 RCS, 0 Phrag, NA none

xs <- sort(unique(grid[[col_x]])); ys <- sort(unique(grid[[col_y]]), decreasing = TRUE)
ci <- match(grid[[col_x]], xs); ri <- match(grid[[col_y]], ys)
as_mat <- function(v, fill = 0) {
  m <- matrix(fill, length(ys), length(xs)); m[cbind(ri, ci)] <- v; m
}
init <- matrix(0L, length(ys), length(xs))
s0 <- grid[[col_rcs0]]
init[cbind(ri, ci)[!is.na(s0) & s0 == 0, , drop = FALSE]] <- 1L
init[cbind(ri, ci)[!is.na(s0) & s0 >= 1, , drop = FALSE]] <- 2L
land <- list(
  mask  = as_mat(grid[[col_water]] < 100, FALSE),
  M_ph  = as_mat(grid[[col_mph]]),
  M_rcs = as_mat(grid[[col_mrcs]]),
  nri   = as_mat(grid[[col_nri]]),
  init  = init,
  cell_km = 1)

# ---------------------------------------------------------------------------
# EDIT 2: observed data (RCS_Phragmites_10_17_26).
# Minimum: cumulative encapsulated area per tick. Strongly recommended: also
# west/east front positions per year (same units as the simulator: km from
# the grid's western edge) -- see README for why.
# ---------------------------------------------------------------------------
obs_tab <- read.csv("data/RCS_observed_by_year.csv")   # columns: tick, area, west, east
storms <- c(4, 7, 4, 5, 8, 4, 3, 2, 5, 1)              # STORMS_PER_TICK, ticks 1..10
pick <- c("area", "west", "east")
pick <- intersect(pick, names(obs_tab))
obs_ticks <- obs_tab$tick                              # ticks with observations
obs <- as.vector(as.matrix(obs_tab[, pick]))           # NA allowed = unobserved

summarise <- function(st) as.vector(st[obs_ticks + 1, pick, drop = FALSE])
simulate <- function(theta) summarise(simulate_rcs(theta, land, storms)$summaries)

# ---------------------------------------------------------------------------
# EDIT 3: prior. Do NOT use the final history-matching box (R10) as the prior
# unless you are sure it contains essentially all posterior mass; the R10
# ranges were tightened partly by hand. These are wider than R7-R10.
# ---------------------------------------------------------------------------
prior <- make_uniform_prior(
  lower = c(b_ph0 = -8, b_ph1 = 1,  b_ph2 = 0, b_rcs0 = -9, b_rcs1 = 3,
            b_rcs2 = 0, b_st0 = -4, b_st1 = 0, r_storm = 10),
  upper = c(b_ph0 = 0,  b_ph1 = 12, b_ph2 = 5, b_rcs0 = -2, b_rcs1 = 13,
            b_rcs2 = 5, b_st0 = 8,  b_st1 = 6, r_storm = 45),
  integer = "r_storm")

# time one simulation so you know what you are signing up for
tt <- system.time(simulate(c(b_ph0 = -3.25, b_ph1 = 5.5, b_ph2 = 0, b_rcs0 = -7,
                             b_rcs1 = 8.25, b_rcs2 = 1.75, b_st0 = 4.25,
                             b_st1 = 0.75, r_storm = 37)))[["elapsed"]]
message(sprintf("one simulation: %.2f s -> ~%.1f h for 30,000 sims on %d cores",
                tt, tt * 30000 / ncores / 3600, ncores))

dir.create("results", showWarnings = FALSE)
fit <- abc_smc(simulate, prior, obs, n_particles = 1000, alpha = 0.5,
               max_gen = 15, min_acc_rate = 0.01, ncores = ncores,
               checkpoint = "results/real_checkpoint.rds")
saveRDS(fit, "results/real_fit.rds")
tab <- posterior_table(fit)
print(round(tab, 3))
write.csv(tab, "results/real_posterior.csv")

# Posterior predictive check: re-simulate 200 posterior draws
draws <- fit$final$theta[sample(nrow(fit$final$theta), 200, TRUE,
                                prob = fit$final$weights), ]
pp <- t(apply(draws, 1, function(th) simulate_rcs(th, land, storms)$summaries[, "area"]))
png("results/real_ppc_area.png", 900, 600)
band <- apply(pp, 2, quantile, c(0.1, 0.5, 0.9))
ticks <- seq_len(ncol(pp)) - 1
plot(ticks, band[2, ], type = "l", lty = 2, ylim = range(c(band, obs_tab$area), na.rm = TRUE),
     xlab = "tick (2016 = 0)", ylab = "encapsulated RCS area (km^2)",
     main = "Posterior predictive check")
polygon(c(ticks, rev(ticks)), c(band[1, ], rev(band[3, ])), col = "#4477AA44", border = NA)
points(obs_tab$tick, obs_tab$area, pch = 19)
dev.off()
