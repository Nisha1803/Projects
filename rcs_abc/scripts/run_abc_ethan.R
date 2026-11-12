# ABC-SMC on the REAL data with Ethan's REAL model (Functions/CellularAutomata.R).
#
# Run from rcs_abc/ on a machine that has Ethan's project folder:
#   Rscript scripts/run_abc_ethan.R /path/to/ethan_project
# where the project folder contains Functions/ and Data/ (Southeast_Data_Clip.csv
# and RCS_Phragmites_Data_10_17_26.csv), exactly as RCS_CA_Procedure.R expects.
#
# Settings are environment variables so a SLURM script can change them:
#   ABC_PARTICLES (500)  ABC_MAX_GEN (12)  ABC_CORES (all detected)
#   ABC_YEARS ("2021,2022,2023")  -- 2025 is excluded by default: 2 records.
#   ABC_STATS ("area,west")  -- east front excluded: surveys moved east over
#                                time, so it tracks survey effort, not spread
#   START_YEAR (2016)  -- Phragmites seeded from records up to this year
#   STORM_MODE ("calendar" | "as_coded")  -- calendar = counts on stated years
#   STORMS_PRE_2016 (only if START_YEAR < 2016 in calendar mode)
suppressPackageStartupMessages({library(data.table); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
proj <- normalizePath(if (length(args)) args[1] else "..")
source("R/rcs_ca.R"); source("R/abc_smc.R"); source("R/fast_diamond.R"); source("R/ethan_adapter.R")
env_num <- function(x, d) as.numeric(Sys.getenv(x, d))
n_particles <- env_num("ABC_PARTICLES", 500); max_gen <- env_num("ABC_MAX_GEN", 12)
ncores <- env_num("ABC_CORES", parallel::detectCores())
obs_years <- as.integer(strsplit(Sys.getenv("ABC_YEARS", "2021,2022,2023"), ",")[[1]])
stats <- strsplit(Sys.getenv("ABC_STATS", "area,west"), ",")[[1]]
storm_mode <- Sys.getenv("STORM_MODE", "calendar")

# ---- 1. Inputs with Ethan's own Initialize.R + PreProcessing.R ----
start_year <- as.integer(Sys.getenv("START_YEAR", "2016"))
source("R/ethan_inputs.R")
rcs_data <- build_ethan_inputs(proj, start_year = start_year)
START_YEAR <- start_year; RCS_INTRO_YEAR <- 2017L; BASE_MODEL <- "COLD"
DIRS <- list(func_dir = file.path(proj, "Functions"))
cat("START_YEAR =", start_year, "| PHRAG seed cells (records <= START_YEAR):",
    sum(rcs_data$Initial_Phrag), "| RCS introduction cells (2017):", sum(rcs_data$RCS_IntroSeed), "\n")

# ---- 2. Observed summaries from the SAME grid (cumulative RCS detections) ----
obs_tab <- observed_summaries(rcs_data, obs_years)
cat("\nObserved summaries (km2; fronts in km from the grid's west edge):\n"); print(obs_tab)
obs <- as.vector(obs_tab[, stats, drop = FALSE])

# ---- 3. Storm schedule ----
# calendar: the manuscript's NWS counts on their own years (2016 = first value).
# With START_YEAR >= 2016 no unknown pre-2016 counts are needed.
n_ticks <- max(obs_years) - START_YEAR
if (storm_mode == "as_coded") {
  storms <- storm_schedule(mode = "as_coded", start_year = START_YEAR, n_ticks = n_ticks)
  warning("STORM_MODE=as_coded reproduces CellularAutomata.R's tick indexing, which does ",
          "not put the counts on their stated years.", call. = FALSE, immediate. = TRUE)
} else {
  pre <- as.integer(strsplit(Sys.getenv("STORMS_PRE_2016", ""), ",")[[1]])
  storms <- storm_schedule(mode = "calendar", start_year = START_YEAR, n_ticks = n_ticks,
                           pre_2016 = if (START_YEAR >= 2016L) 0L else NA_integer_)
  if (START_YEAR < 2016L) {
    if (length(pre) != 2016L - START_YEAR - 1L)
      stop("START_YEAR < 2016 in calendar mode needs STORMS_PRE_2016 for ",
           START_YEAR + 1L, "-2015")
    storms[seq_along(pre)] <- pre
  }
}
cat("Storms per tick (", START_YEAR + 1, "-", max(obs_years), "): ", paste(storms, collapse = " "), "\n", sep = "")

# ---- 4. Simulator and prior ----
sim <- make_ethan_simulator(file.path(proj, "Functions"), rcs_data, start_year = START_YEAR,
                            intro_year = RCS_INTRO_YEAR, obs_years = obs_years,
                            storms = storms, base_model = BASE_MODEL)
simulate <- function(theta) as.vector(sim(theta)[, stats, drop = FALSE])

# Union of the history-matching ranges R7-R10 (manuscript Table 4), truncated
# with HMParameters.R's constraint_reasonable_logits so ABC and HM share a prior.
reasonable_logits <- function(th) {
  (th[["b_ph0"]] + th[["b_ph1"]] + th[["b_ph2"]] <= 3.5) &&
    (th[["b_rcs0"]] + th[["b_rcs1"]] + th[["b_rcs2"]] <= 4.75) &&
    (th[["b_ph0"]] + 0.5 * th[["b_ph1"]] + 0.25 * th[["b_ph2"]] >= -5.5) &&
    (th[["b_rcs0"]] + 0.5 * th[["b_rcs1"]] + 0.25 * th[["b_rcs2"]] >= -5.75)
}
prior <- make_uniform_prior(
  lower = c(b_ph0 = -6.25, b_ph1 = 4.25, b_ph2 = 0, b_rcs0 = -7, b_rcs1 = 5,
            b_rcs2 = 0, b_st0 = 0.25, b_st1 = 0, r_storm = 20),
  upper = c(b_ph0 = -2.75, b_ph1 = 10, b_ph2 = 3.25, b_rcs0 = -3, b_rcs1 = 11,
            b_rcs2 = 4.25, b_st0 = 6, b_st1 = 3.25, r_storm = 40),
  integer = "r_storm", constraint = reasonable_logits)

tt <- system.time(invisible(simulate(prior$sample(1)[1, ])))[["elapsed"]]
message(sprintf("one simulation: %.1f s on this machine -> roughly %.1f h for 15,000 sims on %d cores",
                tt, tt * 15000 / ncores / 3600, ncores))

# ---- 5. ABC-SMC ----
dir.create("results/ethan", recursive = TRUE, showWarnings = FALSE)
fit <- abc_smc(simulate, prior, obs, n_particles = n_particles, alpha = 0.5,
               max_gen = max_gen, min_acc_rate = 0.01, ncores = ncores,
               checkpoint = "results/ethan/abc_checkpoint.rds",
               resume = Sys.getenv("ABC_RESUME", "0") == "1")
fit$obs_tab <- obs_tab; fit$stats <- stats; fit$storms <- storms
saveRDS(fit, "results/ethan/abc_fit.rds")
tab <- posterior_table(fit)
print(round(tab, 3))
write.csv(tab, "results/ethan/abc_posterior.csv")
