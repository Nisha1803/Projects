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
#   ABC_STATS ("area,west,east")
#   STORM_MODE ("as_coded" | "calendar")  -- READ THE WARNING BELOW
#   STORMS_PRE_2016 (needed for calendar mode, comma list for 2001-2015)
suppressPackageStartupMessages({library(data.table); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
proj <- normalizePath(if (length(args)) args[1] else "..")
source("R/rcs_ca.R"); source("R/abc_smc.R"); source("R/fast_diamond.R"); source("R/ethan_adapter.R")
env_num <- function(x, d) as.numeric(Sys.getenv(x, d))
n_particles <- env_num("ABC_PARTICLES", 500); max_gen <- env_num("ABC_MAX_GEN", 12)
ncores <- env_num("ABC_CORES", parallel::detectCores())
obs_years <- as.integer(strsplit(Sys.getenv("ABC_YEARS", "2021,2022,2023"), ",")[[1]])
stats <- strsplit(Sys.getenv("ABC_STATS", "area,west,east"), ",")[[1]]
storm_mode <- Sys.getenv("STORM_MODE", "as_coded")

# ---- 1. Build rcs_data with Ethan's own Initialize.R + PreProcessing.R ----
# Same CONFIG values as RCS_CA_Procedure.R (only the inputs these two files read).
DIRS <- list(root = proj, func_dir = file.path(proj, "Functions"),
             data_dir = file.path(proj, "Data"), outputs_dir = file.path(getwd(), "results", "ethan_pp"))
dir.create(DIRS$outputs_dir, recursive = TRUE, showWarnings = FALSE)
DATA_FILE <- "Southeast_Data_Clip.csv"
OCCURRENCE_FILE <- file.path(DIRS$data_dir, "RCS_Phragmites_Data_10_17_26.csv")
START_YEAR <- 2000L; RCS_INTRO_YEAR <- 2017L; GRID_EPSG <- 32615L; GRID_RESOLUTION_M <- 1000
CLIP_BOUNDS <- c(xmin = 49530.8674345585, xmax = 1417530.86743456,
                 ymin = 3112714.25764237, ymax = 3466714.25764237)
BASE_MODEL <- "COLD"; OUTPUT_LABEL <- "ABC"; USE_FIXED_SEED <- FALSE; RUN_PARALLEL <- FALSE
source(file.path(DIRS$func_dir, "Initialize.R"))
source(file.path(DIRS$func_dir, "PreProcessing.R"))

# ---- 2. Observed summaries from the SAME grid (cumulative RCS detections) ----
first <- rcs_data$RCS_FirstDetectedYear
obs_tab <- t(vapply(obs_years, function(y) {
  occ <- which(!is.na(first) & first <= y)
  c(area = hull_area(rcs_data$row[occ], rcs_data$col[occ], 1), n_rcs = length(occ),
    west = min(rcs_data$col[occ]), east = max(rcs_data$col[occ]))
}, numeric(4)))
rownames(obs_tab) <- obs_years
cat("\nObserved summaries (km2 and km from the grid's west edge):\n"); print(obs_tab)
obs <- as.vector(obs_tab[, stats, drop = FALSE])

# ---- 3. Storm schedule ----
if (storm_mode == "as_coded") {
  storms <- storm_schedule(mode = "as_coded", n_ticks = max(obs_years) - START_YEAR)
  warning("STORM_MODE=as_coded reproduces CellularAutomata.R as it runs today: the 10 ",
          "STORMS_PER_TICK values fall on 2001-2010 and every year from 2011 on gets 1 storm. ",
          "The manuscript says the values are 2016-2025. Set STORM_MODE=calendar once ",
          "pre-2016 counts are known.", call. = FALSE, immediate. = TRUE)
} else {
  pre <- as.integer(strsplit(Sys.getenv("STORMS_PRE_2016", ""), ",")[[1]])
  if (length(pre) != 15L) stop("calendar mode needs STORMS_PRE_2016 with 15 values (2001-2015)")
  storms <- storm_schedule(mode = "calendar", n_ticks = max(obs_years) - START_YEAR,
                           pre_2016 = 0L)
  storms[1:15] <- pre
}
cat("Storms per tick (", START_YEAR + 1, "-", max(obs_years), "): ", paste(storms, collapse = " "), "\n", sep = "")

# ---- 4. Simulator and prior ----
sim <- make_ethan_simulator(DIRS$func_dir, rcs_data, start_year = START_YEAR,
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
               checkpoint = "results/ethan/abc_checkpoint.rds")
fit$obs_tab <- obs_tab; fit$stats <- stats; fit$storms <- storms
saveRDS(fit, "results/ethan/abc_fit.rds")
tab <- posterior_table(fit)
print(round(tab, 3))
write.csv(tab, "results/ethan/abc_posterior.csv")
