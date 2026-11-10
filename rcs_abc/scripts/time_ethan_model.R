# Time Ethan's real CA (Functions/CellularAutomata.R) at full grid size.
# The MaxEnt/NRI layers here are SYNTHETIC placeholders (Southeast_Data_Clip.csv
# was not available); seeds come from the REAL occurrence file run through
# Ethan's own PreProcessing.R. Use only for run time, not for inference.
#
#   Rscript scripts/time_ethan_model.R <path/to/Functions> <occurrence.csv>
suppressPackageStartupMessages({library(data.table); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
func_dir <- args[1]; occ <- args[2]
source("R/rcs_ca.R"); source("R/fast_diamond.R"); source("R/ethan_adapter.R")

CLIP <- c(xmin = 49530.8674345585, xmax = 1417530.86743456, ymin = 3112714.25764237, ymax = 3466714.25764237)
nc <- round((CLIP[["xmax"]] - CLIP[["xmin"]]) / 1000); nr <- round((CLIP[["ymax"]] - CLIP[["ymin"]]) / 1000)
g <- CJ(row = seq_len(nr), col = seq_len(nc))
g[, `:=`(Xmin = CLIP[["xmin"]] + (col - 1) * 1000, Ymin = CLIP[["ymax"]] - row * 1000)]
g[, GridSquare := .I]
set.seed(1)
band <- exp(-((g$row - 0.7 * nr) / (0.15 * nr))^2)              # placeholder "coast"
g[, MaxEnt_Phragmites := pmin(1, band * runif(.N, 0.6, 1))]
g[, MaxEnt_RCS_cold_native := pmin(1, band * (1 - 0.4 * col / nc))]
g[, NRI_Score := band * runif(.N, 0.5, 1)]
rcs_data <- as.data.frame(g)

# Ethan's PreProcessing.R, unchanged, for the occurrence-based seeds
OCCURRENCE_FILE <- occ; START_YEAR <- 2000L; RCS_INTRO_YEAR <- 2017L
GRID_EPSG <- 32615L; GRID_RESOLUTION_M <- 1000; BASE_MODEL <- "COLD"
DIRS <- list(outputs_dir = Sys.getenv("PP_OUT", tempdir()))
invisible(capture.output(source(file.path(func_dir, "PreProcessing.R"))))
cat(sprintf("grid %d x %d = %d cells | PHRAG seeds %d | RCS intro cells %d\n",
            nr, nc, nr * nc, sum(rcs_data$Initial_Phrag), sum(rcs_data$RCS_IntroSeed)))

best <- c(b_ph0 = -3.25, b_ph1 = 5.5, b_ph2 = 0, b_rcs0 = -7, b_rcs1 = 8.25,
          b_rcs2 = 1.75, b_st0 = 4.25, b_st1 = 0.75, r_storm = 37)
obs_years <- c(2017, 2021, 2022, 2023, 2025)
sims <- list(
  original = make_ethan_simulator(func_dir, rcs_data, obs_years = obs_years,
                                  storms = storm_schedule(mode = "as_coded"), fast_diamond = FALSE),
  fast     = make_ethan_simulator(func_dir, rcs_data, obs_years = obs_years,
                                  storms = storm_schedule(mode = "as_coded"), fast_diamond = TRUE))
outs <- list()
for (nm in names(sims)) {
  invisible(sims[[nm]](replace(best, "r_storm", 37)))      # warm-up: builds radius-37 operators
  set.seed(99)
  t0 <- proc.time()[["elapsed"]]
  outs[[nm]] <- sims[[nm]](best, return_raw = TRUE)
  cat(sprintf("%-8s one simulation (2000-2025, %s): %.1f s\n", nm, "r = 37",
              proc.time()[["elapsed"]] - t0))
}
cat("identical simulated arrival ticks (same RNG seed):",
    identical(outs$original$result$rcs_first_tick, outs$fast$result$rcs_first_tick) &&
    identical(outs$original$result$phrag_first_tick, outs$fast$result$phrag_first_tick), "\n")
print(outs$fast$summaries)
cat("PHRAG cells by tick:", outs$fast$phrag_total[c(1, 6, 11, 18, 26)], "(ticks 0,5,10,17,25)\n")
