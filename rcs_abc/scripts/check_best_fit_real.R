# Real layers + real model: how does the manuscript's best-fit parameter set
# (Table 3) behave under (a) the storm schedule as coded and (b) storms on the
# calendar years the paper states? Also: does simulated Phragmites cover the
# cells where Phragmites was actually recorded by 2017?
#   Rscript scripts/check_best_fit_real.R /path/to/ethan_project [n_reps]
suppressPackageStartupMessages({library(data.table); library(Matrix)})
args <- commandArgs(trailingOnly = TRUE); proj <- normalizePath(args[1])
n_reps <- if (length(args) > 1) as.integer(args[2]) else 8L
source("R/rcs_ca.R"); source("R/fast_diamond.R"); source("R/ethan_adapter.R"); source("R/ethan_inputs.R")
set.seed(2026)
start_year <- as.integer(Sys.getenv("START_YEAR", "2000"))
rcs_data <- build_ethan_inputs(proj, start_year = start_year)
cat("START_YEAR =", start_year, "| PHRAG seed cells:", sum(rcs_data$Initial_Phrag), "\n")
years <- c(2017, 2021, 2022, 2023, 2025)
obs <- observed_summaries(rcs_data, years)
cat("Observed:\n"); print(obs)

best <- c(b_ph0 = -3.25, b_ph1 = 5.5, b_ph2 = 0, b_rcs0 = -7, b_rcs1 = 8.25,
          b_rcs2 = 1.75, b_st0 = 4.25, b_st1 = 0.75, r_storm = 37)
pre <- as.integer(Sys.getenv("STORMS_PRE_2016_FILL", "4"))
schedules <- list(
  as_coded = storm_schedule(mode = "as_coded", start_year = start_year, n_ticks = 2025 - start_year),
  calendar = storm_schedule(mode = "calendar", start_year = start_year, n_ticks = 2025 - start_year, pre_2016 = pre))
if (start_year < 2016) cat(sprintf("\ncalendar schedule fills 2001-2015 with %d storms/yr (PLACEHOLDER: real counts unknown)\n", pre))

# Phragmites recorded (first observed) by 2017, from the same grid
ph_obs17 <- which(!is.na(rcs_data$Phrag_FirstObservedYear) & rcs_data$Phrag_FirstObservedYear <= 2017)
dt_order <- order(rcs_data$row, rcs_data$col)   # adapter sorts by row, col

res <- list()
for (nm in names(schedules)) {
  sim <- make_ethan_simulator(file.path(proj, "Functions"), rcs_data, start_year = start_year,
                              obs_years = years,
                              storms = schedules[[nm]])
  reps <- parallel::mclapply(seq_len(n_reps), function(i) {
    o <- sim(best, return_raw = TRUE)
    pf <- o$result$phrag_first_tick
    # map observed-Phragmites cells into the adapter's row-major order
    pos <- match(ph_obs17, dt_order)
    t17 <- 2017L - start_year
    list(S = o$summaries, ph17 = mean(pf[pos] >= 0 & pf[pos] <= t17),
         ph_total17 = o$phrag_total[t17 + 1L], ph_total = o$phrag_total, rcs_total = o$rcs_total)
  }, mc.cores = min(n_reps, parallel::detectCores()), mc.set.seed = TRUE)
  res[[nm]] <- reps
  S <- simplify2array(lapply(reps, `[[`, "S"))
  cat(sprintf("\n== storms %s (%s) | median over %d runs ==\n", nm,
              paste(schedules[[nm]], collapse = " "), n_reps))
  print(round(apply(S, c(1, 2), median)))
  cat("Simulated PHRAG cells by year (median):", paste0(start_year + seq_along(reps[[1]]$ph_total) - 1, ":",
      apply(sapply(reps, `[[`, "ph_total"), 1, median), collapse = " "), "\n")
  cat(sprintf("Simulated PHRAG cells in 2017: median %.0f | share of the %d cells with Phragmites RECORDED by 2017 that the sim has occupied: median %.2f\n",
              median(sapply(reps, `[[`, "ph_total17")), length(ph_obs17),
              median(sapply(reps, `[[`, "ph17"))))
}
saveRDS(list(obs = obs, res = res, schedules = schedules, best = best),
        sprintf("results/ethan/check_best_fit_real_start%d.rds", start_year))
