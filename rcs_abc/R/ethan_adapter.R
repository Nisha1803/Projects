# Adapter: run Ethan's actual CA (Functions/CellularAutomata.R) as an ABC simulator.
#
# Nothing in Ethan's model is re-implemented here. run_ca_two_layer() is called
# unchanged; this file only
#   1. sources CellularAutomata.R into its own environment,
#   2. writes each ABC parameter draw into the globals that CA reads,
#   3. caches the radius-dependent storm operators (built once per NEI_RADIUS),
#   4. turns first-arrival ticks into the summaries ABC compares with data,
#      using the same convex hull of cell corners as PostProcessing.R.
#
# `rcs_data` must be the grid AFTER Ethan's Initialize.R + PreProcessing.R
# (it needs row, col, cell_id, Xmin, Ymin, MaxEnt_Phragmites, the active RCS
# MaxEnt column, NRI_Score, Initial_Phrag, Initial_RCS, RCS_IntroSeed).
# scripts/export_ethan_inputs.R produces exactly that object.

ETHAN_PARAM_MAP <- c(b_ph0 = "PHRAG_INTERCEPT", b_ph1 = "PHRAG_MAXENT_COEF",
                     b_ph2 = "PHRAG_SOURCE_COEF", b_rcs0 = "RCS_INTERCEPT",
                     b_rcs1 = "RCS_MAXENT_COEF", b_rcs2 = "RCS_SOURCE_COEF",
                     b_st0 = "STORM_INTERCEPT", b_st1 = "STORM_NRI_COEF")

#' @param func_dir     folder holding Ethan's CellularAutomata.R
#' @param rcs_data     preprocessed grid (see above)
#' @param start_year,intro_year  as in CONFIG (2000, 2017)
#' @param obs_years    calendar years to return summaries for
#' @param storms       storm count per tick 1..T (tick t = start_year + t).
#'                     Length must cover the last observed tick; see
#'                     storm_schedule() below for why this matters.
#' @param base_model   "COLD" etc., as in CONFIG
#' @param fast_diamond use the exact radius-independent storm-shell sum
#' @param fixed        list of CONFIG values held fixed (NEIGHBORHOOD, DIST_METRIC)
make_ethan_simulator <- function(func_dir, rcs_data, start_year = 2000L,
                                 intro_year = 2017L, obs_years, storms,
                                 base_model = "COLD", fast_diamond = TRUE,
                                 fixed = list(NEIGHBORHOOD = 4L,
                                              DIST_METRIC = "manhattan")) {
  stopifnot(requireNamespace("data.table", quietly = TRUE),
            requireNamespace("Matrix", quietly = TRUE))
  env <- new.env(parent = globalenv())
  # CellularAutomata.R validates these at source time
  env$NEIGHBORHOOD <- as.integer(fixed$NEIGHBORHOOD)
  env$DIST_METRIC <- fixed$DIST_METRIC
  env$NEI_RADIUS <- 30L; env$MAX_TICKS <- 30L; env$BASE_MODEL <- base_model
  env$JUMP_SOURCE_METHOD <- "offset"; env$STORE_PROB_SUMMARY <- FALSE
  env$SPARSE_OPS_USE_CACHE <- FALSE
  for (nm in ETHAN_PARAM_MAP) env[[nm]] <- 0
  env$STORMS_PER_TICK <- storms
  sys.source(file.path(func_dir, "CellularAutomata.R"), envir = env)
  # Optional exact speed-up of the storm-shell sum (see R/fast_diamond.R and
  # tests/test_fast_diamond.R); same numbers, ~4-6x faster per simulation.
  if (isTRUE(fast_diamond)) env$.manhattan_diamond_sum_grid <- manhattan_diamond_sum_fast

  dt <- data.table::as.data.table(rcs_data)
  if (!"cell_id" %in% names(dt)) dt[, cell_id := GridSquare]
  data.table::setorderv(dt, c("row", "col"))
  nr <- max(dt$row); nc <- max(dt$col)
  covs <- env$compute_ca_covariates(dt, base_model, verbose = FALSE)
  dt[, `:=`(M_phrag = covs$M_phrag, M_rcs = covs$M_rcs, N_jump = covs$N_jump,
            valid_jump_source = covs$valid_jump_source)]

  # seeds exactly as RunCode.R::pick_initial_seeds() with LoadedData
  intro_tick <- as.integer(intro_year - start_year)
  init_rcs <- dt$cell_id[dt$Initial_RCS == 1L]
  intro_ids <- if (intro_tick > 0L) setdiff(dt$cell_id[dt$RCS_IntroSeed == 1L], init_rcs) else integer(0)
  init_phrag <- union(dt$cell_id[dt$Initial_Phrag == 1L], init_rcs)

  obs_ticks <- as.integer(obs_years - start_year)
  T_steps <- max(obs_ticks)
  if (length(storms) < T_steps) stop("storms must have one value per tick up to ", T_steps)
  rows <- dt$row; cols <- dt$col

  # Adjacency is radius-independent; jump offsets and denominators are not.
  base_ops <- env$build_sparse_ca_operators(nr, nc, env$NEIGHBORHOOD, 2L, env$DIST_METRIC,
                                            "offset", verbose = FALSE, cache_dir = NULL)
  phrag_src_mask <- env$valid_score_mask(dt$MaxEnt_Phragmites) & env$valid_score_mask(dt$NRI_Score)
  ops_cache <- new.env()
  ops_for <- function(r) {
    key <- as.character(r)
    if (is.null(ops_cache[[key]])) {
      ops <- base_ops
      ops$jump_offsets <- env$build_jump_offsets(r, env$DIST_METRIC, env$NEIGHBORHOOD)
      ops$radius <- r
      ops_cache[[key]] <- list(ops = ops,
                               elig = env$.jump_eligible_counts(dt$valid_jump_source, ops),
                               elig_ph = env$.jump_eligible_counts(phrag_src_mask, ops))
    }
    ops_cache[[key]]
  }

  summarise_tick <- function(first, tick) {
    occ <- which(first >= 0L & first <= tick)
    if (!length(occ)) return(c(area = 0, n_rcs = 0, west = NA, east = NA))
    c(area = hull_area(rows[occ], cols[occ], 1),   # km2 on the 1 km grid
      n_rcs = length(occ), west = min(cols[occ]), east = max(cols[occ]))
  }

  simulate <- function(theta, return_raw = FALSE) {
    for (p in names(ETHAN_PARAM_MAP)) env[[ETHAN_PARAM_MAP[[p]]]] <- unname(theta[[p]])
    r <- as.integer(round(theta[["r_storm"]]))
    oc <- ops_for(r)
    res <- env$run_ca_two_layer(
      rcs_data = dt, nrow_grid = nr, ncol_grid = nc,
      initial_phrag_ids = init_phrag, initial_rcs_ids = init_rcs,
      rcs_intro_ids = intro_ids, rcs_intro_tick = intro_tick,
      base_model = base_model, neighborhood = env$NEIGHBORHOOD, nei_radius = r,
      dist_metric = env$DIST_METRIC, T_steps = T_steps, seed = NULL,
      history_storage = "arrival", store_probabilities = FALSE,
      store_prob_summary = FALSE, sparse_ops = oc$ops, verbose = FALSE,
      jump_eligible_counts = oc$elig, phrag_jump_eligible_counts = oc$elig_ph)
    S <- t(vapply(obs_ticks, function(k) summarise_tick(res$rcs_first_tick, k), numeric(4)))
    rownames(S) <- obs_years
    if (return_raw) list(summaries = S, phrag_total = res$phrag_total,
                         rcs_total = res$rcs_total, result = res) else S
  }
  simulate
}

#' Storm schedule per tick.
#' "as_coded": what CellularAutomata.R does today with START_YEAR = 2000 —
#'   the 10 values hit ticks 1-10 (2001-2010) and the LAST value (1) is
#'   reused for every later tick, i.e. 1 storm/yr for the whole RCS period.
#' "calendar": the 10 values placed on the years the paper says they describe
#'   (2016 = first value ... 2025 = last); years before 2016 get `pre_2016`.
storm_schedule <- function(values = c(4, 7, 4, 5, 8, 4, 3, 2, 5, 1),
                           start_year = 2000L, n_ticks = 30L,
                           mode = c("as_coded", "calendar"), first_year = 2016L,
                           pre_2016 = NA_integer_) {
  mode <- match.arg(mode)
  if (mode == "as_coded")
    return(values[pmin(seq_len(n_ticks), length(values))])
  yrs <- start_year + seq_len(n_ticks)
  out <- rep(as.integer(pre_2016), n_ticks)
  k <- yrs - first_year + 1L
  ok <- k >= 1L & k <= length(values)
  out[ok] <- values[k[ok]]
  out[yrs > first_year + length(values) - 1L] <- tail(values, 1)
  if (anyNA(out)) stop("calendar mode needs `pre_2016` storm counts for ",
                       paste(yrs[is.na(out)], collapse = ", "))
  out
}
