####################################################################################################
############################ RUN THE TWO-LAYER AUTOMATON (PHRAG + RCS) #############################
####################################################################################################

# Purpose
#   Orchestrate a batch of CA simulations:
#     - verify rcs_data is a complete rectangular lattice
#     - attach normalized CA covariates once
#     - build CA operators once in serial mode
#     - in parallel mode, build/load CA operators once per worker chunk
#     - select separate initial PHRAG and RCS seeds from occurrence-derived columns
#     - pass staged RCS introduction seeds to the CA at the configured calendar year
#     - run N_RUNS serially or in parallel
#     - return ca_results for PostProcessing.R
#
# Output structure:
#   - RunCode.R returns ca_results only.
#   - PostProcessing.R builds and saves the single BatchSummary.csv.
#   - BatchSummary.csv is one row per run per tick and includes:
#       * Number_of_Phrag_Cells
#       * Number_of_Infested_Cells
#       * Infested_Area
#       * Encapsulated_Infested_Area
#       * Storm_Event_Count
#       * Storm_Jump_Success_Count (coupled RCS jump)
#       * Phrag_Storm_Jump_Success_Count (after PostProcessing.R is updated)
#
# Memory/output controls:
#   - STORE_HISTORY controls whether spatial histories are retained.
#   - STORE_HISTORY must be TRUE if Encapsulated_Infested_Area is needed.
#   - STORE_PROBABILITIES should remain FALSE for history matching.
#   - STORE_PROB_SUMMARY should remain FALSE for history matching.
#   - returned ca_result objects are stripped of unnecessary heavy objects:
#       * sparse operators
#       * cell covariates
#       * probability matrices
#     while preserving rcs_history and order_indexed_cells when required.
#
# NEI-radius memory update:
#   - JUMP_SOURCE_METHOD controls the storm-jump source calculation:
#       * "offset" = memory-light grid-shift method; recommended for large NEI_RADIUS
#       * "sparse" = previous sparse-matrix method
#   - In parallel mode, each worker receives JUMP_SOURCE_METHOD and builds matching operators.
#
# Key design choices
#   - PHRAG hard-gates RCS
#   - Covariates are computed once and reused across runs
#   - Sparse adjacency operators are computed once and reused across runs in serial mode
#   - In parallel mode, operators are built/loaded inside each worker
#     so they are NOT exported as huge globals
#   - Eligible storm-source counts are computed once per worker chunk,
#     not once per stochastic replicate
#   - Uses multicore on Unix-like systems when not in RStudio
#   - Falls back to multisession in RStudio / non-Unix environments
#   - RCS initialization is controlled by INITIAL_INFESTATION:
#       * "LoadedData"
#       * "Random"
#       * "Susceptible"

cat("Preparing to run two-layer cellular automaton (PHRAG hard-gates RCS; storm jumps = PHRAG-only + coupled PHRAG/RCS radius establishment)\n\n")

####################################################################################################
######################################## CONFIG READ ###############################################
####################################################################################################

get_run_option <- function(name, default = NULL) {
  get0(name, ifnotfound = default, inherits = TRUE)
}

BASE_MODEL <- as.character(
  get_run_option("BASE_MODEL", "Nipponaclerda")
)

INITIAL_INFESTATION <- as.character(
  get_run_option("INITIAL_INFESTATION", "LoadedData")
)

INITIAL_INFESTATION_COUNT <- as.integer(
  get_run_option("INITIAL_INFESTATION_COUNT", 5L)
)

OUTPUT_LABEL   <- as.character(get_run_option("OUTPUT_LABEL", ""))

N_RUNS       <- as.integer(get_run_option("N_RUNS", 1L))
RUN_PARALLEL <- isTRUE(get_run_option("RUN_PARALLEL", FALSE))
N_WORKERS    <- as.integer(get_run_option("N_WORKERS", 1L))

USE_FIXED_SEED <- isTRUE(get_run_option("USE_FIXED_SEED", FALSE))
SEED_BASE      <- as.integer(get_run_option("SEED_BASE", 0L))

MAX_TICKS    <- as.integer(get_run_option("MAX_TICKS", 24L))
START_YEAR   <- as.integer(get_run_option("START_YEAR", NA_integer_))
RCS_INTRO_YEAR <- as.integer(get_run_option("RCS_INTRO_YEAR", NA_integer_))
NEIGHBORHOOD <- as.integer(get_run_option("NEIGHBORHOOD", 8L))
NEI_RADIUS   <- as.integer(get_run_option("NEI_RADIUS", 4L))
DIST_METRIC  <- as.character(get_run_option("DIST_METRIC", "manhattan"))

# NEI-radius jump source method:
#   "offset" = memory-light; recommended for large NEI_RADIUS
#   "sparse" = old sparse A_jump_binary behavior
JUMP_SOURCE_METHOD <- tolower(as.character(get_run_option("JUMP_SOURCE_METHOD", "offset")))

STORE_HISTORY       <- isTRUE(get_run_option("STORE_HISTORY", FALSE))
# Backward compatible: the old STORE_HISTORY flag still selects full/none
# unless an explicit HISTORY_STORAGE mode is supplied by the procedure.
HISTORY_STORAGE <- tolower(as.character(get_run_option(
  "HISTORY_STORAGE", if (STORE_HISTORY) "full" else "none"
)))
if (length(HISTORY_STORAGE) != 1L ||
    !HISTORY_STORAGE %in% c("full", "arrival", "none"))
  stop("HISTORY_STORAGE must be 'full', 'arrival', or 'none'.")
STORE_HISTORY <- identical(HISTORY_STORAGE, "full")
STORE_PROBABILITIES <- isTRUE(get_run_option("STORE_PROBABILITIES", FALSE))
STORE_PROB_SUMMARY  <- isTRUE(get_run_option("STORE_PROB_SUMMARY", FALSE))

# Probability summaries are lightweight, but this selector lets you compute/return
# summaries only for selected simulation indices when desired.
#
# Preferred config name:
#   PROBABILITY_SUMMARY_RUNS = NULL              # all runs
#   PROBABILITY_SUMMARY_RUNS = c(1, 10, 20)      # selected runs
#   PROBABILITY_SUMMARY_RUNS = integer(0)        # no runs
#
# NULL means "all runs"; integer(0) or invalid values means "no runs".
resolve_probability_summary_runs_config <- function() {
  if (exists("PROBABILITY_SUMMARY_RUNS", inherits = TRUE)) {
    return(get("PROBABILITY_SUMMARY_RUNS", inherits = TRUE))
  }
  
  NULL
}

PROBABILITY_SUMMARY_RUNS <- resolve_probability_summary_runs_config()

# Full probability matrices are expensive. This selector controls which simulation
# indices are allowed to allocate and return p_phrag_local_hist, p_rcs_local_hist,
# and p_jump_hist.
#
# Preferred config name:
#   PROBABILITY_MATRIX_RUNS = c(1, 10, 20, 30)
#
# Backward-compatible fallback:
#   PROBABILITY_RUNS = c(1, 10, 20, 30)
#
# NULL means "all runs"; integer(0) or invalid values means "no runs".
resolve_probability_matrix_runs_config <- function() {
  if (exists("PROBABILITY_MATRIX_RUNS", inherits = TRUE)) {
    return(get("PROBABILITY_MATRIX_RUNS", inherits = TRUE))
  }
  
  if (exists("PROBABILITY_RUNS", inherits = TRUE)) {
    return(get("PROBABILITY_RUNS", inherits = TRUE))
  }
  
  NULL
}

PROBABILITY_MATRIX_RUNS <- resolve_probability_matrix_runs_config()

format_run_selector <- function(runs) {
  if (is.null(runs)) {
    return("ALL")
  }
  
  runs <- suppressWarnings(as.integer(unlist(
    runs,
    recursive = TRUE,
    use.names = FALSE
  )))
  runs <- runs[is.finite(runs) & runs >= 1L]
  
  if (!length(runs)) {
    return("NONE")
  }
  
  paste(unique(runs), collapse = ", ")
}

should_store_probability_summary_for_run <- function(run_i,
                                                     store_prob_summary = STORE_PROB_SUMMARY,
                                                     probability_summary_runs = PROBABILITY_SUMMARY_RUNS) {
  if (!isTRUE(store_prob_summary)) {
    return(FALSE)
  }
  
  # NULL means "all runs".
  if (is.null(probability_summary_runs)) {
    return(TRUE)
  }
  
  runs <- suppressWarnings(as.integer(unlist(
    probability_summary_runs,
    recursive = TRUE,
    use.names = FALSE
  )))
  runs <- runs[is.finite(runs) & runs >= 1L]
  
  if (!length(runs)) {
    return(FALSE)
  }
  
  as.integer(run_i) %in% unique(runs)
}

should_store_probability_matrices_for_run <- function(run_i,
                                                      store_probabilities = STORE_PROBABILITIES,
                                                      probability_matrix_runs = PROBABILITY_MATRIX_RUNS) {
  if (!isTRUE(store_probabilities)) {
    return(FALSE)
  }
  
  # NULL means "all runs".
  if (is.null(probability_matrix_runs)) {
    return(TRUE)
  }
  
  runs <- suppressWarnings(as.integer(unlist(
    probability_matrix_runs,
    recursive = TRUE,
    use.names = FALSE
  )))
  runs <- runs[is.finite(runs) & runs >= 1L]
  
  if (!length(runs)) {
    return(FALSE)
  }
  
  as.integer(run_i) %in% unique(runs)
}

# Controls for stripping heavy returned objects.
RETURN_SPARSE_OPS      <- isTRUE(get_run_option("RETURN_SPARSE_OPS", FALSE))
RETURN_CELL_COVARIATES <- isTRUE(get_run_option("RETURN_CELL_COVARIATES", FALSE))
RETURN_CELL_INDEX      <- isTRUE(get_run_option("RETURN_CELL_INDEX", FALSE))

VERBOSE_RUNNER        <- isTRUE(get_run_option("VERBOSE_RUNNER", FALSE))
ENABLE_RUNCODE_TIMERS <- isTRUE(get_run_option("ENABLE_STAGE_TIMERS", TRUE))

validate_run_config <- function() {
  ok_modes <- c("LoadedData", "Random", "Susceptible")
  
  if (!INITIAL_INFESTATION %in% ok_modes) {
    stop("INITIAL_INFESTATION must be one of: ", paste(ok_modes, collapse = ", "))
  }
  
  if (!is.finite(INITIAL_INFESTATION_COUNT) || INITIAL_INFESTATION_COUNT < 0L) {
    stop("INITIAL_INFESTATION_COUNT must be an integer >= 0.")
  }
  
  if (length(START_YEAR) != 1L || is.na(START_YEAR) ||
      length(RCS_INTRO_YEAR) != 1L || is.na(RCS_INTRO_YEAR) ||
      START_YEAR < 1700L || RCS_INTRO_YEAR < START_YEAR ||
      RCS_INTRO_YEAR > START_YEAR + MAX_TICKS) {
    stop("START_YEAR / RCS_INTRO_YEAR must define an introduction within the simulation.")
  }
  
  if (RCS_INTRO_YEAR > START_YEAR && INITIAL_INFESTATION != "LoadedData") {
    stop("Delayed RCS introduction requires INITIAL_INFESTATION='LoadedData'.")
  }
  
  if (!is.finite(N_RUNS) || N_RUNS < 1L) {
    stop("N_RUNS must be an integer >= 1.")
  }
  
  if (!is.finite(N_WORKERS) || N_WORKERS < 1L) {
    stop("N_WORKERS must be an integer >= 1.")
  }
  
  if (!JUMP_SOURCE_METHOD %in% c("offset", "sparse")) {
    stop("JUMP_SOURCE_METHOD must be either 'offset' or 'sparse'.")
  }
  
  if (!is.null(PROBABILITY_SUMMARY_RUNS)) {
    ps_runs <- suppressWarnings(as.integer(unlist(
      PROBABILITY_SUMMARY_RUNS,
      recursive = TRUE,
      use.names = FALSE
    )))
    ps_runs <- ps_runs[is.finite(ps_runs)]
    
    if (!length(ps_runs)) {
      warning(
        "STORE_PROB_SUMMARY may be TRUE, but PROBABILITY_SUMMARY_RUNS contains ",
        "no valid run indices. No probability summaries will be computed."
      )
    } else if (any(ps_runs < 1L | ps_runs > N_RUNS)) {
      warning(
        "PROBABILITY_SUMMARY_RUNS contains indices outside 1..N_RUNS. ",
        "Out-of-range indices will be ignored."
      )
    }
  }
  
  if (!is.null(PROBABILITY_MATRIX_RUNS)) {
    pm_runs <- suppressWarnings(as.integer(unlist(
      PROBABILITY_MATRIX_RUNS,
      recursive = TRUE,
      use.names = FALSE
    )))
    pm_runs <- pm_runs[is.finite(pm_runs)]
    
    if (!length(pm_runs)) {
      warning(
        "STORE_PROBABILITIES may be TRUE, but PROBABILITY_MATRIX_RUNS / ",
        "PROBABILITY_RUNS contains no valid run indices. No full probability ",
        "matrices will be stored."
      )
    } else if (any(pm_runs < 1L | pm_runs > N_RUNS)) {
      warning(
        "PROBABILITY_MATRIX_RUNS / PROBABILITY_RUNS contains indices outside ",
        "1..N_RUNS. Out-of-range indices will be ignored."
      )
    }
  }
  
  if (isTRUE(get_run_option("SAVE_ENCAPSULATED_AREA", FALSE)) &&
      HISTORY_STORAGE == "none") {
    warning("SAVE_ENCAPSULATED_AREA=TRUE but HISTORY_STORAGE='none'. ",
            "PostProcessing.R needs full histories or first-arrival ticks.")
  }
  
  if (isTRUE(get_run_option("SAVE_ENCAPSULATED_AREA", FALSE)) && !isTRUE(RETURN_CELL_INDEX)) {
    warning(
      "SAVE_ENCAPSULATED_AREA=TRUE but RETURN_CELL_INDEX=FALSE. ",
      "PostProcessing.R may not be able to align rcs_history to spatial geometry."
    )
  }
  
  invisible(TRUE)
}

validate_run_config()
RCS_INTRO_TICK <- as.integer(RCS_INTRO_YEAR - START_YEAR)

# Fail before constructing expensive operators if the CA has not been updated.
required_ca_args <- c("rcs_intro_ids", "rcs_intro_tick")
if (!exists("run_ca_two_layer", mode = "function") ||
    !all(required_ca_args %in% names(formals(run_ca_two_layer)))) {
  stop("CellularAutomata.R must support rcs_intro_ids and rcs_intro_tick ",
       "before this RunCode.R can run.")
}

cat(
  "RunCode output / jump-source mode:\n",
  "   → STORE_HISTORY          = ", STORE_HISTORY, "\n",
  "   → HISTORY_STORAGE        = ", HISTORY_STORAGE, "\n",
  "   → STORE_PROBABILITIES    = ", STORE_PROBABILITIES, "\n",
  "   → STORE_PROB_SUMMARY      = ", STORE_PROB_SUMMARY, "\n",
  "   → PROBABILITY_SUMMARY_RUNS = ", format_run_selector(PROBABILITY_SUMMARY_RUNS), "\n",
  "   → PROBABILITY_MATRIX_RUNS  = ", format_run_selector(PROBABILITY_MATRIX_RUNS), "\n",
  "   → RETURN_SPARSE_OPS      = ", RETURN_SPARSE_OPS, "\n",
  "   → RETURN_CELL_COVARIATES = ", RETURN_CELL_COVARIATES, "\n",
  "   → RETURN_CELL_INDEX      = ", RETURN_CELL_INDEX, "\n",
  "   → JUMP_SOURCE_METHOD     = ", JUMP_SOURCE_METHOD, "\n",
  "   → Simulation years       = ", START_YEAR, "–", START_YEAR + MAX_TICKS, "\n",
  "   → RCS introduction       = ", RCS_INTRO_YEAR, " (tick ", RCS_INTRO_TICK, ")\n\n",
  sep = ""
)

####################################################################################################
#################################### PERFORMANCE HELPERS ###########################################
####################################################################################################

RUNCODE_PERF_LOG <- data.frame(
  Step        = character(),
  Elapsed_Sec = numeric(),
  stringsAsFactors = FALSE
)

time_runcode_step <- function(step_name, expr, print_timing = TRUE) {
  if (!isTRUE(ENABLE_RUNCODE_TIMERS)) {
    return(force(expr))
  }
  
  gc(FALSE)
  t0 <- proc.time()[["elapsed"]]
  out <- force(expr)
  dt <- proc.time()[["elapsed"]] - t0
  
  RUNCODE_PERF_LOG <<- rbind(
    RUNCODE_PERF_LOG,
    data.frame(
      Step = as.character(step_name),
      Elapsed_Sec = as.numeric(dt),
      stringsAsFactors = FALSE
    )
  )
  
  if (isTRUE(print_timing)) {
    cat(sprintf(" [RunCode] %-32s : %10.3f sec\n", step_name, dt))
  }
  
  out
}

save_runcode_perf_log <- function() {
  if (!isTRUE(ENABLE_RUNCODE_TIMERS) || nrow(RUNCODE_PERF_LOG) == 0L) {
    return(invisible(NULL))
  }
  
  out_file <- file.path(
    DIRS$outputs_dir,
    paste0("RunCode_Performance_", OUTPUT_LABEL, ".csv")
  )
  
  utils::write.csv(RUNCODE_PERF_LOG, out_file, row.names = FALSE)
  
  cat("\n========================== RUNCODE PERFORMANCE SUMMARY ==========================\n")
  print(RUNCODE_PERF_LOG, row.names = FALSE)
  cat(sprintf("\n%-42s : %10.3f sec\n", "TOTAL", sum(RUNCODE_PERF_LOG$Elapsed_Sec)))
  cat("Saved RunCode performance summary: ", out_file, "\n", sep = "")
  cat("================================================================================\n\n")
  
  invisible(out_file)
}

####################################################################################################
######################################## GRID CHECKS ###############################################
####################################################################################################

prepare_rcs_grid <- function(rcs_data, verbose = TRUE) {
  required_cols <- c("row", "col")
  missing_cols <- setdiff(required_cols, names(rcs_data))
  if (length(missing_cols) > 0L) {
    stop("rcs_data must contain: ", paste(missing_cols, collapse = ", "))
  }
  
  dt <- data.table::as.data.table(rcs_data)
  
  if (!"cell_id" %in% names(dt)) {
    dt[, cell_id := if ("GridSquare" %in% names(dt)) GridSquare else seq_len(.N)]
  }
  
  nrow_grid <- max(dt$row, na.rm = TRUE)
  ncol_grid <- max(dt$col, na.rm = TRUE)
  
  validate_grid_data(dt, nrow_grid, ncol_grid)
  
  if (anyDuplicated(dt$cell_id)) {
    stop("rcs_data$cell_id must be unique.")
  }
  
  if (isTRUE(verbose)) {
    cat(
      "Grid size detected:\n",
      "   → nrow_grid = ", nrow_grid, "\n",
      "   → ncol_grid = ", ncol_grid, "\n\n",
      sep = ""
    )
  }
  
  list(
    rcs_data = dt,
    nrow_grid = as.integer(nrow_grid),
    ncol_grid = as.integer(ncol_grid)
  )
}

####################################################################################################
####################################### COVARIATE SETUP ############################################
####################################################################################################

show_top_covariates <- function(df, var, n = 5L) {
  cat("Top ", n, " cells by ", var, ":\n", sep = "")
  print(
    df %>%
      dplyr::arrange(dplyr::desc(.data[[var]])) %>%
      dplyr::select(cell_id, row, col, M_phrag, M_rcs, N_jump) %>%
      head(n)
  )
  cat("\n")
}

attach_ca_covariates <- function(rcs_data, base_model, verbose = TRUE) {
  covs <- compute_ca_covariates(rcs_data, base_model = base_model, verbose = verbose)
  
  dt <- data.table::copy(data.table::as.data.table(rcs_data))
  dt[, M_phrag := covs$M_phrag]
  dt[, M_rcs   := covs$M_rcs]
  dt[, N_jump  := covs$N_jump]
  dt[, valid_jump_source := covs$valid_jump_source]
  
  if (isTRUE(verbose)) {
    cat("Active RCS MaxEnt base model:\n",
        "   → BASE_MODEL = ", base_model, "\n\n", sep = "")
    
    cat("Normalized CA covariate summaries (attached to rcs_data):\n")
    cat("   M_phrag:\n"); print(summary(dt$M_phrag))
    cat("   M_rcs:\n");   print(summary(dt$M_rcs))
    cat("   N_jump:\n");  print(summary(dt$N_jump))
    cat("\n")
    
    show_top_covariates(dt, "M_phrag", 5L)
    show_top_covariates(dt, "M_rcs", 5L)
    show_top_covariates(dt, "N_jump", 5L)
    
    cat("Active RCS column resolved by CA code:\n",
        "   → ", covs$active_rcs_col, "\n", sep = "")
    cat("Valid jump-source cells attached:\n",
        "   → ", sum(dt$valid_jump_source, na.rm = TRUE), " of ", nrow(dt), "\n\n", sep = "")
  }
  
  list(
    rcs_data = dt,
    active_rcs_col = covs$active_rcs_col
  )
}

make_light_rcs_data_for_return <- function(rcs_data) {
  dt <- data.table::as.data.table(rcs_data)
  
  keep_cols <- intersect(
    c("cell_id", "GridSquare", "row", "col", "Xmin", "Xmax", "Ymin", "Ymax",
      "Phrag_FirstObservedYear", "RCS_FirstDetectedYear",
      "Initial_Phrag", "Initial_RCS", "RCS_IntroSeed"),
    names(dt)
  )
  
  if (!length(keep_cols)) {
    return(NULL)
  }
  
  as.data.frame(dt[, ..keep_cols])
}

####################################################################################################
######################################## SEED HELPERS ##############################################
####################################################################################################

# Occurrence-based PHRAG initialization (from PreProcessing.R).
pick_initial_phrag_ids <- function(rcs_data, initial_seed = NULL) {
  if (!all(c("cell_id", "Initial_Phrag") %in% names(rcs_data)))
    stop("PreProcessing.R must provide cell_id and Initial_Phrag.")
  
  vals <- suppressWarnings(as.integer(rcs_data$Initial_Phrag))
  if (anyNA(vals) || any(!vals %in% c(0L, 1L)))
    stop("Initial_Phrag must contain only 0/1 values.")
  
  ids <- unique(rcs_data$cell_id[vals == 1L])
  if (!length(ids)) stop("No Phragmites seeds at START_YEAR.")
  ids
}

pick_initial_rcs_ids <- function(rcs_data,
                                 initial_phrag_ids,
                                 mode = INITIAL_INFESTATION,
                                 count = INITIAL_INFESTATION_COUNT,
                                 initial_seed = NULL) {
  if (!all(c("cell_id", "M_rcs", "Initial_RCS") %in% names(rcs_data)))
    stop("PreProcessing.R must provide cell_id, Initial_RCS, and M_rcs.")
  
  if (!is.null(initial_seed)) set.seed(as.integer(initial_seed))
  
  if (mode == "LoadedData") {
    vals <- suppressWarnings(as.integer(rcs_data$Initial_RCS))
    if (anyNA(vals) || any(!vals %in% c(0L, 1L)))
      stop("Initial_RCS must contain only 0/1 values.")
    return(unique(rcs_data$cell_id[vals == 1L]))
  }
  
  # Preserve the existing alternative seeding policies for nonhistorical runs.
  datR <- rcs_data[
    (rcs_data$cell_id %in% initial_phrag_ids) & !is.na(rcs_data$M_rcs),
  ]
  if (!nrow(datR) || count == 0L) return(integer(0))
  
  n_take <- min(as.integer(count), nrow(datR))
  if (mode == "Random") return(sample(datR$cell_id, size = n_take))
  if (mode == "Susceptible") {
    datR <- datR %>% dplyr::arrange(dplyr::desc(M_rcs))
    return(datR$cell_id[seq_len(n_take)])
  }
  stop("Unsupported RCS seeding mode: ", mode)
}

pick_initial_seeds <- function(rcs_data,
                               initial_infestation = INITIAL_INFESTATION,
                               initial_infestation_count = INITIAL_INFESTATION_COUNT,
                               initial_seed = NULL) {
  initial_phrag_ids <- pick_initial_phrag_ids(
    rcs_data = rcs_data, initial_seed = initial_seed
  )
  
  initial_rcs_ids <- pick_initial_rcs_ids(
    rcs_data = rcs_data,
    initial_phrag_ids = initial_phrag_ids,
    mode = initial_infestation,
    count = initial_infestation_count,
    initial_seed = initial_seed
  )
  
  if (!"RCS_IntroSeed" %in% names(rcs_data))
    stop("PreProcessing.R must provide RCS_IntroSeed.")
  intro_vals <- suppressWarnings(as.integer(rcs_data$RCS_IntroSeed))
  if (anyNA(intro_vals) || any(!intro_vals %in% c(0L, 1L)))
    stop("RCS_IntroSeed must contain only 0/1 values.")
  
  # A zero introduction tick means RCS seeds are already included at START_YEAR.
  rcs_intro_ids <- if (RCS_INTRO_TICK > 0L) {
    setdiff(unique(rcs_data$cell_id[intro_vals == 1L]), initial_rcs_ids)
  } else {
    integer(0)
  }
  
  # The CA will establish host occupancy at newly introduced RCS sites.
  initial_phrag_ids <- union(initial_phrag_ids, initial_rcs_ids)
  
  list(
    initial_phrag_ids = initial_phrag_ids,
    initial_rcs_ids = initial_rcs_ids,
    rcs_intro_ids = rcs_intro_ids
  )
}

####################################################################################################
############################### RESULT STRIPPING ###################################################
####################################################################################################

strip_ca_result_for_batch <- function(ca_result) {
  if (is.null(ca_result) || !is.list(ca_result)) {
    return(ca_result)
  }
  
  if (!isTRUE(STORE_HISTORY)) {
    ca_result$phrag_history <- NULL
    ca_result$rcs_history   <- NULL
  }
  
  if (!isTRUE(STORE_PROBABILITIES)) {
    ca_result$p_phrag_local_hist <- NULL
    ca_result$p_rcs_local_hist   <- NULL
    ca_result$p_jump_hist        <- NULL
    ca_result$p_phrag_jump_hist  <- NULL
  }
  
  if (!isTRUE(STORE_PROB_SUMMARY)) {
    ca_result$prob_summary <- NULL
  }
  
  if (!isTRUE(RETURN_SPARSE_OPS)) {
    ca_result$sparse_ops <- NULL
  }
  
  if (!isTRUE(RETURN_CELL_COVARIATES)) {
    ca_result$M_phrag <- NULL
    ca_result$M_rcs <- NULL
    ca_result$N_jump <- NULL
    ca_result$valid_jump_source <- NULL
  }
  
  if (!isTRUE(RETURN_CELL_INDEX)) {
    ca_result$order_indexed_cells <- NULL
  }
  
  ca_result
}

####################################################################################################
######################################## RUN ONE SIM ###############################################
####################################################################################################

make_run_tag <- function(run_i, base_model, output_label = OUTPUT_LABEL) {
  run_id <- sprintf("Run_%04d", as.integer(run_i))
  
  tag_parts <- c(
    if (nzchar(output_label)) output_label else NULL,
    as.character(base_model),
    run_id
  )
  
  paste(tag_parts, collapse = "_")
}

run_one_sim <- function(run_i,
                        rcs_data,
                        nrow_grid,
                        ncol_grid,
                        sparse_ops,
                        base_model = BASE_MODEL,
                        initial_infestation = INITIAL_INFESTATION,
                        initial_infestation_count = INITIAL_INFESTATION_COUNT,
                        probability_summary_runs = PROBABILITY_SUMMARY_RUNS,
                        probability_matrix_runs = PROBABILITY_MATRIX_RUNS,
                        verbose = VERBOSE_RUNNER,
                        jump_eligible_counts = NULL,
                        phrag_jump_eligible_counts = NULL) {
  run_seed <- as.integer(SEED_BASE + run_i)
  
  if (isTRUE(USE_FIXED_SEED) && isTRUE(verbose)) {
    cat("[", run_i, "] set.seed(", run_seed, ")\n", sep = "")
  }
  
  run_tag <- make_run_tag(
    run_i = run_i,
    base_model = base_model,
    output_label = OUTPUT_LABEL
  )
  
  seeds <- time_runcode_step(sprintf("run_%04d::pick_initial_seeds", as.integer(run_i)), {
    pick_initial_seeds(
      rcs_data = rcs_data,
      initial_infestation = initial_infestation,
      initial_infestation_count = initial_infestation_count,
      initial_seed = if (isTRUE(USE_FIXED_SEED)) run_seed else NULL
    )
  }, print_timing = !isTRUE(RUN_PARALLEL))
  
  store_probability_summary_this_run <- should_store_probability_summary_for_run(
    run_i = run_i,
    store_prob_summary = STORE_PROB_SUMMARY,
    probability_summary_runs = probability_summary_runs
  )
  
  store_probability_matrices_this_run <- should_store_probability_matrices_for_run(
    run_i = run_i,
    store_probabilities = STORE_PROBABILITIES,
    probability_matrix_runs = probability_matrix_runs
  )
  
  if (isTRUE(verbose)) {
    cat(
      "[", run_i, "] Seeds: PHRAG=", length(seeds$initial_phrag_ids),
      " | RCS=", length(seeds$initial_rcs_ids),
      " | RCS intro=", length(seeds$rcs_intro_ids), " (", RCS_INTRO_YEAR, ")",
      " | Policy=", initial_infestation,
      if (initial_infestation == "LoadedData") "" else paste0(" (Count=", initial_infestation_count, ")"),
      " | Store probability matrices=", store_probability_matrices_this_run,
      "\n",
      sep = ""
    )
  }
  
  ca_result <- time_runcode_step(sprintf("run_%04d::run_ca_two_layer", as.integer(run_i)), {
    run_ca_two_layer(
      rcs_data = rcs_data,
      nrow_grid = nrow_grid,
      ncol_grid = ncol_grid,
      initial_phrag_ids = seeds$initial_phrag_ids,
      initial_rcs_ids = seeds$initial_rcs_ids,
      rcs_intro_ids = seeds$rcs_intro_ids,
      rcs_intro_tick = RCS_INTRO_TICK,
      base_model = base_model,
      neighborhood = NEIGHBORHOOD,
      nei_radius = NEI_RADIUS,
      dist_metric = DIST_METRIC,
      T_steps = MAX_TICKS,
      seed = if (isTRUE(USE_FIXED_SEED)) run_seed else NULL,
      store_history = STORE_HISTORY,
      history_storage = HISTORY_STORAGE,
      store_probabilities = store_probability_matrices_this_run,
      store_prob_summary = store_probability_summary_this_run,
      sparse_ops = sparse_ops,
      verbose = verbose,
      jump_eligible_counts = jump_eligible_counts,
      phrag_jump_eligible_counts = phrag_jump_eligible_counts
    )
  }, print_timing = !isTRUE(RUN_PARALLEL))
  
  ca_result$run_tag <- run_tag
  ca_result$start_year <- START_YEAR
  ca_result$rcs_intro_year <- RCS_INTRO_YEAR
  ca_result$rcs_intro_tick <- RCS_INTRO_TICK
  ca_result$rcs_intro_seed_count <- length(seeds$rcs_intro_ids)
  ca_result$run_i   <- as.integer(run_i)
  ca_result$seed    <- run_seed
  ca_result$stored_probability_summary <- isTRUE(store_probability_summary_this_run)
  ca_result$stored_probability_matrices <- isTRUE(store_probability_matrices_this_run)
  
  strip_ca_result_for_batch(ca_result)
}

####################################################################################################
######################################## PARALLEL HELPERS ##########################################
####################################################################################################

choose_future_backend <- function() {
  is_unix <- identical(.Platform$OS.type, "unix")
  in_rstudio <- requireNamespace("rstudioapi", quietly = TRUE) &&
    isTRUE(tryCatch(rstudioapi::isAvailable(), error = function(e) FALSE))
  
  if (is_unix && !in_rstudio) {
    return("multicore")
  }
  
  "multisession"
}

run_chunk_from_rds <- function(run_chunk,
                               rcs_data_rds,
                               nrow_grid,
                               ncol_grid,
                               base_model,
                               initial_infestation,
                               initial_infestation_count,
                               neighborhood,
                               nei_radius,
                               dist_metric,
                               jump_source_method,
                               probability_summary_runs,
                               probability_matrix_runs) {
  rcs_data_local <- readRDS(rcs_data_rds)
  # The CA sorts by (row, col) before applying the offset-based source counts.
  data.table::setorderv(rcs_data_local, c("row", "col"))
  
  ca_ops_local <- build_sparse_ca_operators(
    nrow_grid = nrow_grid,
    ncol_grid = ncol_grid,
    neighborhood = neighborhood,
    radius = nei_radius,
    metric = dist_metric,
    jump_source_method = jump_source_method,
    verbose = FALSE
  )
  
  # Environmental validity and grid geometry are fixed within this chunk.
  # Calculate the nonlocal eligible-source denominator only once per worker.
  jump_eligible_counts_local <- .jump_eligible_counts(
    rcs_data_local$valid_jump_source, ca_ops_local
  )
  # PHRAG-only storm sources need PHRAG MaxEnt + NRI, but not RCS MaxEnt.
  phrag_jump_eligible_counts_local <- .jump_eligible_counts(
    valid_score_mask(rcs_data_local$MaxEnt_Phragmites) &
      valid_score_mask(rcs_data_local$NRI_Score), ca_ops_local
  )
  
  out <- lapply(run_chunk, function(i) {
    time_runcode_step(sprintf("run_%04d::total", as.integer(i)), {
      run_one_sim(
        run_i = i,
        rcs_data = rcs_data_local,
        nrow_grid = nrow_grid,
        ncol_grid = ncol_grid,
        sparse_ops = ca_ops_local,
        base_model = base_model,
        initial_infestation = initial_infestation,
        initial_infestation_count = initial_infestation_count,
        probability_summary_runs = probability_summary_runs,
        probability_matrix_runs = probability_matrix_runs,
        verbose = FALSE,
        jump_eligible_counts = jump_eligible_counts_local,
        phrag_jump_eligible_counts = phrag_jump_eligible_counts_local
      )
    }, print_timing = FALSE)
  })
  
  rm(ca_ops_local, rcs_data_local, jump_eligible_counts_local,
     phrag_jump_eligible_counts_local)
  gc(FALSE)
  
  out
}

####################################################################################################
######################################## BATCH RUNNER ##############################################
####################################################################################################

run_ca_batch <- function(rcs_data,
                         base_model = BASE_MODEL,
                         n_runs = N_RUNS,
                         run_parallel = RUN_PARALLEL,
                         n_workers = N_WORKERS,
                         verbose = VERBOSE_RUNNER) {
  grid_info <- time_runcode_step("prepare_rcs_grid", {
    prepare_rcs_grid(rcs_data, verbose = verbose)
  })
  
  cov_info <- time_runcode_step("attach_ca_covariates", {
    attach_ca_covariates(
      rcs_data = grid_info$rcs_data,
      base_model = base_model,
      verbose = verbose
    )
  })
  
  rcs_data_use <- cov_info$rcs_data
  required_seed_cols <- c("Initial_Phrag", "Initial_RCS", "RCS_IntroSeed")
  if (length(missing_seed_cols <- setdiff(required_seed_cols, names(rcs_data_use))))
    stop("Occurrence seed columns missing: ", paste(missing_seed_cols, collapse = ", "))
  if (RCS_INTRO_TICK > 0L && any(rcs_data_use$Initial_RCS == 1L))
    stop("Initial RCS presence predates configured RCS_INTRO_YEAR; check the occurrence years.")
  if (RCS_INTRO_TICK > 0L && !any(rcs_data_use$RCS_IntroSeed == 1L))
    stop("No RCS introduction cells were found for RCS_INTRO_YEAR.")
  # Match run_ca_two_layer()'s canonical row-major ordering before caching
  # the fixed denominator; seed selection still uses cell IDs, not positions.
  data.table::setorderv(rcs_data_use, c("row", "col"))
  nrow_grid <- grid_info$nrow_grid
  ncol_grid <- grid_info$ncol_grid
  
  cat(
    "Initial infestation policy (RCS):\n",
    "   → INITIAL_INFESTATION       = ", INITIAL_INFESTATION, "\n",
    "   → INITIAL_INFESTATION_COUNT = ", INITIAL_INFESTATION_COUNT,
    if (INITIAL_INFESTATION == "LoadedData") " (ignored; count comes from data)\n" else "\n",
    "   → Start year / RCS intro    = ", START_YEAR, " / ", RCS_INTRO_YEAR, "\n\n",
    sep = ""
  )
  
  cat(
    "Running ", n_runs, " simulation(s)",
    if (isTRUE(run_parallel)) paste0(" in parallel (N_WORKERS=", n_workers, ")") else " serially",
    "…\n",
    "   → JUMP_SOURCE_METHOD = ", JUMP_SOURCE_METHOD, "\n\n",
    sep = ""
  )
  
  if (isTRUE(run_parallel)) {
    if (!requireNamespace("future", quietly = TRUE) ||
        !requireNamespace("future.apply", quietly = TRUE)) {
      stop(
        "Parallel execution requested, but packages 'future' and/or 'future.apply' are not installed.\n",
        "   Install them before running parallel mode."
      )
    }
    
    rcs_data_rds <- make_outfile_batch("CA_Runtime_rcs_data_use", "rds")
    saveRDS(rcs_data_use, rcs_data_rds)
    
    cat("Saved runtime CA grid for workers:\n",
        "   → ", rcs_data_rds, "\n\n", sep = "")
    
    run_ids <- seq_len(n_runs)
    n_chunks <- min(as.integer(n_workers), length(run_ids))
    chunks <- split(run_ids, cut(run_ids, breaks = n_chunks, labels = FALSE))
    
    backend <- choose_future_backend()
    
    if (isTRUE(verbose)) {
      cat("future backend selected: ", backend, "\n", sep = "")
      if (backend == "multicore") {
        cat("   → using forked workers (good for Linux/macOS terminal/HPC)\n")
      } else {
        cat("   → using separate R sessions (RStudio-safe fallback)\n")
      }
      cat("\n")
    }
    
    future::plan(strategy = backend, workers = n_workers)
    on.exit(future::plan(future::sequential), add = TRUE)
    
    chunk_results <- time_runcode_step("future_lapply_all_chunks", {
      future.apply::future_lapply(
        X = chunks,
        FUN = run_chunk_from_rds,
        rcs_data_rds = rcs_data_rds,
        nrow_grid = nrow_grid,
        ncol_grid = ncol_grid,
        base_model = base_model,
        initial_infestation = INITIAL_INFESTATION,
        initial_infestation_count = INITIAL_INFESTATION_COUNT,
        neighborhood = NEIGHBORHOOD,
        nei_radius = NEI_RADIUS,
        dist_metric = DIST_METRIC,
        jump_source_method = JUMP_SOURCE_METHOD,
        probability_summary_runs = PROBABILITY_SUMMARY_RUNS,
        probability_matrix_runs = PROBABILITY_MATRIX_RUNS,
        future.seed = TRUE
      )
    })
    
    ca_results <- unlist(chunk_results, recursive = FALSE)
    
    rm(chunk_results)
    gc(FALSE)
    
  } else {
    
    ca_ops <- time_runcode_step("build_ca_operators", {
      build_sparse_ca_operators(
        nrow_grid = nrow_grid,
        ncol_grid = ncol_grid,
        neighborhood = NEIGHBORHOOD,
        radius = NEI_RADIUS,
        metric = DIST_METRIC,
        jump_source_method = JUMP_SOURCE_METHOD,
        verbose = verbose
      )
    })
    
    # Fixed for this spatial grid and covariate mask; reuse for every run.
    jump_eligible_counts_batch <- time_runcode_step("build_jump_eligible_counts", {
      .jump_eligible_counts(rcs_data_use$valid_jump_source, ca_ops)
    })
    
    phrag_jump_eligible_counts_batch <- time_runcode_step("build_phrag_jump_eligible_counts", {
      .jump_eligible_counts(valid_score_mask(rcs_data_use$MaxEnt_Phragmites) &
                              valid_score_mask(rcs_data_use$NRI_Score), ca_ops)
    })
    
    run_fun <- function(i) {
      time_runcode_step(sprintf("run_%04d::total", as.integer(i)), {
        run_one_sim(
          run_i = i,
          rcs_data = rcs_data_use,
          nrow_grid = nrow_grid,
          ncol_grid = ncol_grid,
          sparse_ops = ca_ops,
          base_model = base_model,
          initial_infestation = INITIAL_INFESTATION,
          initial_infestation_count = INITIAL_INFESTATION_COUNT,
          probability_summary_runs = PROBABILITY_SUMMARY_RUNS,
          probability_matrix_runs = PROBABILITY_MATRIX_RUNS,
          verbose = verbose,
          jump_eligible_counts = jump_eligible_counts_batch,
          phrag_jump_eligible_counts = phrag_jump_eligible_counts_batch
        )
      }, print_timing = TRUE)
    }
    
    ca_results <- time_runcode_step("lapply_all_runs", {
      lapply(seq_len(n_runs), run_fun)
    })
  }
  
  cat("\nAll simulations complete.\n\n")
  
  cat("Simulation results retained for PostProcessing.R:\n")
  cat("   → ca_results: ", length(ca_results), " run(s)\n", sep = "")
  cat("   → BatchSummary.csv will be built in PostProcessing.R\n\n")
  
  rcs_data_return <- if (isTRUE(RETURN_CELL_COVARIATES)) {
    as.data.frame(rcs_data_use)
  } else {
    make_light_rcs_data_for_return(rcs_data_use)
  }
  
  ca_ops_return <- if (!isTRUE(run_parallel) && isTRUE(RETURN_SPARSE_OPS)) {
    ca_ops
  } else {
    NULL
  }
  
  list(
    ca_results = ca_results,
    rcs_data = rcs_data_return,
    nrow_grid = nrow_grid,
    ncol_grid = ncol_grid,
    active_rcs_col = cov_info$active_rcs_col,
    sparse_ops = ca_ops_return
  )
}

####################################################################################################
######################################## EXECUTE BATCH #############################################
####################################################################################################

batch_out <- time_runcode_step("run_ca_batch_total", {
  run_ca_batch(
    rcs_data = rcs_data,
    base_model = BASE_MODEL,
    n_runs = N_RUNS,
    run_parallel = RUN_PARALLEL,
    n_workers = N_WORKERS,
    verbose = VERBOSE_RUNNER
  )
})

ca_results     <- batch_out$ca_results
rcs_data       <- batch_out$rcs_data
nrow_grid      <- batch_out$nrow_grid
ncol_grid      <- batch_out$ncol_grid
ACTIVE_RCS_COL <- batch_out$active_rcs_col
SPARSE_OPS     <- batch_out$sparse_ops

save_runcode_perf_log()
