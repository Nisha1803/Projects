####################################################################################################
###################################### CELLULAR AUTOMATON ###########################################
####################################################################################################
# Two-layer PHRAG/RCS CA.
# Tick order: PHRAG and coupled RCS storm jumps -> local PHRAG -> local RCS -> timed RCS introduction -> output.
# PHRAG can jump without RCS; RCS jumps establish both PHRAG and RCS. RCS remains PHRAG-gated.
# Timed introduction is imposed AFTER that tick's spread, so newly introduced
# RCS cannot produce local or storm-mediated spread before the following tick.

####################################################################################################
######################################## PARAMETER READ ############################################
####################################################################################################

CA_DEFAULTS <- list(
  NEIGHBORHOOD = 8L, NEI_RADIUS = 4L, DIST_METRIC = "manhattan", MAX_TICKS = 24L,
  BASE_MODEL = "Nipponaclerda",
  PHRAG_INTERCEPT = -4, PHRAG_MAXENT_COEF = 3, PHRAG_SOURCE_COEF = 2,
  RCS_INTERCEPT = -4, RCS_MAXENT_COEF = 3, RCS_SOURCE_COEF = 2,
  STORM_INTERCEPT = -6, STORM_NRI_COEF = 2, STORMS_PER_TICK = 1L,
  USE_FIXED_SEED = FALSE, RNG_SEED = NULL,
  JUMP_SOURCE_METHOD = "offset",
  SPARSE_OPS_USE_CACHE = TRUE, SPARSE_OPS_OVERWRITE = FALSE,
  SPARSE_OPS_CACHE_SUBDIR = "SparseOpsCache",
  STORE_PROB_SUMMARY = TRUE
)

# Read a CA setting from the active environment, falling back to its default.
get_ca_option <- function(name) get0(name, ifnotfound = CA_DEFAULTS[[name]], inherits = TRUE)

NEIGHBORHOOD <- as.integer(get_ca_option("NEIGHBORHOOD"))
NEI_RADIUS <- as.integer(get_ca_option("NEI_RADIUS"))
DIST_METRIC <- tolower(as.character(get_ca_option("DIST_METRIC")))
T_STEPS <- as.integer(get_ca_option("MAX_TICKS"))
BASE_MODEL <- as.character(get_ca_option("BASE_MODEL"))

PHRAG_INTERCEPT <- get_ca_option("PHRAG_INTERCEPT")
PHRAG_MAXENT_COEF <- get_ca_option("PHRAG_MAXENT_COEF")
PHRAG_SOURCE_COEF <- get_ca_option("PHRAG_SOURCE_COEF")
RCS_INTERCEPT <- get_ca_option("RCS_INTERCEPT")
RCS_MAXENT_COEF <- get_ca_option("RCS_MAXENT_COEF")
RCS_SOURCE_COEF <- get_ca_option("RCS_SOURCE_COEF")
STORM_INTERCEPT <- get_ca_option("STORM_INTERCEPT")
STORM_NRI_COEF <- get_ca_option("STORM_NRI_COEF")
STORMS_PER_TICK_RAW <- get_ca_option("STORMS_PER_TICK")

USE_FIXED_SEED <- isTRUE(get_ca_option("USE_FIXED_SEED"))
DEFAULT_SEED <- get_ca_option("RNG_SEED")
JUMP_SOURCE_METHOD <- tolower(as.character(get_ca_option("JUMP_SOURCE_METHOD")))
SPARSE_OPS_USE_CACHE <- isTRUE(get_ca_option("SPARSE_OPS_USE_CACHE"))
SPARSE_OPS_OVERWRITE <- isTRUE(get_ca_option("SPARSE_OPS_OVERWRITE"))
SPARSE_OPS_CACHE_SUBDIR <- as.character(get_ca_option("SPARSE_OPS_CACHE_SUBDIR"))
STORE_PROB_SUMMARY <- isTRUE(get_ca_option("STORE_PROB_SUMMARY"))

# Validate core CA geometry, method choices, tick count, and coefficients.
validate_ca_config <- function() {
  if (!NEIGHBORHOOD %in% c(4L, 8L)) stop("NEIGHBORHOOD must be 4 or 8.")
  if (length(NEI_RADIUS) != 1L || !is.finite(NEI_RADIUS) || NEI_RADIUS < 1L)
    stop("NEI_RADIUS must be a single integer >= 1.")
  if (!DIST_METRIC %in% c("manhattan", "chebyshev"))
    stop("DIST_METRIC must be 'manhattan' or 'chebyshev'.")
  if (!JUMP_SOURCE_METHOD %in% c("offset", "sparse"))
    stop("JUMP_SOURCE_METHOD must be 'offset' or 'sparse'.")
  if (length(T_STEPS) != 1L || !is.finite(T_STEPS) || T_STEPS < 0L)
    stop("MAX_TICKS must be a single integer >= 0.")
  
  coefs <- c(PHRAG_INTERCEPT, PHRAG_MAXENT_COEF, PHRAG_SOURCE_COEF,
             RCS_INTERCEPT, RCS_MAXENT_COEF, RCS_SOURCE_COEF,
             STORM_INTERCEPT, STORM_NRI_COEF)
  if (!all(is.finite(coefs))) stop("All CA coefficients must be finite numeric values.")
  invisible(TRUE)
}

####################################################################################################
######################################## CORE HELPERS ##############################################
####################################################################################################

# Convert linear predictors to probabilities with a numerically stable logistic function.
logistic <- function(x) ifelse(x >= 0, 1 / (1 + exp(-x)), exp(x) / (1 + exp(x)))

# Validate and convert the storm schedule to nonnegative integer counts.
normalize_storm_schedule <- function(x) {
  x <- suppressWarnings(as.numeric(unlist(x, recursive = TRUE, use.names = FALSE)))
  if (!length(x)) stop("STORMS_PER_TICK must contain at least one value.")
  if (any(!is.finite(x))) stop("STORMS_PER_TICK must contain only finite numeric values.")
  if (any(x < 0)) stop("STORMS_PER_TICK values must be >= 0.")
  if (any(abs(x - round(x)) > sqrt(.Machine$double.eps)))
    stop("STORMS_PER_TICK values must be whole-number storm counts.")
  as.integer(round(x))
}

# Return the storm count for a tick, reusing the final value if needed.
resolve_storms_this_tick <- function(storm_schedule, tick_index) {
  tick_index <- as.integer(tick_index)
  if (tick_index < 1L) stop("tick_index must be >= 1.")
  if (!length(storm_schedule)) stop("storm_schedule is empty.")
  as.integer(storm_schedule[min(tick_index, length(storm_schedule))])
}

STORMS_PER_TICK <- normalize_storm_schedule(STORMS_PER_TICK_RAW)
validate_ca_config()

# Min-max scale a numeric vector to [0,1] while handling missing/sentinel values.
scale_minmax_safe <- function(x, na_fill = 0, sentinel_values = -50) {
  x <- suppressWarnings(as.numeric(x))
  if (length(sentinel_values)) x[x %in% sentinel_values] <- NA_real_
  if (all(is.na(x))) return(rep(na_fill, length(x)))
  
  rng <- range(x, na.rm = TRUE)
  if (!all(is.finite(rng)) || diff(rng) == 0) {
    out <- rep(0, length(x)); out[is.na(x)] <- na_fill
    return(out)
  }
  
  out <- (x - rng[1L]) / diff(rng)
  out[is.na(out)] <- na_fill
  pmax(0, pmin(1, out))
}

# Identify finite predictor values that are valid for jump-source calculations.
valid_score_mask <- function(x, sentinel_values = -50) {
  x <- suppressWarnings(as.numeric(x))
  is.finite(x) & (!length(sentinel_values) | !(x %in% sentinel_values))
}

####################################################################################################
####################################### REQUIRED HELPERS ###########################################
####################################################################################################

if (!exists("validate_grid_data", mode = "function")) {
  # Check that the spatial grid is complete, unique, and correctly indexed.
  validate_grid_data <- function(dt, nrow_grid, ncol_grid) {
    if (!data.table::is.data.table(dt)) dt <- data.table::as.data.table(dt)
    missing <- setdiff(c("row", "col"), names(dt))
    if (length(missing)) stop("Grid data is missing required columns: ", paste(missing, collapse = ", "))
    if (anyNA(dt$row) || anyNA(dt$col)) stop("Grid data contains NA values in row/col.")
    if (!is.numeric(dt$row) || !is.numeric(dt$col)) stop("row and col must be numeric/integer.")
    if (any(dt$row < 1) || any(dt$col < 1)) stop("row and col must be >= 1.")
    
    dup_n <- sum(duplicated(dt[, .(row, col)]))
    if (dup_n) stop("Grid contains duplicate (row, col) locations. Duplicates found: ", dup_n)
    
    expected_n <- as.integer(nrow_grid) * as.integer(ncol_grid)
    if (nrow(dt) != expected_n)
      stop("Grid is not complete. Expected cells: ", expected_n, "; actual records: ", nrow(dt))
    
    ranges <- list(row = c(1L, as.integer(nrow_grid)), col = c(1L, as.integer(ncol_grid)))
    for (nm in names(ranges)) {
      observed <- as.integer(range(dt[[nm]]))
      if (!identical(observed, ranges[[nm]]))
        stop("", nm, " indices do not span ", paste(ranges[[nm]], collapse = ".."),
             ". Observed: ", paste(observed, collapse = ":"))
    }
    invisible(TRUE)
  }
}

if (!exists("resolve_seed_indices", mode = "function")) {
  # Match supplied seed cell IDs to row indices in the ordered grid table.
  resolve_seed_indices <- function(ids, dt, arg_name = "ids", allow_empty = FALSE) {
    if (is.null(ids) || !length(ids)) {
      if (isTRUE(allow_empty)) return(integer(0))
      stop("", arg_name, " is empty.")
    }
    idx <- match(ids, dt$cell_id)
    idx <- unique(as.integer(idx[!is.na(idx)]))
    if (!length(idx) && !isTRUE(allow_empty)) stop("None of ", arg_name, " matched dt$cell_id.")
    idx
  }
}

if (!exists("get_sparse_ops_cache_dir", mode = "function")) {
  # Resolve the directory used to store cached CA spatial operators.
  get_sparse_ops_cache_dir <- function() {
    subdir <- get0("SPARSE_OPS_CACHE_SUBDIR", ifnotfound = "SparseOpsCache", inherits = TRUE)
    dirs <- get0("DIRS", ifnotfound = NULL, inherits = TRUE)
    if (is.list(dirs) && !is.null(dirs$outputs_dir)) return(file.path(dirs$outputs_dir, subdir))
    file.path(getwd(), subdir)
  }
}

if (!exists("safe_prob_summary", mode = "function")) {
  # Summarize eligible establishment probabilities while safely handling empty cases.
  safe_prob_summary <- function(prob, eligible) {
    prob <- suppressWarnings(as.numeric(prob)); eligible <- as.logical(eligible)
    use <- prob[eligible & is.finite(prob)]
    n_eligible <- sum(eligible, na.rm = TRUE)
    if (!length(use))
      return(c(n_eligible = n_eligible, p_min = NA_real_, p_q25 = NA_real_, p_median = NA_real_,
               p_mean = NA_real_, p_q75 = NA_real_, p_max = NA_real_))
    
    q <- stats::quantile(use, c(0, .25, .5, .75, 1), na.rm = TRUE, names = FALSE)
    c(n_eligible = n_eligible, p_min = q[1L], p_q25 = q[2L], p_median = q[3L],
      p_mean = mean(use, na.rm = TRUE), p_q75 = q[4L], p_max = q[5L])
  }
}

if (!exists("build_neighbor_offsets_adj", mode = "function")) {
  # Build row/column offsets for the selected 4- or 8-cell neighborhood.
  build_neighbor_offsets_adj <- function(neighborhood = 8L) {
    neighborhood <- as.integer(neighborhood)
    if (neighborhood == 4L)
      return(matrix(c(-1L,0L, 1L,0L, 0L,-1L, 0L,1L), ncol = 2L, byrow = TRUE))
    if (neighborhood == 8L)
      return(matrix(c(-1L,-1L, -1L,0L, -1L,1L, 0L,-1L, 0L,1L,
                      1L,-1L, 1L,0L, 1L,1L), ncol = 2L, byrow = TRUE))
    stop("neighborhood must be 4 or 8.")
  }
}

if (!exists("build_neighbor_offsets_radius", mode = "function")) {
  # Build all offsets inside the selected radius and distance metric.
  build_neighbor_offsets_radius <- function(radius = 4L, metric = c("chebyshev", "manhattan")) {
    metric <- match.arg(metric); radius <- as.integer(radius)
    if (radius < 1L) stop("radius must be >= 1.")
    
    v <- seq.int(-radius, radius)
    g <- expand.grid(dr = v, dc = v, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    g <- g[g$dr != 0L | g$dc != 0L, , drop = FALSE]
    keep <- if (metric == "manhattan") abs(g$dr) + abs(g$dc) <= radius else
      pmax(abs(g$dr), abs(g$dc)) <= radius
    
    offsets <- as.matrix(g[keep, c("dr", "dc"), drop = FALSE]); storage.mode(offsets) <- "integer"
    list(offsets = offsets, radius = radius, metric = metric)
  }
}

# Build nonlocal jump offsets by removing local adjacency offsets from the radius.
build_jump_offsets <- function(radius = 4L, metric = c("chebyshev", "manhattan"), neighborhood = 8L) {
  radius_offsets <- build_neighbor_offsets_radius(radius, match.arg(metric))$offsets
  adjacent_offsets <- build_neighbor_offsets_adj(neighborhood)
  # Convert offset pairs to compact string keys for set comparison.
  key <- function(x) paste(x[, 1L], x[, 2L], sep = ",")
  out <- radius_offsets[!key(radius_offsets) %in% key(adjacent_offsets), , drop = FALSE]
  storage.mode(out) <- "integer"
  out
}

if (!exists("build_sparse_from_offsets", mode = "function")) {
  # Convert spatial offsets into source/destination indices for sparse matrices.
  build_sparse_from_offsets <- function(nrow_grid, ncol_grid, offsets) {
    nrow_grid <- as.integer(nrow_grid); ncol_grid <- as.integer(ncol_grid)
    if (!is.matrix(offsets) || ncol(offsets) != 2L) stop("offsets must be a two-column matrix.")
    
    base <- expand.grid(row = seq_len(nrow_grid), col = seq_len(ncol_grid),
                        KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
    i_all <- j_all <- vector("list", nrow(offsets))
    
    for (k in seq_len(nrow(offsets))) {
      rr <- base$row + offsets[k, 1L]; cc <- base$col + offsets[k, 2L]
      keep <- rr >= 1L & rr <= nrow_grid & cc >= 1L & cc <= ncol_grid
      if (!any(keep)) next
      i_all[[k]] <- (base$row[keep] - 1L) * ncol_grid + base$col[keep]
      j_all[[k]] <- (rr[keep] - 1L) * ncol_grid + cc[keep]
    }
    
    list(i = as.integer(unlist(i_all, use.names = FALSE)),
         j = as.integer(unlist(j_all, use.names = FALSE)))
  }
}

####################################################################################################
#################################### COVARIATE CONSTRUCTION ########################################
####################################################################################################

# Resolve the RCS MaxEnt predictor column associated with the selected base model.
resolve_active_rcs_column <- function(rcs_data, base_model) {
  model <- gsub("\\s+", "_", as.character(base_model))
  aliases <- c(
    NIPPON = "MaxEnt_Nipponaclerda", NIPPONACLERDA = "MaxEnt_Nipponaclerda",
    COLD = "MaxEnt_RCS_cold_native", RCS_COLD_NATIVE = "MaxEnt_RCS_cold_native",
    COLD_NATIVE = "MaxEnt_RCS_cold_native", NATIVE = "MaxEnt_RCS_native",
    RCS_NATIVE = "MaxEnt_RCS_native"
  )
  candidates <- unique(c(unname(aliases[toupper(model)]), model, paste0("MaxEnt_", model)))
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  hit <- candidates[candidates %in% names(rcs_data)][1L]
  if (!length(hit) || is.na(hit))
    stop("Could not resolve active RCS MaxEnt column from BASE_MODEL = '", base_model,
         "'. Looked for: ", paste(candidates, collapse = ", "))
  hit
}

# Scale model predictors and identify cells eligible to act as storm-jump sources.
compute_ca_covariates <- function(rcs_data, base_model = BASE_MODEL, verbose = TRUE) {
  if (!is.data.frame(rcs_data) && !data.table::is.data.table(rcs_data))
    stop("rcs_data must be a data.frame or data.table.")
  
  missing <- setdiff(c("MaxEnt_Phragmites", "NRI_Score"), names(rcs_data))
  if (length(missing)) stop("rcs_data is missing required columns: ", paste(missing, collapse = ", "))
  
  active <- resolve_active_rcs_column(rcs_data, base_model)
  valid <- valid_score_mask(rcs_data$MaxEnt_Phragmites) &
    valid_score_mask(rcs_data[[active]]) & valid_score_mask(rcs_data$NRI_Score)
  
  out <- list(
    M_phrag = scale_minmax_safe(rcs_data$MaxEnt_Phragmites),
    M_rcs = scale_minmax_safe(rcs_data[[active]]),
    N_jump = scale_minmax_safe(rcs_data$NRI_Score),
    valid_jump_source = as.logical(valid),
    active_rcs_col = active
  )
  
  if (isTRUE(verbose))
    cat("Computed normalized covariates for ", nrow(rcs_data), " cells.",
        "\n   → Active RCS MaxEnt column: ", active,
        "\n   → Eligible jump-source cells: ", sum(valid), " of ", nrow(rcs_data), "\n", sep = "")
  out
}

# Use attached normalized covariates when available, otherwise compute them.
get_or_compute_ca_covariates <- function(rcs_data, base_model = BASE_MODEL, verbose = TRUE) {
  active <- resolve_active_rcs_column(rcs_data, base_model)
  if (!all(c("M_phrag", "M_rcs", "N_jump") %in% names(rcs_data)))
    return(compute_ca_covariates(rcs_data, base_model, verbose))
  
  valid <- if ("valid_jump_source" %in% names(rcs_data)) as.logical(rcs_data$valid_jump_source) else
    valid_score_mask(rcs_data$MaxEnt_Phragmites) & valid_score_mask(rcs_data[[active]]) &
    valid_score_mask(rcs_data$NRI_Score)
  
  if (isTRUE(verbose))
    cat("Using preattached normalized covariates from rcs_data.",
        "\n   → Active RCS MaxEnt column: ", active,
        "\n   → Eligible jump-source cells: ", sum(valid), " of ", nrow(rcs_data), "\n", sep = "")
  
  list(M_phrag = as.numeric(rcs_data$M_phrag), M_rcs = as.numeric(rcs_data$M_rcs),
       N_jump = as.numeric(rcs_data$N_jump), valid_jump_source = as.logical(valid),
       active_rcs_col = active)
}

####################################################################################################
###################################### OPERATOR BUILD ##############################################
####################################################################################################

# Check whether cached spatial operators match the required jump-source method.
.valid_cached_ops <- function(x, method) {
  ok <- is.list(x) && !is.null(x$A_adj_mean) && inherits(x$A_adj_mean, "Matrix") &&
    identical(as.character(x$jump_source_method), method)
  if (!ok) return(FALSE)
  if (method == "sparse") return(!is.null(x$A_jump_binary) && inherits(x$A_jump_binary, "Matrix"))
  !is.null(x$jump_offsets) && is.matrix(x$jump_offsets) && ncol(x$jump_offsets) == 2L
}

# Build or load local adjacency and nonlocal jump operators for the CA grid.
build_sparse_ca_operators <- function(nrow_grid, ncol_grid, neighborhood = 8L, radius = 4L,
                                      metric = c("chebyshev", "manhattan"),
                                      jump_source_method = JUMP_SOURCE_METHOD, verbose = TRUE,
                                      cache_dir = get_sparse_ops_cache_dir(),
                                      use_cache = SPARSE_OPS_USE_CACHE,
                                      overwrite_cache = SPARSE_OPS_OVERWRITE) {
  metric <- match.arg(metric)
  jump_source_method <- tolower(as.character(jump_source_method))
  if (!jump_source_method %in% c("offset", "sparse"))
    stop("jump_source_method must be 'offset' or 'sparse'.")
  if (!requireNamespace("Matrix", quietly = TRUE))
    stop("Package 'Matrix' is required for sparse CA adjacency operators.")
  
  nrow_grid <- as.integer(nrow_grid); ncol_grid <- as.integer(ncol_grid)
  neighborhood <- as.integer(neighborhood); radius <- as.integer(radius)
  N <- nrow_grid * ncol_grid
  
  cache_file <- NULL
  if (!is.null(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
    cache_file <- file.path(cache_dir, paste0("CAOps_nr", nrow_grid, "_nc", ncol_grid,
                                              "_nb", neighborhood, "_r", radius, "_m", metric, "_jump", jump_source_method, ".rds"))
    
    if (isTRUE(use_cache) && !isTRUE(overwrite_cache) && file.exists(cache_file)) {
      cached <- readRDS(cache_file)
      if (.valid_cached_ops(cached, jump_source_method)) {
        if (isTRUE(verbose)) cat("Loaded cached CA operators:\n   → ", cache_file, "\n", sep = "")
        return(cached)
      }
      if (isTRUE(verbose)) cat("Cached CA operator file is outdated; rebuilding:\n   → ", cache_file, "\n", sep = "")
    }
  }
  
  adj_edges <- build_sparse_from_offsets(nrow_grid, ncol_grid, build_neighbor_offsets_adj(neighborhood))
  adj_deg <- tabulate(adj_edges$i, nbins = N)
  A_adj_mean <- Matrix::sparseMatrix(i = adj_edges$i, j = adj_edges$j,
                                     x = 1 / adj_deg[adj_edges$i], dims = c(N, N))
  
  jump_offsets <- build_jump_offsets(radius, metric, neighborhood)
  A_jump_binary <- NULL; jump_edge_n <- NA_integer_
  
  if (jump_source_method == "sparse") {
    jump_edges <- build_sparse_from_offsets(nrow_grid, ncol_grid, jump_offsets)
    A_jump_binary <- Matrix::sparseMatrix(
      i = jump_edges$i, j = jump_edges$j,
      x = if (length(jump_edges$i)) 1 else numeric(0), dims = c(N, N)
    )
    jump_edge_n <- length(jump_edges$j)
  }
  
  out <- list(A_adj_mean = A_adj_mean, A_jump_binary = A_jump_binary,
              jump_offsets = jump_offsets, jump_source_method = jump_source_method,
              nrow_grid = nrow_grid, ncol_grid = ncol_grid,
              neighborhood = neighborhood, radius = radius, metric = metric)
  
  if (!is.null(cache_file)) {
    saveRDS(out, cache_file)
    if (isTRUE(verbose)) cat("Saved cached CA operators:\n   → ", cache_file, "\n", sep = "")
  }
  
  if (isTRUE(verbose)) {
    cat("CA operators built for ", N, " cells.",
        "\n   → adjacency edges: ", length(adj_edges$j),
        "\n   → jump source method: ", jump_source_method,
        "\n   → jump offsets: ", nrow(jump_offsets), "\n", sep = "")
    if (jump_source_method == "sparse")
      cat("   → jump-shell sparse edges: ", jump_edge_n, "\n", sep = "")
    else cat("   → jump-shell sparse matrix: not built\n")
  }
  out
}

####################################################################################################
################################ SOURCE-PRESSURE CALCULATORS #######################################
####################################################################################################

# Calculate the local proportion of occupied neighboring cells.
compute_adjacent_source_sparse <- function(state, A_adj_mean)
  as.numeric(A_adj_mean %*% as.numeric(state == 1L))

# Fast sum over a full Manhattan diamond, including the focal cell.
# For vertical distance d, the horizontal window has half-width radius - d.
# A single cumulative sum lets each horizontal window be evaluated in O(1),
# reducing work from O(n_cells * radius^2) to O(n_cells * radius).
.manhattan_diamond_sum_grid <- function(values, radius, nrow_grid, ncol_grid) {
  n <- length(values)
  out <- numeric(n)
  prefix <- c(0, cumsum(values))
  row_base <- rep((seq_len(nrow_grid) - 1L) * ncol_grid, each = ncol_grid)
  columns <- seq_len(ncol_grid)
  
  for (d in 0L:min(radius, nrow_grid - 1L)) {
    half_width <- radius - d
    left <- rep.int(pmax.int(1L, columns - half_width), nrow_grid)
    right <- rep.int(pmin.int(ncol_grid, columns + half_width), nrow_grid)
    horizontal <- prefix[row_base + right + 1L] - prefix[row_base + left]
    
    if (d == 0L) {
      out <- out + horizontal
    } else {
      shift <- d * ncol_grid
      first <- seq_len(n - shift)
      second <- first + shift
      out[first] <- out[first] + horizontal[second]
      out[second] <- out[second] + horizontal[first]
    }
  }
  out
}

# Reference method retained for arbitrary/non-Manhattan offset collections.
.offset_sum_grid_reference <- function(values, offsets, nrow_grid, ncol_grid) {
  src <- matrix(values, nrow_grid, ncol_grid, byrow = TRUE)
  out <- matrix(0, nrow_grid, ncol_grid)
  
  for (k in seq_len(nrow(offsets))) {
    dr <- offsets[k, 1L]; dc <- offsets[k, 2L]
    r <- max(1L, 1L - dr):min(nrow_grid, nrow_grid - dr)
    c <- max(1L, 1L - dc):min(ncol_grid, ncol_grid - dc)
    if (!length(r) || !length(c) || r[1L] > tail(r, 1L) || c[1L] > tail(c, 1L)) next
    out[r, c] <- out[r, c] + src[r + dr, c + dc]
  }
  as.numeric(t(out))
}

# Sum values across grid offsets without building a large sparse jump matrix.
# Recognize the exact canonical Manhattan storm shell and use a faster method;
# all other offset geometries continue to use the original implementation.
offset_sum_grid <- function(values, offsets, nrow_grid, ncol_grid) {
  values <- as.numeric(values); nrow_grid <- as.integer(nrow_grid); ncol_grid <- as.integer(ncol_grid)
  if (length(values) != nrow_grid * ncol_grid)
    stop("values length does not match nrow_grid * ncol_grid.")
  if (is.null(offsets) || !is.matrix(offsets) || ncol(offsets) != 2L)
    stop("offsets must be a two-column matrix.")
  if (nrow(offsets) == 0L) return(numeric(length(values)))
  
  # Only optimize exact, built-in Manhattan shells. This also supports older
  # cached operators because the geometry is recognized from their offsets.
  # Counts from the CA are binary; restrict the exact fast path to binary
  # inputs so arbitrary real-valued callers keep the reference arithmetic.
  if (nrow(offsets) > 8L && all(is.finite(values)) && all(values == 0 | values == 1)) {
    radius <- as.integer(max(abs(offsets)))
    for (neighborhood in c(4L, 8L)) {
      expected <- build_jump_offsets(radius, "manhattan", neighborhood)
      if (identical(dim(offsets), dim(expected)) && all(offsets == expected)) {
        # The diamond includes the focal cell and local neighbors, whereas
        # build_jump_offsets() excludes both. Subtract exactly those cells.
        adjacent <- build_neighbor_offsets_adj(neighborhood)
        adjacent <- adjacent[rowSums(abs(adjacent)) <= radius, , drop = FALSE]
        return(.manhattan_diamond_sum_grid(values, radius, nrow_grid, ncol_grid) -
                 values - .offset_sum_grid_reference(values, adjacent, nrow_grid, ncol_grid))
      }
    }
  }
  
  .offset_sum_grid_reference(values, offsets, nrow_grid, ncol_grid)
}

# Convert jump-source counts into proportions and shared diagnostic outputs.
.finish_jump_source <- function(state, eligible_counts, infected_counts) {
  source <- numeric(length(state)); keep <- eligible_counts > 0
  source[keep] <- infected_counts[keep] / eligible_counts[keep]
  list(source = source, eligible_counts = eligible_counts, infected_counts = infected_counts,
       has_any_infected = infected_counts > 0)
}

# Calculate nonlocal RCS source pressure with the sparse jump matrix.
compute_jump_source_equal_sparse <- function(state, A_jump_binary, eligible_mask, eligible_counts = NULL) {
  if (is.null(A_jump_binary)) stop("A_jump_binary is required for sparse jump source method.")
  infected_counts <- as.numeric(A_jump_binary %*% as.numeric((state == 1L) & eligible_mask))
  if (is.null(eligible_counts)) eligible_counts <- as.numeric(A_jump_binary %*% as.numeric(eligible_mask))
  .finish_jump_source(state, eligible_counts, infected_counts)
}

# Calculate nonlocal RCS source pressure with grid-shift offsets.
compute_jump_source_equal_offset <- function(state, jump_offsets, eligible_mask, nrow_grid, ncol_grid,
                                             eligible_counts = NULL) {
  # Apply the current jump offsets to sum a supplied grid vector.
  count <- function(x) offset_sum_grid(x, jump_offsets, nrow_grid, ncol_grid)
  infected_counts <- count(as.numeric((state == 1L) & eligible_mask))
  if (is.null(eligible_counts)) eligible_counts <- count(as.numeric(eligible_mask))
  .finish_jump_source(state, eligible_counts, infected_counts)
}

# Dispatch jump-source calculation to the configured offset or sparse method.
.jump_source <- function(state, eligible_mask, sparse_ops, eligible_counts = NULL) {
  if (sparse_ops$jump_source_method == "offset")
    return(compute_jump_source_equal_offset(state, sparse_ops$jump_offsets, eligible_mask,
                                            sparse_ops$nrow_grid, sparse_ops$ncol_grid, eligible_counts))
  compute_jump_source_equal_sparse(state, sparse_ops$A_jump_binary, eligible_mask, eligible_counts)
}

# Precompute the number of eligible jump-source cells around each target cell.
.jump_eligible_counts <- function(mask, sparse_ops) {
  if (sparse_ops$jump_source_method == "offset")
    return(offset_sum_grid(as.numeric(mask), sparse_ops$jump_offsets,
                           sparse_ops$nrow_grid, sparse_ops$ncol_grid))
  as.numeric(sparse_ops$A_jump_binary %*% as.numeric(mask))
}

####################################################################################################
########################################### CA UPDATES #############################################
####################################################################################################

# Apply one generic local establishment step using habitat and adjacent source pressure.
.update_local_state <- function(state, habitat, A_adj_mean, intercept, maxent_coef, source_coef,
                                support = NULL, eligible = NULL) {
  adj_source <- compute_adjacent_source_sparse(state, A_adj_mean)
  probability <- logistic(intercept + maxent_coef * habitat + source_coef * adj_source)
  can_establish <- (state == 0L) & (adj_source > 0)
  if (!is.null(support)) can_establish <- can_establish & support
  if (!is.null(eligible)) can_establish <- can_establish & eligible
  
  establish <- can_establish & (runif(length(state)) < probability)
  next_state <- state; next_state[establish] <- 1L
  if (!is.null(support)) next_state[!support] <- 0L
  
  list(next_state = next_state, probability = probability, adj_source = adj_source,
       can_establish = can_establish, establish = establish)
}

# Apply local PHRAG spread using the shared local-state update routine.
update_local_phrag_sparse <- function(phrag_state, M_phrag, A_adj_mean,
                                      phrag_intercept, phrag_maxent_coef, phrag_source_coef,
                                      eligible = NULL) {
  .update_local_state(phrag_state, M_phrag, A_adj_mean,
                      phrag_intercept, phrag_maxent_coef, phrag_source_coef, eligible = eligible)
}

# Apply local RCS spread while enforcing PHRAG support.
update_local_rcs_sparse <- function(rcs_state, phrag_state_after_updates, M_rcs, A_adj_mean,
                                    rcs_intercept, rcs_maxent_coef, rcs_source_coef,
                                    eligible = NULL) {
  .update_local_state(rcs_state, M_rcs, A_adj_mean,
                      rcs_intercept, rcs_maxent_coef, rcs_source_coef,
                      support = phrag_state_after_updates == 1L, eligible = eligible)
}

# Apply storm-mediated joint PHRAG/RCS establishment from nonlocal RCS sources.
update_joint_jump <- function(phrag_state, rcs_state, valid_jump_source, sparse_ops, q_storm_base,
                              jump_eligible_counts = NULL, storm_count_this_tick = 1L,
                              valid_jump_target = NULL) {
  method <- tolower(as.character(sparse_ops$jump_source_method))
  if (!method %in% c("offset", "sparse"))
    stop("sparse_ops$jump_source_method must be 'offset' or 'sparse'.")
  sparse_ops$jump_source_method <- method
  
  jump_src <- .jump_source(rcs_state, valid_jump_source, sparse_ops, jump_eligible_counts)
  can_jump <- (rcs_state == 0L) & (jump_src$eligible_counts > 0) & jump_src$has_any_infected
  if (!is.null(valid_jump_target)) can_jump <- can_jump & valid_jump_target
  
  p_single_storm <- q_storm_base * jump_src$source
  p_single_storm[!can_jump] <- 0
  
  if (length(storm_count_this_tick) != 1L || !is.finite(storm_count_this_tick) || storm_count_this_tick < 0)
    stop("storm_count_this_tick must be a single integer >= 0.")
  storm_count_this_tick <- as.integer(storm_count_this_tick)
  
  probability <- if (storm_count_this_tick == 0L) rep(0, length(rcs_state)) else
    if (storm_count_this_tick == 1L) p_single_storm else
      1 - (1 - p_single_storm)^storm_count_this_tick
  probability[!can_jump] <- 0
  
  jump_event <- can_jump & (runif(length(rcs_state)) < probability)
  phrag_next <- phrag_state; rcs_next <- rcs_state
  phrag_next[jump_event] <- 1L; rcs_next[jump_event] <- 1L
  
  list(phrag_next = phrag_next, rcs_next = rcs_next, probability = probability,
       p_single_storm = p_single_storm, S_radius_rcs = jump_src$source,
       shell_eligible_n = jump_src$eligible_counts, shell_infected_n = jump_src$infected_counts,
       can_jump = can_jump, jump_event = jump_event, storm_count_this_tick = storm_count_this_tick)
}

# Storm-mediated PHRAG-only establishment from nonlocal PHRAG sources.
# This uses the same storms, jump geometry, and NRI factor as the existing RCS process.
update_phrag_jump <- function(phrag_state, valid_phrag_source, sparse_ops, q_phrag_storm_base,
                              phrag_jump_eligible_counts = NULL, storm_count_this_tick = 1L,
                              valid_phrag_target = NULL) {
  jump_src <- .jump_source(phrag_state, valid_phrag_source, sparse_ops, phrag_jump_eligible_counts)
  can_jump <- (phrag_state == 0L) & (jump_src$eligible_counts > 0) & jump_src$has_any_infected
  if (!is.null(valid_phrag_target)) can_jump <- can_jump & valid_phrag_target
  
  p_single_storm <- q_phrag_storm_base * jump_src$source
  p_single_storm[!can_jump] <- 0
  if (length(storm_count_this_tick) != 1L || !is.finite(storm_count_this_tick) ||
      storm_count_this_tick < 0 || storm_count_this_tick != as.integer(storm_count_this_tick))
    stop("storm_count_this_tick must be a single nonnegative integer.")
  storm_count_this_tick <- as.integer(storm_count_this_tick)
  
  probability <- if (storm_count_this_tick == 0L) rep(0, length(phrag_state)) else
    if (storm_count_this_tick == 1L) p_single_storm else
      1 - (1 - p_single_storm)^storm_count_this_tick
  probability[!can_jump] <- 0
  
  jump_event <- can_jump & (runif(length(phrag_state)) < probability)
  phrag_next <- phrag_state
  phrag_next[jump_event] <- 1L
  
  list(phrag_next = phrag_next, probability = probability, can_jump = can_jump,
       jump_event = jump_event, p_single_storm = p_single_storm,
       S_radius_phrag = jump_src$source)
}

####################################################################################################
#################################### RUN THE TWO-LAYER CA ##########################################
####################################################################################################

# Create one tick-level probability-summary record for all spread mechanisms.
.make_prob_summary_row <- function(t, storms, jump_upd, phrag_jump_upd, phrag_upd, rcs_upd) {
  # Prefix probability-summary field names for a specific spread mechanism.
  add_prefix <- function(x, prefix) setNames(as.list(x), paste0(prefix, names(x)))
  cbind(
    data.frame(Tick = as.integer(t), Storms_This_Tick = as.integer(storms), stringsAsFactors = FALSE),
    add_prefix(safe_prob_summary(jump_upd$probability, jump_upd$can_jump), "jump_"),
    add_prefix(safe_prob_summary(phrag_jump_upd$probability, phrag_jump_upd$can_jump), "phrag_jump_"),
    add_prefix(safe_prob_summary(phrag_upd$probability, phrag_upd$can_establish), "phrag_"),
    add_prefix(safe_prob_summary(rcs_upd$probability, rcs_upd$can_establish), "rcs_")
  )
}

# Run the complete two-layer PHRAG/RCS cellular automaton across all ticks.
run_ca_two_layer <- function(rcs_data, nrow_grid, ncol_grid, initial_phrag_ids,
                             initial_rcs_ids = NULL, base_model = BASE_MODEL,
                             neighborhood = NEIGHBORHOOD, nei_radius = NEI_RADIUS,
                             dist_metric = DIST_METRIC, T_steps = T_STEPS,
                             seed = if (isTRUE(USE_FIXED_SEED)) DEFAULT_SEED else NULL,
                             store_history = FALSE, store_probabilities = FALSE,
                             store_prob_summary = STORE_PROB_SUMMARY,
                             sparse_ops = NULL, verbose = TRUE,
                             jump_eligible_counts = NULL,
                             history_storage = if (isTRUE(store_history)) "full" else "none",
                             rcs_intro_ids = NULL, rcs_intro_tick = 0L,
                             phrag_jump_eligible_counts = NULL) {
  history_storage <- tolower(as.character(history_storage))
  if (length(history_storage) != 1L ||
      !history_storage %in% c("full", "arrival", "none"))
    stop("history_storage must be 'full', 'arrival', or 'none'.")
  
  # Tick 0 is the initial state; tick t is the state at START_YEAR + t.
  # Keep introduction optional to preserve the existing CA calling interface.
  if (length(rcs_intro_tick) != 1L || !is.numeric(rcs_intro_tick) ||
      is.na(rcs_intro_tick) || !is.finite(rcs_intro_tick) ||
      rcs_intro_tick != as.integer(rcs_intro_tick) ||
      rcs_intro_tick < 0L || rcs_intro_tick > T_steps)
    stop("rcs_intro_tick must be a whole-number tick between 0 and T_steps.")
  rcs_intro_tick <- as.integer(rcs_intro_tick)
  dt <- data.table::as.data.table(rcs_data)
  if (!"cell_id" %in% names(dt)) dt[, cell_id := if ("GridSquare" %in% names(dt)) GridSquare else .I]
  data.table::setorderv(dt, c("row", "col"))
  validate_grid_data(dt, nrow_grid, ncol_grid)
  if (anyDuplicated(dt$cell_id)) stop("dt$cell_id must be unique.")
  
  if (!is.null(seed)) {
    set.seed(seed)
    if (isTRUE(verbose)) cat("run_ca_two_layer(): RNG seed set to ", seed, "\n", sep = "")
  }
  
  covs <- get_or_compute_ca_covariates(dt, base_model, verbose)
  # MaxEnt NoData (-50) cells cannot receive new simulated establishment.
  # Keep these destination masks separate from the NRI-dependent storm-source mask.
  valid_phrag_habitat <- valid_score_mask(dt$MaxEnt_Phragmites)
  valid_rcs_habitat <- valid_score_mask(dt[[covs$active_rcs_col]])
  valid_joint_habitat <- valid_phrag_habitat & valid_rcs_habitat
  valid_phrag_jump_source <- valid_phrag_habitat & valid_score_mask(dt$NRI_Score)
  if (is.null(sparse_ops))
    sparse_ops <- build_sparse_ca_operators(as.integer(nrow_grid), as.integer(ncol_grid),
                                            as.integer(neighborhood), as.integer(nei_radius), dist_metric, JUMP_SOURCE_METHOD, verbose)
  if (is.null(sparse_ops$jump_source_method))
    sparse_ops$jump_source_method <- if (!is.null(sparse_ops$A_jump_binary)) "sparse" else JUMP_SOURCE_METHOD
  sparse_ops$jump_source_method <- tolower(as.character(sparse_ops$jump_source_method))
  
  N <- nrow(dt)
  phrag_state <- rcs_state <- integer(N)
  phrag_state[resolve_seed_indices(initial_phrag_ids, dt, "initial_phrag_ids")] <- 1L
  
  if (!is.null(initial_rcs_ids) && length(initial_rcs_ids)) {
    idx <- resolve_seed_indices(initial_rcs_ids, dt, "initial_rcs_ids", allow_empty = TRUE)
    if (!length(idx)) warning("initial_rcs_ids did not match dt$cell_id; starting RCS empty.")
    else rcs_state[idx] <- 1L
  }
  
  if (any(rcs_state == 1L & phrag_state == 0L))
    warning("Some initial_rcs_ids were outside initial PHRAG support and were removed.")
  rcs_state[phrag_state == 0L] <- 0L
  
  # Resolve introduction IDs against the same canonical, row-major cell order
  # used by the CA. Fail rather than silently omit requested introduction sites.
  intro_idx <- integer(0)
  if (!is.null(rcs_intro_ids) && length(rcs_intro_ids)) {
    intro_idx <- match(unique(rcs_intro_ids), dt$cell_id)
    if (anyNA(intro_idx))
      stop("Some rcs_intro_ids are not present in the CA grid.")
    intro_idx <- as.integer(intro_idx)
  }
  
  # An introduction scheduled at tick 0 belongs to the initial conditions.
  # For later introductions, host and RCS are added only after that tick's spread.
  if (rcs_intro_tick == 0L && length(intro_idx)) {
    phrag_state[intro_idx] <- 1L
    rcs_state[intro_idx] <- 1L
  }
  
  # Keep the original coupled RCS jump probability unchanged.
  q_storm_base <- logistic(PHRAG_INTERCEPT + PHRAG_MAXENT_COEF * covs$M_phrag) *
    logistic(RCS_INTERCEPT + RCS_MAXENT_COEF * covs$M_rcs) *
    logistic(STORM_INTERCEPT + STORM_NRI_COEF * covs$N_jump)
  # PHRAG-only jumps require PHRAG suitability and storm risk, but not RCS suitability.
  q_phrag_storm_base <- logistic(PHRAG_INTERCEPT + PHRAG_MAXENT_COEF * covs$M_phrag) *
    logistic(STORM_INTERCEPT + STORM_NRI_COEF * covs$N_jump)
  # The eligible-source denominator depends only on grid geometry and valid predictor cells.
  # A batch runner may supply it once for many stochastic replicates.
  if (is.null(jump_eligible_counts)) {
    jump_eligible_counts <- .jump_eligible_counts(covs$valid_jump_source, sparse_ops)
  } else {
    if (!is.numeric(jump_eligible_counts) || length(jump_eligible_counts) != N ||
        any(!is.finite(jump_eligible_counts)) || any(jump_eligible_counts < 0))
      stop("jump_eligible_counts must be a finite nonnegative numeric vector of length nrow(rcs_data).")
  }
  
  # The PHRAG-only storm source denominator can likewise be reused across runs.
  if (is.null(phrag_jump_eligible_counts)) {
    phrag_jump_eligible_counts <- .jump_eligible_counts(valid_phrag_jump_source, sparse_ops)
  } else if (!is.numeric(phrag_jump_eligible_counts) || length(phrag_jump_eligible_counts) != N ||
             any(!is.finite(phrag_jump_eligible_counts)) || any(phrag_jump_eligible_counts < 0)) {
    stop("phrag_jump_eligible_counts must be a finite nonnegative vector of length nrow(rcs_data).")
  }
  
  if (isTRUE(verbose)) {
    cat("Initial PHRAG occupied: ", sum(phrag_state), " of ", N,
        "\nInitial RCS occupied:   ", sum(rcs_state), " of ", N,
        "\n   active RCS model: ", covs$active_rcs_col,
        "\n   jump source method: ", sparse_ops$jump_source_method,
        "\n   jump offsets: ", if (!is.null(sparse_ops$jump_offsets)) nrow(sparse_ops$jump_offsets) else NA_integer_,
        "\n   PHRAG local: intercept=", PHRAG_INTERCEPT, " | maxent=", PHRAG_MAXENT_COEF,
        " | source=", PHRAG_SOURCE_COEF,
        "\n   RCS local:   intercept=", RCS_INTERCEPT, " | maxent=", RCS_MAXENT_COEF,
        " | source=", RCS_SOURCE_COEF,
        "\n   RCS introduction: tick=", rcs_intro_tick,
        " | sites=", length(intro_idx),
        "\n   Update order: PHRAG + RCS storm jumps → local PHRAG → local RCS → introduction → output\n", sep = "")
  }
  
  phrag_hist <- rcs_hist <- NULL
  phrag_first_tick <- rcs_first_tick <- NULL
  p_phrag_local_hist <- p_rcs_local_hist <- p_jump_hist <- p_phrag_jump_hist <- NULL
  prob_summary <- NULL
  
  if (history_storage == "full") {
    phrag_hist <- matrix(0L, N, T_steps + 1L)
    rcs_hist <- matrix(0L, N, T_steps + 1L)
    phrag_hist[, 1L] <- phrag_state
    rcs_hist[, 1L] <- rcs_state
  } else if (history_storage == "arrival") {
    # Occupancy is irreversible, so a first-arrival tick fully determines
    # every subsequent occupancy state. -1 denotes never occupied.
    phrag_first_tick <- rep.int(-1L, N)
    rcs_first_tick <- rep.int(-1L, N)
    phrag_first_tick[phrag_state == 1L] <- 0L
    rcs_first_tick[rcs_state == 1L] <- 0L
  }
  if (isTRUE(store_probabilities)) {
    p_phrag_local_hist <- matrix(NA_real_, N, T_steps)
    p_rcs_local_hist <- matrix(NA_real_, N, T_steps)
    p_jump_hist <- matrix(NA_real_, N, T_steps)
    p_phrag_jump_hist <- matrix(NA_real_, N, T_steps)
  }
  if (isTRUE(store_prob_summary)) prob_summary <- vector("list", T_steps)
  
  phrag_total <- rcs_total <- jump_total <- phrag_jump_total <- storm_count <- integer(T_steps + 1L)
  # Event counts are separate from stochastic spread and are indexed by tick+1.
  rcs_intro_count <- phrag_intro_count <- integer(T_steps + 1L)
  phrag_total[1L] <- sum(phrag_state); rcs_total[1L] <- sum(rcs_state)
  
  for (t in seq_len(T_steps)) {
    storms_this_tick <- resolve_storms_this_tick(STORMS_PER_TICK, t)
    
    # Both storm pathways calculate source pressure from the PRE-storm states.
    # In particular, a new PHRAG jump cannot become a source in this same tick.
    jump_upd <- update_joint_jump(phrag_state, rcs_state, covs$valid_jump_source, sparse_ops,
                                  q_storm_base, jump_eligible_counts, storms_this_tick,
                                  valid_jump_target = valid_joint_habitat)
    phrag_jump_upd <- update_phrag_jump(phrag_state, valid_phrag_jump_source, sparse_ops,
                                        q_phrag_storm_base, phrag_jump_eligible_counts,
                                        storms_this_tick, valid_phrag_target = valid_phrag_habitat)
    # A cell reached by either mechanism gains PHRAG; only coupled jumps add RCS.
    phrag_after_storm <- pmax(jump_upd$phrag_next, phrag_jump_upd$phrag_next)
    rcs_after_storm <- jump_upd$rcs_next
    rcs_after_storm[phrag_after_storm == 0L] <- 0L
    
    phrag_upd <- update_local_phrag_sparse(phrag_after_storm, covs$M_phrag, sparse_ops$A_adj_mean,
                                           PHRAG_INTERCEPT, PHRAG_MAXENT_COEF, PHRAG_SOURCE_COEF,
                                           eligible = valid_phrag_habitat)
    rcs_upd <- update_local_rcs_sparse(rcs_after_storm, phrag_upd$next_state, covs$M_rcs,
                                       sparse_ops$A_adj_mean, RCS_INTERCEPT, RCS_MAXENT_COEF,
                                       RCS_SOURCE_COEF, eligible = valid_rcs_habitat)
    
    phrag_state <- phrag_upd$next_state
    rcs_state <- rcs_upd$next_state
    rcs_state[phrag_state == 0L] <- 0L
    
    # Introduce documented RCS only AFTER this year's spread, and BEFORE its
    # state is recorded. Thus tick 17 is the 2017 introduced state, but newly
    # introduced RCS cannot disperse until tick 18. Supply PHRAG host support
    # to any introduction cell where local PHRAG spread has not yet reached.
    if (rcs_intro_tick > 0L && t == rcs_intro_tick && length(intro_idx)) {
      rcs_intro_count[t + 1L] <- sum(rcs_state[intro_idx] == 0L)
      phrag_intro_count[t + 1L] <- sum(phrag_state[intro_idx] == 0L)
      phrag_state[intro_idx] <- 1L
      rcs_state[intro_idx] <- 1L
      if (isTRUE(verbose))
        cat("Tick ", t, ": introduced RCS at ", rcs_intro_count[t + 1L],
            " cells; added PHRAG support at ", phrag_intro_count[t + 1L],
            " cells.\n", sep = "")
    }
    
    if (history_storage == "full") {
      phrag_hist[, t + 1L] <- phrag_state
      rcs_hist[, t + 1L] <- rcs_state
    } else if (history_storage == "arrival") {
      # Record the first tick only; do not allocate a cell-by-tick matrix.
      phrag_first_tick[phrag_state == 1L & phrag_first_tick < 0L] <- as.integer(t)
      rcs_first_tick[rcs_state == 1L & rcs_first_tick < 0L] <- as.integer(t)
    }
    if (isTRUE(store_probabilities)) {
      p_jump_hist[, t] <- jump_upd$probability
      p_phrag_jump_hist[, t] <- phrag_jump_upd$probability
      p_phrag_local_hist[, t] <- phrag_upd$probability
      p_rcs_local_hist[, t] <- rcs_upd$probability
    }
    if (isTRUE(store_prob_summary))
      prob_summary[[t]] <- .make_prob_summary_row(t, storms_this_tick, jump_upd,
                                                  phrag_jump_upd, phrag_upd, rcs_upd)
    
    phrag_total[t + 1L] <- sum(phrag_state)
    rcs_total[t + 1L] <- sum(rcs_state)
    jump_total[t + 1L] <- sum(jump_upd$jump_event)
    # Count PHRAG-only colonizations, excluding cells also reached by an RCS jump.
    phrag_jump_total[t + 1L] <- sum(phrag_jump_upd$jump_event & !jump_upd$jump_event)
    storm_count[t + 1L] <- as.integer(storms_this_tick)
    
    if (isTRUE(verbose) && (t %% 5L == 0L || t == T_steps)) {
      cat("   → Tick ", t, ": storms=", storm_count[t + 1L],
          " | PHRAG=", phrag_total[t + 1L], " | RCS=", rcs_total[t + 1L],
          " | RCS jumps=", jump_total[t + 1L],
          " | PHRAG-only jumps=", phrag_jump_total[t + 1L], "\n", sep = "")
      if (isTRUE(store_prob_summary)) {
        ps <- prob_summary[[t]]
        cat("      eligible probs | jump median=", sprintf("%.3f", ps$jump_p_median),
            " | phrag median=", sprintf("%.3f", ps$phrag_p_median),
            " | rcs median=", sprintf("%.3f", ps$rcs_p_median), "\n", sep = "")
      }
    }
    
    rm(jump_upd, phrag_jump_upd, phrag_upd, rcs_upd, phrag_after_storm, rcs_after_storm)
  }
  
  prob_summary_df <- if (isTRUE(store_prob_summary) && length(prob_summary))
    dplyr::bind_rows(prob_summary) else NULL
  
  list(
    order_indexed_cells = dt[, .(cell_id, row, col)],
    active_rcs_col = covs$active_rcs_col,
    M_phrag = covs$M_phrag, M_rcs = covs$M_rcs, N_jump = covs$N_jump,
    valid_jump_source = covs$valid_jump_source, sparse_ops = sparse_ops,
    phrag_history = phrag_hist, rcs_history = rcs_hist,
    phrag_first_tick = phrag_first_tick, rcs_first_tick = rcs_first_tick,
    history_storage = history_storage,
    p_phrag_local_hist = p_phrag_local_hist, p_rcs_local_hist = p_rcs_local_hist,
    p_jump_hist = p_jump_hist, p_phrag_jump_hist = p_phrag_jump_hist,
    prob_summary = prob_summary_df,
    stored_prob_summary = isTRUE(store_prob_summary),
    phrag_total = phrag_total, rcs_total = rcs_total, jump_total = jump_total,
    phrag_jump_total = phrag_jump_total, storm_count = storm_count,
    rcs_intro_count = rcs_intro_count, phrag_intro_count = phrag_intro_count,
    rcs_intro_tick = rcs_intro_tick, rcs_intro_seed_count = length(intro_idx),
    nrow_grid = as.integer(nrow_grid), ncol_grid = as.integer(ncol_grid),
    neighborhood = as.integer(neighborhood), nei_radius = as.integer(nei_radius),
    dist_metric = as.character(dist_metric), jump_source_method = as.character(sparse_ops$jump_source_method),
    coefficients = list(
      phrag_intercept = PHRAG_INTERCEPT, phrag_maxent_coef = PHRAG_MAXENT_COEF,
      phrag_source_coef = PHRAG_SOURCE_COEF, rcs_intercept = RCS_INTERCEPT,
      rcs_maxent_coef = RCS_MAXENT_COEF, rcs_source_coef = RCS_SOURCE_COEF,
      storm_intercept = STORM_INTERCEPT, storm_nri_coef = STORM_NRI_COEF,
      storms_per_tick = STORMS_PER_TICK
    )
  )
}
