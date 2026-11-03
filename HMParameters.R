################################################################################
# Latin Hypercube Sampling (LHS) parameter table for CURRENT TWO-LAYER RCS CA
################################################################################

# ---- Packages ----
if (!requireNamespace("lhs", quietly = TRUE)) {
  install.packages("lhs")
}

# ---- User config: edit here only ----
config <- list(
  n = 999,
  
  ranges = list(
    PHRAG_INTERCEPT   = c(-5.75, -3.25),
    PHRAG_MAXENT_COEF = c( 4.50, 10.00),
    PHRAG_SOURCE_COEF = c( 0.25,  3.25),
    RCS_INTERCEPT     = c(-6.25, -3.00),
    RCS_MAXENT_COEF   = c( 5.00, 10.50),
    RCS_SOURCE_COEF   = c( 0.75,  4.25),
     STORM_INTERCEPT   = c( 0.50,  4.50),
    STORM_NRI_COEF    = c( 0.00,  2.50),
    NEI_RADIUS        = c(20L, 32L)
  ),
  
  # Use 0.25 because the updated ranges use quarter-step bounds.
  step = 0.25,
  
  # Set below after constraint_reasonable_logits() is defined.
  constraint_fun = NULL,
  
  oversample = 25,
  max_iter = 300L,
  seed = 1,
  
  out_file = "param_space_lhs_current_ca.csv"
)

# ---- Project output folder helper ----
get_start_dir <- function() {
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getActiveDocumentContext()$path, error = function(e) "")
    if (nzchar(p)) return(dirname(normalizePath(p, winslash = "/", mustWork = TRUE)))
  }
  p <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(p) == 1 && nzchar(p)) return(dirname(normalizePath(p, winslash = "/", mustWork = TRUE)))
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

find_project_root <- function(start, req = c("Functions", "Data"), max_up = 25L) {
  p <- normalizePath(start, winslash = "/", mustWork = TRUE)
  for (i in 0:max_up) {
    if (all(dir.exists(file.path(p, req)))) return(p)
    parent <- normalizePath(file.path(p, ".."), winslash = "/", mustWork = TRUE)
    if (identical(parent, p)) break
    p <- parent
  }
  NULL
}

start_dir <- get_start_dir()
root_dir  <- find_project_root(start_dir)
out_dir   <- if (!is.null(root_dir)) file.path(root_dir, "Outputs") else file.path(getwd(), "Outputs")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Mapping helpers ----
map_to_range <- function(u, lo, hi) {
  lo + u * (hi - lo)
}

map_to_integer_range <- function(u, lo_int, hi_int) {
  lo_int <- as.integer(lo_int)
  hi_int <- as.integer(hi_int)
  x <- lo_int + floor(u * (hi_int - lo_int + 1L))
  x <- pmin(pmax(x, lo_int), hi_int)
  as.integer(x)
}

quantize_step <- function(x, step = 0.25, lo = -Inf, hi = Inf) {
  stopifnot(is.numeric(step), length(step) == 1, step > 0)
  q <- round(x / step) * step
  pmin(pmax(q, lo), hi)
}

# ---- Optional constraint helpers ----
constraint_source_ge_maxent <- function(df) {
  (df$PHRAG_SOURCE_COEF >= df$PHRAG_MAXENT_COEF) &
    (df$RCS_SOURCE_COEF >= df$RCS_MAXENT_COEF)
}

constraint_intercepts_negative <- function(df) {
  (df$PHRAG_INTERCEPT < 0) &
    (df$RCS_INTERCEPT < 0) &
    (df$STORM_INTERCEPT < 0)
}

constraint_reasonable_logits <- function(df) {
  phrag_eta_max <- df$PHRAG_INTERCEPT +
    df$PHRAG_MAXENT_COEF +
    df$PHRAG_SOURCE_COEF
  
  rcs_eta_max <- df$RCS_INTERCEPT +
    df$RCS_MAXENT_COEF +
    df$RCS_SOURCE_COEF
  
  phrag_eta_mid <- df$PHRAG_INTERCEPT +
    0.5 * df$PHRAG_MAXENT_COEF +
    0.25 * df$PHRAG_SOURCE_COEF
  
  rcs_eta_mid <- df$RCS_INTERCEPT +
    0.5 * df$RCS_MAXENT_COEF +
    0.25 * df$RCS_SOURCE_COEF
  
  # Avoid parameter combinations that are likely saturated-high
  # or functionally dead before running full simulations.
  (phrag_eta_max <= 3.5) &
    (rcs_eta_max <= 4.75) &
    (phrag_eta_mid >= -5.5) &
    (rcs_eta_mid >= -5.75)
}

combine_constraints <- function(...) {
  funs <- list(...)
  function(df) {
    keep <- rep(TRUE, nrow(df))
    for (f in funs) keep <- keep & f(df)
    keep
  }
}

# Use the reasonable-logit screen for this pass.
config$constraint_fun <- constraint_reasonable_logits

# ---- Validation ----
validate_config <- function(config) {
  required_names <- c(
    "PHRAG_INTERCEPT", "PHRAG_MAXENT_COEF", "PHRAG_SOURCE_COEF",
    "RCS_INTERCEPT", "RCS_MAXENT_COEF", "RCS_SOURCE_COEF",
    "STORM_INTERCEPT", "STORM_NRI_COEF", "NEI_RADIUS"
  )
  
  stopifnot(is.list(config))
  stopifnot(all(c("n", "ranges", "step", "oversample", "max_iter") %in% names(config)))
  stopifnot(all(required_names %in% names(config$ranges)))
  
  for (nm in required_names) {
    stopifnot(length(config$ranges[[nm]]) == 2)
  }
  
  stopifnot(is.numeric(config$n), length(config$n) == 1, config$n >= 1)
  stopifnot(is.numeric(config$step), length(config$step) == 1, config$step > 0)
  stopifnot(is.numeric(config$oversample), length(config$oversample) == 1, config$oversample >= 1)
  stopifnot(is.numeric(config$max_iter), length(config$max_iter) == 1, config$max_iter >= 1)
  
  invisible(TRUE)
}

# ---- Main LHS builder ----
make_param_table_lhs_current_ca <- function(config) {
  validate_config(config)
  
  if (!is.null(config$seed)) set.seed(config$seed)
  
  r <- config$ranges
  n <- as.integer(config$n)
  
  collect <- list()
  collected <- 0L
  
  for (iter in seq_len(config$max_iter)) {
    m <- max(ceiling((n - collected) * config$oversample), 100L)
    
    U <- lhs::randomLHS(m, 9)
    
    df <- data.frame(
      PHRAG_INTERCEPT = quantize_step(
        map_to_range(U[, 1], r$PHRAG_INTERCEPT[1], r$PHRAG_INTERCEPT[2]),
        step = config$step, lo = r$PHRAG_INTERCEPT[1], hi = r$PHRAG_INTERCEPT[2]
      ),
      PHRAG_MAXENT_COEF = quantize_step(
        map_to_range(U[, 2], r$PHRAG_MAXENT_COEF[1], r$PHRAG_MAXENT_COEF[2]),
        step = config$step, lo = r$PHRAG_MAXENT_COEF[1], hi = r$PHRAG_MAXENT_COEF[2]
      ),
      PHRAG_SOURCE_COEF = quantize_step(
        map_to_range(U[, 3], r$PHRAG_SOURCE_COEF[1], r$PHRAG_SOURCE_COEF[2]),
        step = config$step, lo = r$PHRAG_SOURCE_COEF[1], hi = r$PHRAG_SOURCE_COEF[2]
      ),
      
      RCS_INTERCEPT = quantize_step(
        map_to_range(U[, 4], r$RCS_INTERCEPT[1], r$RCS_INTERCEPT[2]),
        step = config$step, lo = r$RCS_INTERCEPT[1], hi = r$RCS_INTERCEPT[2]
      ),
      RCS_MAXENT_COEF = quantize_step(
        map_to_range(U[, 5], r$RCS_MAXENT_COEF[1], r$RCS_MAXENT_COEF[2]),
        step = config$step, lo = r$RCS_MAXENT_COEF[1], hi = r$RCS_MAXENT_COEF[2]
      ),
      RCS_SOURCE_COEF = quantize_step(
        map_to_range(U[, 6], r$RCS_SOURCE_COEF[1], r$RCS_SOURCE_COEF[2]),
        step = config$step, lo = r$RCS_SOURCE_COEF[1], hi = r$RCS_SOURCE_COEF[2]
      ),
      
      STORM_INTERCEPT = quantize_step(
        map_to_range(U[, 7], r$STORM_INTERCEPT[1], r$STORM_INTERCEPT[2]),
        step = config$step, lo = r$STORM_INTERCEPT[1], hi = r$STORM_INTERCEPT[2]
      ),
      STORM_NRI_COEF = quantize_step(
        map_to_range(U[, 8], r$STORM_NRI_COEF[1], r$STORM_NRI_COEF[2]),
        step = config$step, lo = r$STORM_NRI_COEF[1], hi = r$STORM_NRI_COEF[2]
      ),
      
      NEI_RADIUS = map_to_integer_range(
        U[, 9], r$NEI_RADIUS[1], r$NEI_RADIUS[2]
      )
    )
    
    if (!is.null(config$constraint_fun)) {
      keep <- config$constraint_fun(df)
      if (!is.logical(keep) || length(keep) != nrow(df)) {
        stop("constraint_fun must return logical vector length nrow(df).")
      }
      df <- df[keep, , drop = FALSE]
    }
    
    if (nrow(df) == 0) next
    
    collect[[length(collect) + 1L]] <- df
    all_df <- unique(do.call(rbind, collect))
    
    if (nrow(all_df) >= n) {
      all_df <- all_df[seq_len(n), , drop = FALSE]
      all_df$Run <- seq_len(nrow(all_df))
      rownames(all_df) <- NULL
      
      return(all_df[, c(
        "Run",
        "PHRAG_INTERCEPT", "PHRAG_MAXENT_COEF", "PHRAG_SOURCE_COEF",
        "RCS_INTERCEPT", "RCS_MAXENT_COEF", "RCS_SOURCE_COEF",
        "STORM_INTERCEPT", "STORM_NRI_COEF",
        "NEI_RADIUS"
      )])
    }
    
    collected <- nrow(all_df)
    collect <- list(all_df)
  }
  
  stop(
    "Unable to generate ", n, " unique parameter sets within max_iter. ",
    "Try increasing max_iter, oversample, widening ranges, or relaxing constraints."
  )
}

# ---- Build table ----
df_params <- make_param_table_lhs_current_ca(config)

print(head(df_params, 10))
cat("Total runs:", nrow(df_params), "\n")

# ---- Quick range check ----
cat("\nParameter ranges in generated table:\n")
print(data.frame(
  Parameter = names(df_params)[names(df_params) != "Run"],
  Min = vapply(df_params[names(df_params) != "Run"], min, numeric(1), na.rm = TRUE),
  Max = vapply(df_params[names(df_params) != "Run"], max, numeric(1), na.rm = TRUE),
  row.names = NULL
))

# ---- Write CSV ----
out_file <- file.path(out_dir, config$out_file)
write.csv(df_params, out_file, row.names = FALSE)
cat("\nWrote:", out_file, "\n")