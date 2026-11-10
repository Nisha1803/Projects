####################################################################################################
######################################## INITIALIZATION ############################################
################################ RCS CA SIMULATION PIPELINE ########################################
####################################################################################################
# This module performs:
#   1. Package validation and loading
#   2. Optional animation stack detection
#   3. Output tag construction (batch + per-run safe)
#   4. Optional RNG seeding (single-run only)
#   5. Environmental grid file discovery
#   6. Environmental grid loading + validation
#   7. Active RCS model column validation
#   8. Optional rectangular environmental-grid clipping
#   9. Grid index inference for Cellular Automata
#
# Batch/offset-mode note:
#   - Matrix is required because CellularAutomata.R still uses Matrix::sparseMatrix()
#     for the local adjacency operator, even when JUMP_SOURCE_METHOD = "offset".
####################################################################################################


####################################################################################################
#################################### PACKAGE MANAGEMENT ############################################
####################################################################################################

ensure_pkgs <- function(pkgs) {
  missing_pkgs <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs)) {
    stop(
      "Missing required packages:\n   - ",
      paste(missing_pkgs, collapse = "\n   - "),
      "\nInstall these before running the simulation."
    )
  }
  invisible(TRUE)
}

core_pkgs <- c(
  "dplyr",
  "readr",
  "tidyr",
  "purrr",
  "stringr",
  "data.table",
  "ggplot2",
  "Matrix",
  "sf"  # Coordinate reprojection in PreProcessing.R
)

if (exists("RUN_PARALLEL", inherits = TRUE) && isTRUE(RUN_PARALLEL)) {
  core_pkgs <- unique(c(core_pkgs, "future", "future.apply"))
}

ensure_pkgs(core_pkgs)

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(data.table)
  library(ggplot2)
  library(Matrix)
  
  if (exists("RUN_PARALLEL", inherits = TRUE) && isTRUE(RUN_PARALLEL)) {
    library(future)
    library(future.apply)
  }
})

cat("Core libraries loaded.\n")


####################################################################################################
#################################### OPTIONAL ANIMATION ############################################
####################################################################################################

ANIM_OK <- FALSE

if (requireNamespace("gganimate", quietly = TRUE)) {
  suppressPackageStartupMessages(library(gganimate))
  ANIM_OK <- TRUE
  cat("Animation support enabled (gganimate detected).\n")
} else {
  cat("gganimate not available → animation export will be skipped.\n")
}

cat("ANIM_OK =", ANIM_OK, "\n\n")


####################################################################################################
######################################## OUTPUT TAGGING ############################################
####################################################################################################

clean_tag <- function(x) {
  x <- trimws(as.character(x))
  x <- gsub("[^A-Za-z0-9._-]", "_", x)
  x <- gsub("_+", "_", x)
  gsub("^_|_$", "", x)
}

BASE_MODEL <- if (exists("BASE_MODEL", inherits = TRUE)) {
  as.character(BASE_MODEL)
} else {
  "RCS_COLD_NATIVE"
}

OUTPUT_LABEL <- if (exists("OUTPUT_LABEL", inherits = TRUE)) clean_tag(OUTPUT_LABEL) else ""
MODEL_TAG    <- clean_tag(BASE_MODEL)

BASE_TAG <- paste(
  c(if (nzchar(OUTPUT_LABEL)) OUTPUT_LABEL else NULL, MODEL_TAG),
  collapse = "_"
)

cat("Base tag:", BASE_TAG, "\n")

make_run_id  <- function(i) sprintf("Run_%04d", as.integer(i))
make_run_tag <- function(i) paste0(BASE_TAG, "_", make_run_id(i))
make_run_dir <- function(i) file.path(DIRS$outputs_dir, make_run_tag(i))

make_outfile_run <- function(i, suffix, ext) {
  suffix <- clean_tag(suffix)
  stopifnot(nzchar(suffix), nzchar(ext))
  
  out_dir <- make_run_dir(i)
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  file.path(out_dir, paste0(make_run_tag(i), "_", suffix, ".", ext))
}

make_outfile_batch <- function(suffix, ext) {
  suffix <- clean_tag(suffix)
  stopifnot(nzchar(suffix), nzchar(ext))
  file.path(DIRS$outputs_dir, paste0(BASE_TAG, "_", suffix, ".", ext))
}

cat("Outputs root:", DIRS$outputs_dir, "\n\n")


####################################################################################################
######################################## RNG CONTROL ################################################
####################################################################################################

if (isTRUE(USE_FIXED_SEED)) {
  if (!exists("N_RUNS", inherits = TRUE) || as.integer(N_RUNS) <= 1L) {
    set.seed(RNG_SEED)
    cat("Global RNG seeded | RNG_SEED =", RNG_SEED, "\n\n")
  } else {
    cat("Multi-run mode detected → per-run seeding handled inside run_one_sim().\n\n")
  }
}


####################################################################################################
############################## ENVIRONMENTAL GRID DISCOVERY #########################################
####################################################################################################

DATA_FILE <- if (exists("DATA_FILE", inherits = TRUE)) as.character(DATA_FILE) else ""

all_csvs <- list.files(DIRS$data_dir, pattern = "\\.csv$", full.names = TRUE)

if (!length(all_csvs)) {
  stop("No CSV files found in DIRS$data_dir: ", DIRS$data_dir)
}

if (length(DATA_FILE) != 1L || !nzchar(DATA_FILE)) {
  stop("DATA_FILE is empty. Set DATA_FILE in CONFIG to the environmental grid CSV filename.")
}

rcs_file <- file.path(DIRS$data_dir, DATA_FILE)

if (!file.exists(rcs_file)) {
  stop(
    "Environmental DATA_FILE not found in DIRS$data_dir.\n",
    "   Requested: ", DATA_FILE, "\n",
    "   Directory: ", DIRS$data_dir, "\n",
    "   Available CSV files:\n   - ",
    paste(basename(all_csvs), collapse = "\n   - ")
  )
}

cat("Data directory scanned.\n")
cat("Selected environmental grid from CONFIG:\n   → ", basename(rcs_file), "\n\n", sep = "")


####################################################################################################
#################################### LOAD ENVIRONMENTAL GRID ########################################
####################################################################################################

# Historical survey/presence columns in DATA_FILE are not used for seeding;
# PreProcessing.R rebuilds initial occupancy from OCCURRENCE_FILE.
rcs_data <- read.csv(rcs_file, stringsAsFactors = FALSE)

cat("Loaded environmental grid:", basename(rcs_file),
    " | rows =", nrow(rcs_data),
    " | cols =", ncol(rcs_data), "\n\n")

required_rcs_cols <- c("GridSquare", "Xmin", "Ymin", "MaxEnt_Phragmites", "NRI_Score")

if (!all(required_rcs_cols %in% names(rcs_data))) {
  stop("rcs_data must contain: ", paste(required_rcs_cols, collapse = ", "))
}

if (!"cell_id" %in% names(rcs_data)) {
  rcs_data$cell_id <- rcs_data$GridSquare
}


####################################################################################################
################################## OPTIONAL GEOGRAPHIC CLIPPING ####################################
####################################################################################################

# CLIP_BOUNDS uses projected meters (GRID_EPSG = 32615).
# Retain cells whose centers fall inside the requested rectangle.
# NULL leaves the loaded environmental grid unchanged.
if (exists("CLIP_BOUNDS", inherits = TRUE) && !is.null(CLIP_BOUNDS)) {
  bounds <- CLIP_BOUNDS
  
  if (!is.numeric(bounds) || length(bounds) != 4L ||
      !identical(names(bounds), c("xmin", "xmax", "ymin", "ymax")) ||
      any(!is.finite(bounds)) ||
      bounds[["xmin"]] >= bounds[["xmax"]] ||
      bounds[["ymin"]] >= bounds[["ymax"]]) {
    stop("Invalid CLIP_BOUNDS. Use c(xmin=..., xmax=..., ymin=..., ymax=...).")
  }
  
  resolution <- as.numeric(GRID_RESOLUTION_M)
  if (length(resolution) != 1L || !is.finite(resolution) || resolution <= 0)
    stop("GRID_RESOLUTION_M must be a positive number for clipping.")
  
  x_min <- suppressWarnings(as.numeric(rcs_data$Xmin))
  y_min <- suppressWarnings(as.numeric(rcs_data$Ymin))
  if (any(!is.finite(x_min)) || any(!is.finite(y_min)))
    stop("Cannot clip: Xmin/Ymin must be finite numeric coordinates.")
  
  x_center <- x_min + resolution / 2
  y_center <- y_min + resolution / 2
  keep <- x_center >= bounds[["xmin"]] & x_center <= bounds[["xmax"]] &
    y_center >= bounds[["ymin"]] & y_center <= bounds[["ymax"]]
  
  if (!any(keep))
    stop("CLIP_BOUNDS does not contain any environmental grid-cell centers.")
  
  original_cells <- nrow(rcs_data)
  rcs_data <- rcs_data[keep, , drop = FALSE]
  
  # The CA requires a complete rectangular lattice, not an irregular mask.
  if (length(unique(rcs_data$Xmin)) * length(unique(rcs_data$Ymin)) != nrow(rcs_data) ||
      anyDuplicated(rcs_data[, c("Xmin", "Ymin")]))
    stop("CLIP_BOUNDS produced a nonrectangular or duplicate-cell grid.")
  
  # Cell IDs remain those of DATA_FILE; local row/col must be rebuilt below.
  rcs_data$row <- NULL
  rcs_data$col <- NULL
  
  cat(" Applied CLIP_BOUNDS (EPSG:", GRID_EPSG, "):\n",
      "   X: [", bounds[["xmin"]], ", ", bounds[["xmax"]], "]\n",
      "   Y: [", bounds[["ymin"]], ", ", bounds[["ymax"]], "]\n",
      "   Cells retained: ", nrow(rcs_data), " of ", original_cells, "\n\n", sep = "")
}


####################################################################################################
################################ ACTIVE RCS MODEL VALIDATION #######################################
####################################################################################################

resolve_active_rcs_column_init <- function(df, base_model) {
  base_model_std <- gsub("\\s+", "_", as.character(base_model))
  
  alias_map <- c(
    "NIPPON"            = "MaxEnt_Nipponaclerda",
    "NIPPONACLERDA"     = "MaxEnt_Nipponaclerda",
    "COLD"              = "MaxEnt_RCS_cold_native",
    "RCS_COLD_NATIVE"   = "MaxEnt_RCS_cold_native",
    "COLD_NATIVE"       = "MaxEnt_RCS_cold_native",
    "NATIVE"            = "MaxEnt_RCS_native",
    "RCS_NATIVE"        = "MaxEnt_RCS_native"
  )
  
  candidates <- unique(c(
    unname(alias_map[toupper(base_model_std)]),
    base_model_std,
    paste0("MaxEnt_", base_model_std),
    gsub("\\s+", "_", base_model_std)
  ))
  
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  hit <- candidates[candidates %in% names(df)][1]
  
  if (!length(hit) || is.na(hit)) {
    stop(
      "Could not resolve active RCS MaxEnt column from BASE_MODEL = '", base_model, "'.\n",
      "   Looked for: ", paste(unique(candidates), collapse = ", "), "\n",
      "   Available columns include:\n   - ",
      paste(names(df), collapse = "\n   - ")
    )
  }
  
  hit
}

ACTIVE_RCS_MAXENT_COL <- resolve_active_rcs_column_init(rcs_data, BASE_MODEL)

cat("Active RCS MaxEnt column resolved:\n",
    "   → BASE_MODEL = ", BASE_MODEL, "\n",
    "   → column    = ", ACTIVE_RCS_MAXENT_COL, "\n\n", sep = "")

for (nm in c("MaxEnt_Phragmites", "NRI_Score", ACTIVE_RCS_MAXENT_COL)) {
  suppressWarnings({
    x_num <- as.numeric(rcs_data[[nm]])
  })
  if (all(is.na(x_num))) {
    warning("Column '", nm, "' is entirely NA/non-numeric after coercion.")
  }
}


####################################################################################################
############################# GRID INFERENCE / GEOMETRY CHECK #####################################
####################################################################################################

build_row_col_from_xy <- function(df, x_col = "Xmin", y_col = "Ymin") {
  if (!all(c(x_col, y_col) %in% names(df))) {
    stop("Cannot build row/col: missing ", x_col, " and/or ", y_col)
  }
  
  x_num <- suppressWarnings(as.numeric(df[[x_col]]))
  y_num <- suppressWarnings(as.numeric(df[[y_col]]))
  
  if (any(is.na(x_num)) || any(is.na(y_num))) {
    stop("Cannot build row/col: Xmin/Ymin contain NA or non-numeric values.")
  }
  
  # West -> East for columns
  x_vals <- sort(unique(x_num), decreasing = FALSE)
  
  # North -> South for rows
  # For North America / north-up grids, larger Y is farther north and should be row 1.
  y_vals <- sort(unique(y_num), decreasing = TRUE)
  
  df$col <- match(x_num, x_vals)
  df$row <- match(y_num, y_vals)
  
  attr(df, "x_vals") <- x_vals
  attr(df, "y_vals") <- y_vals
  df
}

validate_existing_row_col <- function(df) {
  if (!all(c("row", "col") %in% names(df))) return(FALSE)
  
  row_num <- suppressWarnings(as.integer(df$row))
  col_num <- suppressWarnings(as.integer(df$col))
  
  if (any(is.na(row_num)) || any(is.na(col_num))) {
    warning("Existing row/col found, but contain NA/non-integer values. Rebuilding from Xmin/Ymin.")
    return(FALSE)
  }
  
  # Rebuild expected row/col from Xmin/Ymin and compare
  rebuilt <- build_row_col_from_xy(df)
  same_row <- identical(row_num, as.integer(rebuilt$row))
  same_col <- identical(col_num, as.integer(rebuilt$col))
  
  if (!same_row || !same_col) {
    warning(
      "Existing row/col do not match Xmin/Ymin-derived north-up orientation.\n",
      "   Rebuilding row/col from Xmin/Ymin to avoid grid misalignment."
    )
    return(FALSE)
  }
  
  TRUE
}

if (validate_existing_row_col(rcs_data)) {
  rcs_data$row <- as.integer(rcs_data$row)
  rcs_data$col <- as.integer(rcs_data$col)
  cat("Using existing row/col from loaded file.\n")
  
  x_vals <- sort(unique(suppressWarnings(as.numeric(rcs_data$Xmin))), decreasing = FALSE)
  y_vals <- sort(unique(suppressWarnings(as.numeric(rcs_data$Ymin))), decreasing = TRUE)
  
} else {
  rcs_data <- build_row_col_from_xy(rcs_data, x_col = "Xmin", y_col = "Ymin")
  x_vals <- attr(rcs_data, "x_vals")
  y_vals <- attr(rcs_data, "y_vals")
  cat("row/col not found or failed validation → rebuilt from Xmin/Ymin.\n")
}

nrow_grid <- length(y_vals)
ncol_grid <- length(x_vals)

# Confirm the environmental grid resolution matches the occurrence-mapping settings.
if (!exists("GRID_EPSG", inherits = TRUE) || !identical(as.integer(GRID_EPSG), 32615L))
  stop("Environmental grid and occurrence preprocessing require GRID_EPSG = 32615.")
if (!exists("GRID_RESOLUTION_M", inherits = TRUE) ||
    length(GRID_RESOLUTION_M) != 1L || !is.finite(GRID_RESOLUTION_M) ||
    GRID_RESOLUTION_M <= 0)
  stop("GRID_RESOLUTION_M must be a positive finite number.")

resolution <- as.numeric(GRID_RESOLUTION_M)
if ((length(x_vals) > 1L && any(abs(diff(x_vals) - resolution) > 1e-3)) ||
    (length(y_vals) > 1L && any(abs(diff(y_vals) + resolution) > 1e-3)))
  stop("DATA_FILE grid spacing disagrees with GRID_RESOLUTION_M = ", resolution, ".")

cat("Coordinate ranges:\n",
    "   Xmin: [", min(as.numeric(rcs_data$Xmin), na.rm = TRUE), ", ",
    max(as.numeric(rcs_data$Xmin), na.rm = TRUE), "]\n",
    "   Ymin: [", min(as.numeric(rcs_data$Ymin), na.rm = TRUE), ", ",
    max(as.numeric(rcs_data$Ymin), na.rm = TRUE), "]\n",
    "   Orientation: row 1 = northmost cells, col 1 = westmost cells\n\n", sep = "")

cat("Grid structure:\n",
    "   nrow_grid =", nrow_grid,
    " | ncol_grid =", ncol_grid, "\n",
    "   expected cells =", nrow_grid * ncol_grid,
    " | records =", nrow(rcs_data), "\n\n", sep = "")

if (nrow_grid * ncol_grid != nrow(rcs_data)) {
  warning(
    "Grid mismatch detected: expected full rectangular grid.\n",
    "   Cellular Automata requires exactly one record per cell."
  )
}

dup_cells <- duplicated(rcs_data[, c("row", "col")])
if (any(dup_cells)) {
  warning(
    "Duplicate row/col combinations detected in rcs_data.\n",
    "   Cellular Automata expects a single record per grid cell."
  )
}

missing_cell_ids <- any(is.na(rcs_data$cell_id))
if (missing_cell_ids) {
  warning("Missing cell_id values detected; downstream matching may fail.")
}

cat("Environmental grid initialization complete; ready for occurrence preprocessing.\n\n")