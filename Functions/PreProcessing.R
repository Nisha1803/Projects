####################################################################################################
######################################## PREPROCESSING #############################################
####################################################################################################

# Prepare rcs_data for the CA:
#   - load required Species, X, Y, Year occurrence data
#   - assign observations to the existing environmental grid (1 km, EPSG:32615)
#   - derive species first-observed years and initial occupancy
#   - validate required environmental inputs / resolve active RCS MaxEnt
#   - coerce CA inputs to numeric and create diagnostics / audit outputs

####################################################################################################
########################################### HELPERS ################################################
####################################################################################################

# Retained for compatibility with CellularAutomata.R.
scale_minmax <- function(v) {
  if (all(is.na(v))) return(v)
  
  mn <- suppressWarnings(min(v, na.rm = TRUE))
  mx <- suppressWarnings(max(v, na.rm = TRUE))
  
  if (is.finite(mn) && is.finite(mx) && mx > mn)
    return((v - mn) / (mx - mn))
  
  ifelse(!is.na(v), 1, NA_real_)
}

resolve_active_rcs_column <- function(df, base_model) {
  model <- toupper(gsub("\\s+", "_", as.character(base_model)))
  
  alias_map <- c(
    NIPPON = "MaxEnt_Nipponaclerda",
    NIPPONACLERDA = "MaxEnt_Nipponaclerda",
    COLD = "MaxEnt_RCS_cold_native",
    RCS_COLD_NATIVE = "MaxEnt_RCS_cold_native",
    COLD_NATIVE = "MaxEnt_RCS_cold_native",
    NATIVE = "MaxEnt_RCS_native",
    RCS_NATIVE = "MaxEnt_RCS_native"
  )
  
  candidates <- unique(c(
    unname(alias_map[model]),
    model,
    paste0("MaxEnt_", model)
  ))
  
  candidates <- candidates[!is.na(candidates) & nzchar(candidates)]
  hit <- candidates[candidates %in% names(df)][1]
  
  if (!length(hit) || is.na(hit)) {
    stop(
      "Could not resolve active RCS MaxEnt column from BASE_MODEL = '",
      base_model, "'.\nLooked for: ",
      paste(candidates, collapse = ", ")
    )
  }
  
  hit
}

summarize_numeric_col <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  
  c(
    n = length(x),
    n_na = sum(is.na(x)),
    min = suppressWarnings(min(x, na.rm = TRUE)),
    q25 = suppressWarnings(quantile(x, 0.25, na.rm = TRUE, names = FALSE)),
    median = suppressWarnings(median(x, na.rm = TRUE)),
    mean = suppressWarnings(mean(x, na.rm = TRUE)),
    q75 = suppressWarnings(quantile(x, 0.75, na.rm = TRUE, names = FALSE)),
    max = suppressWarnings(max(x, na.rm = TRUE))
  )
}

# Batch-output fallback.
if (!exists("make_outfile_batch", inherits = TRUE)) {
  make_outfile_batch <- function(suffix, ext) {
    suffix <- gsub("[^A-Za-z0-9._-]", "_", trimws(as.character(suffix)))
    ext <- gsub("[^A-Za-z0-9._-]", "_", trimws(as.character(ext)))
    
    out_dir <- if (exists("DIRS", inherits = TRUE) &&
                   is.list(DIRS) && !is.null(DIRS$outputs_dir)) {
      DIRS$outputs_dir
    } else {
      getwd()
    }
    
    file.path(out_dir, paste0("Output_", suffix, ".", ext))
  }
}

####################################################################################################
########################################### CONTROLS ###############################################
####################################################################################################

BASE_MODEL <- if (exists("BASE_MODEL", inherits = TRUE))
  as.character(BASE_MODEL) else "RCS_COLD_NATIVE"

REQUIRED_CA_COLS <- c("MaxEnt_Phragmites", "NRI_Score")

OPTIONAL_SEED_COLS <- c(
  "Initial_Infested", "InitialPhrag", "Initial_Phrag",
  "Initial_RCS", "RCS_IntroSeed", "SeedPhrag", "Seed_Phrag", "Phrag_Seed"
)

WRITE_PREPROCESS_CSVS <- FALSE

# Occurrence input is required for every simulation.

OCCURRENCE_FILE <- if (exists("OCCURRENCE_FILE", inherits = TRUE))
  as.character(OCCURRENCE_FILE) else ""

START_YEAR <- if (exists("START_YEAR", inherits = TRUE))
  as.integer(START_YEAR) else stop("START_YEAR must be defined in CONFIG.")

RCS_INTRO_YEAR <- if (exists("RCS_INTRO_YEAR", inherits = TRUE))
  as.integer(RCS_INTRO_YEAR) else stop("RCS_INTRO_YEAR must be defined in CONFIG.")

GRID_EPSG <- if (exists("GRID_EPSG", inherits = TRUE))
  as.integer(GRID_EPSG) else 32615L

GRID_RESOLUTION_M <- if (exists("GRID_RESOLUTION_M", inherits = TRUE))
  as.numeric(GRID_RESOLUTION_M) else 1000

# The model's environmental grid (with MaxEnt/NRI) is loaded by Initialize.R BEFORE this file.
# It must have one row per cell, with row, col, Xmin, Ymin and predictors.

####################################################################################################
########################### MAP OCCURRENCES TO THE ENVIRONMENTAL GRID ###############################
####################################################################################################

occurrence_diagnostics <- NULL

if (!requireNamespace("sf", quietly = TRUE))
  stop("Install the sf package to read geographic occurrence coordinates.")
if (!requireNamespace("data.table", quietly = TRUE))
  stop("Install the data.table package to process the occurrence file.")
if (length(OCCURRENCE_FILE) != 1L || !nzchar(OCCURRENCE_FILE) ||
    !file.exists(OCCURRENCE_FILE))
  stop("OCCURRENCE_FILE is missing or not found: ", OCCURRENCE_FILE)
if (!exists("rcs_data", inherits = TRUE) || !is.data.frame(rcs_data))
  stop("Initialize.R must load the environmental grid as rcs_data first.")
if (length(START_YEAR) != 1L || !is.finite(START_YEAR) ||
    length(RCS_INTRO_YEAR) != 1L || !is.finite(RCS_INTRO_YEAR) ||
    RCS_INTRO_YEAR < START_YEAR ||
    length(GRID_EPSG) != 1L || !is.finite(GRID_EPSG) ||
    length(GRID_RESOLUTION_M) != 1L || !is.finite(GRID_RESOLUTION_M) ||
    GRID_RESOLUTION_M <= 0)
  stop("START_YEAR, RCS_INTRO_YEAR, GRID_EPSG, or GRID_RESOLUTION_M is invalid.")

required_grid <- c("row", "col", "Xmin", "Ymin", "GridSquare")
missing_grid <- setdiff(required_grid, names(rcs_data))
if (length(missing_grid))
  stop("Environmental grid missing columns: ", paste(missing_grid, collapse = ", "))

gx <- suppressWarnings(as.numeric(rcs_data$Xmin))
gy <- suppressWarnings(as.numeric(rcs_data$Ymin))
gr <- suppressWarnings(as.integer(rcs_data$row))
gc <- suppressWarnings(as.integer(rcs_data$col))
resolution <- GRID_RESOLUTION_M
ngrid <- nrow(rcs_data)
if (any(!is.finite(gx) | !is.finite(gy)) ||
    anyNA(gr) || anyNA(gc) || any(gr < 1L | gc < 1L))
  stop("Environmental grid has invalid indices or coordinates.")
if (anyDuplicated(data.frame(row = gr, col = gc)))
  stop("Environmental grid contains duplicate row/col cells.")

# Derive geometry from the environmental grid, not from occurrence locations.
grid_west <- min(gx)
grid_east <- max(gx) + resolution
grid_south <- min(gy)
grid_north <- max(gy) + resolution
if (ngrid != as.double(max(gr)) * max(gc))
  stop("CA needs a complete rectangular grid (one record per row/col).")
expected_xmin <- grid_west + (gc - 1L) * resolution
expected_ymin <- grid_north - gr * resolution
if (any(abs(gx - expected_xmin) > 1e-3) ||
    any(abs(gy - expected_ymin) > 1e-3))
  stop("Grid row/col, Xmin/Ymin, or GRID_RESOLUTION_M are misaligned.")

obs <- data.table::fread(OCCURRENCE_FILE, na.strings = c("", "NA"),
                         showProgress = FALSE)
required_obs <- c("Species", "X", "Y", "Year")
missing_obs <- setdiff(required_obs, names(obs))
if (length(missing_obs))
  stop("Occurrence CSV missing columns: ", paste(missing_obs, collapse = ", "))
if (!nrow(obs)) stop("Occurrence CSV is empty.")

obs[, species := tolower(trimws(as.character(Species)))]
if (anyNA(obs$species) || any(!obs$species %in% c("rcs", "phragmites")))
  stop("Species must be either 'RCS' or 'Phragmites' in every row.")
obs[, `:=`(lon = suppressWarnings(as.numeric(X)),
           lat = suppressWarnings(as.numeric(Y)),
           obs_year = suppressWarnings(as.numeric(Year)))]
if (any(!is.na(obs$obs_year) &
        (!is.finite(obs$obs_year) | obs$obs_year != floor(obs$obs_year) |
         obs$obs_year < 1700 | obs$obs_year > as.integer(format(Sys.Date(), "%Y")))))
  stop("Occurrence Year must be an integer from 1700 through the current year, or blank.")
bad_xy <- !is.finite(obs$lon) | !is.finite(obs$lat) |
  abs(obs$lon) > 180 | abs(obs$lat) > 90
n_invalid <- sum(bad_xy)
if (n_invalid) warning("Dropping ", n_invalid, " rows with invalid geographic coordinates.")
obs <- obs[!bad_xy]
if (!nrow(obs)) stop("No valid geographic occurrences remain.")

pts <- sf::st_as_sf(as.data.frame(obs[, .(lon, lat)]),
                    coords = c("lon", "lat"), crs = 4326)
xy_utm <- sf::st_coordinates(sf::st_transform(pts, GRID_EPSG))
obs[, `:=`(utm_x = xy_utm[, 1L], utm_y = xy_utm[, 2L])]

# Consistent half-open cell assignment for the EPSG projected grid.
inside <- is.finite(obs$utm_x) & is.finite(obs$utm_y) &
  obs$utm_x >= grid_west & obs$utm_x < grid_east &
  obs$utm_y > grid_south & obs$utm_y <= grid_north
n_outside <- sum(!inside)
obs <- obs[inside]
if (!nrow(obs))
  stop("No occurrence locations fall inside the selected environmental grid.")
obs[, `:=`(
  col = as.integer(floor((utm_x - grid_west) / resolution)) + 1L,
  row = as.integer(floor((grid_north - utm_y) / resolution)) + 1L
)]

lookup <- data.table::data.table(row = gr, col = gc,
                                 grid_index = seq_len(ngrid))
data.table::setindexv(lookup, c("row", "col"))
obs[, grid_index := lookup[obs, on = .(row, col), grid_index]]
if (anyNA(obs$grid_index))
  stop("A projected occurrence did not match an existing environmental cell.")

# Keep undated host evidence separately; it must not be assigned an arbitrary year.
rcs_data$Phrag_UndatedPresence <- 0L
rcs_data$RCS_UndatedPresence <- 0L
rcs_data$Phrag_FirstObservedYear <- rep(NA_integer_, ngrid)
rcs_data$RCS_FirstDetectedYear <- rep(NA_integer_, ngrid)

phrag_undated <- unique(obs[species == "phragmites" & is.na(obs_year), grid_index])
rcs_undated <- unique(obs[species == "rcs" & is.na(obs_year), grid_index])
rcs_data$Phrag_UndatedPresence[union(phrag_undated, rcs_undated)] <- 1L
rcs_data$RCS_UndatedPresence[rcs_undated] <- 1L

dated <- obs[!is.na(obs_year)]
if (nrow(dated)) {
  first_by_cell <- dated[, .(FirstYear = as.integer(min(obs_year))),
                         by = .(species, grid_index)]
  rcs_first <- first_by_cell[species == "rcs"]
  phrag_first <- first_by_cell[species == "phragmites"]
  if (nrow(rcs_first))
    rcs_data$RCS_FirstDetectedYear[rcs_first$grid_index] <- rcs_first$FirstYear
  if (nrow(phrag_first))
    rcs_data$Phrag_FirstObservedYear[phrag_first$grid_index] <- phrag_first$FirstYear
}

# Presence of RCS is evidence of host presence by that same observation year.
# The year is FIRST DOCUMENTED PRESENCE, not the biological establishment year.
p <- rcs_data$Phrag_FirstObservedYear
r <- rcs_data$RCS_FirstDetectedYear
improve <- !is.na(r) & (is.na(p) | r < p)
p[improve] <- r[improve]
rcs_data$Phrag_FirstObservedYear <- p

rcs_data$Initial_Phrag <- as.integer(!is.na(p) & p <= START_YEAR)
rcs_data$Initial_RCS <- as.integer(!is.na(r) & r <= START_YEAR)
rcs_data$Initial_Phrag <- pmax(rcs_data$Initial_Phrag, rcs_data$Initial_RCS)

# Staged RCS introduction: documentary RCS presence by the introduction year.
# RunCode.R must activate these cells at RCS_INTRO_YEAR (not at START_YEAR).
rcs_data$RCS_IntroSeed <- as.integer(!is.na(r) & r <= RCS_INTRO_YEAR)

# COMPATIBILITY with Model 1.2 RunCode.R:
#   NA = neither species; 0 = Phragmites only; 1 = Phragmites + RCS.
# This deliberately replaces the legacy Survey_2017-derived Initial_Infested.
rcs_data$Initial_Infested <- rep(NA_integer_, ngrid)
rcs_data$Initial_Infested[rcs_data$Initial_Phrag == 1L] <- 0L
rcs_data$Initial_Infested[rcs_data$Initial_RCS == 1L] <- 1L

occurrence_diagnostics <- list(
  input_file = OCCURRENCE_FILE,
  start_year = START_YEAR,
  rcs_intro_year = RCS_INTRO_YEAR,
  original_records = n_invalid + n_outside + nrow(obs),
  invalid_coordinate_records = n_invalid,
  outside_grid_records = n_outside,
  mapped_records = nrow(obs),
  mapped_phrag_records = sum(obs$species == "phragmites"),
  mapped_rcs_records = sum(obs$species == "rcs"),
  dated_phrag_cells = sum(!is.na(p)),
  dated_rcs_cells = sum(!is.na(r)),
  undated_host_cells = sum(rcs_data$Phrag_UndatedPresence == 1L),
  phrag_seed_cells = sum(rcs_data$Initial_Phrag),
  rcs_seed_cells = sum(rcs_data$Initial_RCS),
  rcs_intro_cells = sum(rcs_data$RCS_IntroSeed)
)
cat("\nOccurrence mapping complete:\n")
print(occurrence_diagnostics)
if (sum(rcs_data$Initial_Phrag) == 0L)
  stop("No dated Phragmites seeds on/before START_YEAR inside the grid.")
if (sum(rcs_data$Initial_RCS) == 0L && RCS_INTRO_YEAR > START_YEAR) {
  cat("No initial RCS at START_YEAR; ",
      sum(rcs_data$RCS_IntroSeed),
      " documented RCS introduction cell(s) prepared for year ",
      RCS_INTRO_YEAR, ".\n", sep = "")
}
if (sum(rcs_data$RCS_IntroSeed) == 0L)
  warning("No dated RCS observations available by RCS_INTRO_YEAR inside the grid.")
cat("\n")

####################################################################################################
######################################## VALIDATE INPUTS ############################################
####################################################################################################

if (!exists("rcs_data", inherits = TRUE) || !is.data.frame(rcs_data))
  stop("rcs_data must already exist before running PreProcessing.R.")

missing_required <- setdiff(REQUIRED_CA_COLS, names(rcs_data))

if (length(missing_required))
  stop(
    "rcs_data is missing required CA input column(s): ",
    paste(missing_required, collapse = ", ")
  )

ACTIVE_RCS_MAXENT_COL <- resolve_active_rcs_column(rcs_data, BASE_MODEL)

cat(
  strrep("=", 80), "\n",
  "PREPROCESSING FOR NEW CA (NO ML WEIGHTS)\n",
  strrep("=", 80), "\n",
  "BASE_MODEL: ", BASE_MODEL, "\n",
  "Resolved active RCS MaxEnt column: ", ACTIVE_RCS_MAXENT_COL, "\n\n",
  sep = ""
)

####################################################################################################
##################################### COERCE / STANDARDIZE ##########################################
####################################################################################################

cols_to_numeric <- unique(c(
  "MaxEnt_Phragmites",
  ACTIVE_RCS_MAXENT_COL,
  "NRI_Score",
  "Initial_Infested", "Phrag_FirstObservedYear", "RCS_FirstDetectedYear",
  "Initial_Phrag", "Initial_RCS", "RCS_IntroSeed"
))

for (nm in intersect(cols_to_numeric, names(rcs_data))) {
  old_na <- sum(is.na(rcs_data[[nm]]))
  rcs_data[[nm]] <- suppressWarnings(as.numeric(rcs_data[[nm]]))
  new_na <- sum(is.na(rcs_data[[nm]]))
  
  cat(
    "Coerced numeric column: ", nm,
    " | NA before=", old_na,
    " | NA after=", new_na, "\n",
    sep = ""
  )
}

cat("\n")

####################################################################################################
########################################## DIAGNOSTICS #############################################
####################################################################################################

column_map <- data.frame(
  Role = c("PHRAG_LOCAL_MAXENT", "RCS_LOCAL_MAXENT", "JUMP_NRI"),
  Column = c("MaxEnt_Phragmites", ACTIVE_RCS_MAXENT_COL, "NRI_Score"),
  stringsAsFactors = FALSE
)

diagnostic_cols <- column_map$Column

input_diagnostics <- do.call(
  rbind,
  lapply(diagnostic_cols, function(nm) {
    data.frame(
      Column = nm,
      t(as.data.frame(summarize_numeric_col(rcs_data[[nm]]))),
      stringsAsFactors = FALSE
    )
  })
)

rownames(input_diagnostics) <- NULL

cat("CA input column map:\n")
print(column_map)

cat("\nInput diagnostics:\n")
print(input_diagnostics)
cat("\n")

present_seed_cols <- intersect(OPTIONAL_SEED_COLS, names(rcs_data))

if (length(present_seed_cols)) {
  cat(
    "Optional seed-related columns present:\n   - ",
    paste(present_seed_cols, collapse = "\n   - "),
    "\n\n", sep = ""
  )
} else {
  cat("No optional seed-related columns detected.\n\n")
}

####################################################################################################
########################################### SUMMARY ################################################
####################################################################################################

preprocess_summary <- list(
  base_model = BASE_MODEL,
  active_rcs_maxent_col = ACTIVE_RCS_MAXENT_COL,
  required_inputs = REQUIRED_CA_COLS,
  resolved_inputs = list(
    phrag_maxent = "MaxEnt_Phragmites",
    rcs_maxent = ACTIVE_RCS_MAXENT_COL,
    jump_nri = "NRI_Score"
  ),
  optional_seed_cols_present = present_seed_cols,
  diagnostics = input_diagnostics,
  occurrence_diagnostics = occurrence_diagnostics
)

cat(
  strrep("=", 80), "\n",
  "SUMMARY: PreProcessing complete for NEW CA inputs\n",
  "  • No ML feature weights computed\n",
  "  • PHRAG uses MaxEnt_Phragmites directly\n",
  "  • RCS uses active MaxEnt column directly\n",
  "  • Jump uses NRI_Score directly\n",
  "  • RCS intro year: ", RCS_INTRO_YEAR, " (seeds prepared; RunCode must activate them)\n",
  strrep("=", 80), "\n\n",
  sep = ""
)

####################################################################################################
######################################## SAVE OUTPUTS ###############################################
####################################################################################################

outputs <- list(
  CA_InputColumnMap = column_map,
  CA_InputDiagnostics = input_diagnostics,
  CA_PreprocessSummary = preprocess_summary
)
if (!is.null(occurrence_diagnostics))
  outputs$CA_OccurrenceDiagnostics <- occurrence_diagnostics

for (nm in names(outputs))
  saveRDS(outputs[[nm]], make_outfile_batch(nm, "rds"))

cat(
  "Saved RDS objects:\n",
  paste0(
    "   • ",
    vapply(names(outputs), function(nm) make_outfile_batch(nm, "rds"), character(1)),
    collapse = "\n"
  ),
  "\n\n",
  sep = ""
)

if (WRITE_PREPROCESS_CSVS) {
  write.csv(column_map, make_outfile_batch("CA_InputColumnMap", "csv"), row.names = FALSE)
  write.csv(input_diagnostics, make_outfile_batch("CA_InputDiagnostics", "csv"), row.names = FALSE)
  if (!is.null(occurrence_diagnostics)) {
    write.csv(data.frame(Metric = names(occurrence_diagnostics),
                         Value = unlist(occurrence_diagnostics),
                         row.names = NULL),
              make_outfile_batch("CA_OccurrenceDiagnostics", "csv"), row.names = FALSE)
  }
  cat("Saved CSV audit files for CA preprocessing.\n")
}