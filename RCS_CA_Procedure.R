
################################################################################################################
############ Set Directory ######################################################################################
################################################################################################################

get_start_dir <- function() {
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getActiveDocumentContext()$path, error = function(e) "")
    if (nzchar(p)) return(dirname(normalizePath(p, winslash = "/", mustWork = TRUE)))
  }
  
  p <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(p) == 1L && nzchar(p))
    return(dirname(normalizePath(p, winslash = "/", mustWork = TRUE)))
  
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
  
  stop("Could not detect project root from ", start,
       " (need: ", paste(req, collapse = ", "), ")")
}

proj_root <- find_project_root(get_start_dir())
func_dir <- file.path(proj_root, "Functions")
data_dir <- file.path(proj_root, "Data")
outputs_dir <- file.path(proj_root, "Outputs")

# Compatibility object used by Initialize.R, RunCode.R, and PostProcessing.R
DIRS <- list(
  root = proj_root,
  func_dir = func_dir,
  data_dir = data_dir,
  outputs_dir = outputs_dir
)

dir.create(outputs_dir, recursive = TRUE, showWarnings = FALSE)

cat("Project root is:     ", proj_root, "\n")
cat("Functions folder is: ", func_dir, "\n")
cat("Data folder is:      ", data_dir, "\n")
cat("Outputs folder is:   ", outputs_dir, "\n")


################################################################################################################
############ Configuration ######################################################################################
################################################################################################################

CONFIG <- list(
  
  # Output / environmental data
  OUTPUT_LABEL = "Run644",
  DATA_FILE = "Southeast_Data_Clip.csv",
  
  # Occurrence data
  OCCURRENCE_FILE = "RCS_Phragmites_Data_10_17_26.csv",
  
  # Simulation timeline
  START_YEAR = 2000,
  RCS_INTRO_YEAR = 2017,
  MAX_TICKS = 30,
  
  # Simulation grid
  GRID_EPSG = 32615,
  GRID_RESOLUTION_M = 1000,
  
  # Optional geographic clip (EPSG:32615, meters); NULL uses the full DATA_FILE grid.
  # Example: c(xmin = 300000, xmax = 700000, ymin = 3100000, ymax = 3500000)
  CLIP_BOUNDS = c(
    xmin = 49530.8674345585,
    xmax = 1417530.86743456,
    ymin = 3112714.25764237,
    ymax = 3466714.25764237
  ),
  
  # Simulation geometry
  NEIGHBORHOOD = 4,
  DIST_METRIC = "manhattan",
  JUMP_SOURCE_METHOD = "offset",
  
  # Base model
  BASE_MODEL = "COLD",
  
  # Local PHRAG coefficients
  PHRAG_INTERCEPT = -3.25, # beta 0 in phragmites probability formula
  PHRAG_MAXENT_COEF = 5.50, # beta 1 in phragmites probability formula
  PHRAG_SOURCE_COEF = 0.00, # beta 2 in phragmites probability formula
  
  # Local RCS coefficients
  RCS_INTERCEPT = -7.00,
  RCS_MAXENT_COEF = 8.25,
  RCS_SOURCE_COEF = 1.75,
  
  # Storm spread
  STORM_INTERCEPT = 4.25,
  STORM_NRI_COEF = 0.75,
  NEI_RADIUS = 37,
  STORMS_PER_TICK = c(4, 7, 4, 5, 8, 4, 3, 2, 5, 1),
  
  # Initial infestation
  INITIAL_INFESTATION = "LoadedData",
  INITIAL_INFESTATION_COUNT = 5,
  
  # RNG
  USE_FIXED_SEED = FALSE,
  RNG_SEED = 12345,
  SEED_BASE = 100000,
  
  # Runs / parallelization
  N_RUNS = 4,
  RUN_PARALLEL = TRUE,
  N_WORKERS = 4,
  
  # Run outputs
  SAVE_BATCH_SUMMARY = TRUE,
  STORE_HISTORY = TRUE,
  RETURN_CELL_INDEX = TRUE,
  SAVE_ENCAPSULATED_AREA = TRUE,
  
  # Probability outputs
  STORE_PROB_SUMMARY = TRUE,
  SAVE_PROBABILITY_SUMMARY = TRUE,
  PROBABILITY_SUMMARY_RUNS = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
  STORE_PROBABILITIES = FALSE,
  SAVE_PROBABILITY_MATRICES = FALSE,
  PROBABILITY_MATRIX_RUNS = integer(0),
  
  # Returned objects
  RETURN_SPARSE_OPS = FALSE,
  RETURN_CELL_COVARIATES = FALSE,
  
  # Operator cache
  SPARSE_OPS_USE_CACHE = TRUE,
  SPARSE_OPS_OVERWRITE = FALSE,
  SPARSE_OPS_CACHE_SUBDIR = "SparseOpsCache",
  
  # Postprocessing
  AUTO_RUN = TRUE,
  
  # GIFs
  MAKE_GIF = TRUE,
  GIF_RUNS = 1
)

# Unpack Config
if (is.null(names(CONFIG)) || anyNA(names(CONFIG)) || any(!nzchar(names(CONFIG))))
  stop("Every CONFIG entry must have a name using NAME = value.")

if (anyDuplicated(names(CONFIG)))
  stop("CONFIG contains duplicated parameter names.")

list2env(CONFIG, envir = .GlobalEnv)


################################################################################################################
############ Validate Configuration ############################################################################
################################################################################################################

# Configuration values are used as entered; no automatic type conversions.

stopifnot(
  BASE_MODEL %in% c("NIPPON", "COLD", "NATIVE", "NIPPONACLERDA",
                    "RCS_COLD_NATIVE", "RCS_NATIVE"),
  DIST_METRIC %in% c("manhattan", "chebyshev"),
  JUMP_SOURCE_METHOD %in% c("offset", "sparse"),
  NEIGHBORHOOD %in% c(4L, 8L),
  MAX_TICKS >= 1,
  NEI_RADIUS >= 1,
  N_RUNS >= 1,
  N_WORKERS >= 1,
  INITIAL_INFESTATION %in% c("LoadedData", "Random", "Susceptible"),
  INITIAL_INFESTATION_COUNT >= 0,
  
  # Occurrence and timeline validation
  length(OCCURRENCE_FILE) == 1L,
  nzchar(OCCURRENCE_FILE),
  is.finite(START_YEAR),
  is.finite(RCS_INTRO_YEAR),
  START_YEAR >= 1700,
  RCS_INTRO_YEAR >= START_YEAR,
  RCS_INTRO_YEAR <= START_YEAR + MAX_TICKS,
  GRID_EPSG == 32615L,
  GRID_RESOLUTION_M == 1000,
  
  all(is.finite(c(
    PHRAG_INTERCEPT, PHRAG_MAXENT_COEF, PHRAG_SOURCE_COEF,
    RCS_INTERCEPT, RCS_MAXENT_COEF, RCS_SOURCE_COEF,
    STORM_INTERCEPT, STORM_NRI_COEF
  ))),
  is.numeric(STORMS_PER_TICK),
  length(STORMS_PER_TICK) >= 1L,
  all(is.finite(STORMS_PER_TICK)),
  all(STORMS_PER_TICK >= 0),
  all(STORMS_PER_TICK == round(STORMS_PER_TICK))
)

# Validate the optional rectangular clipping boundary.
if (!is.null(CLIP_BOUNDS)) {
  if (!is.numeric(CLIP_BOUNDS) || length(CLIP_BOUNDS) != 4L ||
      !identical(names(CLIP_BOUNDS), c("xmin", "xmax", "ymin", "ymax")) ||
      any(!is.finite(CLIP_BOUNDS)) ||
      CLIP_BOUNDS[["xmin"]] >= CLIP_BOUNDS[["xmax"]] ||
      CLIP_BOUNDS[["ymin"]] >= CLIP_BOUNDS[["ymax"]]) {
    stop("CLIP_BOUNDS must be NULL or c(xmin=..., xmax=..., ymin=..., ymax=...) ",
         "with finite EPSG:32615 coordinates and xmin<xmax, ymin<ymax.")
  }
}

# Resolve the occurrence CSV from the same Data folder as DATA_FILE.
OCCURRENCE_FILE <- file.path(data_dir, OCCURRENCE_FILE)

if (!file.exists(OCCURRENCE_FILE))
  stop("Occurrence file not found: ", OCCURRENCE_FILE)

if (RUN_PARALLEL && N_WORKERS > parallel::detectCores())
  warning("N_WORKERS exceeds available cores; consider lowering N_WORKERS.")

if (length(GIF_RUNS) && (any(GIF_RUNS < 1L) || any(GIF_RUNS > N_RUNS)))
  warning("GIF_RUNS contains indices outside 1..N_RUNS.")

if (SAVE_PROBABILITY_SUMMARY && !STORE_PROB_SUMMARY)
  warning("SAVE_PROBABILITY_SUMMARY=TRUE but STORE_PROB_SUMMARY=FALSE.")

if (SAVE_PROBABILITY_MATRICES && !STORE_PROBABILITIES)
  warning("SAVE_PROBABILITY_MATRICES=TRUE but STORE_PROBABILITIES=FALSE.")


################################################################################################################
############ Source Simulation ##################################################################################
################################################################################################################

source(file.path(func_dir, "Initialize.R"))
source(file.path(func_dir, "PreProcessing.R"))
source(file.path(func_dir, "CellularAutomata.R"))
source(file.path(func_dir, "RunCode.R"))
source(file.path(func_dir, "PostProcessing.R"))
