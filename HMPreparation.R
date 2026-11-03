################################################################################
# Make per-run procedure files from Outputs/param_space_lhs_current_ca.csv
# Updated for CURRENT RCS Procedure parameters:
#   PHRAG_INTERCEPT, PHRAG_MAXENT_COEF, PHRAG_SOURCE_COEF,
#   RCS_INTERCEPT,   RCS_MAXENT_COEF,   RCS_SOURCE_COEF,
#   STORM_INTERCEPT, STORM_NRI_COEF,
#   NEI_RADIUS
#
# Parameter CSV may still use legacy column names:
#   JUMP_INTERCEPT, JUMP_NRI_COEF
# Those are mapped onto the template's STORM_* fields.
#
# Reads:
#   Outputs/param_space_lhs_current_ca.csv
# Writes:
#   <PROJECT ROOT>/RCS_CA_Procedure_###.R
################################################################################

# ---- Project root helper ----
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
stopifnot(!is.null(root_dir))

# ---- Directories ----
outputs_dir <- file.path(root_dir, "Outputs")
out_dir     <- root_dir

# ---- Template + params ----
template_file <- file.path(root_dir, "RCS_CA_Procedure.R")
params_file   <- file.path(outputs_dir, "param_space_lhs_current_ca.csv")
stopifnot(file.exists(template_file), file.exists(params_file))

lines  <- readLines(template_file)
params <- read.csv(params_file, stringsAsFactors = FALSE)

# Allow either legacy JUMP_* columns or newer STORM_* columns in params CSV
if (!("JUMP_INTERCEPT" %in% names(params)) && ("STORM_INTERCEPT" %in% names(params))) {
  params$JUMP_INTERCEPT <- params$STORM_INTERCEPT
}
if (!("JUMP_NRI_COEF" %in% names(params)) && ("STORM_NRI_COEF" %in% names(params))) {
  params$JUMP_NRI_COEF <- params$STORM_NRI_COEF
}

need_cols <- c(
  "Run",
  "PHRAG_INTERCEPT", "PHRAG_MAXENT_COEF", "PHRAG_SOURCE_COEF",
  "RCS_INTERCEPT",   "RCS_MAXENT_COEF",   "RCS_SOURCE_COEF",
  "JUMP_INTERCEPT",  "JUMP_NRI_COEF",
  "NEI_RADIUS"
)
stopifnot(all(need_cols %in% names(params)))

# numeric regex (handles decimals + scientific notation)
num_re <- "[-+]?(?:\\d*\\.?\\d+|\\d+\\.?)(?:[eE][-+]?\\d+)?"

# ---- Helpers: replace ONLY within CONFIG <- list(...) block ----
locate_config_block <- function(x) {
  start <- grep("^\\s*CONFIG\\s*<-\\s*list\\s*\\(", x)
  if (length(start) != 1) stop("Could not uniquely locate 'CONFIG <- list(' in template.")
  
  end_candidates <- which(seq_along(x) > start & grepl("^\\s*\\)\\s*$", x))
  if (length(end_candidates) < 1) stop("Could not locate end of CONFIG list block (a line with just ')').")
  
  end <- end_candidates[1]
  c(start = start, end = end)
}

replace_num_line <- function(x, name, value, digits = 6) {
  pat  <- paste0("(^\\s*", name, "\\s*=\\s*)", num_re, "(\\s*,?\\s*(#.*)?$)")
  repl <- paste0("\\1", sprintf(paste0("%.", digits, "f"), value), "\\2")
  gsub(pat, repl, x, perl = TRUE)
}

replace_int_line <- function(x, name, value) {
  pat  <- paste0("(^\\s*", name, "\\s*=\\s*)", num_re, "(\\s*,?\\s*(#.*)?$)")
  repl <- paste0("\\1", as.integer(value), "\\2")
  gsub(pat, repl, x, perl = TRUE)
}

replace_chr_line <- function(x, name, value) {
  pat  <- paste0("(^\\s*", name, "\\s*=\\s*)\".*?\"(\\s*,?\\s*(#.*)?$)")
  repl <- paste0("\\1\"", value, "\"\\2")
  gsub(pat, repl, x, perl = TRUE)
}

# Optional safety check: verify a field exists in CONFIG block before replacement
assert_config_names_present <- function(cfg_lines, names_needed) {
  found <- vapply(
    names_needed,
    function(nm) any(grepl(paste0("^\\s*", nm, "\\s*="), cfg_lines)),
    logical(1)
  )
  if (!all(found)) {
    stop("These CONFIG fields were not found in template: ",
         paste(names_needed[!found], collapse = ", "))
  }
}

# ---- Generate per-run files ----
blk <- locate_config_block(lines)
cfg_idx <- blk["start"]:blk["end"]

assert_config_names_present(
  lines[cfg_idx],
  c(
    "OUTPUT_LABEL", "RUN_TAG_PREFIX",
    "PHRAG_INTERCEPT", "PHRAG_MAXENT_COEF", "PHRAG_SOURCE_COEF",
    "RCS_INTERCEPT",   "RCS_MAXENT_COEF",   "RCS_SOURCE_COEF",
    "STORM_INTERCEPT", "STORM_NRI_COEF",
    "NEI_RADIUS"
  )
)

for (i in seq_len(nrow(params))) {
  
  run_id <- as.integer(params$Run[i])
  
  new_lines <- lines
  cfg <- new_lines[cfg_idx]
  
  # per-run labels
  cfg <- replace_chr_line(cfg, "OUTPUT_LABEL",   sprintf("Run%03d", run_id))
  cfg <- replace_chr_line(cfg, "RUN_TAG_PREFIX", sprintf("Run%03d", run_id))
  
  # sampled parameters
  cfg <- replace_num_line(cfg, "PHRAG_INTERCEPT",   as.numeric(params$PHRAG_INTERCEPT[i]),   digits = 6)
  cfg <- replace_num_line(cfg, "PHRAG_MAXENT_COEF", as.numeric(params$PHRAG_MAXENT_COEF[i]), digits = 6)
  cfg <- replace_num_line(cfg, "PHRAG_SOURCE_COEF", as.numeric(params$PHRAG_SOURCE_COEF[i]), digits = 6)
  
  cfg <- replace_num_line(cfg, "RCS_INTERCEPT",     as.numeric(params$RCS_INTERCEPT[i]),     digits = 6)
  cfg <- replace_num_line(cfg, "RCS_MAXENT_COEF",   as.numeric(params$RCS_MAXENT_COEF[i]),   digits = 6)
  cfg <- replace_num_line(cfg, "RCS_SOURCE_COEF",   as.numeric(params$RCS_SOURCE_COEF[i]),   digits = 6)
  
  # legacy JUMP_* params written into current STORM_* template fields
  cfg <- replace_num_line(cfg, "STORM_INTERCEPT",   as.numeric(params$JUMP_INTERCEPT[i]),    digits = 6)
  cfg <- replace_num_line(cfg, "STORM_NRI_COEF",    as.numeric(params$JUMP_NRI_COEF[i]),     digits = 6)
  
  cfg <- replace_int_line(cfg, "NEI_RADIUS",        as.integer(params$NEI_RADIUS[i]))
  
  # write back config block
  new_lines[cfg_idx] <- cfg
  
  out_file <- file.path(out_dir, sprintf("RCS_CA_Procedure_%03d.R", run_id))
  writeLines(new_lines, out_file)
}

cat("Wrote", nrow(params), "files to project root:\n", out_dir, "\n")