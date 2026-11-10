####################################################################################################
######################################## POSTPROCESSING ##############################################
####################################################################################################
# Batch postprocessing for CA results:
#   - run/tick summary and encapsulated infestation area
#   - probability summaries and optional probability matrices
#   - optional per-run overlay GIFs
#   - calendar-year labels, RCS introduction counts, and separate PHRAG/RCS storm jumps
#
# Encapsulated area requires history plus cell geometry.
# Probability summaries/matrices require their corresponding storage options.
# GIFs require history, cell indices, and magick.

####################################################################################################
######################################## OPTIONS / PATHS ############################################
####################################################################################################

# Read a value from the active environment with a supplied fallback.
pp_get0 <- function(name, default = NULL, env = .GlobalEnv)
  get0(name, ifnotfound = default, envir = env, inherits = TRUE)

# Collect all postprocessing controls into one list.
get_postprocess_opts <- function() {
  matrix_runs <- if (exists("PROBABILITY_MATRIX_RUNS", inherits = TRUE)) {
    get("PROBABILITY_MATRIX_RUNS", inherits = TRUE)
  } else if (exists("PROBABILITY_RUNS", inherits = TRUE)) {
    get("PROBABILITY_RUNS", inherits = TRUE)
  } else 1L
  
  list(
    AUTO_RUN = isTRUE(pp_get0("AUTO_RUN", TRUE)),
    SAVE_BATCH_SUMMARY = isTRUE(pp_get0("SAVE_BATCH_SUMMARY", TRUE)),
    SAVE_ENCAPSULATED_AREA = isTRUE(pp_get0("SAVE_ENCAPSULATED_AREA", FALSE)),
    STORE_HISTORY = isTRUE(pp_get0("STORE_HISTORY", FALSE)),
    HISTORY_STORAGE = tolower(as.character(pp_get0(
      "HISTORY_STORAGE", if (isTRUE(pp_get0("STORE_HISTORY", FALSE))) "full" else "none"
    ))),
    RETURN_CELL_INDEX = isTRUE(pp_get0("RETURN_CELL_INDEX", FALSE)),
    SAVE_PER_RUN_FILES = isTRUE(pp_get0("SAVE_PER_RUN_FILES", FALSE)),
    SAVE_PROBABILITY_SUMMARY = isTRUE(pp_get0("SAVE_PROBABILITY_SUMMARY", FALSE)),
    SAVE_PROBABILITY_MATRICES = isTRUE(pp_get0("SAVE_PROBABILITY_MATRICES", FALSE)),
    PROBABILITY_SUMMARY_RUNS = pp_get0("PROBABILITY_SUMMARY_RUNS", NULL),
    PROBABILITY_MATRIX_RUNS = matrix_runs,
    SAVE_SPREAD_COMPARISON = isTRUE(pp_get0("SAVE_SPREAD_COMPARISON", FALSE)),
    MAKE_GIF = isTRUE(pp_get0("MAKE_GIF", FALSE)),
    GIF_RUNS = as.integer(pp_get0("GIF_RUNS", 1L))
  )
}

# Return the configured Outputs directory, or the working directory as fallback.
pp_get_outputs_dir <- function() {
  dirs <- pp_get0("DIRS", NULL)
  if (is.list(dirs) && !is.null(dirs$outputs_dir) && nzchar(dirs$outputs_dir))
    return(dirs$outputs_dir)
  getwd()
}

# Create a directory if needed and return its path invisibly.
pp_safe_mkdir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  invisible(path)
}

# Build the shared output tag from OUTPUT_LABEL and BASE_MODEL.
pp_base_batch_tag <- function()
  paste0(pp_get0("OUTPUT_LABEL", "Run"), "_", pp_get0("BASE_MODEL", ""))

# Build a batch-level output filename.
pp_make_batch_file <- function(suffix, ext)
  file.path(pp_get_outputs_dir(), paste0(pp_base_batch_tag(), "_", suffix, ".", ext))

# Resolve a stable label for one simulation run.
pp_run_tag <- function(ca_result, run_i = NA_integer_) {
  if (is.list(ca_result) && !is.null(ca_result$run_tag) && nzchar(ca_result$run_tag))
    return(ca_result$run_tag)
  
  run_i <- as.integer(run_i)
  rid <- sprintf("Run_%04d", if (is.finite(run_i)) run_i else 0L)
  
  paste0(pp_get0("OUTPUT_LABEL", "Run"), "_", pp_get0("BASE_MODEL", ""), "_", rid)
}

# Translate CA tick indices into calendar years. Prefer metadata attached to
# each result by RunCode.R; fall back to the active configuration if necessary.
pp_result_start_year <- function(ca_result) {
  year <- if (!is.null(ca_result$start_year)) ca_result$start_year else
    pp_get0("START_YEAR", NA_integer_)
  if (length(year) != 1L || !is.numeric(year) || !is.finite(year) ||
      year != as.integer(year))
    stop("START_YEAR is missing/invalid in the CA result and configuration.")
  as.integer(year)
}

pp_result_intro_year <- function(ca_result) {
  year <- if (!is.null(ca_result$rcs_intro_year)) ca_result$rcs_intro_year else
    pp_get0("RCS_INTRO_YEAR", NA_integer_)
  if (length(year) != 1L || !is.numeric(year) || !is.finite(year) ||
      year != as.integer(year))
    stop("RCS_INTRO_YEAR is missing/invalid in the CA result and configuration.")
  as.integer(year)
}

pp_result_calendar_years <- function(ca_result, ticks) {
  pp_result_start_year(ca_result) + as.integer(ticks)
}

# Return the output directory assigned to one run.
pp_run_dir <- function(run_tag)
  file.path(pp_get_outputs_dir(), as.character(run_tag))

# Build a sanitized per-run output filename.
pp_make_run_file <- function(run_tag, suffix, ext) {
  # Sanitize filename components for portable output paths.
  clean <- function(x) gsub("[^A-Za-z0-9._-]", "_", trimws(as.character(x)))
  file.path(pp_run_dir(run_tag),
            paste0(as.character(run_tag), "_", clean(suffix), ".", clean(ext)))
}

####################################################################################################
######################################## GEOMETRY ###################################################
####################################################################################################

# Compute polygon area from ordered x/y vertices using the shoelace formula.
pp_polygon_area <- function(x, y) {
  x <- as.numeric(x); y <- as.numeric(y)
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]; y <- y[keep]
  if (length(x) < 3L) return(0)
  
  j <- c(seq.int(2L, length(x)), 1L)
  abs(sum(x * y[j] - y * x[j])) / 2
}

# Infer a representative grid-cell area from stored bounds or cell spacing.
infer_cell_area <- function(rcs_data) {
  if (is.null(rcs_data) || !is.data.frame(rcs_data)) return(1)
  
  if (all(c("Xmin", "Xmax", "Ymin", "Ymax") %in% names(rcs_data))) {
    dx <- suppressWarnings(as.numeric(rcs_data$Xmax) - as.numeric(rcs_data$Xmin))
    dy <- suppressWarnings(as.numeric(rcs_data$Ymax) - as.numeric(rcs_data$Ymin))
    area <- abs(dx * dy)
    area <- area[is.finite(area) & area > 0]
    if (length(area)) return(stats::median(area, na.rm = TRUE))
  }
  
  if (!all(c("Xmin", "Ymin") %in% names(rcs_data))) return(1)
  
  ux <- sort(unique(suppressWarnings(as.numeric(rcs_data$Xmin))))
  uy <- sort(unique(suppressWarnings(as.numeric(rcs_data$Ymin))))
  ux <- ux[is.finite(ux)]; uy <- uy[is.finite(uy)]
  
  if (length(ux) < 2L || length(uy) < 2L) return(1)
  
  dx <- stats::median(diff(ux), na.rm = TRUE)
  dy <- stats::median(diff(uy), na.rm = TRUE)
  if (!is.finite(dx) || !is.finite(dy) || dx == 0 || dy == 0) return(1)
  
  abs(dx * dy)
}

# Prepare finite cell bounds and center coordinates for spatial area calculations.
pp_prepare_geometry <- function(rcs_data) {
  if (is.null(rcs_data) || !is.data.frame(rcs_data))
    stop("rcs_data must be available as a data.frame to compute encapsulated area.")
  
  out <- as.data.frame(rcs_data, stringsAsFactors = FALSE)
  if (!all(c("Xmin", "Ymin") %in% names(out)))
    stop("rcs_data must contain Xmin and Ymin to compute encapsulated area.")
  
  out$Xmin <- suppressWarnings(as.numeric(out$Xmin))
  out$Ymin <- suppressWarnings(as.numeric(out$Ymin))
  
  ux <- sort(unique(out$Xmin[is.finite(out$Xmin)]))
  uy <- sort(unique(out$Ymin[is.finite(out$Ymin)]))
  dx <- if (length(ux) >= 2L) stats::median(diff(ux), na.rm = TRUE) else NA_real_
  dy <- if (length(uy) >= 2L) stats::median(diff(uy), na.rm = TRUE) else NA_real_
  
  if (!"Xmax" %in% names(out)) {
    if (!is.finite(dx) || dx == 0) stop("Cannot infer Xmax because X spacing is unavailable.")
    out$Xmax <- out$Xmin + abs(dx)
  } else out$Xmax <- suppressWarnings(as.numeric(out$Xmax))
  
  if (!"Ymax" %in% names(out)) {
    if (!is.finite(dy) || dy == 0) stop("Cannot infer Ymax because Y spacing is unavailable.")
    out$Ymax <- out$Ymin + abs(dy)
  } else out$Ymax <- suppressWarnings(as.numeric(out$Ymax))
  
  bad_x <- !is.finite(out$Xmax)
  bad_y <- !is.finite(out$Ymax)
  
  if (any(bad_x)) {
    if (!is.finite(dx) || dx == 0) stop("Xmax has NA and X spacing is unavailable.")
    out$Xmax[bad_x] <- out$Xmin[bad_x] + abs(dx)
  }
  
  if (any(bad_y)) {
    if (!is.finite(dy) || dy == 0) stop("Ymax has NA and Y spacing is unavailable.")
    out$Ymax[bad_y] <- out$Ymin[bad_y] + abs(dy)
  }
  
  out$center_x <- (out$Xmin + out$Xmax) / 2
  out$center_y <- (out$Ymin + out$Ymax) / 2
  out
}

# Align rcs_data geometry to the cell ordering returned by the CA.
pp_match_geom_to_cells <- function(ca_result, rcs_data) {
  geom <- pp_prepare_geometry(rcs_data)
  cells <- ca_result$order_indexed_cells
  n_cells <- if (!is.null(ca_result$rcs_history)) nrow(ca_result$rcs_history) else
    if (!is.null(ca_result$rcs_first_tick)) length(ca_result$rcs_first_tick) else NA_integer_
  
  if (!is.null(cells) && is.data.frame(cells)) {
    cells <- as.data.frame(cells, stringsAsFactors = FALSE)
    
    if ("cell_id" %in% names(cells) && "cell_id" %in% names(geom)) {
      idx <- match(as.character(cells$cell_id), as.character(geom$cell_id))
      if (all(!is.na(idx))) return(geom[idx, , drop = FALSE])
    }
    
    if (all(c("row", "col") %in% names(cells)) &&
        all(c("row", "col") %in% names(geom))) {
      cell_key <- paste(as.integer(cells$row), as.integer(cells$col), sep = "_")
      geom_key <- paste(as.integer(geom$row), as.integer(geom$col), sep = "_")
      idx <- match(cell_key, geom_key)
      if (all(!is.na(idx))) return(geom[idx, , drop = FALSE])
    }
  }
  
  if (is.finite(n_cells) && nrow(geom) == n_cells) return(geom)
  
  stop("Could not align rcs_data geometry to CA cell order. ",
       "Use RETURN_CELL_INDEX=TRUE and keep cell_id or row/col in rcs_data.")
}

# Compute the convex-hull area enclosing a set of occupied grid cells.
pp_convex_hull_area_cells <- function(geom_subset, cell_area = 1) {
  if (is.null(geom_subset) || !nrow(geom_subset)) return(0)
  
  pts <- unique(rbind(
    data.frame(x = geom_subset$Xmin, y = geom_subset$Ymin),
    data.frame(x = geom_subset$Xmax, y = geom_subset$Ymin),
    data.frame(x = geom_subset$Xmax, y = geom_subset$Ymax),
    data.frame(x = geom_subset$Xmin, y = geom_subset$Ymax)
  ))
  
  pts <- pts[is.finite(pts$x) & is.finite(pts$y), , drop = FALSE]
  if (!nrow(pts)) return(0)
  if (nrow(pts) < 3L) return(as.numeric(cell_area) * nrow(geom_subset))
  
  hull <- grDevices::chull(pts$x, pts$y)
  area <- pp_polygon_area(pts$x[hull], pts$y[hull])
  
  if (!is.finite(area) || area <= 0)
    return(as.numeric(cell_area) * nrow(geom_subset))
  
  as.numeric(area)
}

####################################################################################################
######################################## BATCH SUMMARY ##############################################
####################################################################################################

# Convert one CA result into the run-by-tick batch-summary table.
compute_rcs_run_tick_table <- function(ca_result, run_i, rcs_data = NULL) {
  if (is.null(ca_result) || !is.list(ca_result)) stop("ca_result must be a list.")
  
  pp <- get_postprocess_opts()
  run_tag <- pp_run_tag(ca_result, run_i)
  rcs_hist <- if (!is.null(ca_result$rcs_history)) as.matrix(ca_result$rcs_history) else NULL
  
  if (!is.null(ca_result$rcs_total)) {
    rcs_cells <- as.integer(ca_result$rcs_total)
  } else if (!is.null(rcs_hist)) {
    rcs_cells <- as.integer(colSums(rcs_hist >= 1L, na.rm = TRUE))
  } else stop("ca_result must contain either rcs_total or rcs_history.")
  
  ticks <- seq_along(rcs_cells) - 1L
  
  phrag_cells <- if (!is.null(ca_result$phrag_total) &&
                     length(ca_result$phrag_total) == length(rcs_cells)) {
    as.integer(ca_result$phrag_total)
  } else if (!is.null(ca_result$phrag_history)) {
    as.integer(colSums(as.matrix(ca_result$phrag_history) >= 1L, na.rm = TRUE))
  } else rep(NA_integer_, length(rcs_cells))
  
  valid_cell_area <- !is.null(ca_result$cell_area_used) &&
    is.numeric(ca_result$cell_area_used) &&
    length(ca_result$cell_area_used) == 1L &&
    is.finite(ca_result$cell_area_used) &&
    ca_result$cell_area_used > 0
  
  cell_area <- if (valid_cell_area) as.numeric(ca_result$cell_area_used) else infer_cell_area(rcs_data)
  infested_area <- as.numeric(rcs_cells) * cell_area
  
  # Return a result vector when its length matches the tick series, otherwise fill with NA.
  same_length_or_na <- function(x, mode = c("integer", "numeric")) {
    mode <- match.arg(mode)
    if (!is.null(x) && length(x) == length(ticks))
      return(if (mode == "integer") as.integer(x) else as.numeric(x))
    if (mode == "integer") rep(NA_integer_, length(ticks)) else rep(NA_real_, length(ticks))
  }
  
  storm_count <- same_length_or_na(ca_result$storm_count)
  jump_count <- same_length_or_na(ca_result$jump_total)
  phrag_jump_count <- same_length_or_na(ca_result$phrag_jump_total)
  rcs_intro_count <- same_length_or_na(ca_result$rcs_intro_count)
  phrag_intro_count <- same_length_or_na(ca_result$phrag_intro_count)
  intro_year <- pp_result_intro_year(ca_result)
  intro_seed_cells <- if (!is.null(ca_result$rcs_intro_seed_count) &&
                          length(ca_result$rcs_intro_seed_count) == 1L) {
    as.integer(ca_result$rcs_intro_seed_count)
  } else NA_integer_
  seed <- if (!is.null(ca_result$seed) && length(ca_result$seed) == 1L &&
              is.finite(ca_result$seed)) as.integer(ca_result$seed) else NA_integer_
  jump_method <- as.character(
    if (!is.null(ca_result$jump_source_method)) ca_result$jump_source_method
    else pp_get0("JUMP_SOURCE_METHOD", NA_character_)
  )
  
  encapsulated_area <- rep(NA_real_, length(ticks))
  
  if (pp$SAVE_ENCAPSULATED_AREA) {
    arrival <- ca_result$rcs_first_tick
    if (is.null(rcs_hist) && is.null(arrival)) {
      warning("SAVE_ENCAPSULATED_AREA=TRUE but neither RCS history nor first-arrival ",
              "ticks are available for run ", run_i, ".")
    } else if (is.null(rcs_data) || !is.data.frame(rcs_data)) {
      warning("SAVE_ENCAPSULATED_AREA=TRUE but rcs_data is unavailable for run ", run_i, ".")
    } else {
      geom <- pp_match_geom_to_cells(ca_result, rcs_data)
      n_expected <- if (!is.null(rcs_hist)) nrow(rcs_hist) else length(arrival)
      if (nrow(geom) != n_expected)
        stop("Geometry/history mismatch for run ", run_i,
             ": aligned cells=", nrow(geom), " but history cells=", n_expected)
      
      if (!is.null(rcs_hist)) {
        for (j in seq_len(min(length(ticks), ncol(rcs_hist)))) {
          occupied <- which(rcs_hist[, j] >= 1L)
          encapsulated_area[j] <- pp_convex_hull_area_cells(
            geom[occupied, , drop = FALSE], cell_area
          )
        }
      } else {
        # Exact same occupied set at tick t, reconstructed without a history matrix.
        for (j in seq_along(ticks)) {
          occupied <- which(arrival >= 0L & arrival <= ticks[j])
          encapsulated_area[j] <- pp_convex_hull_area_cells(
            geom[occupied, , drop = FALSE], cell_area
          )
        }
      }
    }
  }
  
  data.frame(
    Run_Number = as.integer(run_i),
    Run_Label = run_tag,
    Seed = seed,
    Tick = as.integer(ticks),
    Year = pp_result_calendar_years(ca_result, ticks),
    RCS_Introduction_Year = intro_year,
    Number_of_Phrag_Cells = phrag_cells,
    Number_of_Infested_Cells = rcs_cells,
    Infested_Area = infested_area,
    Encapsulated_Infested_Area = encapsulated_area,
    Storm_Event_Count = storm_count,
    Storm_Jump_Success_Count = jump_count,
    Phrag_Storm_Jump_Success_Count = phrag_jump_count,
    RCS_Introduction_Count = rcs_intro_count,
    Phrag_Introduction_Support_Count = phrag_intro_count,
    RCS_Introduction_Seed_Count = intro_seed_cells,
    Jump_Source_Method = jump_method,
    stringsAsFactors = FALSE
  )
}

# Save the combined run-by-tick batch summary as CSV.
save_batch_summary_csv <- function(ca_results, rcs_data = NULL) {
  if (!get_postprocess_opts()$SAVE_BATCH_SUMMARY) {
    cat("SAVE_BATCH_SUMMARY=FALSE; skipping BatchSummary.csv.\n")
    return(invisible(NULL))
  }
  
  if (is.null(ca_results) || !is.list(ca_results) || !length(ca_results)) {
    warning("save_batch_summary_csv: ca_results empty; nothing to save.")
    return(invisible(NULL))
  }
  
  out_df <- do.call(rbind, lapply(seq_along(ca_results), function(i)
    compute_rcs_run_tick_table(ca_results[[i]], i, rcs_data)))
  
  out_file <- pp_make_batch_file("BatchSummary", "csv")
  utils::write.csv(out_df, out_file, row.names = FALSE)
  cat("Saved batch run/tick summary to:\n   → ", out_file, "\n\n", sep = "")
  invisible(out_df)
}

####################################################################################################
######################################## RUN SELECTION ##############################################
####################################################################################################

# Normalize a run selector to valid unique indices within 1..n.
pp_normalize_run_selector <- function(runs, n, null_means_all = TRUE) {
  n <- as.integer(n)
  if (!is.finite(n) || n < 1L) return(integer(0))
  if (is.null(runs)) return(if (null_means_all) seq_len(n) else integer(0))
  
  runs <- suppressWarnings(as.integer(unlist(runs, recursive = TRUE, use.names = FALSE)))
  unique(runs[is.finite(runs) & runs >= 1L & runs <= n])
}

# Format a run selector for console reporting.
pp_format_run_selector <- function(runs) {
  if (is.null(runs)) return("ALL")
  runs <- suppressWarnings(as.integer(unlist(runs, recursive = TRUE, use.names = FALSE)))
  runs <- unique(runs[is.finite(runs) & runs >= 1L])
  if (!length(runs)) "NONE" else paste(runs, collapse = ",")
}

# Resolve runs selected for lightweight probability summaries.
pp_probability_summary_runs <- function(ca_results)
  pp_normalize_run_selector(pp_get0("PROBABILITY_SUMMARY_RUNS", NULL), length(ca_results), TRUE)

# Resolve runs selected for full probability matrices, including the legacy fallback.
pp_probability_matrix_runs <- function(ca_results) {
  runs <- if (exists("PROBABILITY_MATRIX_RUNS", inherits = TRUE)) {
    get("PROBABILITY_MATRIX_RUNS", inherits = TRUE)
  } else if (exists("PROBABILITY_RUNS", inherits = TRUE)) {
    get("PROBABILITY_RUNS", inherits = TRUE)
  } else 1L
  
  pp_normalize_run_selector(runs, length(ca_results), TRUE)
}

# Resolve runs selected for overlay GIF generation.
pp_gif_runs <- function(ca_results)
  pp_normalize_run_selector(pp_get0("GIF_RUNS", 1L), length(ca_results), TRUE)

####################################################################################################
######################################## PROBABILITY OUTPUTS ########################################
####################################################################################################

# Save selected tick-level probability summaries as one CSV.
save_probability_summary_csv <- function(ca_results) {
  if (!get_postprocess_opts()$SAVE_PROBABILITY_SUMMARY) {
    cat("SAVE_PROBABILITY_SUMMARY=FALSE; skipping probability summary.\n")
    return(invisible(NULL))
  }
  
  if (is.null(ca_results) || !is.list(ca_results) || !length(ca_results)) {
    warning("save_probability_summary_csv: ca_results empty; nothing to save.")
    return(invisible(NULL))
  }
  
  run_ids <- pp_probability_summary_runs(ca_results)
  if (!length(run_ids)) {
    warning("No valid PROBABILITY_SUMMARY_RUNS selected; skipping probability summary.")
    return(invisible(NULL))
  }
  
  out_rows <- lapply(run_ids, function(i) {
    cr <- ca_results[[i]]
    if (is.null(cr$prob_summary) || !is.data.frame(cr$prob_summary) || !nrow(cr$prob_summary)) {
      warning("Run ", i, " has no prob_summary. Was STORE_PROB_SUMMARY=TRUE?")
      return(NULL)
    }
    
    df <- as.data.frame(cr$prob_summary, stringsAsFactors = FALSE)
    if (!"Tick" %in% names(df))
      stop("CA probability summary for run ", i, " has no Tick column.")
    df$Year <- pp_result_calendar_years(cr, df$Tick)
    df$Run_Number <- as.integer(i)
    df$Run_Label <- pp_run_tag(cr, i)
    id_cols <- c("Run_Number", "Run_Label", "Tick", "Year")
    df[, c(id_cols, setdiff(names(df), id_cols)), drop = FALSE]
  })
  
  out_rows <- Filter(Negate(is.null), out_rows)
  if (!length(out_rows)) {
    warning("No probability summary rows found; nothing written.")
    return(invisible(NULL))
  }
  
  out_df <- do.call(rbind, out_rows)
  out_file <- pp_make_batch_file("ProbabilitySummary", "csv")
  utils::write.csv(out_df, out_file, row.names = FALSE)
  cat("Saved probability summary to:\n   → ", out_file, "\n\n", sep = "")
  invisible(out_df)
}

# Save selected full probability matrices as per-run RDS files.
save_probability_matrices_rds <- function(ca_results) {
  if (!get_postprocess_opts()$SAVE_PROBABILITY_MATRICES) {
    cat("SAVE_PROBABILITY_MATRICES=FALSE; skipping probability matrices.\n")
    return(invisible(NULL))
  }
  
  if (is.null(ca_results) || !is.list(ca_results) || !length(ca_results)) {
    warning("save_probability_matrices_rds: ca_results empty; nothing to save.")
    return(invisible(NULL))
  }
  
  run_ids <- pp_probability_matrix_runs(ca_results)
  if (!length(run_ids)) {
    warning("No valid PROBABILITY_MATRIX_RUNS / PROBABILITY_RUNS selected; skipping probability matrices.")
    return(invisible(NULL))
  }
  
  out_dir <- file.path(pp_get_outputs_dir(), paste0(pp_base_batch_tag(), "_ProbabilityMatrices"))
  pp_safe_mkdir(out_dir)
  
  written <- character(0)
  
  for (i in run_ids) {
    cr <- ca_results[[i]]
    run_tag <- pp_run_tag(cr, i)
    mats <- list(
      Run_Number = as.integer(i),
      Run_Label = run_tag,
      Start_Year = pp_result_start_year(cr),
      RCS_Introduction_Year = pp_result_intro_year(cr),
      # Probability column t represents the transition ending at START_YEAR + t.
      Probability_Years = pp_result_calendar_years(
        cr, seq_len(length(cr$rcs_total) - 1L)
      ),
      p_phrag_local_hist = cr$p_phrag_local_hist,
      p_rcs_local_hist = cr$p_rcs_local_hist,
      p_jump_hist = cr$p_jump_hist,
      p_phrag_jump_hist = cr$p_phrag_jump_hist,
      order_indexed_cells = cr$order_indexed_cells
    )
    
    has_matrix <- any(vapply(
      mats[c("p_phrag_local_hist", "p_rcs_local_hist", "p_jump_hist", "p_phrag_jump_hist")],
      Negate(is.null), logical(1)
    ))
    
    if (!has_matrix) {
      warning("Run ", i, " has no probability matrices. Was STORE_PROBABILITIES=TRUE?")
      next
    }
    
    out_file <- file.path(out_dir, paste0(run_tag, "_ProbabilityMatrices.rds"))
    saveRDS(mats, out_file)
    written <- c(written, out_file)
    cat("Saved probability matrices for run ", i, ":\n   → ", out_file, "\n", sep = "")
  }
  
  if (length(written)) cat("\n") else warning("No probability matrix files were written.")
  invisible(written)
}

####################################################################################################
######################################## GIF OUTPUTS ################################################
####################################################################################################

# Render GIF frames directly from cell-state matrices, avoiding ggplot/gganimate.
# This retains every annual frame, the four original colors, and the existing output filenames.
save_overlay_gifs <- function(ca_results) {
  if (!get_postprocess_opts()$MAKE_GIF) {
    cat("MAKE_GIF=FALSE; skipping GIF saving.\n")
    return(invisible(character(0)))
  }
  
  if (is.null(ca_results) || !is.list(ca_results) || !length(ca_results)) {
    warning("save_overlay_gifs: ca_results empty; nothing to save.")
    return(invisible(character(0)))
  }
  
  if (!requireNamespace("magick", quietly = TRUE)) {
    warning("Missing package required to save GIFs: magick")
    return(invisible(character(0)))
  }
  
  run_ids <- pp_gif_runs(ca_results)
  if (!length(run_ids)) {
    warning("No valid GIF_RUNS selected; skipping GIF saving.")
    return(invisible(character(0)))
  }
  
  colors <- c("white", "palegreen3", "orange", "red")
  written <- character(0)
  
  for (i in run_ids) {
    cr <- ca_results[[i]]
    has_full_history <- !is.null(cr$phrag_history) && !is.null(cr$rcs_history)
    has_arrival_history <- !is.null(cr$phrag_first_tick) && !is.null(cr$rcs_first_tick)
    
    if (!has_full_history && !has_arrival_history) {
      warning("Run ", i, " cannot build GIF: no full history or first-arrival ticks.")
      next
    }
    if (is.null(cr$order_indexed_cells) ||
        !all(c("row", "col") %in% names(cr$order_indexed_cells))) {
      warning("Run ", i, " cannot build GIF: row/col cell indices are missing.")
      next
    }
    
    cells <- cr$order_indexed_cells
    rows <- as.integer(cells$row)
    cols <- as.integer(cells$col)
    if (!length(rows) || anyNA(rows) || anyNA(cols)) {
      warning("Run ", i, " cannot build GIF: invalid row/col indices.")
      next
    }
    
    # Convert sparse/irregular cell locations into positions in a dense image.
    nrows <- as.integer(max(rows) - min(rows) + 1L)
    ncols <- as.integer(max(cols) - min(cols) + 1L)
    image_index <- (cols - min(cols)) * nrows + (rows - min(rows) + 1L)
    
    if (anyDuplicated(image_index))
      stop("Run ", i, ": duplicate row/col cell positions in order_indexed_cells.")
    
    if (has_full_history) {
      n_frames <- min(ncol(cr$phrag_history), ncol(cr$rcs_history))
      if (nrow(cr$phrag_history) != length(image_index) ||
          nrow(cr$rcs_history) != length(image_index))
        stop("Run ", i, ": history rows do not match cell indices.")
    } else {
      n_frames <- length(cr$rcs_total)
      if (length(cr$phrag_first_tick) != length(image_index) ||
          length(cr$rcs_first_tick) != length(image_index))
        stop("Run ", i, ": first-arrival vectors do not match cell indices.")
    }
    
    if (n_frames < 2L) {
      warning("Run ", i, " has fewer than two frames; skipping GIF.")
      next
    }
    
    run_tag <- pp_run_tag(cr, i)
    out_file <- pp_make_run_file(run_tag, "Overlay", "gif")
    pp_safe_mkdir(dirname(out_file))
    cat("Saving fast overlay GIF for run ", i, ":\n   ", out_file, "\n", sep = "")
    
    start_time <- proc.time()[["elapsed"]]
    frames <- vector("list", n_frames)
    start_year <- pp_result_start_year(cr)
    intro_year <- pp_result_intro_year(cr)
    model <- pp_get0("BASE_MODEL", "")
    
    description <- paste0(
      "Run ", i, "  |  Model: ", model,
      "  |  Start: ", start_year, "  |  RCS introduction: ", intro_year,
      "  |  Neighborhood: ", pp_get0("NEIGHBORHOOD", ""),
      "  |  Storm shell: 2..", pp_get0("NEI_RADIUS", ""),
      " (", pp_get0("DIST_METRIC", ""), ")"
    )
    
    # Occupancy is permanent in this model. Convert full history to first-arrival
    # ticks only if the CA did not already return its more compact arrival vectors.
    if (has_arrival_history) {
      phrag_first <- as.integer(cr$phrag_first_tick)
      rcs_first <- as.integer(cr$rcs_first_tick)
    } else {
      first_hit <- function(history) {
        first <- rep.int(-1L, nrow(history))
        for (j in seq_len(n_frames)) {
          newly <- which(history[, j] >= 1L & first < 0L)
          first[newly] <- j - 1L
        }
        first
      }
      phrag_first <- first_hit(cr$phrag_history)
      rcs_first <- first_hit(cr$rcs_history)
    }
    
    # Group new arrivals by tick once, instead of scanning every cell every year.
    arrivals_by_frame <- function(first) {
      valid <- which(!is.na(first) & first >= 0L & first < n_frames)
      split(valid, factor(first[valid] + 1L, levels = seq_len(n_frames)))
    }
    phrag_additions <- arrivals_by_frame(phrag_first)
    rcs_additions <- arrivals_by_frame(rcs_first)
    
    # Persistent cell states and image pixel colors (including empty grid cells).
    phrag_state <- integer(length(image_index))
    rcs_state <- integer(length(image_index))
    pixel_colors <- rep.int("white", nrows * ncols)
    
    for (j in seq_len(n_frames)) {
      tick <- j - 1L
      newly_phrag <- phrag_additions[[j]]
      newly_rcs <- rcs_additions[[j]]
      phrag_state[newly_phrag] <- 1L
      rcs_state[newly_rcs] <- 1L
      
      # Recolor only the cells that changed since the previous year.
      changed <- union(newly_phrag, newly_rcs)
      if (length(changed)) {
        combo <- phrag_state[changed] + 2L * rcs_state[changed]
        pixel_colors[image_index[changed]] <- colors[combo + 1L]
      }
      
      # ImageMagick still receives a complete frame, but its data are updated
      # incrementally and the unchanged grid is retained between years.
      frame <- magick::image_read(as.raster(matrix(
        pixel_colors, nrow = nrows, ncol = ncols
      )))
      
      # Keep year, model metadata, and state legend visible in every frame.
      frame <- magick::image_border(frame, color = "white", geometry = "12x68")
      frame <- magick::image_annotate(
        frame, paste0("Two-layer CA Spread (PHRAG + RCS) - Year: ", start_year + tick),
        gravity = "north", location = "+0+8", size = 18, color = "black"
      )
      frame <- magick::image_annotate(
        frame, description,
        gravity = "north", location = "+0+37", size = 12, color = "black"
      )
      frame <- magick::image_annotate(
        frame, "White: None  |  Green: PHRAG only  |  Orange: RCS only  |  Red: PHRAG + RCS",
        gravity = "south", location = "+0+18", size = 13, color = "black"
      )
      frames[[j]] <- frame
    }
    
    render_seconds <- proc.time()[["elapsed"]] - start_time
    cat("  Rendered ", n_frames, " frames in ",
        round(render_seconds, 1), " seconds.\n", sep = "")
    
    encode_start <- proc.time()[["elapsed"]]
    # GIF optimization stores frame differences rather than full repeated images.
    # It usually reduces file size, but its extra comparison work can raise encode time.
    gif <- magick::image_animate(do.call(c, frames), fps = 4, optimize = TRUE)
    magick::image_write(gif, path = out_file, format = "gif")
    encode_seconds <- proc.time()[["elapsed"]] - encode_start
    cat("  Encoded and saved GIF in ", round(encode_seconds, 1),
        " seconds.\n", sep = "")
    written <- c(written, out_file)
  }
  
  if (!length(written))
    cat("MAKE_GIF=TRUE but no GIF files were written. Check GIF_RUNS, STORE_HISTORY, and RETURN_CELL_INDEX.\n")
  
  invisible(written)
}

####################################################################################################
######################################## SAVE / AUTO-RUN ############################################
####################################################################################################

# Save all enabled postprocessing products and return their in-memory references.
save_all_outputs <- function(ca_results, rcs_data = NULL) {
  if (is.null(ca_results) || !is.list(ca_results) || !length(ca_results))
    stop("ca_results must be a non-empty list.")
  
  pp_safe_mkdir(pp_get_outputs_dir())
  
  invisible(list(
    batch_summary = save_batch_summary_csv(ca_results, rcs_data),
    probability_summary = save_probability_summary_csv(ca_results),
    probability_matrix_files = save_probability_matrices_rds(ca_results),
    overlay_gif_files = save_overlay_gifs(ca_results)
  ))
}

# Warn about enabled outputs whose required run data or packages are unavailable.
pp_validate_output_requirements <- function(pp, ca_results) {
  if (pp$SAVE_ENCAPSULATED_AREA && pp$HISTORY_STORAGE == "none")
    warning("SAVE_ENCAPSULATED_AREA=TRUE but HISTORY_STORAGE='none'. ",
            "A history or first-arrival vector is required.")
  
  if (pp$SAVE_ENCAPSULATED_AREA && !pp$RETURN_CELL_INDEX)
    warning("SAVE_ENCAPSULATED_AREA=TRUE but RETURN_CELL_INDEX=FALSE. ",
            "Geometry alignment may fail unless ca_result$order_indexed_cells is still present.")
  
  if (pp$SAVE_PROBABILITY_SUMMARY &&
      !any(vapply(ca_results, function(x)
        !is.null(x$prob_summary) && is.data.frame(x$prob_summary) && nrow(x$prob_summary) > 0L,
        logical(1))))
    warning("SAVE_PROBABILITY_SUMMARY=TRUE but no ca_results contain prob_summary. ",
            "Check that STORE_PROB_SUMMARY=TRUE during RunCode.R / CellularAutomata.R.")
  
  if (pp$SAVE_PROBABILITY_MATRICES &&
      !any(vapply(ca_results, function(x)
        !is.null(x$p_phrag_local_hist) || !is.null(x$p_rcs_local_hist) ||
        !is.null(x$p_jump_hist) || !is.null(x$p_phrag_jump_hist),
        logical(1))))
    warning("SAVE_PROBABILITY_MATRICES=TRUE but no probability matrices were found. ",
            "Check that STORE_PROBABILITIES=TRUE during RunCode.R / CellularAutomata.R.")
  
  if (pp$MAKE_GIF) {
    if (pp$HISTORY_STORAGE == "none")
      warning("MAKE_GIF=TRUE but HISTORY_STORAGE='none'. GIF generation requires history.")
    if (!pp$RETURN_CELL_INDEX)
      warning("MAKE_GIF=TRUE but RETURN_CELL_INDEX=FALSE. GIF generation requires cell indices.")
    if (!requireNamespace("magick", quietly = TRUE))
      warning("MAKE_GIF=TRUE but package 'magick' is missing.")
  }
  
  invisible(TRUE)
}

# Ensure each CA result has its expected run number and run label.
pp_label_results <- function(ca_results) {
  lapply(seq_along(ca_results), function(i) {
    cr <- ca_results[[i]]
    if (is.null(cr$run_tag) || !nzchar(cr$run_tag)) cr$run_tag <- pp_run_tag(cr, i)
    if (is.null(cr$run_i)) cr$run_i <- as.integer(i)
    cr
  })
}

pp <- get_postprocess_opts()

if (pp$AUTO_RUN) {
  if (exists("ca_results", inherits = TRUE) && is.list(ca_results) && length(ca_results)) {
    cat(
      "PostProcessing\n",
      "   → SAVE_BATCH_SUMMARY        = ", pp$SAVE_BATCH_SUMMARY, "\n",
      "   → SAVE_ENCAPSULATED_AREA    = ", pp$SAVE_ENCAPSULATED_AREA, "\n",
      "   → SAVE_PROBABILITY_SUMMARY  = ", pp$SAVE_PROBABILITY_SUMMARY, "\n",
      "   → SAVE_PROBABILITY_MATRICES = ", pp$SAVE_PROBABILITY_MATRICES, "\n",
      "   → PROBABILITY_SUMMARY_RUNS  = ", pp_format_run_selector(pp$PROBABILITY_SUMMARY_RUNS), "\n",
      "   → PROBABILITY_MATRIX_RUNS   = ", pp_format_run_selector(pp$PROBABILITY_MATRIX_RUNS), "\n",
      "   → MAKE_GIF                  = ", pp$MAKE_GIF, "\n",
      "   → GIF_RUNS                  = ", paste(pp$GIF_RUNS, collapse = ","), "\n\n",
      sep = ""
    )
    
    pp_validate_output_requirements(pp, ca_results)
    ca_results <- pp_label_results(ca_results)
    
    pp_saved <- save_all_outputs(ca_results, pp_get0("rcs_data", NULL))
    batch_summary <- pp_saved$batch_summary
    probability_summary <- pp_saved$probability_summary
    probability_matrix_files <- pp_saved$probability_matrix_files
    overlay_gif_files <- pp_saved$overlay_gif_files
    
  } else warning("PostProcessing: ca_results not found; skipping postprocessing.")
} else cat("AUTO_RUN=FALSE; PostProcessing loaded but did not execute.\n")
