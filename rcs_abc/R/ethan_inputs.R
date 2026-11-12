# Build Ethan's preprocessed grid (rcs_data) with his own Initialize.R +
# PreProcessing.R, using the same CONFIG values as RCS_CA_Procedure.R, and the
# observed RCS summaries on that same grid.
build_ethan_inputs <- function(proj, out_dir = file.path(getwd(), "results", "ethan_pp"),
                               start_year = 2000L, intro_year = 2017L, base_model = "COLD",
                               quiet = TRUE) {
  ge <- globalenv()
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  assign("DIRS", list(root = proj, func_dir = file.path(proj, "Functions"),
                      data_dir = file.path(proj, "Data"), outputs_dir = out_dir), ge)
  assign("DATA_FILE", "Southeast_Data_Clip.csv", ge)
  assign("OCCURRENCE_FILE", file.path(proj, "Data", "RCS_Phragmites_Data_10_17_26.csv"), ge)
  assign("START_YEAR", as.integer(start_year), ge); assign("RCS_INTRO_YEAR", as.integer(intro_year), ge)
  assign("GRID_EPSG", 32615L, ge); assign("GRID_RESOLUTION_M", 1000, ge)
  assign("CLIP_BOUNDS", c(xmin = 49530.8674345585, xmax = 1417530.86743456,
                          ymin = 3112714.25764237, ymax = 3466714.25764237), ge)
  assign("BASE_MODEL", base_model, ge); assign("OUTPUT_LABEL", "ABC", ge)
  assign("USE_FIXED_SEED", FALSE, ge); assign("RUN_PARALLEL", FALSE, ge)
  run <- function() {
    sys.source(file.path(proj, "Functions", "Initialize.R"), envir = ge)
    sys.source(file.path(proj, "Functions", "PreProcessing.R"), envir = ge)
  }
  if (quiet) invisible(capture.output(run())) else run()
  get("rcs_data", envir = ge)
}

observed_summaries <- function(rcs_data, years) {
  first <- rcs_data$RCS_FirstDetectedYear
  out <- t(vapply(years, function(y) {
    occ <- which(!is.na(first) & first <= y)
    c(area = hull_area(rcs_data$row[occ], rcs_data$col[occ], 1), n_rcs = length(occ),
      west = min(rcs_data$col[occ]), east = max(rcs_data$col[occ]))
  }, numeric(4)))
  rownames(out) <- years
  out
}
