# Checks the fast diamond sum equals Ethan's original exactly.
#   Rscript tests/test_fast_diamond.R ../Functions
args <- commandArgs(trailingOnly = TRUE); func_dir <- if (length(args)) args[1] else "../Functions"
source("R/fast_diamond.R")
env <- new.env()
txt <- readLines(file.path(func_dir, "CellularAutomata.R"))
s <- grep("^\\.manhattan_diamond_sum_grid <- function", txt); e <- s + which(txt[(s + 1):length(txt)] == "}")[1]
eval(parse(text = txt[s:e]), envir = env)
set.seed(42); fails <- 0
for (case in 1:200) {
  nr <- sample(1:40, 1); nc <- sample(1:60, 1); rad <- sample(1:45, 1)
  x <- as.numeric(runif(nr * nc) < runif(1))
  a <- env$.manhattan_diamond_sum_grid(x, rad, nr, nc)
  b <- manhattan_diamond_sum_fast(x, rad, nr, nc)
  if (!isTRUE(all.equal(a, b, tolerance = 0))) { fails <- fails + 1; cat("MISMATCH", nr, nc, rad, "\n") }
}
nr <- 354; nc <- 1368; x <- as.numeric(runif(nr * nc) < 0.05)
ta <- system.time(a <- env$.manhattan_diamond_sum_grid(x, 37, nr, nc))[["elapsed"]]
invisible(manhattan_diamond_sum_fast(x, 37, nr, nc))  # builds the cached index once
tb <- system.time(b <- manhattan_diamond_sum_fast(x, 37, nr, nc))[["elapsed"]]
cat(sprintf("random grids: %d / 200 mismatches | full grid r=37 identical: %s | original %.2f s, fast %.2f s\n",
            fails, identical(a, b), ta, tb))
if (fails > 0 || !identical(a, b)) quit(status = 1)
