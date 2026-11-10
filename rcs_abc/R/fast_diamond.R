# Drop-in replacement for Ethan's .manhattan_diamond_sum_grid() (CellularAutomata.R).
# Same inputs/outputs: for each cell, the sum of `values` over the full Manhattan
# diamond |dr| + |dc| <= radius (focal cell included), grid clipped at edges,
# cells in row-major order. Instead of one pass per row offset (cost ~ N * radius),
# it rotates the grid 45 degrees (u = r + c, v = r - c), where the diamond becomes
# an axis-aligned square, and uses one summed-area table (cost ~ N, any radius).
# Index arrays are cached per (grid, radius). Exactness is checked against the
# original in tests/test_fast_diamond.R. Ethan only calls this on 0/1 counts,
# where double-precision cumulative sums are exact.

.diamond_cache <- new.env()

# Column-wise cumulative sum in one vectorized pass.
.colcumsum <- function(X) {
  n <- nrow(X); cs <- cumsum(as.vector(X))
  base <- c(0, cs[seq.int(n, length.out = ncol(X) - 1L, by = n)])
  matrix(cs - rep(base, each = n), n)
}

.diamond_index <- function(nrow_grid, ncol_grid, radius) {
  key <- paste(nrow_grid, ncol_grid, radius)
  if (!is.null(.diamond_cache[[key]])) return(.diamond_cache[[key]])
  r <- rep(seq_len(nrow_grid), each = ncol_grid)              # row-major order
  c <- rep.int(seq_len(ncol_grid), nrow_grid)
  u <- r + c - 1L; v <- r - c + ncol_grid
  M <- nrow_grid + ncol_grid - 1L; P1 <- M + 1L
  ulo <- pmax(1L, u - radius); uhi <- pmin(M, u + radius) + 1L
  vlo <- pmax(1L, v - radius); vhi <- pmin(M, v + radius) + 1L
  lin <- function(i, j) (j - 1L) * P1 + i                     # linear index into P
  idx <- list(M = M, cell = (v - 1L) * M + u,
              a = lin(uhi, vhi), b = lin(ulo, vhi), c = lin(uhi, vlo), d = lin(ulo, vlo))
  .diamond_cache[[key]] <- idx
  idx
}

manhattan_diamond_sum_fast <- function(values, radius, nrow_grid, ncol_grid) {
  ix <- .diamond_index(as.integer(nrow_grid), as.integer(ncol_grid), as.integer(radius))
  R <- matrix(0, ix$M, ix$M)
  R[ix$cell] <- values
  P <- matrix(0, ix$M + 1L, ix$M + 1L)
  P[-1L, -1L] <- t(.colcumsum(t(.colcumsum(R))))
  P[ix$a] - P[ix$b] - P[ix$c] + P[ix$d]
}
