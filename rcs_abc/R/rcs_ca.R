# Reference cellular automaton for Phragmites / Roseau cane scale (RCS) spread.
#
# Written from the equations in "RCS Watch" (Sec. 2.5) and the PhragODE notes.
# It is a stand-in: if your own simulator differs, wrap YOUR simulator in a
# function with the same signature as `simulate_rcs()` and the ABC code works
# unchanged.
#
# States: 0 = uncolonized, 1 = Phragmites-colonized, 2 = RCS-colonized.
# Transitions only move forward (0 -> 1, 0 -> 2, 1 -> 2).
#
# ASSUMPTIONS that you must check against your own code (the paper is ambiguous):
#   A1. Local spread uses the 4-cell (von Neumann) neighbourhood ("diagonal
#       spread is not allowed"); storm spread uses the full square window of
#       Chebyshev radius r_storm ("diagonal spread is allowed"), excluding self.
#   A2. Local spread needs at least one colonized neighbour (require_source).
#       Without this, beta_ph2 = 0 (the paper's best fit) would make every
#       suitable cell sprout Phragmites spontaneously each tick.
#   A3. RCS cells count as Phragmites for the Phragmites source term (RCS lives
#       on Phragmites).
#   A4. Source pressure S = colonized neighbours / eligible (land) neighbours.
#   A5. Coupled storm events can hit uncolonized AND Phragmites cells; a cell
#       that receives RCS in a tick ends the tick as RCS.
#   A6. All probabilities are computed from the state at the start of the tick
#       (synchronous update).

PARAM_NAMES <- c("b_ph0", "b_ph1", "b_ph2",
                 "b_rcs0", "b_rcs1", "b_rcs2",
                 "b_st0", "b_st1", "r_storm")

# ---------------------------------------------------------------------------
# Neighbourhood helpers
# ---------------------------------------------------------------------------

# Sum of the 4 von Neumann neighbours of every cell (zero padding).
neigh4_sum <- function(M) {
  nr <- nrow(M); nc <- ncol(M)
  out <- matrix(0, nr, nc)
  out[-1, ]  <- out[-1, ]  + M[-nr, ]
  out[-nr, ] <- out[-nr, ] + M[-1, ]
  out[, -1]  <- out[, -1]  + M[, -nc]
  out[, -nc] <- out[, -nc] + M[, -1]
  out
}

# Sum over the (2r+1) x (2r+1) square window around every cell, O(N) via a
# summed-area table. Window is clipped at the grid edge.
box_sum <- function(M, r) {
  nr <- nrow(M); nc <- ncol(M)
  C <- apply(M, 2, cumsum)
  if (nr == 1) C <- matrix(C, 1)
  C <- t(apply(C, 1, cumsum))
  if (nc == 1) C <- matrix(C, ncol = 1)
  P <- matrix(0, nr + 1, nc + 1)
  P[-1, -1] <- C
  rlo <- pmax(1, seq_len(nr) - r); rhi <- pmin(nr, seq_len(nr) + r)
  clo <- pmax(1, seq_len(nc) - r); chi <- pmin(nc, seq_len(nc) + r)
  P[rhi + 1, chi + 1] - P[rlo, chi + 1] - P[rhi + 1, clo] + P[rlo, clo]
}

# ---------------------------------------------------------------------------
# Summary statistics of one RCS map
# ---------------------------------------------------------------------------

# Convex-hull ("encapsulated") area of the RCS-colonized cells, in km^2.
hull_area <- function(rows, cols, cell_km = 1) {
  n <- length(rows)
  if (n == 0) return(0)
  if (n < 3) return(n * cell_km^2)
  # use cell corners so a line of cells still has area
  x <- c(cols - 0.5, cols + 0.5, cols - 0.5, cols + 0.5)
  y <- c(rows - 0.5, rows - 0.5, rows + 0.5, rows + 0.5)
  h <- grDevices::chull(x, y)
  xs <- x[h]; ys <- y[h]
  abs(sum(xs * c(ys[-1], ys[1]) - c(xs[-1], xs[1]) * ys)) / 2 * cell_km^2
}

map_summaries <- function(state, cell_km = 1) {
  idx <- which(state == 2L, arr.ind = TRUE)
  if (nrow(idx) == 0) {
    return(c(area = 0, n_rcs = 0, west = NA_real_, east = NA_real_,
             n_ph = sum(state >= 1L)))
  }
  c(area  = hull_area(idx[, 1], idx[, 2], cell_km),
    n_rcs = nrow(idx) * cell_km^2,
    west  = min(idx[, 2]) * cell_km,
    east  = max(idx[, 2]) * cell_km,
    n_ph  = sum(state >= 1L) * cell_km^2)
}

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

#' @param theta    named numeric vector with PARAM_NAMES
#' @param land     list with matrices: mask (TRUE = eligible land cell),
#'                 M_ph, M_rcs (MaxEnt 0-1), nri (normalized composite NRI),
#'                 init (integer state matrix at tick 0), cell_km
#' @param storms   integer vector, storms per tick (length = number of ticks)
#' @param require_source assumption A2
#' @param return_maps keep the full state at every tick (memory heavy)
#' @return list(summaries = matrix ticks x stats, maps = optional list)
simulate_rcs <- function(theta, land, storms, require_source = TRUE,
                         return_maps = FALSE) {
  th <- as.list(theta)
  r <- as.integer(round(th$r_storm))
  mask <- land$mask
  maskn <- mask * 1
  state <- land$init
  state[!mask] <- 0L
  n_ticks <- length(storms)

  # Pieces that do not change over time
  lp_ph_hab  <- th$b_ph0 + th$b_ph1 * land$M_ph
  lp_rcs_hab <- th$b_rcs0 + th$b_rcs1 * land$M_rcs
  q_ph  <- plogis(lp_ph_hab)
  q_rcs <- plogis(lp_rcs_hab)
  q_nri <- plogis(th$b_st0 + th$b_st1 * land$nri)
  n_land4 <- pmax(neigh4_sum(maskn), 1)
  n_landR <- pmax(box_sum(maskn, r) - maskn, 1)

  stats <- matrix(NA_real_, n_ticks + 1, 5)
  colnames(stats) <- names(map_summaries(state))
  stats[1, ] <- map_summaries(state, land$cell_km)
  maps <- if (return_maps) list(state) else NULL

  for (t in seq_len(n_ticks)) {
    ph <- (state >= 1L) * 1
    rc <- (state == 2L) * 1
    nb_ph <- neigh4_sum(ph)
    nb_rc <- neigh4_sum(rc)

    # local spread
    p_ph  <- plogis(lp_ph_hab  + th$b_ph2  * nb_ph / n_land4)
    p_rcs <- plogis(lp_rcs_hab + th$b_rcs2 * nb_rc / n_land4)
    if (require_source) {
      p_ph  <- p_ph  * (nb_ph > 0)
      p_rcs <- p_rcs * (nb_rc > 0)
    }

    # storm-mediated jump spread
    nt <- storms[t]
    if (nt > 0) {
      S_ph_rad  <- (box_sum(ph, r) - ph) / n_landR
      S_rcs_rad <- (box_sum(rc, r) - rc) / n_landR
      pj_ph  <- 1 - (1 - q_ph * q_nri * S_ph_rad)^nt
      pj_rcs <- 1 - (1 - q_ph * q_rcs * q_nri * S_rcs_rad)^nt
    } else {
      pj_ph <- pj_rcs <- 0
    }

    u1 <- matrix(runif(length(state)), nrow(state))
    u2 <- matrix(runif(length(state)), nrow(state))
    is0 <- mask & state == 0L
    is1 <- mask & state == 1L

    to_rcs_from0 <- is0 & (u1 < pj_rcs)
    to_ph_from0  <- is0 & !to_rcs_from0 & (u2 < 1 - (1 - p_ph) * (1 - pj_ph))
    to_rcs_from1 <- is1 & (u1 < 1 - (1 - p_rcs) * (1 - pj_rcs))

    state[to_ph_from0] <- 1L
    state[to_rcs_from0 | to_rcs_from1] <- 2L

    stats[t + 1, ] <- map_summaries(state, land$cell_km)
    if (return_maps) maps[[t + 1]] <- state
  }
  list(summaries = stats, maps = maps)
}

# ---------------------------------------------------------------------------
# Synthetic coastal landscape for testing (replace with Southeast_Data_Clip)
# ---------------------------------------------------------------------------
make_synthetic_landscape <- function(nr = 100, nc = 220, seed = 1) {
  set.seed(seed)
  cols <- seq_len(nc); rows <- seq_len(nr)
  coast <- round(nr * 0.55 + 6 * sin(cols / 17) + 3 * sin(cols / 5))
  R <- matrix(rows, nr, nc); Cc <- matrix(cols, nr, nc, byrow = TRUE)
  coastM <- matrix(coast, nr, nc, byrow = TRUE)
  mask <- R <= coastM                      # land above the coastline
  dist_coast <- pmax(coastM - R, 0)
  # Phragmites suitability: high in a coastal fringe
  M_ph <- pmin(1, exp(-dist_coast / 8) * (0.7 + 0.3 * runif(nr * nc)))
  # RCS suitability: coastal and declining to the east (colder)
  M_rcs <- pmin(1, exp(-dist_coast / 15) * (1 - 0.5 * Cc / nc))
  nri <- pmin(1, exp(-dist_coast / 20) * (0.6 + 0.4 * runif(nr * nc)))
  M_ph[!mask] <- 0; M_rcs[!mask] <- 0; nri[!mask] <- 0

  init <- matrix(0L, nr, nc)
  # existing Phragmites stands along the coast
  init[mask & dist_coast < 4 & runif(nr * nc) < 0.25] <- 1L
  # RCS introduction: a small cluster near column 70
  seedc <- 68:72
  for (cc in seedc) {
    rr <- coast[cc] - 0:2
    init[rr, cc] <- 2L
  }
  list(mask = mask, M_ph = M_ph, M_rcs = M_rcs, nri = nri,
       init = init, cell_km = 1)
}
