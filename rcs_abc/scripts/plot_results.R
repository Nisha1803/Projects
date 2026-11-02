# Figures for the ABC-SMC synthetic recovery tests.
#
#   Rscript scripts/plot_results.R
#
# Reads results/synthetic_{area,rich}_fit.rds (from run_synthetic_test.R) and
# writes PNGs to results/figures/. For real-data fits, point `fits` at
# results/real_fit.rds and drop the `truth` lines.

source("R/rcs_ca.R")
source("R/abc_smc.R")
set.seed(7)
out <- "results/figures"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# palette: categorical slots 1-3 (validated all-pairs), ink, grid
COL <- list(rich = "#2a78d6", area = "#eb6834", aqua = "#1baf7a",
            ink = "#0b0b0b", ink2 = "#52514e", grid = "#e4e3df",
            prior = "#d9d8d3", land = "#efeeea", water = "#ffffff")

fits <- list(area = readRDS("results/synthetic_area_fit.rds"),
             rich = readRDS("results/synthetic_rich_fit.rds"))
LAB <- c(area = "Area only", rich = "Area + fronts + RCS cells")
truth <- c(b_ph0 = -3.25, b_ph1 = 5.50, b_ph2 = 1.00, b_rcs0 = -7.00,
           b_rcs1 = 8.25, b_rcs2 = 1.75, b_st0 = 4.25, b_st1 = 0.75,
           r_storm = 12)
PRETTY <- c(b_ph0 = "beta_ph0", b_ph1 = "beta_ph1", b_ph2 = "beta_ph2",
            b_rcs0 = "beta_rcs0", b_rcs1 = "beta_rcs1", b_rcs2 = "beta_rcs2",
            b_st0 = "beta_storm0", b_st1 = "beta_storm1", r_storm = "r_storm")
prior <- fits$rich$prior

base_par <- function(...) par(family = "sans", col.axis = COL$ink2,
                              col.lab = COL$ink2, fg = COL$ink2, las = 1,
                              tcl = -0.25, mgp = c(2, 0.5, 0), xpd = FALSE, ...)
open_png <- function(f, w, h) png(file.path(out, f), w, h, res = 150)
legend_top <- function(labels, cols, lty = 1, lwd = 2, pch = NA, inset = -0.01) {
  par(xpd = NA)
  legend("top", inset = c(0, inset), text.width = max(strwidth(labels)), horiz = TRUE, bty = "n", legend = labels,
         col = cols, lty = lty, lwd = lwd, pch = pch, text.col = COL$ink, cex = 0.95)
}

# ---------------------------------------------------------------------------
# 1. Prior vs posterior marginals, with the true value
# ---------------------------------------------------------------------------
open_png("01_posterior_marginals.png", 1800, 1500)
layout(rbind(1, matrix(2:10, 3, byrow = TRUE)), heights = c(0.12, 1, 1, 1))
base_par(mar = c(0, 0, 0, 0)); plot.new()
legend_top(c("Prior (uniform)", LAB[["area"]], LAB[["rich"]], "True value"),
           c(COL$prior, COL$area, COL$rich, COL$ink),
           lty = c(1, 1, 1, 2), lwd = c(8, 2, 2, 1.5))
for (nm in prior$names) {
  base_par(mar = c(3.2, 1, 2, 1))
  lo <- prior$lower[[nm]]; hi <- prior$upper[[nm]]
  dens <- lapply(fits, function(f) {
    x <- f$final$theta[, nm]; w <- f$final$weights
    if (nm == "r_storm") {
      k <- lo:hi; list(x = k, y = sapply(k, function(v) sum(w[x == v])))
    } else density(x, weights = w / sum(w), from = lo, to = hi, bw = "SJ")
  })
  ymax <- max(sapply(dens, function(d) max(d$y)), 1 / (hi - lo)) * 1.08
  plot(NA, xlim = c(lo, hi), ylim = c(0, ymax), axes = FALSE, xlab = "", ylab = "",
       main = PRETTY[[nm]], col.main = COL$ink, font.main = 1, cex.main = 1.2)
  axis(1, col = COL$grid, col.ticks = COL$grid)
  rect(lo, 0, hi, 1 / (hi - lo), col = COL$prior, border = NA)
  for (s in names(fits)) {
    d <- dens[[s]]
    if (nm == "r_storm") lines(d$x, d$y, type = "s", col = COL[[s]], lwd = 2)
    else lines(d$x, d$y, col = COL[[s]], lwd = 2)
  }
  abline(v = truth[[nm]], lty = 2, col = COL$ink, lwd = 1.5)
  abline(h = 0, col = COL$grid)
}
dev.off()

# ---------------------------------------------------------------------------
# 2. How much each parameter was learned (credible-interval shrinkage)
# ---------------------------------------------------------------------------
tabs <- lapply(fits, posterior_table, truth = truth)
open_png("02_shrinkage.png", 1500, 1000)
base_par(mar = c(4, 7, 3, 2))
nms <- rev(prior$names); y <- seq_along(nms)
plot(NA, xlim = c(-0.1, 1), ylim = c(0.5, length(nms) + 0.5), axes = FALSE,
     xlab = "Shrinkage of 90% interval vs prior  (0 = learned nothing, 1 = pinned down)",
     ylab = "")
abline(v = seq(0, 1, 0.25), col = COL$grid)
axis(1, at = seq(0, 1, 0.25), col = NA, col.ticks = NA)
axis(2, at = y, labels = PRETTY[nms], col = NA, col.ticks = NA)
segments(tabs$area[nms, "shrink"], y - 0.12, tabs$rich[nms, "shrink"], y + 0.12,
         col = COL$grid, lwd = 2)
points(tabs$area[nms, "shrink"], y - 0.12, pch = 21, bg = COL$area, col = "white", cex = 1.6)
points(tabs$rich[nms, "shrink"], y + 0.12, pch = 21, bg = COL$rich, col = "white", cex = 1.6)
legend_top(LAB, c(COL$area, COL$rich), lty = NA, pch = 19, inset = -0.1)
dev.off()

# ---------------------------------------------------------------------------
# 3. 90% credible intervals vs truth
# ---------------------------------------------------------------------------
open_png("03_credible_intervals.png", 1800, 1100)
layout(rbind(1, matrix(2:10, 3, byrow = TRUE)), heights = c(0.12, 1, 1, 1))
base_par(mar = c(0, 0, 0, 0)); plot.new()
legend_top(c(paste(LAB, "(90% CI, median)"), "True value"),
           c(COL$area, COL$rich, COL$ink), lty = c(1, 1, 2), lwd = c(3, 3, 1.5))
for (nm in prior$names) {
  base_par(mar = c(3, 1, 2, 1))
  lo <- prior$lower[[nm]]; hi <- prior$upper[[nm]]
  plot(NA, xlim = c(lo, hi), ylim = c(0.5, 3.5), axes = FALSE, xlab = "", ylab = "",
       main = PRETTY[[nm]], col.main = COL$ink, font.main = 1, cex.main = 1.2)
  axis(1, col = COL$grid, col.ticks = COL$grid)
  segments(lo, 3, hi, 3, col = COL$prior, lwd = 6, lend = 1)
  for (k in 1:2) {
    s <- names(fits)[k]; t <- tabs[[s]][nm, ]; yy <- 3 - k
    segments(t$q05, yy, t$q95, yy, col = COL[[s]], lwd = 3, lend = 1)
    points(t$median, yy, pch = 21, bg = COL[[s]], col = "white", cex = 1.5)
  }
  abline(v = truth[[nm]], lty = 2, col = COL$ink, lwd = 1.5)
  text(lo, 3, "prior", pos = 3, cex = 0.8, col = COL$ink2, offset = 0.4)
}
dev.off()

# ---------------------------------------------------------------------------
# 4. Tolerance schedule (convergence)
# ---------------------------------------------------------------------------
open_png("04_tolerance_schedule.png", 1500, 900)
base_par(mar = c(4, 4.5, 3, 2))
sched <- lapply(fits, function(f) {
  h <- f$history[-1]
  data.frame(sims = sapply(h, `[[`, "n_sim"), eps = sapply(h, `[[`, "eps"))
})
plot(NA, xlim = range(unlist(lapply(sched, `[[`, "sims"))), log = "xy",
     ylim = range(unlist(lapply(sched, `[[`, "eps"))), axes = FALSE,
     xlab = "Cumulative simulations", ylab = "Tolerance ε (scaled distance)")
grid(col = COL$grid, lty = 1, equilogs = FALSE)
axis(1, col = NA); axis(2, col = NA)
for (s in names(sched)) {
  lines(sched[[s]]$sims, sched[[s]]$eps, col = COL[[s]], lwd = 2)
  points(sched[[s]]$sims, sched[[s]]$eps, pch = 21, bg = COL[[s]], col = "white", cex = 1.4)
}
legend_top(LAB, c(COL$area, COL$rich), pch = 19)
dev.off()

# ---------------------------------------------------------------------------
# 5. Posterior predictive check: re-simulate posterior draws
# ---------------------------------------------------------------------------
land <- make_synthetic_landscape()
storms <- c(4, 7, 4, 5, 8, 4, 3, 2, 5, 1)
obs_full <- simulate_rcs(truth, land, storms)$summaries   # not the exact obs draw
obs_rich <- matrix(fits$rich$obs, ncol = 4,
                   dimnames = list(NULL, c("area", "n_rcs", "west", "east")))
ppc <- lapply(fits, function(f) {
  idx <- sample(nrow(f$final$theta), 200, TRUE, prob = f$final$weights)
  sims <- lapply(idx, function(i) simulate_rcs(f$final$theta[i, ], land, storms)$summaries)
  sapply(c("area", "west", "east"), function(v) sapply(sims, function(s) s[, v]),
         simplify = FALSE)
})
open_png("05_posterior_predictive.png", 1800, 1250)
layout(rbind(c(1, 1, 1), matrix(2:7, 2, byrow = TRUE)), heights = c(0.1, 1, 1))
base_par(mar = c(0, 0, 0, 0)); plot.new()
legend_top(c("Posterior predictive median", "10-90% band", "Observed (synthetic)"),
           c(COL$ink2, adjustcolor(COL$ink2, 0.3), COL$ink), lty = c(1, 1, NA),
           lwd = c(2, 8, NA), pch = c(NA, NA, 19))
ticks <- 0:length(storms)
ylabs <- c(area = "Encapsulated area (km2)", west = "West front (km)",
           east = "East front (km)")
for (s_row in names(ppc)) for (v in c("area", "west", "east")) {
  base_par(mar = c(4, 4.5, 1.6, 1))
  b <- apply(ppc[[s_row]][[v]], 1, quantile, c(0.1, 0.5, 0.9), na.rm = TRUE)
  yr <- range(unlist(lapply(ppc, function(p) apply(p[[v]], 1, quantile, c(.1, .9), na.rm = TRUE))),
              obs_rich[, v], na.rm = TRUE)
  plot(NA, xlim = range(ticks), ylim = yr,
       axes = FALSE, xlab = "Tick (year)", ylab = ylabs[[v]])
  mtext(sprintf("%s", LAB[[s_row]]), side = 3, line = 0.2, adj = 0, cex = 0.85, col = COL$ink)
  abline(h = pretty(yr), col = COL$grid)
  axis(1, col = NA); axis(2, col = NA)
  polygon(c(ticks, rev(ticks)), c(b[1, ], rev(b[3, ])),
          col = adjustcolor(COL[[s_row]], 0.25), border = NA)
  lines(ticks, b[2, ], col = COL[[s_row]], lwd = 2)
  points(1:length(storms), obs_rich[, v], pch = 21, bg = COL$ink, col = "white", cex = 1.3)
}
dev.off()

# ---------------------------------------------------------------------------
# 6. The b_rcs0 / b_rcs1 ridge: data fixes a combination, not each coefficient
# ---------------------------------------------------------------------------
open_png("06_rcs_ridge.png", 1300, 1100)
base_par(mar = c(4.5, 4.5, 3, 2))
th <- fits$rich$final$theta; w <- fits$rich$final$weights
plot(NA, xlim = c(prior$lower[["b_rcs0"]], prior$upper[["b_rcs0"]]),
     ylim = c(prior$lower[["b_rcs1"]], prior$upper[["b_rcs1"]]), axes = FALSE,
     xlab = "beta_rcs0 (intercept)", ylab = "beta_rcs1 (MaxEnt slope)")
abline(v = pretty(par("usr")[1:2]), h = pretty(par("usr")[3:4]), col = COL$grid)
axis(1, col = NA); axis(2, col = NA)
points(th[, "b_rcs0"], th[, "b_rcs1"], pch = 21, col = "white",
       bg = adjustcolor(COL$rich, 0.65), cex = 0.6 + 2.5 * sqrt(w / max(w)))
points(truth[["b_rcs0"]], truth[["b_rcs1"]], pch = 4, col = COL$ink, lwd = 2.5, cex = 2)
r <- cov.wt(th[, c("b_rcs0", "b_rcs1")], wt = w / sum(w), cor = TRUE)$cor[1, 2]
mtext(sprintf("Posterior particles (rich summaries); weighted correlation = %.2f;  x = truth", r),
      side = 3, line = 1, col = COL$ink, cex = 0.95)
dev.off()

# ---------------------------------------------------------------------------
# 7. What the CA does: maps at the true parameters
# ---------------------------------------------------------------------------
set.seed(3)
sim <- simulate_rcs(truth, land, storms, return_maps = TRUE)
open_png("07_ca_maps.png", 1800, 800)
layout(matrix(c(1, 1, 1, 2, 3, 4), 2, byrow = TRUE), heights = c(0.12, 1))
base_par(mar = c(0, 0, 0, 0)); plot.new()
legend_top(c("Land, uncolonized", "Phragmites", "RCS", "Water"),
           c(COL$land, COL$aqua, COL$rich, "#d3dbe3"), lty = NA, pch = 15)
for (t in c(0, 5, 10)) {
  base_par(mar = c(1, 1, 2.2, 1))
  m <- sim$maps[[t + 1]]
  z <- ifelse(land$mask, m + 1, 0)        # 0 water, 1 land, 2 phrag, 3 RCS
  image(t(z[nrow(z):1, ]), col = c("#d3dbe3", COL$land, COL$aqua, COL$rich),
        breaks = c(-0.5, 0.5, 1.5, 2.5, 3.5), axes = FALSE, useRaster = TRUE)
  box(col = "#bdbcb6")
  title(sprintf("Tick %d  -  RCS hull area %.0f km2", t, sim$summaries[t + 1, "area"]),
        col.main = COL$ink, font.main = 1, cex.main = 1.1)
}
dev.off()

message("wrote ", length(list.files(out, "png$")), " figures to ", out)
