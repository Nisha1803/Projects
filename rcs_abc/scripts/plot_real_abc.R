# Figures + posterior table for the real-data ABC run on Ethan's model.
#
#   Rscript scripts/plot_real_abc.R /path/to/ethan_project
#
# Reads results/ethan/abc_checkpoint.rds (last completed generation), rebuilds
# the simulator exactly as run_abc_ethan.R does, and writes
#   results/ethan/abc_posterior.csv
#   results/ethan/figures/{01..05}_*.png
# Shrinkage is measured against the TRUNCATED prior (constraint applied), not
# the bounding box, so a parameter the data says nothing about scores ~0.
Sys.setenv(ABC_SETUP_ONLY = "1")
source("scripts/run_abc_ethan.R")      # gives prior, rcs_data, storms, stats, obs, ...
set.seed(11)
n_pp <- as.integer(Sys.getenv("PP_DRAWS", "60"))
out <- "results/ethan/figures"; dir.create(out, recursive = TRUE, showWarnings = FALSE)

ck <- readRDS("results/ethan/abc_checkpoint.rds")
H <- ck$history; last <- H[[length(H)]]
X <- last$theta; w <- last$weights / sum(last$weights)
P <- prior$sample(20000)
nms <- colnames(X)

COL <- list(post = "#2a78d6", prior = "#d9d8d3", prior_line = "#8a8984",
            obs = "#0b0b0b", ink2 = "#52514e", grid = "#e4e3df", pp_prior = "#eb6834")
PRETTY <- c(b_ph0 = "beta_ph0 (Phrag intercept)", b_ph1 = "beta_ph1 (Phrag MaxEnt)",
            b_ph2 = "beta_ph2 (Phrag source)", b_rcs0 = "beta_rcs0 (RCS intercept)",
            b_rcs1 = "beta_rcs1 (RCS MaxEnt)", b_rcs2 = "beta_rcs2 (RCS source)",
            b_st0 = "beta_storm0", b_st1 = "beta_storm1 (NRI)", r_storm = "r_storm (km)")
base_par <- function(...) par(family = "sans", col.axis = COL$ink2, col.lab = COL$ink2,
                              fg = COL$ink2, las = 1, tcl = -0.25, mgp = c(2, 0.5, 0),
                              xpd = FALSE, ...)
open_png <- function(f, w, h) png(file.path(out, f), w, h, res = 150)

# ---- posterior table ----
q <- function(x, w, p) wquantile(x, w, p)
tab <- t(sapply(nms, function(nm) {
  pq <- quantile(P[, nm], c(.05, .95)); qq <- sapply(c(.05, .5, .95), function(p) q(X[, nm], w, p))
  c(post_median = qq[2], post_q05 = qq[1], post_q95 = qq[3],
    post_mean = sum(w * X[, nm]), prior_q05 = pq[[1]], prior_q95 = pq[[2]],
    shrink = 1 - (qq[3] - qq[1]) / (pq[[2]] - pq[[1]]))
}))
write.csv(round(tab, 4), "results/ethan/abc_posterior.csv")
cat("\nPosterior after generation", last$gen, "(eps =", round(last$eps, 3), ",",
    last$n_sim, "simulations, ESS =", round(last$ess), ")\n"); print(round(tab, 2))
C <- cov2cor(cov.wt(X, w)$cov); write.csv(round(C, 3), "results/ethan/abc_posterior_cor.csv")

# ---- 1. marginals ----
open_png("01_posterior_marginals.png", 1800, 1450)
layout(rbind(1, matrix(2:10, 3, byrow = TRUE)), heights = c(0.1, 1, 1, 1))
base_par(mar = c(0, 0, 0, 0)); plot.new()
legend("center", horiz = TRUE, bty = "n", legend = c("Prior (truncated uniform)", paste("Posterior, gen", last$gen)),
       col = c(COL$prior, COL$post), lwd = c(8, 2.5), cex = 1)
for (nm in nms) {
  base_par(mar = c(3.2, 1, 2, 1))
  lo <- prior$lower[[nm]]; hi <- prior$upper[[nm]]
  if (nm == "r_storm") {
    k <- lo:hi; dp <- list(x = k, y = tabulate(P[, nm] - lo + 1, length(k)) / nrow(P))
    dq <- list(x = k, y = sapply(k, function(v) sum(w[X[, nm] == v])))
  } else {
    dp <- density(P[, nm], from = lo, to = hi); dq <- density(X[, nm], weights = w, from = lo, to = hi)
  }
  ym <- max(dp$y, dq$y) * 1.1
  plot(NA, xlim = c(lo, hi), ylim = c(0, ym), axes = FALSE, xlab = "", ylab = "",
       main = PRETTY[[nm]], font.main = 1, cex.main = 1.05)
  abline(v = pretty(c(lo, hi)), col = COL$grid)
  if (nm == "r_storm") {
    segments(dp$x, 0, dp$x, dp$y, col = COL$prior, lwd = 4)
    points(dq$x, dq$y, type = "h", col = COL$post, lwd = 2)
  } else {
    polygon(c(dp$x, rev(dp$x)), c(dp$y, 0 * dp$y), col = COL$prior, border = NA)
    lines(dq, col = COL$post, lwd = 2.5)
  }
  axis(1); mtext(sprintf("shrink %.2f", tab[nm, "shrink"]), 3, -1.2, adj = 0.98, cex = 0.75, col = COL$ink2)
}
dev.off()

# ---- 2. shrinkage ----
open_png("02_shrinkage.png", 1500, 950)
base_par(mar = c(4, 13, 2, 2))
o <- order(tab[, "shrink"]); s <- tab[o, "shrink"]
barplot(s, horiz = TRUE, names.arg = PRETTY[rownames(tab)[o]], col = COL$post, border = NA,
        xlim = c(min(0, s) - 0.05, 1), xlab = "Shrinkage of the 90% interval vs the prior (0 = learned nothing, 1 = pinned)",
        cex.names = 0.9)
abline(v = 0, col = COL$ink2)
dev.off()

# ---- 3. tolerance schedule ----
open_png("03_tolerance_schedule.png", 1500, 850)
base_par(mar = c(4, 4, 2, 2))
eps <- sapply(H, `[[`, "eps"); ns <- sapply(H, `[[`, "n_sim"); acc <- sapply(H, function(h) h$acc_rate %||% NA)
plot(ns, eps, type = "b", pch = 19, col = COL$post, log = "y", xlab = "Cumulative simulations",
     ylab = "Tolerance eps (log scale)")
text(ns, eps, ifelse(is.na(acc), "", sprintf("acc %.0f%%", 100 * acc)), pos = 4, cex = 0.8, col = COL$ink2)
dev.off()

# ---- 4. posterior + prior predictive, 2017-2025 (2024/2025 and east front not fitted) ----
pp_years <- 2017:2025
sim_pp <- make_ethan_simulator(file.path(proj, "Functions"), rcs_data, start_year = START_YEAR,
                               intro_year = RCS_INTRO_YEAR, obs_years = pp_years,
                               storms = storm_schedule(mode = "calendar", start_year = START_YEAR,
                                                       n_ticks = max(pp_years) - START_YEAR, pre_2016 = 0L),
                               base_model = BASE_MODEL)
draw_post <- X[sample(nrow(X), n_pp, replace = TRUE, prob = w), , drop = FALSE]
draw_prior <- prior$sample(n_pp)
run <- function(M) par_map(seq_len(nrow(M)), function(i) sim_pp(M[i, ]), ncores)
t0 <- Sys.time(); pp_post <- run(draw_post); pp_prior <- run(draw_prior)
cat("predictive sims:", round(difftime(Sys.time(), t0, units = "mins"), 1), "min\n")
saveRDS(list(post = pp_post, prior = pp_prior, years = pp_years), "results/ethan/predictive.rds")
obs_pp <- observed_summaries(rcs_data, pp_years)

band <- function(L, v) apply(sapply(L, function(s) s[, v]), 1, quantile, c(.05, .5, .95), na.rm = TRUE)
open_png("04_posterior_predictive.png", 1800, 1300)
layout(rbind(c(1, 1), c(2, 3), c(4, 5)), heights = c(0.12, 1, 1)); base_par(mar = c(0, 0, 0, 0)); plot.new()
legend("center", horiz = TRUE, bty = "n", cex = 0.95,
       legend = c("Prior predictive 90%", "Posterior predictive 90%", "Posterior median", "Observed (fitted)", "Observed (not fitted)"),
       col = c(COL$pp_prior, COL$post, COL$post, COL$obs, COL$obs), lwd = c(8, 8, 2, NA, NA), pch = c(NA, NA, NA, 19, 1))
labs <- c(area = "RCS hull area (km2)", n_rcs = "RCS-occupied cells", west = "West front (km)", east = "East front (km)")
for (v in names(labs)) {
  base_par(mar = c(3.5, 5, 2, 1))
  bp <- band(pp_prior, v); bq <- band(pp_post, v)
  yl <- range(bp, bq, obs_pp[, v], na.rm = TRUE)
  plot(NA, xlim = range(pp_years), ylim = yl, xlab = "", ylab = "", main = labs[[v]], font.main = 1)
  abline(h = pretty(yl), col = COL$grid)
  polygon(c(pp_years, rev(pp_years)), c(bp[1, ], rev(bp[3, ])), col = adjustcolor(COL$pp_prior, 0.25), border = NA)
  polygon(c(pp_years, rev(pp_years)), c(bq[1, ], rev(bq[3, ])), col = adjustcolor(COL$post, 0.45), border = NA)
  lines(pp_years, bq[2, ], col = COL$post, lwd = 2)
  fitted <- v %in% stats & pp_years %in% obs_years
  points(pp_years, obs_pp[, v], pch = ifelse(fitted, 19, 1), col = COL$obs, cex = 1.2)
}
dev.off()

# ---- 5. ridges: the data fixes combinations, not single coefficients ----
open_png("05_parameter_ridges.png", 1800, 900)
par(mfrow = c(1, 2))
for (pr in list(c("b_ph0", "b_ph1"), c("b_rcs0", "b_rcs1"))) {
  base_par(mar = c(4, 4.5, 2.5, 1))
  plot(P[1:3000, pr[1]], P[1:3000, pr[2]], pch = 16, cex = 0.4, col = COL$prior,
       xlab = PRETTY[[pr[1]]], ylab = PRETTY[[pr[2]]],
       main = sprintf("weighted correlation %.2f", C[pr[1], pr[2]]), font.main = 1)
  points(X[, pr[1]], X[, pr[2]], pch = 19, col = adjustcolor(COL$post, 0.7), cex = 0.4 + 8 * w)
}
dev.off()
cat("figures written to", out, "\n")
