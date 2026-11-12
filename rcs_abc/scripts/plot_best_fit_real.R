# Figure for scripts/check_best_fit_real.R output.
start <- Sys.getenv("START_YEAR", "2000")
x <- readRDS(sprintf("results/ethan/check_best_fit_real_start%s.rds", start))
COL <- c(as_coded = "#eb6834", calendar = "#2a78d6")
LAB <- c(as_coded = "Storms as coded (1/yr after 2010)", calendar = "Storms on stated years (2001-15 = 4, placeholder)")
yrs <- as.numeric(rownames(x$obs))
png(sprintf("results/ethan/best_fit_real_start%s.png", start), 1800, 700, res = 150)
layout(matrix(c(1, 1, 1, 2, 3, 4), 2, byrow = TRUE), heights = c(0.14, 1))
par(mar = c(0, 0, 0, 0)); plot.new()
legend("center", horiz = TRUE, bty = "n", legend = c(LAB, "Observed"), col = c(COL, "#0b0b0b"),
       lty = c(1, 1, NA), lwd = 2, pch = c(NA, NA, 19), text.col = "#0b0b0b")
spec <- list(area = c(1e-3, "RCS hull area (1000 km2)"), east = c(1, "East front (km from grid edge)"),
             west = c(1, "West front (km from grid edge)"))
for (v in names(spec)) {
  par(mar = c(4, 4.8, 1.5, 1), las = 1, col.axis = "#52514e", col.lab = "#52514e", fg = "#52514e",
      tcl = -0.25, mgp = c(3.2, 0.5, 0))
  k <- as.numeric(spec[[v]][1])
  runs <- lapply(x$res, function(r) sapply(r, function(o) o$S[, v] * k))
  yl <- range(unlist(runs), x$obs[, v] * k)
  plot(NA, xlim = range(yrs), ylim = yl, axes = FALSE, xlab = "Year", ylab = spec[[v]][2])
  abline(h = pretty(yl), col = "#e4e3df"); axis(1, at = yrs, col = NA); axis(2, col = NA)
  for (s in names(runs)) matlines(yrs, runs[[s]], col = adjustcolor(COL[[s]], 0.45), lty = 1, lwd = 1.5)
  points(yrs, x$obs[, v] * k, pch = 21, bg = "#0b0b0b", col = "white", cex = 1.4)
}
dev.off()
