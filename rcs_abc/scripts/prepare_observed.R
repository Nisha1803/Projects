# Turn the occurrence file into the observed summary statistics ABC fits to.
#
#   Rscript scripts/prepare_observed.R [path/to/RCS_Phragmites_Data_10_17_26.csv]
#
# Uses the same grid as RCS_CA_Procedure.R: EPSG:32615, 1 km cells, CLIP_BOUNDS.
# RCS points are cumulative: a cell detected in year y counts in every later
# year (the manuscript reports RCS is always re-detected once present).
# Writes results/observed/observed_rcs_by_year.csv and a map + time-series figure.

suppressPackageStartupMessages(library(sf))
source("R/rcs_ca.R")                       # hull_area()

args <- commandArgs(trailingOnly = TRUE)
occ_file <- if (length(args)) args[1] else "data/RCS_Phragmites_Data_10_17_26.csv"
CLIP <- c(xmin = 49530.8674345585, xmax = 1417530.86743456,
          ymin = 3112714.25764237, ymax = 3466714.25764237)
RES <- 1000
out <- "results/observed"; dir.create(out, recursive = TRUE, showWarnings = FALSE)

d <- read.csv(occ_file)
d <- d[!is.na(d$Year) & !duplicated(d), ]
xy <- st_coordinates(st_transform(st_as_sf(d, coords = c("X", "Y"), crs = 4326), 32615))
d$E <- xy[, 1]; d$N <- xy[, 2]
inside <- d$E >= CLIP["xmin"] & d$E < CLIP["xmax"] & d$N >= CLIP["ymin"] & d$N < CLIP["ymax"]
d$col <- floor((d$E - CLIP["xmin"]) / RES) + 1           # 1 = western edge
d$row <- floor((CLIP["ymax"] - d$N) / RES) + 1           # 1 = northern edge

rcs <- d[d$Species == "RCS", ]
cat(sprintf("RCS records: %d total, %d inside the CA clip, %d outside (dropped)\n",
            nrow(rcs), sum(inside[d$Species == "RCS"]), sum(!inside[d$Species == "RCS"])))
rcs <- rcs[inside[d$Species == "RCS"], ]

years <- sort(unique(rcs$Year))
obs <- do.call(rbind, lapply(years, function(y) {
  s <- unique(rcs[rcs$Year <= y, c("row", "col")])
  new <- unique(rcs[rcs$Year == y, c("row", "col")])
  data.frame(year = y, records_this_year = sum(rcs$Year == y),
             cum_cells = nrow(s),
             area = hull_area(s$row, s$col, RES / 1000),
             west = min(s$col) * RES / 1000, east = max(s$col) * RES / 1000)
}))
print(obs, row.names = FALSE)
write.csv(obs, file.path(out, "observed_rcs_by_year.csv"), row.names = FALSE)

# Phragmites records per year inside the clip: sampling effort, not spread
ph <- d[d$Species == "Phragmites" & inside, ]
ph_tab <- as.data.frame(table(year = ph$Year))
write.csv(ph_tab, file.path(out, "phragmites_records_by_year.csv"), row.names = FALSE)

# ---- figure ----
COL <- c("2017" = "#2a78d6", "2021" = "#eb6834", "2022" = "#1baf7a",
         "2023" = "#6250d6", "2025" = "#e34948")
png(file.path(out, "observed_rcs.png"), 1800, 1200, res = 150)
layout(matrix(c(1, 1, 2, 3), 2, byrow = TRUE), heights = c(1.1, 1))
par(family = "sans", las = 1, col.axis = "#52514e", col.lab = "#52514e", fg = "#52514e",
    tcl = -0.25, mgp = c(3.6, 0.5, 0), mar = c(4.5, 5.5, 3, 3), xpd = FALSE)
xr <- range(rcs$col) + c(-20, 20); yr <- range(rcs$row) + c(-20, 20)
phw <- ph[ph$col >= xr[1] & ph$col <= xr[2] & ph$row >= yr[1] & ph$row <= yr[2], ]
plot(phw$col, -phw$row, pch = 16, cex = 0.35, col = "#c9c8c2", xlim = xr, ylim = -rev(yr),
     asp = 1, axes = FALSE, xlab = "km east of clip edge", ylab = "km south of clip top")
axis(1, col = NA); axis(2, at = pretty(-yr), labels = -pretty(-yr), col = NA)
for (y in names(COL)) {
  s <- rcs[rcs$Year == as.numeric(y), ]
  points(s$col, -s$row, pch = 21, bg = COL[[y]], col = "white", cex = 1.1)
}
mtext("RCS first detections by survey year (grey = Phragmites records)", 3, 1, adj = 0, col = "#0b0b0b")
legend("bottomleft", c(names(COL), "Phragmites"), pt.bg = c(COL, "#c9c8c2"),
       col = c(rep("white", 5), "#c9c8c2"), pch = c(rep(21, 5), 16), bty = "n",
       text.col = "#0b0b0b", ncol = 2)
plot(obs$year, obs$area / 1000, type = "b", pch = 21, bg = "#2a78d6", col = "#2a78d6", lwd = 2,
     xlab = "Survey year", ylab = "Hull area (1000 km2)", axes = FALSE)
abline(h = pretty(obs$area / 1000), col = "#e4e3df"); axis(1, at = obs$year, col = NA); axis(2, col = NA)
mtext("Observed RCS hull area", 3, 0.5, adj = 0, col = "#0b0b0b")
plot(NA, xlim = range(obs$year), ylim = range(c(obs$west, obs$east)), axes = FALSE,
     xlab = "Survey year", ylab = "Front position (km east of clip edge)")
abline(h = pretty(c(obs$west, obs$east)), col = "#e4e3df")
axis(1, at = obs$year, col = NA); axis(2, col = NA)
lines(obs$year, obs$west, type = "b", pch = 21, bg = "#eb6834", col = "#eb6834", lwd = 2)
lines(obs$year, obs$east, type = "b", pch = 21, bg = "#2a78d6", col = "#2a78d6", lwd = 2)
text(max(obs$year), tail(obs$east, 1), "east", pos = 4, col = "#0b0b0b", xpd = NA)
text(max(obs$year), tail(obs$west, 1), "west", pos = 4, col = "#0b0b0b", xpd = NA)
mtext("Observed range edges", 3, 0.5, adj = 0, col = "#0b0b0b")
dev.off()
