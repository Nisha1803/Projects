# ABC-SMC (population Monte Carlo ABC) in base R.
#
# Algorithm: Beaumont et al. (2009) / Toni et al. (2009) with an adaptive
# tolerance schedule (Del Moral et al. 2012 style: each new epsilon is the
# `alpha` quantile of the previous population's distances) and the
# "2 x weighted covariance" Gaussian perturbation kernel. Integer parameters
# (r_storm) are perturbed with a discretized normal kernel.
#
# The simulator is a black box: any function(theta) -> numeric summary vector
# works, so you can swap in your own R simulator.

# ---------------------------------------------------------------------------
# Prior: independent uniforms (optionally integer-valued)
# ---------------------------------------------------------------------------
make_uniform_prior <- function(lower, upper, integer = character(0),
                               constraint = NULL) {
  # constraint: optional function(named theta) -> TRUE/FALSE. The prior is the
  # uniform box truncated to where it is TRUE (e.g. HMParameters.R's
  # constraint_reasonable_logits). Truncation only rescales a uniform density,
  # so the ABC weights need no other change.
  stopifnot(identical(names(lower), names(upper)), all(upper > lower))
  is_int <- names(lower) %in% integer
  ok <- function(theta) is.null(constraint) || isTRUE(constraint(theta))
  draw_box <- function(n) {
    m <- sapply(seq_along(lower), function(k) {
      if (is_int[k]) sample(lower[k]:upper[k], n, replace = TRUE)
      else runif(n, lower[k], upper[k])
    })
    m <- matrix(m, n)
    colnames(m) <- names(lower)
    m
  }
  list(
    names = names(lower), lower = lower, upper = upper, is_int = is_int,
    sample = function(n) {
      out <- matrix(numeric(0), 0, length(lower), dimnames = list(NULL, names(lower)))
      while (nrow(out) < n) {
        m <- draw_box(max(2L * (n - nrow(out)), 100L))
        keep <- apply(m, 1, function(v) ok(setNames(v, names(lower))))
        out <- rbind(out, m[keep, , drop = FALSE])
      }
      out[seq_len(n), , drop = FALSE]
    },
    # prior density up to a constant: 1 inside the (truncated) box, 0 outside
    inside = function(theta) {
      all(theta >= lower & theta <= upper) &&
        all(abs(theta[is_int] - round(theta[is_int])) < 1e-9) &&
        ok(setNames(theta, names(lower)))
    }
  )
}

# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------
# Weighted Euclidean distance between simulated and observed summaries,
# each summary divided by a scale (MAD of prior-predictive simulations).
# NA in observed = not observed (ignored). NA in simulated = Inf distance.
make_distance <- function(obs, scale) {
  keep <- !is.na(obs) & is.finite(scale) & scale > 0
  function(sim) {
    s <- sim[keep]
    if (any(!is.finite(s))) return(Inf)
    sqrt(sum(((s - obs[keep]) / scale[keep])^2))
  }
}

# ---------------------------------------------------------------------------
# Perturbation kernel
# ---------------------------------------------------------------------------
weighted_cov <- function(X, w) {
  w <- w / sum(w)
  mu <- colSums(X * w)
  Xc <- sweep(X, 2, mu)
  crossprod(Xc * sqrt(w)) / (1 - sum(w^2))
}

make_kernel <- function(X, w, is_int) {
  cont <- which(!is_int); disc <- which(is_int)
  Sigma <- 2 * weighted_cov(X[, cont, drop = FALSE], w)
  Sigma <- Sigma + diag(1e-10, length(cont))
  L <- chol(Sigma)
  Sinv <- chol2inv(L)
  logdet <- 2 * sum(log(diag(L)))
  sd_int <- if (length(disc)) {
    pmax(1, sqrt(2 * diag(weighted_cov(X[, disc, drop = FALSE], w))))
  } else numeric(0)

  list(
    perturb = function(x) {
      y <- x
      y[cont] <- x[cont] + drop(rnorm(length(cont)) %*% L)
      if (length(disc)) y[disc] <- round(x[disc] + rnorm(length(disc), 0, sd_int))
      y
    },
    # log K(y | each row of X0), vectorized over rows of X0
    logdens = function(y, X0) {
      D <- sweep(X0[, cont, drop = FALSE], 2, y[cont])
      ld <- -0.5 * rowSums((D %*% Sinv) * D) - 0.5 * logdet
      if (length(disc)) {
        for (k in seq_along(disc)) {
          j <- disc[k]
          d <- y[j] - X0[, j]
          ld <- ld + log(pmax(pnorm(d + 0.5, 0, sd_int[k]) -
                              pnorm(d - 0.5, 0, sd_int[k]), 1e-300))
        }
      }
      ld
    }
  )
}

# ---------------------------------------------------------------------------
# Parallel map helper (forking on Linux/macOS, serial on Windows)
# ---------------------------------------------------------------------------
par_map <- function(X, FUN, ncores) {
  if (ncores > 1 && .Platform$OS.type == "unix") {
    parallel::mclapply(X, FUN, mc.cores = ncores, mc.preschedule = TRUE,
                       mc.set.seed = TRUE)
  } else lapply(X, FUN)
}

# ---------------------------------------------------------------------------
# Main ABC-SMC
# ---------------------------------------------------------------------------
#' @param simulate  function(theta) -> numeric summary vector (same layout as obs)
#' @param prior     from make_uniform_prior()
#' @param obs       observed summary vector (NA = missing)
#' @param n_particles population size
#' @param alpha     quantile used to set the next epsilon (0.3-0.6 typical)
#' @param max_gen   maximum number of SMC generations
#' @param min_acc_rate stop when acceptance rate drops below this
#' @param n_pilot   prior-predictive sims used for summary scaling
#' @param ncores    cores for parallel simulation
#' @param checkpoint optional .rds path, written after every generation
abc_smc <- function(simulate, prior, obs, n_particles = 500, alpha = 0.5,
                    max_gen = 15, min_acc_rate = 0.01, n_pilot = NULL,
                    ncores = 1, checkpoint = NULL, verbose = TRUE,
                    batch_size = NULL, resume = FALSE) {
  # resume = TRUE continues from `checkpoint` (written after every generation)
  # if it exists and was made with the same observed data.
  p <- length(prior$names)
  if (is.null(n_pilot)) n_pilot <- n_particles
  if (is.null(batch_size)) batch_size <- max(ncores * 25, 100)
  sim_safe <- function(th) {
    out <- tryCatch(simulate(th), error = function(e) rep(NA_real_, length(obs)))
    if (length(out) != length(obs)) stop("simulate() returned wrong length")
    out
  }

  ck <- if (isTRUE(resume) && !is.null(checkpoint) && file.exists(checkpoint)) readRDS(checkpoint) else NULL
  if (!is.null(ck) && !isTRUE(all.equal(ck$obs, obs))) {
    warning("checkpoint was made with different observed data; starting fresh")
    ck <- NULL
  }
  if (!is.null(ck)) {
    history <- ck$history; scale <- ck$scale
    dist <- make_distance(obs, scale)
    last <- history[[length(history)]]
    X <- last$theta; w <- last$weights; d <- last$dist; S <- last$stats
    n_sim_total <- last$n_sim
    if (verbose) message(sprintf("resuming after gen %d (eps = %.3f, %d sims so far)",
                                 last$gen, last$eps, n_sim_total))
  } else {
    # ---- generation 0: prior predictive, also used to scale summaries ----
    t0 <- Sys.time()
    X <- prior$sample(max(n_pilot, n_particles))
    S <- do.call(rbind, par_map(split(X, row(X)),
                                function(v) sim_safe(setNames(v, prior$names)),
                                ncores))
    scale <- apply(S, 2, mad, na.rm = TRUE)
    zero <- !is.finite(scale) | scale == 0
    scale[zero] <- apply(S[, zero, drop = FALSE], 2, sd, na.rm = TRUE)
    dist <- make_distance(obs, scale)
    d <- apply(S, 1, dist)
    keep <- order(d)[seq_len(n_particles)]      # best N from the prior sample
    X <- X[keep, , drop = FALSE]; d <- d[keep]; S <- S[keep, , drop = FALSE]
    w <- rep(1 / n_particles, n_particles)
    eps <- max(d)
    n_sim_total <- max(n_pilot, n_particles)
  
    history <- list(list(gen = 0, eps = eps, theta = X, weights = w, dist = d,
                         stats = S, n_sim = n_sim_total, acc_rate = NA,
                         time = Sys.time() - t0))
    if (verbose) message(sprintf("gen 0: eps = %.3f  (prior sample of %d)",
                                 eps, n_sim_total))
  }
  g_start <- length(history)

  for (g in seq.int(g_start, length.out = max(0, max_gen - g_start + 1))) {
    t0 <- Sys.time()
    eps <- as.numeric(quantile(d, alpha))
    kern <- make_kernel(X, w, prior$is_int)
    cdf <- cumsum(w)

    newX <- matrix(NA_real_, 0, p); newS <- matrix(NA_real_, 0, ncol(S))
    newd <- numeric(0); n_sim_gen <- 0
    while (nrow(newX) < n_particles) {
      # propose a batch of in-prior candidates
      cand <- matrix(NA_real_, batch_size, p)
      for (b in seq_len(batch_size)) {
        repeat {
          j <- findInterval(runif(1), cdf) + 1
          y <- kern$perturb(X[min(j, n_particles), ])
          if (prior$inside(y)) break
        }
        cand[b, ] <- y
      }
      colnames(cand) <- prior$names
      sims <- do.call(rbind, par_map(split(cand, row(cand)),
                                     function(v) sim_safe(setNames(v, prior$names)),
                                     ncores))
      dd <- apply(sims, 1, dist)
      n_sim_gen <- n_sim_gen + batch_size
      ok <- which(dd <= eps)
      if (length(ok)) {
        newX <- rbind(newX, cand[ok, , drop = FALSE])
        newS <- rbind(newS, sims[ok, , drop = FALSE])
        newd <- c(newd, dd[ok])
      }
      if (n_sim_gen > n_particles / min_acc_rate) break
    }
    acc_rate <- nrow(newX) / n_sim_gen
    if (nrow(newX) < n_particles) {
      if (verbose) message(sprintf("gen %d: acceptance %.4f < min_acc_rate, stopping",
                                   g, acc_rate))
      break
    }
    newX <- newX[1:n_particles, , drop = FALSE]
    newS <- newS[1:n_particles, , drop = FALSE]
    newd <- newd[1:n_particles]

    # importance weights: prior (uniform, constant) / mixture of kernels
    logw_old <- log(w)
    neww <- apply(newX, 1, function(y) {
      lk <- kern$logdens(y, X) + logw_old
      m <- max(lk)
      -(m + log(sum(exp(lk - m))))
    })
    neww <- exp(neww - max(neww)); neww <- neww / sum(neww)

    X <- newX; S <- newS; d <- newd; w <- neww
    n_sim_total <- n_sim_total + n_sim_gen
    ess <- 1 / sum(w^2)
    history[[g + 1]] <- list(gen = g, eps = eps, theta = X, weights = w,
                             dist = d, stats = S, n_sim = n_sim_total,
                             acc_rate = acc_rate, ess = ess,
                             time = Sys.time() - t0)
    if (verbose) message(sprintf(
      "gen %d: eps = %.3f  acc = %.3f  ESS = %.0f  sims so far = %d  (%.1f s)",
      g, eps, acc_rate, ess, n_sim_total,
      as.numeric(history[[g + 1]]$time, units = "secs")))
    if (!is.null(checkpoint))
      saveRDS(list(history = history, scale = scale, obs = obs), checkpoint)
    if (acc_rate < min_acc_rate) break
  }
  list(history = history, final = history[[length(history)]],
       scale = scale, obs = obs, prior = prior)
}

# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------
wquantile <- function(x, w, probs) {
  o <- order(x); x <- x[o]; cw <- cumsum(w[o]) / sum(w)
  sapply(probs, function(p) x[which(cw >= p)[1]])
}

posterior_table <- function(fit, truth = NULL) {
  f <- fit$final
  tab <- t(sapply(colnames(f$theta), function(nm) {
    x <- f$theta[, nm]; w <- f$weights
    q <- wquantile(x, w, c(0.05, 0.5, 0.95))
    c(mean = sum(w * x), q05 = q[1], median = q[2], q95 = q[3],
      prior_lo = fit$prior$lower[[nm]], prior_hi = fit$prior$upper[[nm]],
      # how much the 90% interval shrank relative to the prior width
      shrink = 1 - (q[3] - q[1]) / (0.9 * (fit$prior$upper[[nm]] - fit$prior$lower[[nm]])))
  }))
  tab <- as.data.frame(tab)
  if (!is.null(truth)) {
    tab$truth <- truth[rownames(tab)]
    tab$covered <- tab$truth >= tab$q05 & tab$truth <= tab$q95
  }
  tab
}
