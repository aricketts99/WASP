library(sparsevb)
library(parallel)

# ============================================================
# SETTINGS
# ============================================================

n <- 100
p <- 200

n_runs <- 1000
n_workers <- 8

seed <- 123

# AR(1) correlation between adjacent predictors
rho <- 0.7

# Degrees of freedom for Student-t errors
df <- 3

# Target error variance
sigma2 <- 2

# ============================================================
# GENERATE X
# ============================================================

set.seed(seed)

X <- matrix(
  rnorm(n * p),
  nrow = n,
  ncol = p
)

# AR(1) correlation structure
#
# X[, 1] ~ N(0, 1)
#
# X[, j] = rho * X[, j-1] +
#          sqrt(1-rho^2) * Z_j
#
# This gives
# Corr(X_j, X_k) = rho^|j-k|

b <- sqrt(1 - rho^2)

for (j in 2:p) {
  X[, j] <- rho * X[, j - 1] +
    b * X[, j]
}

# ============================================================
# GENERATE BETA
# ============================================================

beta <- numeric(p)

beta[1:20] <- c(
  2, -3, 2, 2, -3,
  3, -2, 3, -2, 3,
  3, -2, 3, -2, 3,
  3, -2, 3, -2, 3
)

# ============================================================
# GENERATE HEAVY-TAILED ERRORS
# ============================================================

# rstudent() generates a standard Student-t random variable
# with variance df/(df-2), provided df > 2.
#
# Rescale so that Var(error) = sigma2.

epsilon <- sqrt(
  sigma2 * (df - 2) / df
) * rt(
  n,
  df = df
)

# ============================================================
# GENERATE RESPONSE
# ============================================================

Y <- as.vector(
  X %*% beta + epsilon
)

# ============================================================
# BASIC CHECKS
# ============================================================

cat("n =", n, "\n")
cat("p =", p, "\n")
cat("rho =", rho, "\n")
cat("Student-t df =", df, "\n")
cat("sigma2 =", sigma2, "\n")
cat("Number of active variables =", sum(beta != 0), "\n")

cat("\nX dimensions:\n")
print(dim(X))

cat("\nY dimensions:\n")
print(length(Y))

cat("\nBeta dimensions:\n")
print(length(beta))

cat("\nFirst 10 beta values:\n")
print(beta[1:10])

cat("\nEmpirical variance of Y:\n")
print(var(Y))

cat("\nEmpirical error variance:\n")
print(var(epsilon))

cat("\nCorrelation between first two predictors:\n")
print(cor(X[, 1], X[, 2]))

cat("\nCorrelation between first and tenth predictors:\n")
print(cor(X[, 1], X[, 10]))

# ============================================================
# RUN ONE BATCH
# ============================================================

run_batch <- function(n_runs) {
  
  results <- vector("list", n_runs)
  
  for (i in seq_len(n_runs)) {
    
    # Random update ordering.
    #
    # R indexing is 1,...,p, while sparsevb expects the
    # ordering used here to be converted to 0,...,p-1.
    update_order <- sample(seq_len(ncol(X))) - 1L
    
    fit <- svb.fit(
      X = X,
      Y = Y,
      update_order = update_order
    )
    
    results[[i]] <- list(
      mu = fit$mu,
      sigma = fit$sigma,
      gamma = fit$gamma,
      intercept = fit$intercept,
      update_order = update_order
    )
  }
  
  results
}

# ============================================================
# START WORKERS
# ============================================================

cl <- makeCluster(n_workers)

# Reproducible independent RNG streams
clusterSetRNGStream(cl, seed)

# Give workers X and Y
clusterExport(
  cl,
  varlist = c(
    "X",
    "Y",
    "ncol"
  ),
  envir = .GlobalEnv
)

# Load sparsevb on every worker
clusterEvalQ(cl, {
  library(sparsevb)
})

# ============================================================
# SPLIT WORK INTO BATCHES
# ============================================================

base <- n_runs %/% n_workers
remainder <- n_runs %% n_workers

batch_sizes <- rep(
  base,
  n_workers
)

if (remainder > 0) {
  batch_sizes[seq_len(remainder)] <-
    batch_sizes[seq_len(remainder)] + 1
}

print(batch_sizes)

# ============================================================
# RUN IN PARALLEL
# ============================================================

results_by_worker <- parLapply(
  cl,
  batch_sizes,
  run_batch
)

# ============================================================
# STOP WORKERS
# ============================================================

stopCluster(cl)

# ============================================================
# FLATTEN RESULTS
# ============================================================

results <- unlist(
  results_by_worker,
  recursive = FALSE
)

cat("\nNumber of results:\n")
print(length(results))