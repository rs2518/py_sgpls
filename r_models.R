# Load packages
library(sgPLS)    # no PLS(da)

## Other packages to consider for comparison
# library(mixOmics)    # only PLS(da) and sPLS(da)
# library(plsdepot) # only PLS(da)
# library(PLSgenomics) # only PLS and sPLS for univariate outcomes


#### Code copied from "https://cran.r-project.org/web/packages/sgPLS/sgPLS.pdf" 
## Simulation of datasets X and Y with group variables
n <- 100
sigma.gamma <- 1
sigma.e <- 1.5
p <- 400
q <- 500
theta.x1 <- c(rep(1, 15), rep(0, 5), rep(-1, 15), rep(0, 5),
              rep(1.5, 15), rep(0, 5), rep(-1.5, 15), rep(0, 325))
theta.x2 <- c(rep(0, 320), rep(1, 15), rep(0, 5), rep(-1, 15),
              rep(0, 5), rep(1.5, 15), rep(0, 5), rep(-1.5, 15),
              rep(0, 5))

theta.y1 <- c(rep(1, 15), rep(0, 5), rep(-1, 15), rep(0, 5),
              rep(1.5, 15), rep(0, 5), rep(-1.5, 15), rep(0, 425))
theta.y2 <- c(rep(0, 420), rep(1, 15), rep(0, 5), rep(-1, 15)
              ,rep(0, 5), rep(1.5, 15), rep(0, 5), rep(-1.5, 15)
              , rep(0, 5))

Sigmax <- matrix(0, nrow = p, ncol = p)
diag(Sigmax) <- sigma.e ^ 2
Sigmay <- matrix(0, nrow = q, ncol = q)
diag(Sigmay) <- sigma.e ^ 2

set.seed(125)

gam1 <- rnorm(n)
gam2 <- rnorm(n)

X <- matrix(c(gam1, gam2), ncol = 2, byrow = FALSE) %*% matrix(c(theta.x1, theta.x2),
                                                               nrow = 2, byrow = TRUE) + rmvnorm(n, mean = rep(0, p), sigma =
                                                                                                   Sigmax, method = "svd")

Y <- matrix(c(gam1, gam2), ncol = 2, byrow = FALSE) %*% matrix(c(theta.y1, theta.y2),
                                                               nrow = 2, byrow = TRUE) + rmvnorm(n, mean = rep(0, q), sigma =
                                                                                                   Sigmay, method = "svd")


#### Model parameters

ncomp = 2

keepX <- keepY <- c(60, 60)    # sPLS sparsity

keepX.groups <- keepY.groups <- c(4, 4)    # gPLS/sgPLS sparsity
ind.block.x <- seq(20, 380, 20)
ind.block.y <- seq(20, 480, 20)

alpha.x <- alpha.y <- c(0.95, 0.95)    # sgPLS sparsity mixin

#############################################################################################
# sgPLS package - Compare results
#############################################################################################
#### sPLS model
sgPLS_spls <- sPLS(X, Y, ncomp = ncomp, mode = "regression", keepX = keepX, keepY = keepY)

result.sPLS <- select.spls(sgPLS_spls)    # Returns the indices of non-zero variables
result.sPLS$select.X
result.sPLS$select.Y

#### gPLS model
sgPLS_gpls <- gPLS(X, Y, ncomp = ncomp, mode = "regression", keepX = keepX.groups,
                   keepY = keepY.groups, ind.block.x = ind.block.x , ind.block.y = ind.block.y)

result.gPLS <- select.sgpls(sgPLS_gpls)    # Returns the indices of non-zero groups of variables (and group sizes)
result.gPLS$group.size.X
result.gPLS$group.size.Y

#### sgPLS model
sgPLS_sgpls <- sgPLS(X, Y, ncomp = ncomp, mode = "regression", keepX = keepX.groups,
                     keepY = keepY.groups, ind.block.x = ind.block.x, ind.block.y = ind.block.y,
                     alpha.x = alpha.x, alpha.y = alpha.y)

result.sgPLS <- select.sgpls(sgPLS_sgpls)    # Returns the indices of non-zero groups and number of non-zero variables in each group
result.sgPLS$group.size.X
result.sgPLS$group.size.Y


##### Save model elements into individual files. Create new directory if one doesn't already exist
# NOTE: R models cannot be directly saved and loaded into Python. Python packages 'rpy2' and 'pyreadr' exist
# to try and fix this however, pyreadr cannot parse the R object and rpy2 does not work on new versions of Python
path <- "~/Desktop/py_sgpls/Data/"
model_list <- c("sgPLS_spls", "sgPLS_gpls", "sgPLS_sgpls")
skip <- c("call", "names")

for (model in model_list){
  ifelse(dir.exists(paste0(path, model)), "", dir.create(paste0(path, model), showWarnings=FALSE))
  for (item in names(get(model))){
    if (!(item %in% skip)){
      if (is.list(get(model)[[item]])){
        for (index in 1:length(get(model)[[item]])){
         write.csv(as.data.frame(get(model)[[item]][index]),
                   file=paste0(path, model, "/", item, "_", index, ".csv"))}}
      else{
        write.csv(as.data.frame(get(model)[[item]]),
                  file=paste0(path, model, "/", item, ".csv"))}}}}

#############################################################################################
# sgPLS package - Run times (regression mode)
#############################################################################################
n_runs <- 100
mode <- "regression"
spls.times <- gpls.times <- sgpls.times <- NULL

# Could parallelise process to handle more runs
for (n in (1:n_runs)){
  # sPLS
  t0 <- Sys.time()
  spls.model <- sPLS(X, Y, ncomp = ncomp, mode = mode, keepX = keepX, keepY = keepY)
  t1 <- Sys.time()
  spls.times <- c(spls.times, (t1 - t0))
  
  # gPLS
  t0 <- Sys.time()
  gpls.model <- gPLS(X, Y, ncomp = ncomp, mode = mode, keepX = keepX.groups,
                     keepY = keepY.groups, ind.block.x = ind.block.x , ind.block.y = ind.block.y)
  t1 <- Sys.time()
  gpls.times <- c(gpls.times, (t1 - t0))
  
  # sgPLS
  t0 <- Sys.time()
  sgpls.model <- sgPLS(X, Y, ncomp = ncomp, mode = mode, keepX = keepX.groups,
                      keepY = keepY.groups, ind.block.x = ind.block.x, ind.block.y = ind.block.y,
                      alpha.x = alpha.x, alpha.y = alpha.y)
  t1 <- Sys.time()
  sgpls.times <- c(sgpls.times, (t1 - t0))
}

run_times <- as.data.frame(rbind(spls.times, gpls.times, sgpls.times),
                     row.names = c("sPLS", "gPLS", "sgPLS"))    # NOTE: col.names is inactive in as.data.frame
colnames(run_times) <- paste("t", seq(n_runs), sep="")

mean_times <- as.data.frame(apply(run_times, MARGIN=1, FUN = mean))
sd_times <- as.data.frame(apply(run_times, MARGIN=1, FUN = sd))     # NOTE: Dropping colnames removes values (BUG)