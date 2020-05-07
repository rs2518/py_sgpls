#############################################################################################
# sgPLS package - Run times (regression mode)
#############################################################################################
n_runs <- 100
mode <- "regression"
spls.times <- gpls.times <- sgpls.times <- NULL


#### Data and model parameters
path <- "~/Desktop/py_sgpls/data/dataset1/"
X <- read.csv(paste0(path, "X.csv"))
Y <- read.csv(paste0(path, "Y.csv"))

ncomp = 2
keepX <- keepY <- c(60, 60)    # sPLS sparsity
keepX.groups <- keepY.groups <- c(4, 4)    # gPLS/sgPLS sparsity
ind.block.x <- seq(20, 380, 20)
ind.block.y <- seq(20, 480, 20)
alpha.x <- alpha.y <- c(0.95, 0.95)    # sgPLS sparsity mixin

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