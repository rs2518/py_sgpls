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

# Overwrite internal sgPLS function with more stringent uniroot tolerance (because Python's has higher accuracy)
new.step1.sgpls.sparsity <- function(X,Y,ind.block.x,ind.block.y,sparsity.x,sparsity.y,epsilon,iter.max,alpha.x,alpha.y,upper.lambda=upper.lambda, lambda.tol=10*.Machine$double.eps){
  print("sgPLS successfully overwritten")
  
  n <- dim(X)[1]
  Z <- t(X)%*%Y
  svd.Z <- svd(Z,nu=1,nv=1)
  
  u.tild.old <- svd.Z$u
  v.tild.old <- svd.Z$v
  u.tild.previous <- v.tild.previous <- 0
  iter <- 0
  
  ### Step c
  #|(norm(v.tild.old-v.tild.previous)>epsilon)
  while (((normv(u.tild.old-u.tild.previous)>epsilon) ) & (iter <iter.max))  {
    vecZV <- Z%*%matrix(v.tild.old,ncol=1)
    tab.ind <- c(0,ind.block.x,length(vecZV))
    lamb.x <- NULL
    lamb.max <- upper.lambda
    for (i in 1:(length(ind.block.x)+1)){
      ji <- tab.ind[i+1]-tab.ind[i]  
      vecx <- vecZV[((tab.ind[i]+1):tab.ind[i+1])]
      lamb.x <- c(lamb.x,uniroot(lambda.quadra,lower=0,upper=lamb.max,vec=vecx,alpha=alpha.x, tol=lambda.tol)$root)
    }   
    if(sparsity.x==0){lambda.x <- sort(lamb.x)[1]-1} else {
      lambda.x <- sort(lamb.x)[sparsity.x]}
    
    ####block to zero
    index.block.zero.x <- which(lamb.x<=lambda.x)
    
    
    u.tild.new <- soft.thresholding.sparse.group(Z%*%matrix(v.tild.old,ncol=1),ind=ind.block.x,lambda=lambda.x,alpha=alpha.x,ind.block.zero=index.block.zero.x)
    
    u.tild.new <- u.tild.new/sqrt(sum(u.tild.new**2))
    
    if(sparsity.y==0) {lambda.y <- 0} else { 
      vecZV <- t(Z)%*%matrix(u.tild.new,ncol=1)
      tab.ind <- c(0,ind.block.y,length(vecZV))
      lamb.y <- NULL
      lamb.max <- 100000
      res <- NULL
      for (i in 1:(length(ind.block.y)+1)){
        ji <- tab.ind[i+1]-tab.ind[i]  
        vecx <- vecZV[((tab.ind[i]+1):tab.ind[i+1])]
        lamb.y <- c(lamb.y,uniroot(lambda.quadra,lower=0,upper=lamb.max,vec=vecx,alpha=alpha.y, tol=lambda.tol)$root)
      }
      lambda.y <- sort(lamb.y)[sparsity.y]
      index.block.zero.y <- which(lamb.y<=lambda.y)
    }
    
    if(sparsity.y==0) {v.tild.new <- t(Z)%*%matrix(u.tild.new,ncol=1)} else {
      v.tild.new <- soft.thresholding.sparse.group(t(Z)%*%matrix(u.tild.new,ncol=1),ind=ind.block.y,lambda=lambda.y,alpha=alpha.y,ind.block.zero=index.block.zero.y)
    }
    
    v.tild.new <- v.tild.new/sqrt(sum(v.tild.new**2))
    
    u.tild.previous <- u.tild.old
    v.tild.previous <- v.tild.old
    
    u.tild.old <- u.tild.new
    v.tild.old <- v.tild.new
    
    iter <- iter +1
  }  
  res <- list(iter=iter, u.tild.new=u.tild.new,v.tild.new=v.tild.new) 
  
}
assignInNamespace("step1.sparse.group.spls.sparsity", value=new.step1.sgpls.sparsity, ns="sgPLS")




# Save regression and canonical model results
modes <- c("regression", "canonical")
scale <- TRUE

path <- "~/Desktop/py_sgpls/data/dataset1/"
ifelse(dir.exists(path), "", dir.create(path, showWarnings=FALSE))

model_list <- c("sgPLS_spls", "sgPLS_gpls", "sgPLS_sgpls")
skip <- c("call", "X", "Y", "ncomp", "mode", "keepX", "keepY", "names", "tol", "max.iter",
          "iter", "ind.block.x", "ind.block.y", "alpha.x", "alpha.y", "upper.lambda")

for (i in length(modes)){
  
  #### sPLS model
  sgPLS_spls <- sPLS(X, Y, ncomp = ncomp, mode = modes[i], keepX = keepX, keepY = keepY, scale = scale)
  
  # result.sPLS <- select.spls(sgPLS_spls)    # Returns the indices of non-zero variables
  # result.sPLS$select.X
  # result.sPLS$select.Y
  
  #### gPLS model
  sgPLS_gpls <- gPLS(X, Y, ncomp = ncomp, mode = modes[i], keepX = keepX.groups,
                     keepY = keepY.groups, ind.block.x = ind.block.x , ind.block.y = ind.block.y,
                     scale = scale)
  
  # result.gPLS <- select.sgpls(sgPLS_gpls)    # Returns the indices of non-zero groups of variables (and group sizes)
  # result.gPLS$group.size.X
  # result.gPLS$group.size.Y
  
  #### sgPLS model
  sgPLS_sgpls <- sgPLS(X, Y, ncomp = ncomp, mode = modes[i], keepX = keepX.groups,
                       keepY = keepY.groups, ind.block.x = ind.block.x, ind.block.y = ind.block.y,
                       alpha.x = alpha.x, alpha.y = alpha.y, scale = scale)
  
  # result.sgPLS <- select.sgpls(sgPLS_sgpls)    # Returns the indices of non-zero groups and number of non-zero variables in each group
  # result.sgPLS$group.size.X
  # result.sgPLS$group.size.Y
  
  
  ##### Save model elements into individual files. Create new directory if one doesn't already exist
  # NOTE: R models cannot be directly saved and loaded into Python. Python packages 'rpy2' and 'pyreadr' exist
  # to try and fix this however, pyreadr cannot parse the R object and rpy2 does not work on new versions of Python
  for (model in model_list){
    ifelse(dir.exists(paste0(path, model)), "", dir.create(paste0(path, model), showWarnings=FALSE))
    for (item in names(get(model))){
      if (!(item %in% skip)){
        if (is.list(get(model)[[item]])){
          for (index in 1:length(get(model)[[item]])){
           write.csv(as.data.frame(get(model)[[item]][index]),
                     file=paste0(path, model, "/", modes[i], "_", item, "_", index, ".csv"))}}
        else{
          write.csv(as.data.frame(get(model)[[item]]),
                    file=paste0(path, model, "/", modes[i], "_", item, ".csv"))}}}}
}


write.csv(X, file=paste0(path,"X.csv"))
write.csv(Y, file=paste0(path,"Y.csv"))

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



#############################################################################################
# internal functions test
#############################################################################################
# #### Compare to _lambda_quadratic
# set.seed(101096)
# vec <- round(runif(5, -10, 10))/10
# alpha <- 1/2
# lambda.space <- seq(0, 5, by=0.5)
# lq <- NULL
# for (i in 1:length(lambda.space)){
#   lq <- c(lq, lambda.quadra(lambda.space[i], vec, alpha))}



# #### Compare to sgpls_inner_loop
test_model <- sgPLS(X, Y, ncomp = ncomp, mode = "regression", keepX = keepX.groups, keepY = keepY.groups,
                    ind.block.x = ind.block.x , ind.block.y = ind.block.y,
                    alpha.x = alpha.x, alpha.y = alpha.y, scale = TRUE)
View(test_model[["loadings"]][["X"]])    # x_weights
View(test_model[["loadings"]][["Y"]])    # y_weights
View(test_model[["variates"]][["X"]])    # x_scores
View(test_model[["variates"]][["Y"]])    # y_scores
View(test_model[["mat.c"]])    # x_loadings
View(test_model[["mat.d"]])    # y_loadings
# View(test_model[["mat.e"]])    # y_loadings


test_dir <- "~/Desktop/py_sgpls/data/dataset1/"
data <- c("loadings","variates","mat.c","mat.d", "mat.e")

for (item in names(test_model)){
  if ((item %in% data)){
    if (is.list(test_model[[item]])){
      for (index in 1:length(test_model[[item]])){
        write.csv(as.data.frame(test_model[[item]][index]),
                  file=paste0(test_dir, "test", "_", item, "_", index, ".csv"))}}
    else{
      write.csv(as.data.frame(test_model[[item]]),
                file=paste0(test_dir, "test", "_", item, ".csv"))}}} 



# # Save array
# write.csv(as.data.frame(*INSERT ARRAY HERE*), file = paste0(test_dir, "arr", ".csv"))