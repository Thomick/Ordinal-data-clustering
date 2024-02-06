library(ordinalClust)

data("dataqol")

seed <- 100
set.seed(1)

nbSEM <- 150
nbSEMburn <- 100
nbindmini <- 1
init <- "randomBurnin"

percentRandomB <- c(50, 50)

x <- as.matrix(dataqol.classif[, 2:29])
v <- as.vector(dataqol.classif$death)

nb.sample <- ceiling(nrow(x) *7/10)
sample.train <- sample(1:nrow(x), nb.sample, replace=FALSE)

x.train <- x[sample.train, ]
x.validation <- x[-sample.train, ]


v.train <- v[sample.train]
v.validation <- v[-sample.train]

kr <- 2
m <- 4
kcol <- c(0, 1, 2, 3, 4)
preds <- matrix(0, nrow=length(kcol), ncol=nrow(x.validation))

# for( kc in 1:length(kcol) ){
#   classif <- bosclassif(x = x.train, y = v.train, kr = kr, kc = kcol[kc],
#                         m = m, nbSEM = nbSEM, nbSEMburn = nbSEMburn,
#                         nbindmini = nbindmini, init = init,
#                         percentRandomB = percentRandomB)
#   new.prediction <- predict(classif, x.validation)
#   preds[kc,] <- new.prediction@zr_topredict
# }
# 
# 
# preds <- as.data.frame(preds)
# row.names <- c()
# for(kc in kcol){
#   name <- paste0("kc = ",kc)
#   row.names <- c(row.names,name)
# }
# rownames(preds)=row.names
# preds
# v.validation

source("generate_data.R")

# Generate BOS data:
n <- 100

mu <- 2 
pi <- 0.5
n_cat = 4
x <- bos_model(n_cat, mu, pi)

n <- 100
p <- 1
k <- 1
n_cat <- c(5)
alpha <- c(1)
mu <- lapply(1:k, function(x) sapply(n_cat, function(nc) sample(1:nc, 1)))
pi <- lapply(1:k, function(x) runif(p))

output_synthetic_univariate <- generate_data(n, p, n_cat, k, alpha, mu, pi, seed)
x_synthetic_univariate <- output_synthetic_univariate[1]
w_synthetic_univariate <- output_synthetic_univariate[2]

# Plot univariate data:

library(ggplot2)
library(reshape2)

x_synthetic_univariate <- as.data.frame(x_synthetic_univariate)
x_synthetic_univariate$w <- w_synthetic_univariate

x_synthetic_univariate <- melt(x_synthetic_univariate, id.vars = "w")


# set mu and pi in the title (don't hardcode)
ggplot(x_synthetic_univariate, aes(x = value, fill = factor(w))) + 
  geom_histogram(position = "identity", alpha = 0.5, bins = 10) + 
  facet_wrap(~variable, ncol = 1) + 
  theme_bw() + 
  theme(legend.position = "none") +
  labs(x = "x", y = "Frequency") + 
  ggtitle(paste0("mu = ", mu[[1]], ", pi = ", pi[[1]]))

run_univariate <- function(n_cat, n, k, p){
  alpha <- runif(k)
  alpha <- alpha/sum(alpha)
  
  mu <- lapply(1:k, function(x) sapply(n_cat, function(nc) sample(1:nc, 1)))
  pi <- lapply(1:k, function(x) runif(p))
  mu
  pi
  
  
  seed <- 12345
  set.seed(seed)
  
  output_synthetic_multivariate <- generate_data(n, p, n_cat, k, alpha, mu, pi, seed)
  x_synthetic_multivariate <- output_synthetic_multivariate[1]
  w_synthetic_multivariate <- output_synthetic_multivariate[2]
  
  nbSEM <- 120
  nbSEMburn <- 80
  nbindmini <- 1
  init <- "kmeans" #kmeans or random or randomBurnin
  # # clustering setting:
  data_synthetic_multivariate <- as.matrix(as.data.frame(x_synthetic_multivariate))
  start_time <- Sys.time()
  clust <- bosclust(x=data_synthetic_multivariate, kr=k, m=max(n_cat), nbSEM=nbSEM,
                    nbSEMburn=nbSEMburn,
                    nbindmini=nbindmini,
                    init = init
  )
  run_time <- Sys.time() - start_time
  summary(clust)
  # print a data frame of the estimated parameters
  estimated_pis <- clust@params[[1]]$pis
  estimated_mus <- clust@params[[1]]$mus
  
  print("Run time")
  print(run_time)
  print("True pis:")
  print(pi)
  print("True mus:")
  print(mu)
  print("Estimated pis:")
  print(estimated_pis)
  print("Estimated mus:")
  print(estimated_mus)
  
  return(list(estimated_pis, estimated_mus, pi, mu, run_time))
}

n <- 1000
p <- 2
k <- 2
m <-4
n_cat <- matrix(p)
for (j in 1:p) {
  n_cat
  n_cat[j] <- sample(2:m, 1)
}
output <- run_univariate(n_cat, n, k, p)

n <- 1000
p <- 1
k <- 1
n_cats <- 2:3
run_times <- c()
for(n_cat in n_cats){
  output <- run_univariate(n_cat, n, k, p)
  run_times <- c(run_times, output[[5]])
}

# plot run times with ggplot
run_times <- data.frame(run_times, n_cats)
ggplot(run_times, aes(x = n_cats, y = run_times)) + 
  geom_point() + 
  geom_line() + 
  labs(x = "Number of categories", y = "Run time (s)") + 
  theme_bw()



# set.seed(1)
# library(ordinalClust)
# # loading the real dataset
# data("dataqol")
# # loading the ordinal data
# x <- as.matrix(dataqol[,2:31])
# # defining different number of categories:
# m <- c(4,7)
# # defining number of row and column clusters
# krow <- 3
# kcol <- c(3,1)
# # configuration for the inference
# nbSEM <- 20
# nbSEMburn <- 15
# nbindmini <- 2
# init <- 'random'
# d.list <- c(1,29)
# object <- boscoclust(x = x,kr = krow, kc = kcol, m = m,
#                      idx_list = d.list, nbSEM = nbSEM,
#                      nbSEMburn = nbSEMburn, nbindmini = nbindmini,
#                      init = init)
