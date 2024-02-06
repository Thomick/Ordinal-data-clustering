source("generate_data.R")

# Generate BOS data:
n <- 100

mu <- 2 
pi <- 0.5
n_cat = 4
x <- bos_model(n_cat, mu, pi)

n <- 1000
p <- 2
k <- 2
m <-4
n_cat <- matrix(p)
for (j in 1:p) {
  n_cat
  n_cat[j] <- sample(2:m, 1)
}
alpha <- runif(k)
alpha <- alpha/sum(alpha)
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
n_cats <- 2:6
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
