source("generate_data.R")
library(ordinalClust)

# Multivariate plot
ms <- 2:5
n <- 10000
d <- 10
n_cluster <- 4
set.seed(19)

data_path <- "../data/comparison_curves/"
seeds <- c(0:9)

run_times <- matrix(0, length(seeds), length(ms))

i = 0
for (seed in seeds) {
  i <- i + 1
  for (m in ms) {
    print(seed) 
    # Read data
    data <- read.csv(paste0(data_path, paste0("bos_data_n", n, "_d", d, "_m", m, "_k", n_cluster, "_seed", seed, ".csv")))
    y <- data$y
    x_cols <- colnames(data)[!colnames(data) %in% c("y")]
    x <- as.matrix(data[, x_cols])
    
    init = "random"
    nbSEM <- 300 # Number of SEM-Gibbs iterations
    nbSEMburn <- 200 # Burn-in interations (ie. internal EM)
    nbindmini <- 1 # Number of cells in a block, not useful for our case (only for co-clustering)
    k <- length(unique(y))
    if (m <= 5) {
      k <- 1
      # The ordinalClust implementation crashes when one cluster is empty, this happens for values of m where clusters are not separarable easily
      # This doesn't really affect run times a lot for the comparison plots (and actually might underestimate the runtime for small values of m of ordinalClust)
    }
    n_cat <- max(x)
    
    start_time <- Sys.time()
    clust <- bosclust(x=x, kr=k, m=n_cat, nbSEM=nbSEM,
                      nbSEMburn=nbSEMburn,
                      nbindmini=nbindmini,
                      init = init,
    )
    run_time <- difftime(Sys.time(), start_time, units = "secs")
    run_times[i, m-1] <- run_time
    
    summary(clust)
    # print a data frame of the estimated parameters
    estimated_pis <- clust@params[[1]]$pis
    estimated_mus <- clust@params[[1]]$mus
    
    print("Run time")
    print(run_time)
    print("Estimated pis:")
    print(estimated_pis)
    print("Estimated mus:")
    print(estimated_mus)
    
      
  }
}

#save run_times to csv
# run_times_df <- data.frame(ms, run_times[2:length(run_times)])
seed <- c()
m <- c()
run <- c()
for (s in seeds) {
  for (m_ in ms) {
    seed <- c(seed, s)
    m <- c(m, m_)
    run <- c(run, run_times[s+1, m_-1])
  }
}
run_times_df <- data.frame(seed, m, run)
write.csv(run_times_df, paste0(data_path, "run_times_multivariate_r.csv"), row.names = FALSE)

library(ggplot2)

# Univariate plot
ms <- 2:6
n <- 10000
d <- 1
n_cluster <- 1
set.seed(19)

data_path <- "../data/comparison_curves/"
seeds <- c(0:9)
run_times <- matrix(0, length(seeds), length(ms))

i <- 0
for (seed in seeds) {
  i <- i + 1
  for (m in ms) {
    
    # Read data
    data <- read.csv(paste0(data_path, paste0("bos_data_n", n, "_d", d, "_m", m, "_k", n_cluster, "_seed", seed, ".csv")))
    y <- data$y
    x_cols <- colnames(data)[!colnames(data) %in% c("y")]
    x <- as.matrix(data[, x_cols])
    
    init = "random"
    nbSEM <- 300 # Number of SEM-Gibbs iterations
    nbSEMburn <- 200 # Burn-in interations (ie. internal EM)
    nbindmini <- 1 # Number of cells in a block, not useful for our case (only for co-clustering)
    k <- length(unique(y))
    n_cat <- max(x)
    
    start_time <- Sys.time()
    clust <- bosclust(x=x, kr=k, m=n_cat, nbSEM=nbSEM,
                      nbSEMburn=nbSEMburn,
                      nbindmini=nbindmini,
                      init = init,
    )
    run_time <- difftime(Sys.time(), start_time, units = "secs")
    run_times[i, m-1] <- run_time
    
    summary(clust)
    # print a data frame of the estimated parameters
    estimated_pis <- clust@params[[1]]$pis
    estimated_mus <- clust@params[[1]]$mus
    
    print("Run time")
    print(run_time)
    print("Estimated pis:")
    print(estimated_pis)
    print("Estimated mus:")
    print(estimated_mus)
    
      
  }
}

#save run_times to csv
seed <- c()
m <- c()
run <- c()
for (s in seeds) {
  for (m_ in ms) {
    seed <- c(seed, s)
    m <- c(m, m_)
    run <- c(run, run_times[s+1, m_-1])
  }
}
run_times_df <- data.frame(seed, m, run)
write.csv(run_times_df, paste0(data_path, "run_times_univariate_r.csv"), row.names = FALSE)


# Multivariate
run_times_multivariate <- read.csv(paste0(data_path, "run_times_multivariate_r.csv"))
run_times_multivariate_std <- aggregate(run_times_multivariate$run, by=list(run_times_multivariate$m), FUN=sd)
run_times_multivariate_mean <- aggregate(run_times_multivariate$run, by=list(run_times_multivariate$m), FUN=mean)
ggplot(run_times_multivariate_mean, aes(x=Group.1, y=x)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin=x-run_times_multivariate_std$x, ymax=x+run_times_multivariate_std$x), width=.1) +
  labs(title="Run time vs. number of categories (multivariate)", x="Number of categories", y="Run time (s)")

# Univariate
run_times_univariate <- read.csv(paste0(data_path, "run_times_univariate_r.csv"))
run_times_univariate_std <- aggregate(run_times_univariate$run, by=list(run_times_univariate$m), FUN=sd)
run_times_univariate_mean <- aggregate(run_times_univariate$run, by=list(run_times_univariate$m), FUN=mean)
ggplot(run_times_univariate_mean, aes(x=Group.1, y=x)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin=x-run_times_univariate_std$x, ymax=x+run_times_univariate_std$x), width=.1) +
  labs(title="Run time vs. number of categories (univariate)", x="Number of categories", y="Run time (s)")



