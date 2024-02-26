source("generate_data.R")
library(ordinalClust)

# Multivariate plot
ms <- 2:5
n <- 10000
d <- 10
n_cluster <- 4
set.seed(19)

run_times <- rep(0, length(ms))
data_path <- "../data/comparison_curves/"

for (m in ms) {
  
  # Read data
  data <- read.csv(paste0(data_path, paste0("bos_data_n", n, "_d", d, "_m", m, "_k", n_cluster, "_seed", 3, ".csv")))
  y <- data$y
  x_cols <- colnames(data)[!colnames(data) %in% c("y")]
  x <- as.matrix(data[, x_cols])
  
  init = "kmeans"
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
  run_times[m] <- run_time
  
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

#save run_times to csv
run_times_df <- data.frame(ms, run_times[2:length(run_times)])
write.csv(run_times_df, paste0(data_path, "run_times_multivariate_r.csv"), row.names = FALSE)

library(ggplot2)

# Univariate plot
ms <- 2:6
n <- 10000
d <- 1
n_cluster <- 1
set.seed(19)

run_times <- rep(0, length(ms))
data_path <- "../data/comparison_curves/"

for (m in ms) {
  
  # Read data
  data <- read.csv(paste0(data_path, paste0("bos_data_n", n, "_d", d, "_m", m, "_k", n_cluster, "_seed", 3, ".csv")))
  y <- data$y
  x_cols <- colnames(data)[!colnames(data) %in% c("y")]
  x <- as.matrix(data[, x_cols])
  
  init = "kmeans"
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
  run_times[m] <- run_time
  
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

#save run_times to csv
run_times_df <- data.frame(ms, run_times[2:length(run_times)])
write.csv(run_times_df, paste0(data_path, "run_times_univariate_r.csv"), row.names = FALSE)


# Multivariate
run_times_multivariate <- read.csv(paste0(data_path, "run_times_multivariate_r.csv"))
ggplot(run_times_multivariate, aes(x=ms, y=run_times_multivariate[,2])) +
  geom_point() +
  geom_line() +
  labs(title="Run time vs. number of categories (multivariate)", x="Number of categories", y="Run time (s)")

# Univariate
run_times_univariate <- read.csv(paste0(data_path, "run_times_univariate_r.csv"))
ggplot(run_times_univariate, aes(x=ms, y=run_times_univariate[,2])) +
  geom_point() +
  geom_line() +
  labs(title="Run time vs. number of categories (univariate)", x="Number of categories", y="Run time (s)")


