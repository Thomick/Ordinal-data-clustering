bos_model <- function(n_cat, mu, pi) {
cur_e <- seq(1, n_cat)
for (i in seq_len(n_cat)) {
  y <- sample(cur_e, 1)
  z <- rbinom(1, 1, pi)
  e_minus <- cur_e[cur_e <y]
  e_plus <- cur_e[cur_e > y]
  e_equal <- cur_e[cur_e == y]
  
  if (z == 0) {
    p <- c(length(e_minus) / length(cur_e), length(e_plus) / length(cur_e), length(e_equal) / length(cur_e))
    id <- sample(c(1, 2, 3), 1, prob = p)
    cur_e <- list(e_minus, e_plus, e_equal)[[id]]
  } else {
    min_e <- e_equal
    min_dist <- abs(mu - e_equal[1])
    
    if (length(e_minus) != 0) {
      d_e_minus <- min(abs(mu - e_minus[1]), abs(mu - e_minus[length(e_minus)]))
      if (d_e_minus < min_dist) {
        min_e <- e_minus
        min_dist <- d_e_minus
      }
    }
    
    if (length(e_plus) != 0) {
      d_e_plus <- min(abs(mu - e_plus[1]), abs(mu - e_plus[length(e_plus)]))
      if (d_e_plus < min_dist) {
        min_e <- e_plus
      }
    }
    cur_e <- min_e
  }
  
  if (length(cur_e) == 1) {
    return(cur_e[1])
  }
  
}
  return(cur_e[1])

}

generate_data <- function(n, p, n_cat, k, alpha, mu, pi, seed) {
  set.seed(seed)
  w <- sample(1:k, n, replace=TRUE, alpha)
  x <- matrix(0, nrow=n, ncol=p)
  
  for (i in 1:n) {
    for (j in 1:p) {
      x[i, j] <- bos_model(n_cat[j], mu[[w[i]]][j], pi[[w[i]]][j])
    }
  }
  
  return(list(x, w))
  
}
