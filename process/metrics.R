metrics_index <- c("AE", "MAE","RAE","KLD","NAE","NRAE","NKLD","TrainBiasQ")


smooth <- function(p, eps){
  return( (eps+p)/(eps*length(p)+1) )
}

# Absolute Error
AE <- function(p_true, p_hat){
  return(sum(abs(p_hat-p_true)))
}

# Mean Absolute Error
MAE <- function(p_true, p_hat){
  return(1/length(p_hat)*sum(abs(p_hat-p_true)))
}

# Normalized Absolute Error
NAE <- function(p_true, p_hat){
  p_min <- min(p_true)
  return(sum(abs(p_hat-p_true))/(2*(1-p_min)))
}

# Relative Absolute Error
RAE <- function(p_true, p_hat, eps = 1e-08){
  if(eps>0.0){
    p_true <- smooth(p_true,eps)
    p_hat <- smooth(p_hat,eps)
  }
  
  return(1/length(p_hat)*sum(abs(p_hat-p_true)/p_true))
  
}

# Normalized Relative Absolute Error
NRAE <- function(p_true, p_hat, eps = 1e-08){
  if(eps>0.0){
    p_true <- smooth(p_true,eps)
    p_hat <- smooth(p_hat,eps)
  }
  L <- length(p_true)
  p_min <- min(p_true)
  return(sum(abs(p_hat-p_true)/p_true)/(L-1 + (1-p_min)/p_min) )
  
}

# Kullback-Leibler Divergence
KLD <- function(p_true, p_hat, eps = 1e-08){
  if(eps>0){
    p_true <- smooth(p_true,eps)
    p_hat <- smooth(p_hat,eps)
  }
  return(sum(p_true*log2(p_true/p_hat)))
}

# Normalized Kullback-Leibler Divergence
NKLD <- function(p_true, p_hat, eps = 1e-08){
  ekld <- exp(KLD(p_true, p_hat, eps))
  return(max(0,2*ekld/(1+ekld) - 1))
}

# Absolute Distance Ratio
TrainBiasQ <- function(p_true, p_hat, p_train){
  den <- sum(abs(p_hat-p_train))
  if (den == 0) den <- 1e-10
  return(sum(abs(p_hat-p_true))/den)
}
