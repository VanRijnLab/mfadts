require(purrr)


# Model functions ---------------------------------------------------------

# Time lags are split into a within-session component (tjw) and a between-session component (tjb).
# The between-session component can be scaled using h.
activation <- function(tjw, tb, h, d = 0.5) {
  log(sum((tjw + h * tb)^-d))
}

# Recall probability is a sigmoid function
p_recall <- function(a, tau, s) {
  1/(1 + exp(-(a - tau)/s))
}




# Generating lists of learning sequences ----------------------------------


generate_seq_list <- function (data) {
  
  n_seq <- length(unique(data$sequence))
  seq_list <- as.list(rep(NA, n_seq)) 
  
  for (i in 1:n_seq) {
    
    seq_item <- data[sequence == i]
    
    seq_list[[i]] <- list(id = seq_item[1, id], 
                          time_within = seq_item[1:.N - 1, time_within],
                          time_between = seq_item[1, time_between],
                          window = seq_item[1, window],
                          correct = seq_item[.N, correct])
    
  }
  
  return(seq_list)
  
}
