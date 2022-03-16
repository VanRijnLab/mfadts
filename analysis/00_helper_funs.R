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

# generate_seq_list <- function (data) {
#   
#   if (!is.data.table(data) |
#       !is_empty(setdiff(c("id", "time_within", "time_between", "time_between_bin", "correct"), colnames(data)))) {
#     stop("Expected 'data' to be a data.table with columns 'id', 'time_within', 'time_between', 'time_between_bin', 'correct'")
#   }
#   
#   data_by_bin <- split(data, data$time_between_bin, drop = TRUE)
#   
#   seq_list <- map(data_by_bin, function (data_bin) {
#     
#     ids <- unique(data_bin$id)
#     n_seq <- length(ids)
#     seq_list_bin <- as.list(rep(NA, n_seq))
#     
#     for(i in 1:n_seq) {
#       
#       seq_item <- data_bin[id == ids[i]]
#       
#       seq_list_bin[[i]] <- list(id = ids[i], 
#                                 time_within = seq_item[1:.N - 1, time_within],
#                                 time_between = seq_item[1, time_between],
#                                 time_between_bin = seq_item[1, time_between_bin],
#                                 correct = seq_item[.N, correct])
#     }
#     
#     return (seq_list_bin)
#     
#   })
#   
#   return (seq_list)
#   
# }

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




generate_seq_list_windows <- function (data) {
  
  if (!is.data.table(data) |
      !is_empty(setdiff(c("id", "time_within", "time_between", "window", "correct"), colnames(data)))) {
    stop("Expected 'data' to be a data.table with columns 'id', 'time_within', 'time_between', 'window', 'correct'")
  }
  
  data_by_window <- split(data, data$window, drop = TRUE)
  
  seq_list <- map(data_by_window, function (data_window) {
    
    ids <- unique(data_window$id)
    n_seq <- length(ids)
    seq_list_window <- as.list(rep(NA, n_seq))
    
    for(i in 1:n_seq) {
      
      seq_item <- data_window[id == ids[i]]
      
      seq_list_window[[i]] <- list(id = ids[i], 
                                time_within = seq_item[1:.N - 1, time_within],
                                time_between = seq_item[1, time_between],
                                window = seq_item[1, window],
                                correct = seq_item[.N, correct])
    }
    
    return (seq_list_window)
    
  })
  
  return (seq_list)
  
}