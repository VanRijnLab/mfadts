require(purrr)
require(data.table)
require(tidyr)
require(ggplot2)

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

# Fitting functions -------------------------------------------------------

calculate_tau_from_model <- function (model) {
  tau <- - coef(model)[[1]] * model_params$s
  return (tau)
}

calculate_activation_from_model <- function (model) {
  ac <- coef(model)[[1]] * model_params$s
  return (ac)
}

calculate_sequence_activation <- function (sequence_list) {
  ac <- map_dbl(sequence_list, function (x) {
    activation(x$time_within, x$time_between, model_params$h, model_params$decay)
  })
  return (ac)
}

fit_tau <- function (d_seq_list, model_params) {
  ids <- map_chr(d_seq_list, ~.$id)
  correct <- map_int(d_seq_list, ~.$correct)
  
  ac <- calculate_sequence_activation(d_seq_list)
  
  m <- glm(correct ~ 1 + offset((1/model_params$s) * ac),
           family = binomial)
  tau <- calculate_tau_from_model(m)
  
  return (data.table(id = ids, tau = tau))
}


fit_activation <- function (d_seq_list, model_params) {
  correct <- map_int(d_seq_list, ~.$correct)
  
  tau <- rep(model_params$tau, length(correct))
  
  m <- glm(correct ~ 1 + offset((-1/model_params$s) * tau),
           family = binomial)
  ac <- calculate_activation_from_model(m)

  return (ac)
}


fit_d <- function (d_seq_list, model_params, bs_iter = 100) {
  ac <- fit_activation(d_seq_list, model_params)
  
  d_bs <- map(d_seq_list, function (x) {
    
    d_i <- 1
    d_upper <- 2
    d_lower <- 0
    
    # Binary search
    for (i in seq_len(bs_iter)) {
      ac_i <- activation(x$time_within, x$time_between, model_params$h, d_i)
      
      ac_diff <- ac_i - ac
      
      if (ac_diff > 0) { # Predicted activation too high, so d is too small
        d_lower <- d_i
      } else { # Predicted activation too low, so d is too large
        d_upper <- d_i
      }
      
      d_i <- (d_lower + d_upper) / 2
      
    }
    
    return (list(id = x$id, d = d_i))
    
  }) |>
    rbindlist()
  
  return (d_bs)
}

fit_h <- function (d_seq_list, model_params, bs_iter = 100) {
  ac <- fit_activation(d_seq_list, model_params)
  
  h_bs <- map(d_seq_list, function (x) {
    
    h_i <- .5
    h_upper <- 1
    h_lower <- 0
    
    # Binary search
    for (i in seq_len(bs_iter)) {
      ac_i <- activation(x$time_within, x$time_between, h_i, model_params$decay)
      
      ac_diff <- ac_i - ac
      
      if (ac_diff > 0) { # Predicted activation too high, so h is too small
        h_lower <- h_i
      } else { # Predicted activation too low, so h is too large
        h_upper <- h_i
      }
      
      h_i <- (h_lower + h_upper) / 2
      
    }
    
    return (list(id = x$id, h = h_i))
    
  }) |>
    rbindlist()
  
  return (h_bs)
}


fit_d_and_h <- function (d_seq_list, model_params, bs_iter = 100) {
  ac <- fit_activation(d_seq_list, model_params)
  
  d_bs <- map(d_seq_list, function (x) {
    
    d_i <- 1
    d_upper <- 2
    d_lower <- 0
    
    # Binary search
    for (i in seq_len(bs_iter)) {
      ac_i <- activation(x$time_within, x$time_between, model_params$h, d_i)
      
      ac_diff <- ac_i - ac
      
      if (ac_diff > 0) { # Predicted activation too high, so d is too small
        d_lower <- d_i
      } else { # Predicted activation too low, so d is too large
        d_upper <- d_i
      }
      
      d_i <- (d_lower + d_upper) / 2
      
    }
    
    return (list(id = x$id, d = d_i))
    
  }) |>
    rbindlist()
  
  h_bs <- map(d_seq_list, function (x) {
    
    h_i <- .5
    h_upper <- 1
    h_lower <- 0
    
    # Binary search
    for (i in seq_len(bs_iter)) {
      ac_i <- activation(x$time_within, x$time_between, h_i, model_params$decay)
      
      ac_diff <- ac_i - ac
      
      if (ac_diff > 0) { # Predicted activation too high, so h is too small
        h_lower <- h_i
      } else { # Predicted activation too low, so h is too large
        h_upper <- h_i
      }
      
      h_i <- (h_lower + h_upper) / 2
      
    }
    
    return (list(id = x$id, h = h_i))
    
  }) |>
    rbindlist()
  
  dh_bs <- d_bs[h_bs, on = .(id)]
  
  return (dh_bs)
}


fit_parameters <- function (d_seq_list, model_params) {
  tau <- fit_tau(d_seq_list, model_params)
  d_and_h <- fit_d_and_h(d_seq_list, model_params)
  
  fitted_params <- tau[d_and_h, on = .(id)]
  
  return (fitted_params)
}


calculate_activation_over_time <- function (d, timestep = 0.5, h = 1.0, decay = 0.5, tau = -3.0, s = 0.5) {
  
  act <- crossing(time = seq(0, max(d$presentation_start_time), by = timestep),
                  h = h,
                  decay = decay,
                  tau = tau,
                  activation = -Inf)
  setDT(act)
  
  s1_end <- d[1, time_until_session_end + presentation_start_time]
  s2_start <- d[.N, presentation_start_time - time_since_session_start]
  
  for (i in 1:nrow(act)) {
    
    ts <- act[i,]$time
    h <- act[i,]$h
    decay <- act[i,]$decay
    
    obs <- d[presentation_start_time <= ts]
    
    if (ts <= s1_end) { # Within session 1
      
      tjw <- ts - obs$presentation_start_time
      tb <- 0
      
    } else if (ts <= s2_start) { # Between sessions
      
      tjw <- obs$time_until_session_end
      tb <- ts - s1_end
      
      
    } else { # Within session 2
      
      tjw <- obs$time_until_session_end + (ts - s2_start)
      tb <- s2_start - s1_end
      
    }
    
    act[i,]$activation <- activation(tjw, tb, h, d = decay)
    
  }
  
  act[, p_recall := p_recall(activation, tau, s = s)]
  
  return (act)
}


split_data_into_windows <- function (data, n_windows) {
  if (n_windows == 1) {
    data[, window := 1]
  } else {
    data[, window := cut(log(time_between), breaks = n_windows, labels = FALSE)]
  }
  
  data_split <- split(data, data$window)
  return (data_split)
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



# Goodness of fit ---------------------------------------------------------

log_likelihood <- function(y, p, eps = 1e-15, average = TRUE) {
  p <- pmin(pmax(p, eps), 1 - eps)
  ll <- sum(y * log(p) + (1 - y) * log(1 - p))
  if (average) {
    ll <- ll / length(y)
  }
  return (ll)
}

aic <- function(k, y, p, average = TRUE) {
  n <- length(y)
  ll <- log_likelihood(y, p, average = average)
  if (average) {
    return ((2 * k) / n - 2 * ll)
  }
  return (2 * k - 2 * ll)
}

akaike_weights <- function(aic) {
  delta_aic <- aic - min(aic)
  return (exp(-delta_aic / 2) / sum(exp(-delta_aic / 2)))
}



# Plotting ----------------------------------------------------------------

label_x <- c(1, 10, 60, 6*60, 24*60, 7*24*60, 7*7*24*60)
label_y <- 1.09
label_txt <- c("1 min", "10 min", "1 h", "6 h", "24 h", "1 wk", "7 wks")

plot_timescales <- function () {
  list(
    scale_x_log10(
      breaks = scales::trans_breaks("log10", function(x) 10^x),
      labels = scales::trans_format("log10", scales::math_format(10^.x)),
      expand = c(0, 0),
      sec.axis = sec_axis(~.x, breaks = label_x, labels = label_txt)
    ),
    annotation_logticks(sides = "b", outside = T),
    coord_cartesian(xlim = c(min(label_x)/2, max(label_x)*2), clip = "off"),
    theme_bw(base_size = 14),
    theme(plot.margin = margin(7, 14, 7, 7),
          panel.grid.major.x = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank())
  )
}

