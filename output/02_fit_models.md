Fit memory models
================
Maarten van der Velde
Last updated: 2022-05-06

-   [Overview](#overview)
-   [Setup](#setup)
    -   [Model fitting setup](#model-fitting-setup)
    -   [Data setup](#data-setup)
-   [Fit model](#fit-model)
    -   [Optimal retrieval threshold](#optimal-retrieval-threshold)
        -   [Short intervals only](#short-intervals-only)
    -   [Optimal activation](#optimal-activation)
    -   [Optimal decay](#optimal-decay)
    -   [Optimal scaling factor h](#optimal-scaling-factor-h)
-   [Session info](#session-info)

# Overview

This code fits the ACT-R memory model to the retrieval practice data. We vary which parameter is tuned and which are kept constant, as well as the number of windows in which the data are split before we fit the model.

The model as we use it here consists of two functions:

-   **Activation**: *A* = ln (∑<sub>*j*</sub>(*t*<sub>*w**j*</sub>+*h*\**t*<sub>*b*</sub>)<sup>−*d*</sup>)<br>Parameters: scaling factor *h* and decay *d*. Here, *t*<sub>*w**j*</sub> refers to within-session intervals and *t*<sub>*b*</sub> to between-session intervals.
-   **Recall probability**: $p = \\frac{1}{1 + e^{-(A - \\tau)/s}}$<br>Parameters: retrieval threshold *τ* (and activation noise *s*, which we don't vary here).

# Setup

``` r
library(data.table)
library(purrr)
library(furrr)

source("00_helper_funs.R")

future::plan("multiprocess", workers = 6) # Set to desired number of cores

set.seed(0)
```

## Model fitting setup

To save time, try to load model fitting results from cache. Set to FALSE to refit the models.

``` r
use_cache <- TRUE
```

Set default parameters for the ACT-R memory model. These values are used whenever a parameter is held constant.

``` r
model_params <- list(
  s = .5,
  decay = .5,
  h = 1
)
```

We want to try different splits of the data, ranging from a single window that includes everything to 100 windows.

``` r
n_windows <- c(1:10, 20, 25, 50, 100)
```

## Data setup

``` r
d_f <- fread(file.path("..", "data", "cogpsych_data_formatted.csv"))
```

``` r
str(d_f)
```

    ## Classes 'data.table' and 'data.frame':   171219 obs. of  11 variables:
    ##  $ id                      : chr  "17_18_EN_anon-001_1174_1" "17_18_EN_anon-001_1174_1" "17_18_EN_anon-001_1174_1" "17_18_EN_anon-001_1174_1" ...
    ##  $ user                    : chr  "17_18_EN_anon-001" "17_18_EN_anon-001" "17_18_EN_anon-001" "17_18_EN_anon-001" ...
    ##  $ fact                    : chr  "1174_1" "1174_1" "1174_1" "1174_1" ...
    ##  $ time                    : POSIXct, format: "2017-10-14 14:59:39" "2017-10-14 14:59:47" ...
    ##  $ presentation_start_time : num  0.00 6.02 6.92e+01 7.73e+05 0.00 ...
    ##  $ time_since_session_start: num  22.9 28.9 92.1 83.5 38 ...
    ##  $ time_until_session_end  : num  270 264 201 156 255 ...
    ##  $ correct                 : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ rt                      : int  6018 4442 3418 4002 4790 2592 2409 4886 1406 5565 ...
    ##  $ time_between            : num  772370 772370 772370 772370 772370 ...
    ##  $ time_within             : num  354 348 284 240 327 ...
    ##  - attr(*, ".internal.selfref")=<externalptr>

Each user-fact pair has a single learning sequence associated with it, consisting of three or more trials in one session and the first trial in the next session.

Isolate the last observation per learning sequence (i.e., the one after the between-session interval). This is the observation that the model has to predict, given all prior observations in the sequence.

``` r
d_last <- d_f[, .SD[.N], by = id]
setorder(d_last, time_between)
```

Define the time windows for all splits of the data (time values are in seconds).

``` r
window_range <- map_dfr(n_windows, function (n_w) {

  d_windows <- copy(d_last)
  
  if (n_w == 1) {
    d_windows[, window := 1]
  } else {
    d_windows[, window := cut(log(time_between), breaks = n_w, labels = FALSE)]
  }
  
  # Get the window range(s)
  window_range <- d_windows[, .(start = min(time_between), end = max(time_between)), by = .(window)]
  window_range[, geom_mean := sqrt(start*end), by = .(window)]
  setorder(window_range, window)
  window_range[, window := window]
  window_range[, n_windows := n_w]

  return (window_range)
})

window_range
```

    ##      window       start         end    geom_mean n_windows
    ##   1:      1      38.363 4605678.952   13292.3911         1
    ##   2:      1      38.363   13238.121     712.6388         2
    ##   3:      2   13371.307 4605678.952  248161.1315         2
    ##   4:      1      38.363    1889.051     269.2019         3
    ##   5:      2    1902.540   93316.012   13324.3178         3
    ##  ---                                                      
    ## 242:     96 2573482.617 2864244.105 2714973.7411       100
    ## 243:     97 2922474.771 3169998.677 3043721.5966       100
    ## 244:     98 3280127.598 3525867.324 3400778.5456       100
    ## 245:     99 3650704.112 3812538.241 3730743.7641       100
    ## 246:    100 4605678.952 4605678.952 4605678.9520       100

# Fit model

## Optimal retrieval threshold

For each window split, find the optimal retrieval threshold using logistic regression, keeping other parameters constant.

``` r
logreg_tau_file <- file.path("..", "data", "logistic_regression_tau.csv")

if (!use_cache | !file.exists(logreg_tau_file)) {
  
  lr_tau <- future_map_dfr(n_windows, function (n_w) {
    
    d_windows <- copy(d_last)
    
    # Split the data into windows
    if (n_w == 1) {
      d_windows[, window := 1]
    } else {
      d_windows[, window := cut(log(time_between), breaks = n_w, labels = FALSE)]
    }
    
    d_windows_split <- split(d_windows, d_windows$window)
    
    # Within each quantile, perform logistic regression
    lr_nw <- imap_dfr(d_windows_split, function (window, n) {
      
      window[, sequence := 1:.N]
      d_window <- d_f[window[, .(id, window, sequence)], on = .(id)]
      
      d_seq_list <- generate_seq_list(d_window)
  
      correct <- map_int(d_seq_list, ~.$correct)
      
      ac <- map_dbl(d_seq_list, function (x) {
          activation(x$time_within, x$time_between, model_params$h, model_params$decay)
      })
      
      m <- glm(correct ~ 1 + offset((1/model_params$s) * ac),
                 family = binomial)
      tau <- - coef(m)[[1]] * model_params$s
    
      return (list(n_windows = n_w, window = n, tau = tau))
    })
    
    return (lr_nw)  
  }, .progress = interactive())
  
  setDT(lr_tau)
  fwrite(lr_tau, logreg_tau_file)
}

lr_tau <- fread(logreg_tau_file)
```

### Short intervals only

For comparison, find the best threshold for reasonably short intervals: 0-10 minutes (Figure 1B in the paper).

``` r
logreg_tau_short_file <- file.path("..", "data", "logistic_regression_tau_short.csv")

if (!use_cache | !file.exists(logreg_tau_short_file)) {
  
  d_short <- copy(d_last)
  d_short <- d_short[time_between <= 10*60]
  d_short[, sequence := 1:.N]
  
  d_short <- d_f[d_short[, .(id, sequence)], on = .(id)]
  d_short[, window := 1]
  
  d_seq_list <- generate_seq_list(d_short)
  
  correct <- map_int(d_seq_list, ~.$correct)
  
  ac <- map_dbl(d_seq_list, function (x) {
    activation(x$time_within, x$time_between, model_params$h, model_params$decay)
  })
  
  m <- glm(correct ~ 1 + offset((1/model_params$s) * ac),
           family = binomial)
  tau_short <- - coef(m)[[1]] * model_params$s
  
  tau_short <- as.data.table(tau_short)
  fwrite(tau_short, logreg_tau_short_file)
}

tau_short <- fread(logreg_tau_short_file)
```

## Optimal activation

Find the optimal activation per window using logistic regression, keeping other parameters constant.

``` r
logreg_activation_file <- file.path("..", "data", "logistic_regression_activation.csv")

if (!use_cache | !file.exists(logreg_activation_file)) {
  
  lr_ac <- future_map_dfr(n_windows, function (n_w) {
    
    d_windows <- copy(d_last)
    
    # Split the data into windows
    if (n_w == 1) {
      d_windows[, window := 1]
    } else {
      d_windows[, window := cut(log(time_between), breaks = n_w, labels = FALSE)]
    }

    d_windows_split <- split(d_windows, d_windows$window)
    
    # Find the appropriate tau value
    window_1 <- d_windows_split[[1]]
    window_1[, sequence := 1:.N]
    d_window_1 <- d_f[window_1[, .(id, window, sequence)], on = .(id)]
    window_1_seq_list <- generate_seq_list(d_window_1)
    correct <- map_int(window_1_seq_list, ~.$correct)
    
    ac <- map_dbl(window_1_seq_list, function (x) {
      activation(x$time_within, x$time_between, model_params$h, model_params$decay)
    })
    
    m <- glm(correct ~ 1 + offset((1/model_params$s) * ac),
             family = binomial)
    tau <- - coef(m)[[1]] * model_params$s
    
    
    # Within each quantile, perform logistic regression to find A
    lr_nw <- imap_dfr(d_windows_split, function (window, n) {
      
      window[, sequence := 1:.N]
      d_window <- d_f[window[, .(id, window, sequence)], on = .(id)]
      
      d_seq_list <- generate_seq_list(d_window)
      
      correct <- map_int(d_seq_list, ~.$correct)
      
      tau_fixed <- rep(tau, length(correct))
      
      m <- glm(correct ~ 1 + offset((-1/model_params$s) * tau_fixed),
               family = binomial)
      
      ac <- coef(m)[[1]] * model_params$s
      
      return (list(n_windows = n_w, window = n, tau = tau, activation = ac))
    })
    
    return (lr_nw)
    
  }, .progress = interactive())
  
  setDT(lr_ac)
  fwrite(lr_ac, logreg_activation_file)
}

lr_ac <- fread(logreg_activation_file)
```

## Optimal decay

Using the optimal activation values determined in the previous step, do a binary search to find the associated decay for each learning sequence.

``` r
bs_d_indiv_file <- file.path("..", "data", "binary_search_indiv_d.csv")

if (!use_cache | !file.exists(bs_d_indiv_file)) {
  
  bs_d_indiv <- future_map_dfr(n_windows, function (n_w) {
    
    d_windows <- copy(d_last)
    
    # Split the data into windows
    if (n_w == 1) {
      d_windows[, window := 1]
    } else {
      d_windows[, window := cut(log(time_between), breaks = n_w, labels = FALSE)]
    }

    d_windows_split <- split(d_windows, d_windows$window)
    
    # Within each quantile, use binary search to find d from A
    d_nw <- imap_dfr(d_windows_split, function (window, n) {
      
      window[, sequence := 1:.N]
      d_window <- d_f[window[, .(id, window, sequence)], on = .(id)]
      
      ac_window <- lr_ac[n_windows == n_w & window == n, activation]
      
      d_seq_list <- generate_seq_list(d_window)
      
      d_bs <- map_dfr(d_seq_list, function (x) {
        
        d_i <- 1
        d_upper <- 2
        d_lower <- 0
        
        # Binary search
        for (i in 1:100) {
          ac <- activation(x$time_within, x$time_between, model_params$h, d_i)
          
          ac_diff <- ac - ac_window
          
          if (ac_diff > 0) { # Predicted activation too high, so d is too small
            d_lower <- d_i
          } else { # Predicted activation too low, so d is too large
            d_upper <- d_i
          }
          
          d_i <- (d_lower + d_upper) / 2
          
        }
        
        return(list(n_windows = n_w, window = n, id = x$id, d = d_i))
        
      })
      
      return (d_bs)

    })
    
    return (d_nw)
    
  }, .progress = interactive())
  
  setDT(bs_d_indiv)
  fwrite(bs_d_indiv, bs_d_indiv_file)
}

bs_d_indiv <- fread(bs_d_indiv_file)
```

## Optimal scaling factor h

In the same way, do a binary search to find the scaling factor for each learning sequence, based on the optimal activation.

``` r
bs_h_indiv_file <- file.path("..", "data", "binary_search_indiv_h.csv")

if (!use_cache | !file.exists(bs_h_indiv_file)) {
  
  bs_h_indiv <- future_map_dfr(n_windows, function (n_w) {
    
    d_windows <- copy(d_last)
    
    # Split the data into windows
    if (n_w == 1) {
      d_windows[, window := 1]
    } else {
      d_windows[, window := cut(log(time_between), breaks = n_w, labels = FALSE)]
    }
    
    d_windows_split <- split(d_windows, d_windows$window)
    
    # Within each quantile, use binary search to find h from A
    h_nw <- imap_dfr(d_windows_split, function (window, n) {
      
      window[, sequence := 1:.N]
      d_window <- d_f[window[, .(id, window, sequence)], on = .(id)]
      
      ac_window <- lr_ac[n_windows == n_w & window == n, activation]
      
      d_seq_list <- generate_seq_list(d_window)
      
      h_bs <- map_dfr(d_seq_list, function (x) {
        
        h_i <- .5
        h_upper <- 1
        h_lower <- 0
        
        # Binary search
        for (i in 1:100) {
          ac <- activation(x$time_within, x$time_between, h_i, model_params$decay)
          
          ac_diff <- ac - ac_window
          
          if (ac_diff > 0) { # Predicted activation too high, so h is too small
            h_lower <- h_i
          } else { # Predicted activation too low, so h is too large
            h_upper <- h_i
          }
          
          h_i <- (h_lower + h_upper) / 2
          
        }
        
        return(list(n_windows = n_w, window = n, id = x$id, h = h_i))
        
      })
      
      return (h_bs)
      
    })
    
    return (h_nw)
    
  }, .progress = interactive())
  
  setDT(bs_h_indiv)
  fwrite(bs_h_indiv, bs_h_indiv_file)
}

bs_h_indiv <- fread(bs_h_indiv_file)
```

# Session info

``` r
sessionInfo()
```

    ## R version 3.6.3 (2020-02-29)
    ## Platform: x86_64-pc-linux-gnu (64-bit)
    ## Running under: Ubuntu 18.04.6 LTS
    ## 
    ## Matrix products: default
    ## BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.7.1
    ## LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1
    ## 
    ## locale:
    ##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
    ##  [3] LC_TIME=en_GB.UTF-8        LC_COLLATE=en_US.UTF-8    
    ##  [5] LC_MONETARY=en_GB.UTF-8    LC_MESSAGES=en_US.UTF-8   
    ##  [7] LC_PAPER=en_GB.UTF-8       LC_NAME=C                 
    ##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
    ## [11] LC_MEASUREMENT=en_GB.UTF-8 LC_IDENTIFICATION=C       
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ## [1] furrr_0.1.0       future_1.13.0     purrr_0.3.2       data.table_1.13.6
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] Rcpp_1.0.6       knitr_1.23       magrittr_2.0.1   tidyselect_1.1.1
    ##  [5] R6_2.4.0         rlang_0.4.10     fansi_0.4.0      stringr_1.4.0   
    ##  [9] dplyr_1.0.7      globals_0.12.4   tools_3.6.3      parallel_3.6.3  
    ## [13] xfun_0.21        utf8_1.1.4       DBI_1.1.0        ellipsis_0.3.2  
    ## [17] htmltools_0.3.6  yaml_2.2.0       digest_0.6.19    tibble_2.1.3    
    ## [21] lifecycle_1.0.1  crayon_1.4.1     vctrs_0.3.8      codetools_0.2-16
    ## [25] glue_1.4.2       evaluate_0.14    rmarkdown_2.6    stringi_1.4.3   
    ## [29] pillar_1.6.3     compiler_3.6.3   generics_0.1.0   jsonlite_1.6    
    ## [33] listenv_0.7.0    pkgconfig_2.0.2
