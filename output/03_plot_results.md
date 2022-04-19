Plot modelling results
================
Maarten van der Velde
Last updated: 2022-04-19

# Setup

``` r
library(data.table)
library(dplyr)
library(purrr)
library(ggplot2)
library(cowplot)

source("00_helper_funs.R")
```

Load data:

``` r
d_f <- fread(file.path("..", "data", "cogpsych_data_formatted.csv"))
```

Get the last observation per learning sequence:

``` r
d_last <- d_f[, .SD[.N], by = id]
setorder(d_last, time_between)
```

Load fitted models:

``` r
lr_tau <- fread(file.path("..", "data", "logistic_regression_tau.csv"))
lr_activation <- fread(file.path("..", "data", "logistic_regression_activation.csv"))
bs_d_indiv <- fread(file.path("..", "data", "binary_search_indiv_d.csv"))
bs_h_indiv <- fread(file.path("..", "data", "binary_search_indiv_h.csv"))
tau_short <- fread(file.path("..", "data", "logistic_regression_tau_short.csv"))
```

Set default parameters for the memory model:

``` r
model_params <- list(
  s = .5,
  decay = .5,
  h = 1
)
```

Set parameters for splitting the data into windows:

``` r
n_windows <- c(1:10, 20, 25, 50, 100)
```

General function for predicting recall for a given parameter
configuration:

``` r
predict_recall <- function (seq_list, h = model_params$h, decay = model_params$decay, tau, s = model_params$s, windows = 1) {
  
  pred_correct <- map_dbl(seq_list, function (x) {
    
    seq_d <- ifelse(is.data.table(decay), decay[id == x$id, d], decay)
    seq_h <- ifelse(is.data.table(h), h[id == x$id, h], h)
    
    if(is.data.table(tau)) {
      if("id" %in% names(tau)) seq_tau <- tau[id == x$id, tau]
      if("n_windows" %in% names(tau)) seq_tau <- tau[n_windows == windows & window == x$window, tau]
    } else {
      seq_tau <- tau
    }
    
    ac <- activation(x$time_within, x$time_between, seq_h, seq_d)
    p_recall(ac, seq_tau, s)
    
  })
  
  pred <- data.table(id = map_chr(seq_list, ~.$id), pred_correct = pred_correct)
  
  return (pred)
}
```

## General plot elements

Define the time windows for all splits of the data:

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
```

Plot colours:

``` r
pred_col <- "#CC3311"    # red
obs_col <- "#000000"     # black
window_col <- "#33BBEE"  # blue
section_col <- "#FD520F" # orange
```

Interpretable time labels:

``` r
label_x <- c(1, 10, 60, 6*60, 24*60, 7*24*60, 7*7*24*60)
label_y <- 1.09
label_txt <- c("1 min", "10 min", "1 h", "6 h", "24 h", "1 wk", "7 wks")
```

General function for plotting comparison between data and model:

``` r
plot_comparison <- function (d_model,
                             n_w = 1,
                             label_pos = list(data = list(x = 35000, y = .54), 
                                              model = list(x = 35000, y = .46)),
                             print_plot = TRUE) {
  
  
  plot_dodge <- function(y, dodge = .1) {
    return (y * (1 + dodge) - dodge/2)
  }
  
  p <- ggplot() +
    # Window background
    geom_rect(data = window_range[n_windows == n_w],
              aes(xmin = start/60, xmax = ifelse(is.na(shift(start, -1)), end, shift(start, -1))/60,
                  ymin = -Inf, ymax = Inf, alpha = as.factor(window)),
              fill = window_col) +
    # Jittered observations along edges
    geom_point(data = d_last, 
               aes(x = time_between/60, y = plot_dodge(correct, .05)),
               position = position_jitter(width = 0, height = .025, seed = 123),
               colour = "grey20", size = .001, pch = ".", alpha = .1) +
    # GAM: data
    geom_smooth(data = d_last,
                aes(x = time_between/60, y = correct),
                method = "gam", formula = y ~ s(x, bs = "cs"),
                colour = obs_col, lty = 1, lwd = 1) +
    # GAM: model
    geom_smooth(data = d_model, 
                aes(x = time_between/60, y = pred_correct),
                method = "gam", formula = y ~ s(x, bs = "cs"), 
                colour = pred_col, fill = pred_col, lty = 1, lwd = .75) +
    # Labels
    annotate("text", x = label_pos$data$x, y = label_pos$data$y,
             label = "Data", colour = obs_col) +
    annotate("text", x = label_pos$model$x, y = label_pos$model$y,
             label = "Model", colour = pred_col) +
    # Plot setup
    scale_x_log10(
      breaks = scales::trans_breaks("log10", function(x) 10^x),
      labels = scales::trans_format("log10", scales::math_format(10^.x)),
      expand = c(0, 0),
      sec.axis = sec_axis(~.x, breaks = label_x, labels = label_txt)
    ) +
    scale_y_continuous(breaks = seq(0, 1, by = .25), labels = scales::percent_format()) +
    scale_alpha_manual(values = rep(c(.1, .25), ceiling(n_w/2))) +
    guides(colour = "none",
           alpha = "none") +
    labs(x = "Between-session interval (minutes)",
         y = "Response accuracy") +
    annotation_logticks(sides = "b", outside = T) +
    coord_cartesian(ylim = c(0, 1), xlim = c(window_range[1, start], window_range[.N, end])/60, clip = "off") +
    theme_bw(base_size = 14) +
    theme(plot.margin = margin(7, 14, 7, 7),
          panel.grid.major.x = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank())
  
  if (print_plot) print(p)
  return (p)
    
}
```

General function for plotting parameter change over time:

``` r
plot_parameter <- function(d_parameter,
                           parameter_name = "",
                           n_w = 1,
                           log_x = TRUE,
                           log_y = FALSE,
                           print_plot = TRUE) {
  
  # Calculate R-squared
  x <- d_parameter[n_windows == n_w, geom_mean]/60
  y <- d_parameter[n_windows == n_w, parameter]
  if (log_x) x <- log(x)
  if (log_y) y <- log(y)
  
  m <- lm(y ~ x)

  eq <- substitute(parameter_name == a-b %*% ln(italic(t)), 
                   list(parameter_name = parameter_name,
                        a = format(unname(coef(m)[1]), digits = 3),
                        b = format(abs(unname(coef(m)[2])), digits = 3)))
  
  eq <- as.character(as.expression(eq))
  
  rsq <- paste("R^2 ==", scales::number(summary(m)$r.squared, accuracy = .01))

  p <- ggplot() +
    # Window background
    geom_rect(data = window_range[n_windows == n_w],
              aes(xmin = start/60, xmax = ifelse(is.na(shift(start, -1)), end, shift(start, -1))/60,
                  ymin = ifelse(log_y, 0, -Inf), ymax = Inf, alpha = as.factor(window)),
              fill = window_col) +
    # Regression line
    geom_smooth(data = d_parameter[n_windows == n_w], 
                aes(y = parameter, x = geom_mean/60), 
                method = "lm", formula = y ~ x, 
                colour = pred_col, fill = pred_col) +
    # Parameter values
    geom_point(data = d_parameter[n_windows == n_w],
               aes(y = parameter, x = geom_mean/60)) +
    scale_alpha_manual(values = rep(c(.1, .25), ceiling(n_w/2))) +
    # R-squared
    geom_label(aes(x = Inf, y = Inf, label = rsq),
              label.padding = unit(.5, "lines"),
              label.size = NA,
              fill = NA,
              hjust = "inward", vjust = "inward",
              parse = TRUE) +
    geom_label(aes(x = ifelse(log_x, 0, -Inf), y = ifelse(log_y, 0, -Inf), label = eq),
              label.padding = unit(.5, "lines"),
              label.size = NA,
              fill = NA,
              hjust = "inward", vjust = "inward",
              parse = TRUE) +
    # Plot setup
    guides(alpha = "none") +
    labs(x = "Between-session interval (minutes)",
         y = "Fitted parameter") +
    scale_x_continuous(sec.axis = sec_axis(~.x, breaks = label_x, labels = label_txt)) +
    coord_cartesian(xlim = c(window_range[1, start], window_range[.N, end])/60,
                    ylim = c(min(y) - .1*diff(range(y)), max(y) + .1*diff(range(y))),
                    clip = "off") +
    theme_bw(base_size = 14) +
    theme(plot.margin = margin(7, 14, 7, 7),
          panel.grid.major.x = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank())
  
  
  # Transform scales if required
  if (log_x) {
    p <- p +
      scale_x_log10(
        breaks = scales::trans_breaks("log10", function(x) 10^x),
        labels = scales::trans_format("log10", scales::math_format(10^.x)),
        expand = c(0, 0),
        sec.axis = sec_axis(~.x, breaks = label_x, labels = label_txt)
      ) +
      annotation_logticks(sides = "b", outside = T)
  }
  
  if (log_y) {
    p <- p +
      scale_y_log10() +
      annotation_logticks(sides = "l", outside = T)
  }

  if (print_plot) print(p)
  return (p)

}
```

# Single parameter

What does predicted retention look like if we use a single parameter
configuration fitted across the whole range of intervals?

Divide the data into learning sequences for plotting:

``` r
d_single <- copy(d_last)
d_single[, sequence := 1:.N]
d_single <- d_f[d_single[, .(id, sequence)], on = .(id)]
d_single[, window := 1]
d_seq_list_single <- generate_seq_list(d_single)
```

## Whole data set

### Threshold

``` r
single_tau <- lr_tau[n_windows == 1, tau]
```

The best-fitting threshold for the entire range of intervals is
-4.6072046. If we consistently use this threshold in the model, it
initially overpredicts retention (indicating that the threshold is too
low relative to the activation), gets the prediction right around the
mode of the interval distribution, and eventually underpredicts
retention (indicating that the threshold is too
high).

``` r
pred_single_tau <- predict_recall(seq_list = d_seq_list_single, tau = single_tau)
d_single_tau <- d_last[pred_single_tau, on = .(id)]

p_single_tau <- plot_comparison(d_model = d_single_tau, 
                                n_w = 1, 
                                label_pos = list(data = list(x = 35000, y = .5), 
                                                 model = list(x = 7000, y = .29)))
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

### Decay

We did not determine the decay directly, but derived it from the
best-fitting activation. Across the whole data set, the best-fitting
activation is -4.1130682. We determined the optimal decay for each item,
such that it ended up at this activation. Since activation (and the
threshold) are constant across the range, the predicted recall is also
simply the best-fitting horizontal line through the data.

``` r
pred_single_d <- predict_recall(d_seq_list_single,
                                decay = bs_d_indiv[n_windows == 1], 
                                tau = lr_activation[n_windows == 1, tau])
d_single_d <- d_last[pred_single_d, on = .(id)]

p_single_d <- plot_comparison(d_model = d_single_d, 
                              n_w = 1, 
                              label_pos = list(data = list(x = 35000, y = .5), 
                                               model = list(x = 35000, y = .78)))
```

    ## Warning in newton(lsp = lsp, X = G$X, y = G$y, Eb = G$Eb, UrS = G$UrS, L =
    ## G$L, : Fitting terminated with step failure - check results carefully

![](03_plot_results_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

### Scaling factor

Like the decay, the optimal scaling factor was derived from the
best-fitting activation. In principle this should again produce a
horizontal line. However, getting this activation value would require
scaling by a factor larger than 1 for a large portion of the data. Since
h is limited to 1, we cannot get the activation low enough in these
cases, and so we overpredict retention.

``` r
pred_single_h <- predict_recall(d_seq_list_single,
                                h = bs_h_indiv[n_windows == 1],
                                tau = lr_activation[n_windows == 1, tau])
d_single_h <- d_last[pred_single_h, on = .(id)]

p_single_h <- plot_comparison(d_model = d_single_h, 
                              n_w = 1, 
                              label_pos = list(data = list(x = 35000, y = .5), 
                                               model = list(x = 35000, y = .78)))
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

## 24-hour interval

Rather than determining the best parameter fit from the whole data set,
we also explore a scenario in which we only have intervals around 24
hours, which is more similar to many controlled experiments.

We use a subset of the data with intervals around 24 hours. In a
25-window fit this happens to fall quite neatly within window number 17,
so we’ll use that.

``` r
ggplot(window_range[n_windows == 25]) +
  geom_vline(aes(xintercept = 24*60*60), colour = "red", lty = 2) +
  geom_hline(aes(yintercept = 17), colour = "red", lty = 2) +
  geom_segment(aes(x = start, xend = end, y = window, yend = window)) +
  geom_point(aes(x = geom_mean, y = window)) +
  geom_text(aes(x = geom_mean, y = window + .5, label = window), colour = "blue") +
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x)),
    expand = c(0, 0)
  ) +
  labs(x = "Between-session interval (minutes)",
       y = "Window") +
  annotation_logticks(sides = "b", outside = T) +
  coord_cartesian(clip = "off")
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

### Threshold

``` r
tau_24h <- lr_tau[n_windows == 25 & window == 17, tau]
```

The optimal threshold for the 24h interval is -4.702159, quite similar
to the optimal threshold for the whole data set. The model’s predictions
also look very
similar.

``` r
pred_tau_24h <- predict_recall(seq_list = d_seq_list_single, tau = tau_24h)
d_tau_24h <- d_last[pred_tau_24h, on = .(id)]

p_tau_24h <- plot_comparison(d_model = d_tau_24h, 
                             n_w = 1, 
                             label_pos = list(data = list(x = 35000, y = .5), 
                                              model = list(x = 5000, y = .29))) +
  geom_rect(data = window_range[n_windows == 25 & window == 17], aes(xmin  = start/60, xmax = end/60, ymin = -0.05, ymax = 1.05), fill = section_col, alpha = .25)
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
p_tau_24h
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-20-2.png)<!-- -->

### Decay

Because optimal decay is derived separately for each individual learning
sequence, there is not a single decay value associated with the 24h
interval, and we only know the best decay for the individual sequences
that fall within the 24h set.

``` r
d_24h <- median(bs_d_indiv[n_windows == 25 & window == 17, d])
```

To get a parameter that we can use for general prediction, we’ll take
the median: d = 0.2847328.

``` r
ggplot(bs_d_indiv[n_windows == 25 & window == 17], aes(x = d)) +
  geom_histogram(binwidth = .005, fill = "midnightblue") +
  geom_vline(aes(xintercept = d_24h), colour = "red", lty = 2)
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->

These predictions show the typical pattern: the model initially
overpredicts recall, gets it right around 24h, and eventually
underpredicts recall.

``` r
pred_d_24h <- predict_recall(d_seq_list_single,
                             decay = d_24h, 
                             tau = lr_activation[n_windows == 25 & window == 17, tau])
d_d_24h <- d_last[pred_d_24h, on = .(id)]

p_d_24h <- plot_comparison(d_model = d_d_24h, 
                           n_w = 1, 
                           label_pos = list(data = list(x = 35000, y = .5), 
                                            model = list(x = 5000, y = .78))) +
  geom_rect(data = window_range[n_windows == 25 & window == 17], aes(xmin  = start/60, xmax = end/60, ymin = -0.05, ymax = 1.05), fill = section_col, alpha = .25)
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->

``` r
p_d_24h
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-23-2.png)<!-- -->

### Scaling factor

``` r
h_24h <- median(bs_h_indiv[n_windows == 25 & window == 17, h])
```

As with decay, we’ll take the median scaling factor from the 24h set: h
= 0.0029473.

``` r
ggplot(bs_h_indiv[n_windows == 25 & window == 17], aes(x = h)) +
  geom_histogram(binwidth = .01, fill = "midnightblue") +
  geom_vline(aes(xintercept = h_24h), colour = "red", lty = 2)
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

This model already does surprisingly well: short-term predictions are
good, though the model is too optimistic on intervals of 1-24 hours, and
too pessimistic on intervals longer than a week.

``` r
pred_h_24h <- predict_recall(d_seq_list_single,
                             h = h_24h,
                             tau = lr_activation[n_windows == 25 & window == 17, tau])
d_h_24h <- d_last[pred_h_24h, on = .(id)]

p_h_24h <- plot_comparison(d_model = d_h_24h, 
                           n_w = 1, 
                           label_pos = list(data = list(x = 35000, y = .5), 
                                            model = list(x = 12000, y = .2))) +
  geom_rect(data = window_range[n_windows == 25 & window == 17], aes(xmin  = start/60, xmax = end/60, ymin = -0.05, ymax = 1.05), fill = section_col, alpha = .25)
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

``` r
p_h_24h
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-26-2.png)<!-- -->

## Short intervals

### Threshold

Finally, we’ll look at a model with default parameters other than the
threshold, which is fitted to short intervals (0-10 minutes) only.

``` r
pred_tau_short <- predict_recall(d_seq_list_single,
                                 tau = tau_short$tau_short)
d_tau_short <- d_last[pred_tau_short, on = .(id)]

p_tau_short <- plot_comparison(d_model = d_tau_short, 
                               n_w = 1, 
                               label_pos = list(data = list(x = 35000, y = .5), 
                                                model = list(x = 35000, y = .08))) +
  geom_rect(aes(xmin  = min(window_range$start)/60, xmax = 10, ymin = -0.05, ymax = 1.05), fill = section_col, alpha = .25)
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

``` r
p_tau_short
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-27-2.png)<!-- -->

# Time-dependent parameters

Rather than using a single parameter configuration across the range, we
can also fit parameters separately per time bin. Here we’ll look at the
results when the data are split into 20 bins, each containing 5% of the
total.

Prepare the data:

``` r
d_20 <- copy(d_last)
d_20 <- d_20[, sequence := 1:.N][, .(id, sequence, time_between, window = cut(log(time_between), breaks = 20, labels = FALSE))]
d_20_seq <- generate_seq_list(d_20[d_f, on = .(id)])
```

## Threshold

``` r
pred_tau_20 <- predict_recall(seq_list = d_20_seq, tau = lr_tau, windows = 20)
d_tau_20 <- d_last[pred_tau_20, on = .(id)]

p_tau_20 <- plot_comparison(d_model = d_tau_20, 
                            n_w = 20, 
                            label_pos = list(data = list(x = 35000, y = .5), 
                                             model = list(x = 20000, y = .35)))
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

Parameter change over time:

``` r
lr_tau_viz <- lr_tau[window_range, on = .(n_windows, window)]
setnames(lr_tau_viz, "tau", "parameter")

p_tau_time <- plot_parameter(d_parameter = lr_tau_viz,
                             parameter_name = quote(tau),
                             n_w = 20)
```

    ## Scale for 'x' is already present. Adding another scale for 'x', which
    ## will replace the existing scale.

    ## Warning: Transformation introduced infinite values in continuous x-axis

![](03_plot_results_files/figure-gfm/unnamed-chunk-30-1.png)<!-- -->

LM-based
fit:

``` r
m_tau <- lm(parameter ~ log(geom_mean), data = lr_tau_viz[n_windows == 20])

d_tau_lm <- map_dfr(d_20_seq, function(x) {
  list(id = x$id, geom_mean = x$time_between)
})

d_tau_lm$tau <- predict(m_tau, newdata = d_tau_lm)
setDT(d_tau_lm)

pred_tau_lm <- predict_recall(seq_list = d_20_seq, tau = d_tau_lm)
d_tau_lm <- d_last[pred_tau_lm, on = .(id)]

p_tau_lm <- plot_comparison(d_model = d_tau_lm, 
                            n_w = 20, 
                            label_pos = list(data = list(x = 35000, y = .5), 
                                             model = list(x = 20000, y = .29)))
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-31-1.png)<!-- -->

## Decay

``` r
pred_d_20 <- predict_recall(seq_list = d_20_seq,
                            decay = bs_d_indiv[n_windows == 20], 
                            tau = lr_activation,
                            windows = 20)

d_d_20 <- d_last[pred_d_20, on = .(id)]

p_d_20 <- plot_comparison(d_model = d_d_20, 
                          n_w = 20, 
                          label_pos = list(data = list(x = 35000, y = .5), 
                                           model = list(x = 20000, y = .35)))
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-32-1.png)<!-- -->

Parameter change over
time:

``` r
bs_d_viz <- bs_d_indiv[window_range, on = .(n_windows, window)][, .(d = median(d)), by = .(n_windows, window, start, end, geom_mean)]
setnames(bs_d_viz, "d", "parameter")

p_d_time <- plot_parameter(d_parameter = bs_d_viz,
                           parameter_name = quote(italic(d)),
                           n_w = 20)
```

    ## Scale for 'x' is already present. Adding another scale for 'x', which
    ## will replace the existing scale.

    ## Warning: Transformation introduced infinite values in continuous x-axis

![](03_plot_results_files/figure-gfm/unnamed-chunk-33-1.png)<!-- -->

LM-based fit:

``` r
m_d <- lm(parameter ~ log(geom_mean), data = bs_d_viz[n_windows == 20])

d_d_lm <- map_dfr(d_20_seq, function(x) {
  list(id = x$id, geom_mean = x$time_between)
})

d_d_lm$d <- predict(m_d, newdata = d_d_lm)
setDT(d_d_lm)

pred_d_lm <- predict_recall(seq_list = d_20_seq, decay = d_d_lm, tau = lr_activation, windows = 20)
d_d_lm <- d_last[pred_d_lm, on = .(id)]

p_d_lm <- plot_comparison(d_model = d_d_lm,
                          n_w = 20, 
                          label_pos = list(data = list(x = 35000, y = .5), 
                                           model = list(x = 30000, y = .64)))
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-34-1.png)<!-- -->

## Scaling factor

``` r
pred_h_20 <- predict_recall(seq_list = d_20_seq,
                            h = bs_h_indiv[n_windows == 20], 
                            tau = lr_activation,
                            windows = 20)

d_h_20 <- d_last[pred_h_20, on = .(id)]

p_h_20 <- plot_comparison(d_model = d_h_20, 
                          n_w = 20, 
                          label_pos = list(data = list(x = 35000, y = .5), 
                                           model = list(x = 20000, y = .35)))
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-35-1.png)<!-- -->

Parameter change over
time:

``` r
bs_h_viz <- bs_h_indiv[window_range, on = .(n_windows, window)][, .(h = median(h)), by = .(n_windows, window, start, end, geom_mean)]
setnames(bs_h_viz, "h", "parameter")

p_h_time <- plot_parameter(d_parameter = bs_h_viz,
                           parameter_name = quote(ln(italic(h))),
                           log_y = TRUE,
                           n_w = 20) +
  coord_cartesian(xlim = c(window_range[1, start], window_range[.N, end])/60, ylim = c(.001, 1.2), clip = "off") +
  theme(axis.text.y = element_text(margin = margin(r = 8)))
```

    ## Scale for 'x' is already present. Adding another scale for 'x', which
    ## will replace the existing scale.

    ## Warning: Transformation introduced infinite values in continuous y-axis

    ## Warning: Transformation introduced infinite values in continuous x-axis

    ## Warning: Transformation introduced infinite values in continuous y-axis

    ## Warning in scale$trans$transform(coord_limits): NaNs produced

    ## Coordinate system already present. Adding new coordinate system, which will replace the existing one.

![](03_plot_results_files/figure-gfm/unnamed-chunk-36-1.png)<!-- -->

``` r
p_h_time
```

    ## Warning: Transformation introduced infinite values in continuous y-axis

    ## Warning: Transformation introduced infinite values in continuous x-axis

    ## Warning: Transformation introduced infinite values in continuous y-axis

![](03_plot_results_files/figure-gfm/unnamed-chunk-36-2.png)<!-- -->

# Interval distribution

A histogram of the between-session intervals in the data.

``` r
p_histogram <- ggplot() +
  # Window background
  geom_rect(data = window_range[n_windows == 1], aes(xmin = start/60, xmax = end/60, ymin = -Inf, ymax = Inf), fill = window_col, alpha = .1) +
  # Histogram
  geom_histogram(data = d_last, aes(x = time_between/60, y = ..ncount..), bins = 100, fill = obs_col) +
  # Plot setup
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
    labels = scales::trans_format("log10", scales::math_format(10^.x)),
    expand = c(0, 0),
    sec.axis = sec_axis(~.x, breaks = label_x, labels = label_txt)
  ) +
  scale_y_continuous(breaks = seq(0, 1, by = .25)) +
  labs(x = "Between-session interval (minutes)",
       y = "Density") +
  annotation_logticks(sides = "b", outside = T) +
  coord_cartesian(ylim = c(0, 1), xlim = c(window_range[1, start], window_range[.N, end])/60, clip = "off") +
  theme_bw(base_size = 14) +
  theme(plot.margin = margin(7, 14, 7, 7),
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank())

p_histogram
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-37-1.png)<!-- -->

# Combined figures

## Model fits

``` r
plot_grid(
  p_histogram,
  p_tau_short,
  p_tau_24h,
  p_tau_20,
  p_d_24h,
  p_d_20,
  p_h_24h,
  p_h_20,
  ncol = 2,
  labels = c("A\t\tInterval distribution", "B\t\tThreshold optimised for 0 - 10 min", "C\t\tThreshold optimised for 24h", "D\t\tInterval-dependent threshold", "E\t\tDecay optimised for 24h", "F\t\tInterval-dependent decay", "G\t\tScaling factor optimised for 24h", "H\t\tInterval-dependent scaling factor"),
  align = "hv",
  label_x = .025,
  hjust = 0,
  scale = .9
) +
  theme(plot.background = element_rect(fill = "white", colour = NA))
```

![](03_plot_results_files/figure-gfm/unnamed-chunk-38-1.png)<!-- -->

``` r
ggsave(file.path("..", "output", "model_fitting_results.png"), width = 10, height = 15)
```

## Parameters over time

``` r
plot_grid(
  p_tau_time, p_d_time, p_h_time,
  ncol = 3, 
  labels = c("A\t\tInterval-dependent threshold", "B\t\tInterval-dependent decay", "C\t\tInterval-dependent h"),
  align = "hv",
  label_x = .025,
  hjust = 0,
  scale = .9
) +
  theme(plot.background = element_rect(fill = "white", colour = NA))
```

    ## Warning: Transformation introduced infinite values in continuous x-axis
    
    ## Warning: Transformation introduced infinite values in continuous x-axis

    ## Warning: Transformation introduced infinite values in continuous y-axis

    ## Warning: Transformation introduced infinite values in continuous x-axis

    ## Warning: Transformation introduced infinite values in continuous y-axis

![](03_plot_results_files/figure-gfm/unnamed-chunk-39-1.png)<!-- -->

``` r
ggsave(file.path("..", "output", "params_time.png"), width = 12, height = 4)
```
