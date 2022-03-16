.PHONY: all clean

all: analyis/02_fit_models.nb.html analysis/03_plot_results.nb.html

clean:
	rm -f data/cogpsych_data_formatted.csv data/logistic_regression_tau.csv data/logistic_regression_activation.csv data/binary_search_indiv_d.csv data/binary_search_indiv_h.csv

# Data
data/cogpsych_data_formatted.csv: data/cogpsych_data_anon_2017.rda data/cogpsych_data_anon_2018.rda
	Rscript analysis/01_prepare_data.R
	
# Fit models
analyis/02_fit_models.nb.html: analysis/02_fit_models.Rmd data/cogpsych_data_formatted.csv analysis/00_helper_funs.R
	Rscript -e "rmarkdown::render('analysis/02_fit_models.Rmd')"

# Plot results
analysis/03_plot_results.nb.html: analysis/03_plot_results.Rmd data/cogpsych_data_formatted.csv data/logistic_regression_tau.csv data/logistic_regression_activation.csv data/binary_search_indiv_d.csv data/binary_search_indiv_h.csv
	Rscript -e "rmarkdown::render('analysis/03_plot_results.Rmd')"