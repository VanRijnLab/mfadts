.PHONY: all clean

all: output/02_fit_models.nb.html output/03_plot_results.nb.html

clean:
	rm -f data/cogpsych_data_formatted.csv data/logistic_regression_tau.csv data/logistic_regression_activation.csv data/binary_search_indiv_d.csv data/binary_search_indiv_h.csv output/*

# Data
data/cogpsych_data_formatted.csv: data/cogpsych_data_anon_2017.rda data/cogpsych_data_anon_2018.rda
	Rscript scripts/01_prepare_data.R
	
# Fit models
output/02_fit_models.nb.html: scripts/02_fit_models.Rmd scripts/00_helper_funs.R data/cogpsych_data_formatted.csv scripts/00_helper_funs.R
	Rscript -e "rmarkdown::render('scripts/02_fit_models.Rmd')"
	mv scripts/02_fit_models.nb.html output

# Plot results
output/03_plot_results.nb.html: scripts/03_plot_results.Rmd scripts/00_helper_funs.R data/cogpsych_data_formatted.csv data/logistic_regression_tau.csv data/logistic_regression_activation.csv data/binary_search_indiv_d.csv data/binary_search_indiv_h.csv
	Rscript -e "rmarkdown::render('scripts/03_plot_results.Rmd')"
	mv scripts/03_plot_results.nb.html output