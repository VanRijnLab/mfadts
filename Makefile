.PHONY: all clean

all: output/02_walkthrough.nb.html output/02_walkthrough.md output/03_fit_models.nb.html output/03_fit_models.md 

clean:
	rm -f data/cogpsych_data_formatted.csv data/fit* data/pred* output/*

# Data
data/cogpsych_data_formatted.csv: data/cogpsych_data_anon_2017.rda data/cogpsych_data_anon_2018.rda
	Rscript scripts/01_prepare_data.R
	
# Walkthrough
output/02_walkthrough.nb.html output/02_walkthrough.md: scripts/02_walkthrough.Rmd scripts/00_helper_funs.R data/cogpsych_data_formatted.csv 
	Rscript -e "rmarkdown::render('scripts/02_walkthrough.Rmd', output_format = 'all')"
	mv scripts/02_walkthrough.nb.html scripts/02_walkthrough.html scripts/02_walkthrough.md scripts/02_walkthrough_files output

# Fit models
output/03_fit_models.nb.html output/03_fit_models.md: scripts/03_fit_models.Rmd scripts/00_helper_funs.R data/cogpsych_data_formatted.csv
	Rscript -e "rmarkdown::render('scripts/03_fit_models.Rmd', output_format = 'all')"
	mv scripts/03_fit_models.nb.html scripts/03_fit_models.html scripts/03_fit_models.md scripts/03_fit_models_files output