

save-environment:
	conda env export --no-builds > setup/environment.yml

load-environment:
	conda env create -f setup/environment.yml

