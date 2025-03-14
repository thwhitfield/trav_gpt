

save-environment:
	conda env export --no-builds > setup/environment.yml

load-environment:
	conda env create -f setup/environment.yml

kill-port-5000:
	kill -9 $$(lsof -t -i:5000)
