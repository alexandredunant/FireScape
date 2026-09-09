.PHONY: report check clean

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=FireScape_forestry_report report.tex

check:
	python scripts/check_repository.py

clean:
	cd report && latexmk -c -jobname=FireScape_forestry_report report.tex
