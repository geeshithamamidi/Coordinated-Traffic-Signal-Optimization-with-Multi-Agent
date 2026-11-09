.PHONY: report clean-report results

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-report:
	cd report && latexmk -C main.tex

results:
	python scripts/aggregate_results.py --scenario single grid4x4
