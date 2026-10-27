.PHONY: report clean-report

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-report:
	cd report && latexmk -C main.tex
