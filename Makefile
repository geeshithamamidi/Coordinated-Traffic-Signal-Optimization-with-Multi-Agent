.PHONY: report clean-report results curves curves-single curves-grid4x4

CURVE_METHODS := ql dqn ippo shared_ppo

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-report:
	cd report && latexmk -C main.tex

curves-single:
	python scripts/plot_train_curves.py --scenario single --method $(CURVE_METHODS)

curves-grid4x4:
	python scripts/plot_train_curves.py --scenario grid4x4 --method $(CURVE_METHODS)

curves: curves-single curves-grid4x4

results: curves
	python scripts/aggregate_results.py --scenario single grid4x4
	python scripts/ablation_table.py
