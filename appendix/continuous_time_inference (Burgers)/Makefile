# Makefile to run Python scripts in sequence

.PHONY: all clean run_Burgers_ti_main run_Burgers_ti_plots

# Define default target
all: run_Burgers_ti_main run_Burgers_ti_plots

# Target to run Burgers_ti_main.py
run_Schrodinger_main:
	@echo "Running Burgers_ti_main.py..."
	@python Burgers_ti_main.py
	@echo "Finished Burgers_ti_main.py..."

# Target to run Burgers_ti_plots.py after Burgers_ti_main.py
run_Schrodinger_plots:
	@echo "Running Burgers_ti_plots.py..."
	@python Burgers_ti_plots.py
	@echo "Finished Burgers_ti_plots.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
