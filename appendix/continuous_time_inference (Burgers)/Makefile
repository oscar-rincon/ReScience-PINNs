# Makefile to run Python scripts in sequence

.PHONY: all clean run_Burgers_ti_main run_Burgers_ti_plots run_Burgers_ti_main Burgers_ti_main_systematic

# Define default target
all: run_Burgers_ti_main run_Burgers_ti_plots Burgers_ti_main_systematic

# Target to run Burgers_ti_main.py
run_Burgers_ti_main:
	@echo "Running Burgers_ti_main.py..."
	@python Burgers_ti_main.py
	@echo "Finished Burgers_ti_main.py..."

# Target to run Burgers_ti_plots.py after Burgers_ti_main.py
Burgers_ti_plots:
	@echo "Running Burgers_ti_plots.py..."
	@python Burgers_ti_plots.py
	@echo "Finished Burgers_ti_plots.py..."

# Target to run Burgers_ti_main_systematic.py
Burgers_ti_main_systematic:
	@echo "Running Burgers_ti_main_systematic.py..."
	@python Burgers_ti_main_systematic.py
	@echo "Finished Burgers_ti_main_systematic.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
