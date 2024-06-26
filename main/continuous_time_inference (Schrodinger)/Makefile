# Makefile to run Python scripts in sequence

.PHONY: all clean run_Schrodinger_main run_Schrodinger_plots

# Define default target
all: run_Schrodinger_main run_Schrodinger_plots

# Target to run Schrodinger_main.py
run_Schrodinger_main:
	@echo "Running Schrodinger_main.py..."
	@python Schrodinger_main.py

# Target to run Schrodinger_plots.py after Schrodinger_main.py
run_Schrodinger_plots:
	@echo "Running Schrodinger_plots.py..."
	@python Schrodinger_plots.py

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
