# Makefile to run Python scripts in sequence

.PHONY: all clean run_AC_main run_AC_plots

# Define default target
all: run_AC_main run_AC_plots

# Target to run AC_main.py
run_AC_main:
	@echo "Running AC_main.py..."
	@python AC_main.py
	@echo "Finished AC_main.py..."

# Target to run AC_plots.py after AC_main.py
run_AC_plots:
	@echo "Running Schrodinger_plots.py..."
	@python Schrodinger_plots.py
	@echo "Finished Schrodinger_plots.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
