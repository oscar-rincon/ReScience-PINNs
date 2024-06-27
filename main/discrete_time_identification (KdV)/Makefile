# Makefile to run Python scripts in sequence

.PHONY: all clean run_kdV_clean_main run_kdV_noisy_main run_kdV_plots

# Define default target
all: run_kdV_clean_main run_kdV_clean_plots

# Target to run kdV_clean_main.py
run_kdV_clean_main:
	@echo "Running kdV_clean_main.py..."
	@python kdV_clean_main.py
	@echo "Finished kdV_clean_main.py..."

# Target to run kdV_noisy_main.py
run_kdV_noisy_main:
	@echo "Running kdV_noisy_main.py..."
	@python kdV_noisy_main.py
	@echo "Finished kdV_noisy_main.py..."	

# Target to run kdV_plots.py 
run_kdV_plots:
	@echo "Running kdV_plots.py..."
	@python kdV_plots.py
	@echo "Finished kdV_plots.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
