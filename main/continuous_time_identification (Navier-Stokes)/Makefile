# Makefile to run Python scripts in sequence

.PHONY: all clean run_NS_clean_main run_NS_noisy_main run_NS_plots

# Define default target
all: run_NS_clean_main run_NS_clean_plots

# Target to run NS_clean_main.py
run_NS_clean_main:
	@echo "Running NS_clean_main.py..."
	@python NS_clean_main.py
	@echo "Finished NS_clean_main.py..."

# Target to run NS_noisy_main.py
run_kdV_noisy_main:
	@echo "Running NS_noisy_main.py..."
	@python NS_noisy_main.py
	@echo "Finished NS_noisy_main.py..."	

# Target to run NS_plots.py 
run_kdV_plots:
	@echo "Running NS_plots.py..."
	@python NS_plots.py
	@echo "Finished NS_plots.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
