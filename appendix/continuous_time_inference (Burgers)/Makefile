# Makefile to run Python scripts in sequence

.PHONY: all clean run_Burgers_ctin_main run_Burgers_ctin_plots Burgers_ctin_main_systematic

# Define default target
all: run_Burgers_ctin_main run_Burgers_ctin_plots Burgers_ctin_main_systematic

# Target to run Burgers_ctin_main.py
run_Burgers_ctin_main:
	@echo "Running Burgers_ctin_main.py..."
	@python Burgers_ctin_main.py
	@echo "Finished Burgers_ctin_main.py..."

# Target to run Burgers_ctin_plots.py  
Burgers_ctin_plots:
	@echo "Running Burgers_ctin_plots.py..."
	@python Burgers_ctin_plots.py
	@echo "Finished Burgers_ctin_plots.py..."

# Target to run Burgers_ctin_main_systematic.py
Burgers_ctin_main_systematic:
	@echo "Running Burgers_ctin_main_systematic.py..."
	@python Burgers_ctin_main_systematic.py
	@echo "Finished Burgers_ctin_main_systematic.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
