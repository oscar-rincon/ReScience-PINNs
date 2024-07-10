# Makefile to run Python scripts in sequence

.PHONY: all clean run_Burgers_dtin_main run_Burgers_dtin_plots run_Burgers_dtin_main Burgers_dtin_main_systematic

# Define default target
all: run_Burgers_dtin_main run_Burgers_dtin_plots Burgers_dtin_main_systematic

# Target to run Burgers_dtin_main.py
run_Burgers_dtin_main:
	@echo "Running Burgers_dtin_main.py..."
	@python Burgers_dtin_main.py
	@echo "Finished Burgers_dtin_main.py..."

# Target to run Burgers_dtin_plots.py  
Burgers_dtin_plots:
	@echo "Running Burgers_dtin_plots.py..."
	@python Burgers_dtin_plots.py
	@echo "Finished Burgers_dtin_plots.py..."

# Target to run Burgers_dtin_main_systematic.py
Burgers_dtin_main_systematic:
	@echo "Running Burgers_dtin_main_systematic.py..."
	@python Burgers_dtin_main_systematic.py
	@echo "Finished Burgers_dtin_main_systematic.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
