# Makefile to run Python scripts in sequence

.PHONY: all clean run_Burgers_dtid_main run_Burgers_dtid_plots Burgers_dtid_main_systematic

# Define default target
all: run_Burgers_dtid_main run_Burgers_dtid_plots Burgers_dtid_main_systematic

# Target to run Burgers_dtid_main.py
run_Burgers_dtid_main:
	@echo "Running Burgers_dtid_main.py..."
	@python Burgers_dtid_main.py
	@echo "Finished Burgers_dtid_main.py..."

# Target to run Burgers_dtid_plots.py  
Burgers_dtid_plots:
	@echo "Running Burgers_dtid_plots.py..."
	@python Burgers_dtid_plots.py
	@echo "Finished Burgers_dtid_plots.py..."

# Target to run Burgers_dtid_main_systematic.py
Burgers_dtid_main_systematic:
	@echo "Running Burgers_dtid_main_systematic.py..."
	@python Burgers_dtid_main_systematic.py
	@echo "Finished Burgers_dtid_main_systematic.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
