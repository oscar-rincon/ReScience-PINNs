# Makefile to run Python scripts in sequence

.PHONY: all clean run_Schrodinger_main run_Schrodinger_plots run_AC_main run_AC_plots run_NS_clean_main run_NS_noisy_main run_NS_plots run_kdV_clean_main run_kdV_noisy_main run_kdV_plots

# Define default target
all: run_Schrodinger_main run_Schrodinger_plots run_AC_main run_AC_plots run_NS_clean_main run_NS_plots run_kdV_clean_main run_kdV_plots

# Target to run Schrodinger_main.py
run_Schrodinger_main:
	@echo "Running Schrodinger_main.py..."
	@python main/continuous_time_inference\ \(Schrodinger\)/Schrodinger_main.py
	@echo "Finished Schrodinger_main.py..."

# Target to run Schrodinger_plots.py after Schrodinger_main.py
run_Schrodinger_plots:
	@echo "Running Schrodinger_plots.py..."
	@python main/continuous_time_inference\ \(Schrodinger\)/Schrodinger_plots.py
	@echo "Finished Schrodinger_plots.py..."

# Target to run AC_main.py
run_AC_main:
	@echo "Running AC_main.py..."
	@python main/discrete_time_inference\ \(AC\)/AC_main.py
	@echo "Finished AC_main.py..."

# Target to run AC_plots.py
run_AC_plots:
	@echo "Running AC_plots.py..."
	@python main/discrete_time_inference\ \(AC\)/AC_plots.py
	@echo "Finished AC_plots.py..."

# Target to run NS_clean_main.py
run_NS_clean_main:
	@echo "Running NS_clean_main.py..."
	@python main/continuous_time_identification\ \(Navier-Stokes\)/NS_clean_main.py
	@echo "Finished NS_clean_main.py..."

# Target to run NS_noisy_main.py
run_NS_noisy_main:
	@echo "Running NS_noisy_main.py..."
	@python main/continuous_time_identification\ \(Navier-Stokes\)/NS_noisy_main.py
	@echo "Finished NS_noisy_main.py..."

# Target to run NS_plots.py
run_NS_plots:
	@echo "Running NS_plots.py..."
	@python main/continuous_time_identification\ \(Navier-Stokes\)/NS_plots.py
	@echo "Finished NS_plots.py..."

# Target to run kdV_clean_main.py
run_kdV_clean_main:
	@echo "Running kdV_clean_main.py..."
	@python main/discrete_time_identification\ \(KdV\)/kdV_clean_main.py
	@echo "Finished kdV_clean_main.py..."

# Target to run kdV_noisy_main.py
run_kdV_noisy_main:
	@echo "Running kdV_noisy_main.py..."
	@python main/discrete_time_identification\ \(KdV\)/kdV_noisy_main.py
	@echo "Finished kdV_noisy_main.py..."

# Target to run kdV_plots.py
run_kdV_plots:
	@echo "Running kdV_plots.py..."
	@python main/discrete_time_identification\ \(KdV\)/kdV_plots.py
	@echo "Finished kdV_plots.py..."

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
