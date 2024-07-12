# Makefile to run Python scripts in sequence

.PHONY: all clean run_Schrodinger_main run_Schrodinger_plots run_AC_main run_AC_plots run_NS_clean_main run_NS_noisy_main run_NS_plots run_kdV_clean_main run_kdV_noisy_main run_kdV_plots

# Define default target
all: run_Schrodinger_main run_Schrodinger_plots run_AC_main run_AC_plots run_NS_clean_main run_NS_plots run_kdV_clean_main run_kdV_plots run_Burgers_ctin_main Burgers_ctin_plots Burgers_ctin_main_systematic run_Burgers_dtin_main run_Burgers_dtin_plots run_Burgers_dtin_main_systematic run_Burgers_ctid_main run_Burgers_ctid_plots run_Burgers_ctid_main_systematic run_Burgers_dtid_main run_Burgers_dtid_plots run_Burgers_dtid_main_systematic

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

# Target to run KdV_clean_main.py
run_kdV_clean_main:
	@echo "Running KdV_clean_main.py..."
	@python main/discrete_time_identification\ \(KdV\)/KdV_clean_main.py
	@echo "Finished KdV_clean_main.py..."

# Target to run KdV_noisy_main.py
run_kdV_noisy_main:
	@echo "Running kdV_noisy_main.py..."
	@python main/discrete_time_identification\ \(KdV\)/KdV_noisy_main.py
	@echo "Finished KdV_noisy_main.py..."

# Target to run KdV_plots.py
run_kdV_plots:
	@echo "Running kdV_plots.py..."
	@python main/discrete_time_identification\ \(KdV\)/KdV_plots.py
	@echo "Finished KdV_plots.py..."

# Appendix

# Target to run Burgers_ctin_main.py
run_Burgers_ctin_main:
	@echo "Running Burgers_ctin_main.py..."
	@python appendix/continuous_time_inference\ \(Burgers\)/Burgers_ctin_main.py
	@echo "Finished Burgers_ctin_main.py..."

# Target to run Burgers_ctin_plots.py  
run_Burgers_ctin_plots:
	@echo "Running Burgers_ctin_plots.py..."
	@python appendix/continuous_time_inference\ \(Burgers\)/Burgers_ctin_plots.py
	@echo "Finished Burgers_ctin_plots.py..."

# Target to run Burgers_ctin_main_systematic.py
run_Burgers_ctin_main_systematic:
	@echo "Running Burgers_ctin_main_systematic.py..."
	@python appendix/continuous_time_inference\ \(Burgers\)/Burgers_ctin_main_systematic.py
	@echo "Finished Burgers_ctin_main_systematic.py..."

# Target to run Burgers_dtin_main.py
run_Burgers_dtin_main:
	@echo "Running Burgers_dtin_main.py..."
	@python appendix/discrete_time_inference\ \(Burgers\)/Burgers_dtin_main.py
	@echo "Finished Burgers_dtin_main.py..."

# Target to run Burgers_dtin_plots.py  
run_Burgers_dtin_plots:
	@echo "Running Burgers_dtin_plots.py..."
	@python appendix/discrete_time_inference\ \(Burgers\)/Burgers_dtin_plots.py
	@echo "Finished Burgers_dtin_plots.py..."

# Target to run Burgers_dtin_main_systematic.py
Burgers_dtin_main_systematic:
	@echo "Running Burgers_dtin_main_systematic.py..."
	@python appendix/discrete_time_inference\ \(Burgers\)/Burgers_dtin_main_systematic.py
	@echo "Finished Burgers_dtin_main_systematic.py..."	


# Target to run Burgers_ctid_main.py
run_Burgers_ctid_main:
	@echo "Running Burgers_ctid_main.py..."
	@python appendix/continuous_time_identification\ \(Burgers\)/Burgers_ctid_clean_main.py
	@python appendix/continuous_time_identification\ \(Burgers\)/Burgers_ctid_noisy_main.py
	@echo "Finished Burgers_ctid_main.py..."

# Target to run Burgers_ctid_plots.py  
run_Burgers_ctid_plots:
	@echo "Running Burgers_ctid_plots.py..."
	@python appendix/continuous_time_identification\ \(Burgers\)/Burgers_ctid_plots.py
	@echo "Finished Burgers_ctid_plots.py..."

# Target to run Burgers_ctid_main_systematic.py
run_Burgers_ctid_main_systematic:
	@echo "Running Burgers_ctid_main_systematic.py..."
	@python appendix/continuous_time_identification\ \(Burgers\)/Burgers_ctid_main_systematic.py
	@echo "Finished Burgers_ctid_main_systematic.py..."

# Target to run Burgers_dtid_main.py
run_Burgers_dtid_main:
	@echo "Running Burgers_dtid_main.py..."
	@python appendix/discrete_time_identification\ \(Burgers\)/Burgers_dtid_clean_main.py
	@python appendix/discrete_time_identification\ \(Burgers\)/Burgers_dtid_noisy_main.py
	@echo "Finished Burgers_dtid_main.py..."

# Target to run Burgers_dtid_plots.py  
run_Burgers_dtid_plots:
	@echo "Running Burgers_dtid_plots.py..."
	@python appendix/discrete_time_identification\ \(Burgers\)/Burgers_dtid_plots.py
	@echo "Finished Burgers_dtid_plots.py..."

# Target to run Burgers_dtid_main_systematic.py
run_Burgers_dtid_main_systematic:
	@echo "Running Burgers_dtid_main_systematic.py..."
	@python appendix/discrete_time_identification\ \(Burgers\)/Burgers_dtid_main_systematic.py
	@echo "Finished Burgers_dtid_main_systematic.py..."	

# Clean up generated files
clean:
	@echo "Cleaning up..."
	@rm -rf figures/* models_iters/* training/*
	@echo "Clean up complete."
