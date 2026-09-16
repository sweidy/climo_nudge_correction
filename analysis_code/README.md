Contains code and scripts used for pre/postprocessing data for the tendency generation step, as well as analysis code used in the paper. 

postprocessing:
- merra_smooth_rolling.py: performs seasonal smoothing on MERRA climatology data to reduce noise and isolate seasonal signal. 
- split_running_to_tends_avg.sh: average tendencies from tendency_generation_step case over specified time period (typically 1980-2010 for historical SST runs) and convert to correction tendencies used by the model. 

figure_code: 
- calc_norm_error_boot.py: calculate bootstrapped average normalized error for Figure 1. 
- calc_variance_bias.py: calculate and save transients for Figure 7
- corrector_analysis.py: common functions used for analysis in figures
- paper_figures.ipynb: notebook containing plotting functions, using functions from corrector_analysis.py
- phase_calculation.py: function also referenced by paper_figures.ipynb
