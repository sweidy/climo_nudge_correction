Included are sourcemod changes to CESM2.1.5 for running the climatological nudging with integrated control. Example scripts for running the model and setting the case parameters are also included.  

Sourcemods:
- bld/namelist_files: addition of namelist variables for the running_mean_nl category. Most namelist parameters are comparable to the nudging_nl parameters from the original CESM toolbox, with some added parameters for the integrated control and restarting functionality. 
- cime_config: addition of compset used for the climatological SST nudging case referenced in the paper (F2000_DARTC6) used to test sensitivity to SST forcing during the tendency generation step. Not critical for using the climatological nudging.  
- src/control: adding restart functionality for the running mean which is used to calculate the nudging tendency. During a simulation, the running mean is saved in memory, but needs to be saved to a file and reread whenever the simulation stops and restarts. 
- src/physics/cam: primarily adding running_mean.F90 and corrector.F90 files for calculating and applying the tendency during the running mean step, and for applying the tendency as a correction in the correction step.

runscripts
- example user_nl_cam files for spinup and for running mean convergence after spinup  
- example scripts for setting up the cases for spinup (first iteration) and subsequent iterations with the same case (hybrid restarts) 
