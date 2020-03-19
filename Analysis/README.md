This folder contains all data and files necessary to reproduce the analysis section in the study.

DDM/ folder contains all code, fits and convergence check for the different DDM alternatives fitted using HDDM. The fit files were removed because they are extremely heavy (~30 GB) but code can be run to generate new traces or send an email to gabriel.[first author last name](at)gmail.com

MixedModels/ contain all code and convergence check for the mixed models used in the study. Fits were remove due to heavy file size, but code can be run to estimate the models.

Spec_files/ contains the package specification needed to run the ipython notebooks. In order to run pystan and HDDM we had to use two different conda environment. 
In order to reproduce the notebooks containing pystan objects (1,2,3,4) and the code contained in the MixedModels folder one should use the spec-file_for_pystan.txt specification file. 
To reproduce notebooks containing containing HDDM objects (5,6,7) and the code contained in the DDM folder one should use the spec-file_for_HDDM.txt specification file. 

0-Data_trimming.ipynb : contains trimming operated on data as described and generates the trimmed_data.csv file on which all analysis are based.

1-EXP1-Behavior_EMG.ipynb : contains LME analysis (bayesian and frequentist) on each behavioral and EMG variable for Experiment 1

2-EXP1_PMT-MT_correlation.ipynb : contains correlation analysis between PMT and MT for Experiment 1

3-EXP2-Behavior_EMG.ipynb : contains LME analysis (bayesian and frequentist) on each behavioral and EMG variable for Experiment 2

4-EXP2_PMT-MT_correlation.ipynb : contains correlation analysis between PMT and MT for Experiment 2

5-Model_selection.ipynb : contains model selection analysis

6-Testing_DDM_exp1.ipynb : contains the test on the selected DDM model for Experiment 1

7-Testing_DDM_exp1.ipynb : contains the test on the selected DDM model for Experiment 2

All notebooks are given with an html version, all other files are used internally for the analysis and needs to be in the same directory as the notebooks
