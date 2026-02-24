Spiking Neural Network (SNN) modeling and analysis code for "Continuous input drives motor cortical dynamics during reaching"

1. Overview
This repository contains:
- Python code implementing the spiking neural network (SNN) model and optimization pipeline.
- MATLAB scripts used to generate manuscript figures.
- Utilities for reproducing analyses described in the manuscript.
The Python code generates model outputs and intermediate data that are subsequently used by MATLAB scripts to produce figures.

2. System Requirements
Operating Systems Tested
- Windows 11 (primary development system)

Software Requirements
- Python Environment
- Python 3.9
- Anaconda (recommended)

Python dependencies (see environment.yml):
- brian2
- scikit-learn
- scikit-optimize
- matplotlib
- bayesian-optimization

MATLAB
-MATLAB R2023b or newer

Hardware Requirements
- No non-standard hardware required.
- GPU not required.
- Tested on standard desktop PC (Intel i7 CPU, 16GB RAM).


3. Installation Guide
Step 1 – Clone Repository
	conda env create -f environment.yml
	conda activate brian2_snn_env

Step 2 – Install Anaconda (if not already installed)
Download from:
https://www.anaconda.com/download

Step 3 – Create Conda Environment

	conda env create -f environment.yml
	conda activate Brian2

Typical installation time: 5–10 minutes on a standard desktop computer.

4. Demo
Because the manuscript data are not shareable via github, demo data are available:
	Data/MonkCExampleData.mat
	Data/MonkNExampleData.mat
Additional data is avalible upon request from authors. 

Running the Python SNN Code
	python Production_scripts.py
This will:
- Run the SNN simulations
- Perform optimization
- Generate output files for downstream MATLAB analysis

Expected runtime:
~7–21 days depending on system specifications.

Expected output:
- Saved output files in the working directory (ensure flags to save are toggled on)
- Printed performance metrics in console

Generating Figure Inputs
To generate processed data for SNN figures:
run Libs/Utilities.py 
This prepares formatted data used by MATLAB scripts.


5. Generating Manuscript Figures (MATLAB)

To generate manuscript figures:
- Open MATLAB R2023b or later.
- Set working directory to:
	MatlabCode/code_share

Run each script in this directory.
- Use MonkeyC or MonkeyN datasets (contact authors for data access).

Additional figures are generated via:
	MatlabCode/NSTitMCdR.m
Note: The Python pipeline must be executed prior to running NSTitMCdR figure scripts.


6. Instructions for Using the Code on New Data

To use the SNN model on new data:
	1. Format your input data to match the structure used in Production_scripts.py.
	2. Replace dataset loading sections accordingly.
	3. Run: Production_scripts.py

7. License

This repository is licensed under the MIT License.
See LICENSE file for details.
