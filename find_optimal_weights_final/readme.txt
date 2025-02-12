
1. Run findBestScale_MonkC_MultiSteps.py or findBestScale_MonkN_MultiSteps.py corresponding to the monkey data to be used. 
 The data is available in the folders MonkCData or MonkNData. 

Line 156 can be uncommented and modified to run the code for specified units.

The python script with *creates a new folder and save the RMSE output for each Unit and the parameters search space defined in the code. 

* the lines 159/160 can be change to use a static name of the output folder. But make sure the previous results are manually deleted otherwise the code will skip the parameters for which the files are already present. 
This was done to avoid re-running the computation for the parameters that have already been processed.

!Beware: This code takes a lot of time if not run in parallel. For testing or debugging you can reduce the search space defined in lines 117 to 140.

**To avoid confusion due to date names I created two set of data and files for each monkey.


2. Run code readResults_wsteps_epoch.py to get the RMSE values written by the previous code and collate them into a single output file.
! Please make sure to change the line 50 to enter the correct output folder created by the previous code. 