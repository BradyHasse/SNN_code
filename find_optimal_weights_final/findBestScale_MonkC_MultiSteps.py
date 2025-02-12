
# %% import
import numpy as np
import Libs.Helper_Functions as HF
import Libs.Input_generation as IG
import os
import multiprocessing

import sys

from time import time as wall_time
from datetime import datetime

from brian2 import second

#%%  Read input indices array
inp_indices=[[[]for j in range(16)] for i in range(1)]


# these dates correspond to the input indices, weights and events files that Andy sent. 
# the files are in folders MonkCData or MonkNData
date_ = 'July30'
monk = 'MonkC'
date_events = 'Sept18'

additionalPathTags = ''

# I think these are not used anymore. 
#slopesW4MonkC = [0.19401348, 0.30850939, 0.23415761, 0.10993228, 0.43510231, 0.04693501, 0.04905402, 0.10954162, 0.40073764, 0.12070232, 0.05393362, 0.02640521, 0.21301616, 0.02927811, 0.06651825, 0.02238517, 0.21540685, 0.10211742, 0.1161748 , 0.13954582, 0.20371332, 0.17211011, 0.08040707, 0.07439363, 0.3024469 , 0.51354074, 0.08786162, 0.44560824, 0.04139607, 0.18085206, 0.04379441, 0.58378646, 0.20455174, 0.06106598, 0.61508065, 0.07574733, 0.02944586, 0.08156803, 0.23931328, 0.01887588, 0.01969583, 0.06353518, 0.10951975, 0.35114286, 0.10453103, 0.27116546, 0.13000551, 0.65993809, 0.03500946, 0.08120575, 0.51861114, 0.42398155, 0.05636304, 0.2203825 , 0.43588027, 0.27068338, 0.10614126, 0.03287649, 0.29642382, 0.13091832, 0.1389375 , 0.08028673, 0.26314568, 0.21977253, 0.05599628, 0.23613622, 0.01942684]
#slopesW4MonkN = [0.46018302, 0.09533469, 0.52427498, 0.10134658, 0.06814447,0.16220584, 0.27965731, 0.30781256, 0.59478299, 0.62698187,0.76277736, 1.47242911, 0.5386918 , 0.25916438, 0.0571596 ,0.05087924, 0.18781712, 0.21682462, 0.37628543, 0.19085907,0.01892255, 0.07953764, 0.23825523, 0.20183682, 0.11828916, 0.39950681, 0.38959118, 0.02711883, 0.07185855, 0.24886906, 0.0721993 , 0.52506322, 0.21868481, 0.12361669, 0.07918899, 0.08376601, 0.15506244, 0.06300287, 0.30522643, 0.0749404 , 0.37954069, 1.26453004, 0.53973275, 0.07566684, 0.07646446, 0.27647286, 0.2600171 , 0.77891962, 0.0341158 , 0.05197552, 0.40060392, 0.49568956, 0.6980215 , 0.18372937, 0.5573465 ,0.71463945, 0.32569133, 0.0634245 , 0.36566641, 0.3408889 ,0.18608109, 0.00732951, 0.25427053, 0.04879866, 0.39242285,0.07240435, 0.05687793, 0.08007053, 0.05938618, 0.75250851, 0.11423064, 0.09504917, 0.26611557, 0.02579174, 0.04197267, 0.06866484, 0.02212078, 0.02704695]

#print(len(slopesW4))

#%% open files containing the input indices
with open(monk + 'Data/' + monk + '_input_indices' + additionalPathTags + '[' + date_ + '].npy', 'rb') as f:
	num_sources= int(np.load(f))
	num_targets= int(np.load(f))

	print(f'num_sources {num_sources}')
	print(f'num_targets {num_targets}')

	rep_num= np.load(f)
	inp_indices=[[[[] for k in range(rep_num[j])] for j in range(num_targets)] for i in range(num_sources)]
	#input_spikes=[[[[] for k in range(rep_num[j])] for j in range(num_targets)] for i in range(num_sources)]
	for i in range(num_sources):
		for j in range(num_targets):
			nrep = rep_num[j]
			for k in range(nrep):
				g_tmp=np.load(f)
				#print('Rep',k)
				inp_indices[i][j][k]=g_tmp
	#print(f'indices {input_indices}')inp_indices

#%%  Read input spike array

rep_num = []

#str_spikes_name = 'MonkNData/MonkN_r_all[July12].npy'
str_spikes_name = monk+ 'Data/' + monk + '_r_all[' + date_events + '].npy'

#str_spikes_name = 'actual spikes.npy'
with open(str_spikes_name, 'rb') as f:
	r_all = np.array(np.load(f, allow_pickle=True))

print(r_all.shape)
print(len(r_all))

# READ INPUT SPIKES
with open(monk + 'Data/' + monk + '_input_spikes' + additionalPathTags + '['+ date_ + '].npy', 'rb') as f:
	num_sources= int(np.load(f))
	num_targets= int(np.load(f))
	rep_num= np.load(f)
	inp_spikes=[[[[] for k in range(rep_num[j])] for j in range(num_targets)] for i in range(num_sources)]

	for i in range(num_sources):
		for j in range(num_targets):
			nrep = rep_num[j]
			for k in range(nrep):
				g_tmp=np.load(f)
				#print('Rep',k)
				inp_spikes[i][j][k]=g_tmp*second
#    print(len(input_spikes[0][0][0]))

#%% Read events
events = []
with open(monk + 'Data/' + monk + '_events' + '['+ date_events + '].npy', 'rb') as f:
	events = np.load(f)

#print(events.shape)

#%% Read weights

# Read weights
weights = []
weights1_multi = []
weights2_multi = []
weights3_multi = []
weights4_multi = []

# weights for MonkC were generated in Aug1 file. 
with open(monk + 'Data/' + monk + '_weights[' + 'Aug1' + '].npy', 'rb') as f:
	#weights = np.load(f)
	weights1_multi = np.load(f)
	#print(weights1_multi.shape)
	weights2_multi = np.load(f)
	weights3_multi = np.load(f)
	weights4_multi = np.load(f)

weights = HF.make_weights(weights1_multi, weights2_multi, weights3_multi, weights4_multi)

# print shapes of weights for sanity check
# print(len(weights))
# print(len(weights[0]))
# print(len(weights[1]))
# print(len(weights[2]))
# print(len(weights[3]))

#%% Define the range for steps and threshold to calculate RMSE of fit
print("Run Scale Optimizer")

w_steps_ = []

steps = np.linspace(0.03, 0.09, num=8)
num_steps = len(steps)

thresholds = np.linspace(0.1, 0.6, 5)

for th in thresholds:
	for ws1 in range(num_steps):
	    for ws2 in range(num_steps):
	        for ws3 in range(num_steps):
	            temp = np.zeros(4)
	            temp[0] = steps[ws1]
	            temp[1] = steps[ws2]
	            temp[2] = steps[ws3]
	            temp[3] = th
	            w_steps_.append(temp)

w_steps = w_steps_ #[w_steps_[0], w_steps_[1794]]

#%% Call the getRMSE from the library Optimal_weights_hybrid.py

rmse_accum= []

duration = 1000

numUnits = r_all.shape[0]

unit_args = [i for i in range(numUnits)]

#if len(sys.argv) > 1:
#	unit_args = [int(sys.argv[1])]

# Set unit_args to single unit for debugging.
#unit_args = [58]

# Set name of directory where the results will be saved. This will create a new directory everytime the code is run. Can be changed to something static.
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
outputDir = 'RMSE_' + monk + '_' + dt_string + '_Results/'

#	print(new_iterable[0])

n_cores = 1

## if the code is run in parallel. This will parallelize the code over the combinations of parameters for each unit.

# for units in unit_args:
#     args = [units, weights, inp_indices, inp_spikes, r_all, events, duration, outputDir]
#     new_iterable = ([x, args] for x in w_steps)
#     with multiprocessing.Pool(n_cores) as p:
#         p.map(IG.get_rmse_v2, new_iterable)



# to run code in sequential system.
for units in unit_args:
    for r in range(np.shape(w_steps)[0]):
        args = []
        args.append(w_steps[r])
        args.append([units, weights, inp_indices, inp_spikes, r_all, events, duration, outputDir])
        IG.get_rmse_v2(args);
		# new_iterable = ([x, args] for x in w_steps)
		
    # with multiprocessing.Pool(n_cores) as p:
    #     p.map(IG.get_rmse_v2, new_iterable)


#	print(mrmse)

#for unit in range(1):
#	for ws in range(len(w_steps)):
#	    mrmse= IG.get_rmse(unit,weights,inp_indices,inp_spikes,r_all,events,duration,w_steps[ws])
#	    print ('RMSE for',w_steps[ws], '=', mrmse)
#	    rmse_accum.append(mrmse)
#	best_weight_step= w_steps[rmse_accum.index(np.min(rmse_accum))] 
#	print(f"best scale for unit {unit} is {best_weight_step}")

