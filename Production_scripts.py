#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:11:29 2023

@author: Andrew, Brady
"""

#%%   Initialize the enviornment-  Make sure you are in the correct directory
import os
from datetime import datetime
import logging
import multiprocessing
import pickle
import numpy as np
import scipy.io as sio
from brian2 import second
from time import time as wall_time

from random import randint

# Import helper functions from libraries
 
from Libs.Helper_Functions import simple_regress, make_norm_histos_nbins, w_steps_gen # cant use paretheses
from Libs.Helper_Functions import differential_evolution, CreateBestValues
from Libs.Helper_Functions import Create_BV_hist2, ready_make_out_all_spikes_par, score_run

from Libs.Input_generation import make_individual_input_spikes_par, make_offset_weights_par
from Libs.Input_generation import  make_out_all_spikes_par, par_w_step


#%% Global Settings and Configuration
MONK_FLAG = 'C' # Options: 'N' or 'C'

CREATE_INPUTS = False # if you remake inputs you should also do weights, offsets,scale, and thresholds.
SAVE_INPUTS = False

CREATE_W_AND_O = False
SAVE_W_AND_O = False

CREATE_S_AND_T = False
SAVE_S_AND_T = False

CREATE_OUTPUTS = True
SAVE_OUTPUTS = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

n_cores = multiprocessing.cpu_count()-2 #number of core for parallel processing
logging.info("CPU usage: %d/%d", n_cores, multiprocessing.cpu_count())
num_iters = 500 # number of iterations for differential evolution
t00 = wall_time()
# Paths and file suffixes
CODE_DIR = os.getcwd()
DATA_DIR = os.path.join(CODE_DIR, 'Data', f'Monk{MONK_FLAG}')
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

#%% Load in files
if MONK_FLAG == 'N':
    file_suffix = '_05-04-2024-18-01-17.npy'
    spk_struct = sio.loadmat(os.path.join(CODE_DIR, 'Data', 'MonkN359Selected.mat'))
elif MONK_FLAG == 'C':    
    file_suffix = '_28-06-2024-15-09-50.npy'
    spk_struct = sio.loadmat(os.path.join(CODE_DIR, 'Data', 'MonkCDataSelected.mat'))
else:
    raise ValueError("Invalid MONK_FLAG. Choose 'N' or 'C'.")
    
# Define file names
IND_FILE = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_input_indices{file_suffix[:-3]}pickle')
SPK_FILE = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_input_spikes{file_suffix[:-3]}pickle')
W_FILE = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_Weights{file_suffix}')
OFFSET_FILE = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_W_offset{file_suffix}')
SCALE_THRESH_FILE = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_RMSE_Scale_Thresh{file_suffix}')
OSP_FILE = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_output_spikes{file_suffix}')

# Suffix files for saving with timestamp
IND_FILE_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_input_indices_{dt_string}.pickle')
SPK_FILE_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_input_spikes_{dt_string}.pickle')
W_FILE_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_Weights_{dt_string}.npy')
OFFSET_FILE_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_W_offset_{dt_string}.npy')
SCALE_THRESH_FILE_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_RMSE_Scale_Thresh_{dt_string}.npy')
OSP_FILE_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_output_spikes_{dt_string}.npy')
O_POT_FILE_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_output_potential_{dt_string}.npy')
BV_HIST_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_BV_hist_{dt_string}.pickle')
PREV_PAR_S = os.path.join(DATA_DIR, f'Monk{MONK_FLAG}_Prev_Par_{dt_string}.pickle')


#%%Get Data  for Monkey C or N
all_units = spk_struct['spk_all']

r_all = all_units
r_allL = r_all.tolist()#convert to list to match out_all_spikes
for i in range(len(r_allL)):#fix orientation
    for ii in range(len(r_allL[0])):
        for iii in range(len(r_allL[0][0])):
            r_allL[i][ii][iii] = np.squeeze(np.transpose(r_allL[i][ii][iii]))
num_units = r_all.shape[0]
r_raw= spk_struct['spk_raw']
events = spk_struct['events_out']
event_names = spk_struct['event_names']
rctemp = spk_struct['rep_cnt']    
rcl =rctemp.tolist()
rca= np.array(rcl)
rep_cnt = rca.flatten()     
speed_all = spk_struct['speed_out']
max_speeds = []
for i in range(speed_all.shape[0]):
    for j in range(speed_all.shape[0]):
        max_speeds.append(np.max(speed_all[i,j]))
max_speed = np.max(np.array(max_speeds))
speed_times = spk_struct['speed_time']
gauss_center= spk_struct['gauss_mu'].flatten()
gauss_sigma= spk_struct['gauss_sigma'].flatten()
event_landmarks= spk_struct['landmarks'].flatten()
duration = np.ceil(events[:, :, 5]*10000, )/10#reward time in ms to ceil 0.1ms limit
num_neurons= 90

#%% inp_indices and inp_spikes
if CREATE_INPUTS :
    #Generate input indices array and input spike array
    targ_rep = []
    for targ in range(rep_cnt.shape[0]):
        for rep in range(rep_cnt[targ]):
            targ_rep.append([targ, rep, randint(0, 9**9)])
    args = [events.shape[0], rep_cnt, events, duration, speed_all, speed_times,
            gauss_center, gauss_sigma, event_landmarks, max_speed]
    new_iterable = ([x, args] for x in targ_rep)
    if __name__ == '__main__':
        multiprocessing.freeze_support()
        with multiprocessing.Pool(n_cores) as p:
            results = p.map(make_individual_input_spikes_par, new_iterable)
    out_inp_indices = np.ndarray([4, events.shape[0], np.max(rep_cnt)], dtype='object')
    out_inp_spikes = np.ndarray([4, events.shape[0], np.max(rep_cnt)], dtype='object')
    for i in range(len(results)):
        r = results[i]
        for j in range(4):
            out_inp_indices[j, r[2], r[3]] = r[0][j]
            out_inp_spikes[j, r[2], r[3]]  = r[1][j]*second
    inp_indices = np.ndarray.tolist(out_inp_indices)
    inp_spikes = np.ndarray.tolist(out_inp_spikes)
    logging.info("Finished creating inputs in %.2f seconds", wall_time() - t00)
    
    if SAVE_INPUTS:
        with open(SPK_FILE_S, 'wb') as file:
            pickle.dump(inp_spikes, file)
        with open(IND_FILE_S, 'wb') as file:
            pickle.dump(inp_indices, file)
else:
    #Read input indices array and input spike array
    with open(SPK_FILE, 'rb') as file:
        inp_spikes = pickle.load(file)
    with open(IND_FILE, 'rb') as file:
        inp_indices = pickle.load(file)
    
#%% Weights
if CREATE_W_AND_O:
    #Generate offsets and initial weights. 
    num_reps = 20#train on first 20 repititions
    calc_offset = np.zeros([num_units,len(inp_spikes),1])
    tmp_offsets = [0.002, 0.0025]
    
    
    inps = []
    for rep in range(num_reps):
        for target in range(len(inp_indices[0])):
            inp_ind_t = []
            inp_spk_t = []
            for group in range(len(inp_indices)):
                inp_ind_t.append(inp_indices[group][target][rep])
                inp_spk_t.append(inp_spikes[group][target][rep])
            inps.append([r_all[:,target,rep], inp_ind_t, 
                         inp_spk_t, target, rep, duration])
                
    args = calc_offset+tmp_offsets[0]
    new_iterable = ([x, args] for x in inps)
    if __name__ == '__main__':
        with multiprocessing.Pool(n_cores) as p:
            results0 = p.map(make_offset_weights_par, new_iterable)
            
    args = calc_offset+tmp_offsets[1]
    new_iterable = ([x, args] for x in inps)
    if __name__ == '__main__':
        with multiprocessing.Pool(n_cores) as p:
            results1 = p.map(make_offset_weights_par, new_iterable)
    
    sum_weights0 = np.zeros(results0[0][0].shape)
    sum_weights1 = np.zeros(results1[0][0].shape)    
    for i in range(len(results0)):
        sum_weights0 += results0[i][0]
        sum_weights1 += results1[i][0]
    
    for epoch in range(len(inp_spikes)):
        for unit in range(num_units):
            temp_weights1 = sum_weights0[epoch,unit::num_units] #unwrap from vector
            temp_weights2 = sum_weights1[epoch,unit::num_units] #unwrap from vector
            (b0,slope)= simple_regress(
                np.array(tmp_offsets), 
                np.array([np.mean(temp_weights1), np.mean(temp_weights2)]))
            calc_offset[unit,epoch,:] = -b0/slope 
    
    final_offsets = np.copy(calc_offset)
    final_offsets[:,-1,:] = 0#set offsets for vel epoch to 0
    
    args = final_offsets
    new_iterable = ([x, args] for x in inps)
    if __name__ == '__main__':
        with multiprocessing.Pool(n_cores) as p:
            results = p.map(make_offset_weights_par, new_iterable)
    
    weight_multi = np.zeros(results[0][0].shape)
    for i in range(len(results0)):
        weight_multi += results[i][0]
    
    
    weight1_multi = np.transpose(weight_multi[0,:])
    weight2_multi = np.transpose(weight_multi[1,:])
    weight3_multi = np.transpose(weight_multi[2,:])
    weight4_multi = np.transpose(weight_multi[3,:])
    
    logging.info("Finished creating weights in %.2f seconds", wall_time() - t00)
    
    if SAVE_W_AND_O:
        with open(OFFSET_FILE_S, 'wb') as f:
            np.save(f, final_offsets)
        with open(W_FILE_S, 'wb') as f:
            np.save(f, weight_multi)
else:
    #Load in previously created weights and offsets
    # Get stored offsets that came from next code section
    with open(OFFSET_FILE, 'rb') as f:
        final_offsets = np.load(f)
    
    #offsets have already been "applied" to these weights
    with open(W_FILE, 'rb') as f:
        weight_multi = np.load(f)
    weight1_multi = np.transpose(weight_multi[0,:])
    weight2_multi = np.transpose(weight_multi[1,:])
    weight3_multi = np.transpose(weight_multi[2,:])
    weight4_multi = np.transpose(weight_multi[3,:])
        
weight_multi_3d = np.reshape(weight_multi,[len(inp_spikes), -1, num_units])

#%% Scale for weights and threshold for output units. 
if CREATE_S_AND_T:
    #make parameter search space
    bounds = np.array([[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[-0.02,0.02],[-5, 5]])#bounds for the wsteps (0-3) and threshold (4)
    pop_size = 14# A multiplier for setting the total population size - number of individuals per dimension
    min_pop = pop_size*bounds.shape[0]
    pop_size = int(np.ceil(min_pop/n_cores)*n_cores)#set the total population to be a multiple of the number of cores
    population = w_steps_gen(num_units, bounds, pop_size)
    w_steps = list(population)
    
    reps = [0,20]#make actual hist for comparison 
    actual_spikes= r_allL[0]#first unit to init
    [actual_hist,d,e,n_bins,f] =  make_norm_histos_nbins(
        actual_spikes,events,reps,4,o_binwidth=0.005)
    actual_hist = np.zeros(
        [num_units, actual_hist.shape[0], actual_hist.shape[1]])
    for u in range(num_units):
        actual_spikes= r_allL[u]
        [actual_hist[u, :, :], d, e, n_bins, f] = make_norm_histos_nbins(
            actual_spikes, events, reps, 4, n_bins, o_binwidth=0.005)
    av_FR = np.mean(actual_hist, axis=(2, 1))

    BV_hist = []
    reps = [0, 20]
    args = [num_units, weight_multi_3d, inp_spikes, inp_indices, 
            reps, duration, events, actual_hist, n_bins]
    new_iterable = ([x, args] for x in w_steps)
    if __name__ == '__main__':
        with multiprocessing.Pool(n_cores) as p:
            results = p.map(par_w_step, new_iterable)
    Prev_Par = results
    t0 = wall_time()
    t1 = wall_time()
    for i in range(num_iters):#do that many iterations
        Prev_Par, w_steps = differential_evolution(
            results, av_FR, bounds, Prev_Par, (i/num_iters)*3)
        best_values = CreateBestValues(Prev_Par, av_FR)
        
        if SAVE_S_AND_T:
            with open(SCALE_THRESH_FILE_S, 'wb') as f:
             	np.save(f, best_values)
        BV_hist.append(np.copy(best_values))
        new_iterable = ([x, args] for x in w_steps)
        if __name__ == '__main__':
            with multiprocessing.Pool(n_cores) as p:
                results = p.map(par_w_step, new_iterable)
            
        logging.info(
            "S_AND_T iteration %i of %i. iteration time: %.2f ",
            "average time: %.2f total time: %.2f seconds",
            i+1, num_iters, wall_time()-t1, (wall_time()-t0)/(i+1), wall_time()-t0)
        t1 = wall_time()
        
    Prev_Par, w_steps = differential_evolution(
        results, av_FR, bounds, Prev_Par, i/num_iters)
    best_values = CreateBestValues(Prev_Par, av_FR)
    BV_hist.append(np.copy(best_values))
     
    if SAVE_S_AND_T:
        with open(SCALE_THRESH_FILE_S, 'wb') as f:
         	np.save(f, best_values)
        with open(BV_HIST_S, "wb") as fp:   #Pickling
            pickle.dump(BV_hist, fp)
        with open(PREV_PAR_S, "wb") as fp:   #Pickling
            pickle.dump(Prev_Par, fp)
    
    logging.info("S_AND_T saved values, now fine-tuning")
    
    BV_hist.append(best_values)
    BV_hist2 = Create_BV_hist2(BV_hist)
    reps = [0,20]#make actual hist for comparison 
    new_res = []
    for j in range(BV_hist2.shape[2]):
        params = np.copy(BV_hist2[:,:,j])
        inps, args = ready_make_out_all_spikes_par(
            reps, inp_indices, inp_spikes, num_units, 
            weight_multi_3d, duration, params)
        new_iterable = ([x, args] for x in inps)
        if __name__ == '__main__':
            with multiprocessing.Pool(n_cores) as p:
                results = p.map(make_out_all_spikes_par, new_iterable)
        oas = np.ndarray(
            [num_units, len(inp_indices[0]), reps[1]-reps[0]], dtype='object')
        for i in range(len(results)):
            r = results[i]
            oas[:,r[2],r[3]] = r[0]
        out_all_spikes = oas.tolist()
        
        RMSE_neur, correlation, correlation2 = score_run(
            actual_hist, num_units, out_all_spikes, events, reps, n_bins)
        new_res.append((RMSE_neur, params[:,:5], correlation, correlation2))
        
        logging.info(
            "Iteration %i of %i Run time: %.2f ",
            j+1, BV_hist2.shape[2], wall_time()-t00)

    best_values = CreateBestValues(new_res, av_FR)
    
    
    logging.info("Finished creating weight scales and threshold offsets")
    logging.info("Run time: %.2f", wall_time()-t00)
    

    if SAVE_S_AND_T:
        with open(SCALE_THRESH_FILE_S, 'wb') as f:
         	np.save(f, best_values)
        with open(BV_HIST_S, "wb") as fp:   #Pickling
            pickle.dump(BV_hist, fp)
        with open(PREV_PAR_S, "wb") as fp:   #Pickling
            pickle.dump(Prev_Par, fp)
else:
    #Read Best Values from server batch-    Needed to optimize testing phase
    with open(SCALE_THRESH_FILE, 'rb') as f:
        best_values = np.load(f)#0-2 are scale factors for first 3 weights
        #, 3 is threshold for output neuron, 4 is minimum error achieved
    
    
#%% Testing of network--- production version multi-unit with specific weight steps and thresholds
#param comes from the optimization routine where param[0:2] ar the weight_steps
# for the three epochs and param[3] is the optimal threshold
params = np.copy(best_values)
if CREATE_OUTPUTS:
    rep_start = 0
    rep_end = len(inp_indices[0][0])
    inps, args = ready_make_out_all_spikes_par(
        [rep_start, rep_end], inp_indices, inp_spikes, num_units, 
        weight_multi_3d, duration, params)

    new_iterable = ([x, args] for x in inps)
    if __name__ == '__main__':
        with multiprocessing.Pool(n_cores) as p:
            results = p.map(make_out_all_spikes_par, new_iterable)
    oas = np.ndarray(
        [num_units, len(inp_indices[0]), rep_end-rep_start], dtype='object')
    oap = np.ndarray(
        [len(inp_indices[0]), rep_end-rep_start], dtype='object')
    for i in range(len(results)):
        r = results[i]
        oas[:,r[2],r[3]] = r[0]
        oap[r[2],r[3]] = r[1]
            
    out_all_spikes = oas.tolist()
    
    logging.info("Finished creating outputs")
    logging.info("Run time: %.2f", wall_time()-t00)
    
    if SAVE_OUTPUTS:
        with open(OSP_FILE_S, 'wb') as f:
         	np.save(f, oas)
        with open(O_POT_FILE_S, 'wb') as f:
         	np.save(f, oap)
else:
    with open(OSP_FILE, 'rb') as f:
            oas = np.load(f,allow_pickle=True)
    out_all_spikes = oas.tolist()

