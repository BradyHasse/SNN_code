# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:11:29 2023

@author: Andrew
"""

#%%   Initialize the enviornment-  Make sure you are in the correct directory
import os
CodeDir = os.getcwd()#'C:/Users/BAH150/.spyder-py3/Brian2/Brady'
os.chdir(CodeDir)

from brian2 import second

import pickle
import multiprocessing
n_cores = multiprocessing.cpu_count()-2 #number of core for parallel processing
print('cpu usage: ' +str(n_cores)+'/' +str(multiprocessing.cpu_count()))#server has 80 
numiters = 500#number of iterations for differential evolution
import sys
from time import time as wall_time
t00 = wall_time()
from datetime import datetime

import numpy as np
import scipy.io as sio
from random import randint

# from Libs.Histograms import * #change this to only get the ones we need from helper functions
from Libs.Helper_Functions import make_norm_histos, simple_regress, magnitude, make_norm_histos_nbins, w_steps_gen
from Libs.Helper_Functions import differential_evolution,CreateBestValues, Create_BVhist2,ready_make_out_all_spikes_par,score_run
from Libs.Input_generation import make_individual_input_spikes_par, make_offset_weights_par, make_out_all_spikes_par, par_w_step
#%% which nhp
Monk = 'N'
# Monk = 'C'
#%% Load in vs. create data
CreateInputs = True #if you remake inputs you should also do weights, offsets,scale, and thresholds.
SaveInputs = True

CreateWandO = True
SaveWandO = True

CreateSandT = True
SaveSandT = True

CreateOutputs = True
SaveOutputs = True

#%% set up loaded in files
if Monk == 'N':
    filesuffix = '_05-04-2024-18-01-17.npy'
    spkstruct = sio.loadmat(CodeDir+'/Data/MonkN359Selected.mat')
if Monk == 'C':    
    filesuffix = '_06-04-2024-21-46-41.npy'
    spkstruct = sio.loadmat(CodeDir+'/Data/MonkCDataSelected.mat')
    
indFile = CodeDir+'/Data/Monk' + Monk + '_input_indices' + filesuffix[:-3]+'pickle'
spkFile = CodeDir+'/Data/Monk' + Monk + '_input_spikes' + filesuffix[:-3]+'pickle'
wFile = CodeDir+'/Data/Monk' + Monk + '_Weights' + filesuffix
offsetFile = CodeDir+'/Data/Monk' + Monk + '_W_offset' + filesuffix
scalethreshFile = CodeDir+'/Data/Monk' + Monk + '_RMSE_Scale_Thresh' + filesuffix
OspkFile = CodeDir+'/Data/Monk' + Monk + '_output_spikes' + filesuffix
    
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
spkFile_s = CodeDir+'/Data/Monk' + Monk + '_input_spikes_' +dt_string + '.pickle'
indFile_s = CodeDir+'/Data/Monk' + Monk + '_input_indices_' +dt_string + '.pickle'
wFile_s = CodeDir+'/Data/Monk' + Monk + '_Weights_' +dt_string + '.npy'
offsetFile_s = CodeDir+'/Data/Monk' + Monk + '_W_offset_' +dt_string + '.npy'
scalethreshFile_s = CodeDir+'/Data/Monk' + Monk + '_RMSE_Scale_Thresh_' +dt_string + '.npy'
OspkFile_s = CodeDir+'/Data/Monk' + Monk + '_output_spikes_' +dt_string + '.npy'
OpotFile_s = CodeDir+'/Data/Monk' + Monk + '_output_potential_' +dt_string + '.npy'
BVhist_s = CodeDir+'/Data/Monk' + Monk + '_BVhist_' +dt_string + '.pickle'
PrevPar_s = CodeDir+'/Data/Monk' + Monk + '_PrevPar_' +dt_string + '.pickle'

#%%Get Data  for Monkey C or N
all_units = spkstruct['spk_all']

r_all = all_units
r_allL = r_all.tolist()#convert to list to match out_all_spikes
for i in range(len(r_allL)):#fix orientation
    for ii in range(len(r_allL[0])):
        for iii in range(len(r_allL[0][0])):
            r_allL[i][ii][iii] = np.squeeze(np.transpose(r_allL[i][ii][iii]))
num_units = r_all.shape[0]
r_raw= spkstruct['spk_raw']
events = spkstruct['events_out']
event_names = spkstruct['event_names']
rctemp = spkstruct['rep_cnt']    
rcl =rctemp.tolist()
rca= np.array(rcl)
rep_cnt = rca.flatten()     
speed_all = spkstruct['speed_out']
maxspeeds = []
for i in range(speed_all.shape[0]):
    for j in range(speed_all.shape[0]):
        maxspeeds.append(np.max(speed_all[i,j]))
maxspeed = np.max(np.array(maxspeeds))
speed_times = spkstruct['speed_time']
gauss_center= spkstruct['gauss_mu'].flatten()
gauss_sigma= spkstruct['gauss_sigma'].flatten()
event_landmarks= spkstruct['landmarks'].flatten()
duration = np.ceil(events[:, :, 5]*10000, )/10#reward time in ms to ceil 0.1ms limit
num_neurons= 90

#%% inp_indices and inp_spikes
if CreateInputs:
    #Generate input indices array and input spike array
    targrep = []
    for targ in range(rep_cnt.shape[0]):
        for rep in range(rep_cnt[targ]):
            targrep.append([targ,rep,randint(0,9**9)])
    args = [events.shape[0],rep_cnt,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks, maxspeed]
    new_iterable = ([x, args] for x in targrep)
    if __name__ == '__main__':
        with multiprocessing.Pool(n_cores) as p:
            results = p.map(make_individual_input_spikes_par, new_iterable)
    out_inp_indices = np.ndarray([4, events.shape[0], np.max(rep_cnt)], dtype='object')
    out_inp_spikes = np.ndarray([4, events.shape[0], np.max(rep_cnt)], dtype='object')
    for i in range(len(results)):
        r = results[i]
        for j in range(4):
            out_inp_indices[j,r[2],r[3]] = r[0][j]
            out_inp_spikes[j,r[2],r[3]]  = r[1][j]*second
    inp_indices = np.ndarray.tolist(out_inp_indices)
    inp_spikes = np.ndarray.tolist(out_inp_spikes)
    print('finished creating inputs')
    print('run time:' +str(np.round(wall_time()-t00, decimals=2)))
    
    if SaveInputs:
        with open(spkFile_s, 'wb') as file:
            pickle.dump(inp_spikes, file)
        with open(indFile_s, 'wb') as file:
            pickle.dump(inp_indices, file)
else:
    #Read input indices array and input spike array
    with open(spkFile, 'rb') as file:
        inp_spikes = pickle.load(file)
    with open(indFile, 'rb') as file:
        inp_indices = pickle.load(file)
    
#%% Weights
if CreateWandO:
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
            inps.append([r_all[:,target,rep], inp_ind_t, inp_spk_t, target, rep, duration])
                
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
            (b0,slope)= simple_regress(np.array(tmp_offsets),np.array([np.mean(temp_weights1), np.mean(temp_weights2)]))
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
    
    print('finished creating weights')
    print('run time:' +str(np.round(wall_time()-t00, decimals=2)))
    
    if SaveWandO:
        with open(offsetFile_s, 'wb') as f:
            np.save(f, final_offsets)
        with open(wFile_s, 'wb') as f:
            np.save(f, weight_multi)
else:
    #Load in previously created weights and offsets
    # Get stored offsets that came from next code section
    with open(offsetFile, 'rb') as f:
        final_offsets = np.load(f)
    
    #offsets have already been "applied" to these weights
    with open(wFile, 'rb') as f:
        weight_multi = np.load(f)
    weight1_multi = np.transpose(weight_multi[0,:])
    weight2_multi = np.transpose(weight_multi[1,:])
    weight3_multi = np.transpose(weight_multi[2,:])
    weight4_multi = np.transpose(weight_multi[3,:])
        
weight_multi_3d = np.reshape(weight_multi,[len(inp_spikes), -1, num_units])

#%% Scale for weights and threshold for output units. 
if CreateSandT:
    #make parameter search space
    bounds = np.array([[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[-0.02,0.02],[-5, 5]])#bounds for the wsteps (0-3) and threshold (4)
    popsize = 14# A multiplier for setting the total population size - number of individuals per dimension
    minpop = popsize*bounds.shape[0]
    popsize = int(np.ceil(minpop/n_cores)*n_cores)#set the total population to be a multiple of the number of cores
    population = w_steps_gen(num_units, bounds, popsize)
    w_steps = list(population)
    
    reps = [0,20]#make actual hist for comparison 
    actual_spikes= r_allL[0]#first unit to init
    [actual_hist,d,e,nbins] =  make_norm_histos_nbins(actual_spikes,events,reps,4,o_binwidth=0.005)
    actual_hist = np.zeros([num_units, actual_hist.shape[0], actual_hist.shape[1]])
    for u in range(num_units):
        actual_spikes= r_allL[u]
        [actual_hist[u,:,:],d,e,nbins] = make_norm_histos_nbins(actual_spikes,events,reps,4,nbins,o_binwidth=0.005)
    Av_FR = np.mean(actual_hist,axis=(2,1))

    BVhist = []
    reps = [0,20]
    args = [num_units, weight_multi_3d, inp_spikes, inp_indices, reps, duration, events, actual_hist, nbins]
    new_iterable = ([x, args] for x in w_steps)
    if __name__ == '__main__':
        with multiprocessing.Pool(n_cores) as p:
            results = p.map(par_w_step, new_iterable)
    PrevPar = results
    t0 = wall_time()
    t1 = wall_time()
    for i in range(numiters):#do that many iterations
        PrevPar, w_steps = differential_evolution(results, Av_FR,bounds,PrevPar,(i/numiters)*3)
        BestValues = CreateBestValues(PrevPar,Av_FR)
        
        if SaveSandT:
            with open(scalethreshFile_s, 'wb') as f:
             	np.save(f, BestValues)
        BVhist.append(np.copy(BestValues))
        new_iterable = ([x, args] for x in w_steps)
        if __name__ == '__main__':
            with multiprocessing.Pool(n_cores) as p:
                results = p.map(par_w_step, new_iterable)
        print('iteration ' +str(i+1)+'/'+str(numiters)  +' done')
        print('iteration time:' +str(np.round(wall_time()-t1, decimals=2)))
        print('average time:' +str(np.round((wall_time()-t0)/(i+1), decimals=2)))
        print('total time:' +str(np.round(wall_time()-t0, decimals=2)))
        t1 = wall_time()
        
    PrevPar, w_steps = differential_evolution(results, Av_FR,bounds,PrevPar,i/numiters)
    BestValues = CreateBestValues(PrevPar,Av_FR)
    BVhist.append(np.copy(BestValues))
     
    if SaveSandT:
        with open(scalethreshFile_s, 'wb') as f:
         	np.save(f, BestValues)
        with open(BVhist_s, "wb") as fp:   #Pickling
            pickle.dump(BVhist, fp)
        with open(PrevPar_s, "wb") as fp:   #Pickling
            pickle.dump(PrevPar, fp)
    
    print('Saved values, now fine-tuning')
    
    BVhist.append(BestValues)
    BVhist2 = Create_BVhist2(BVhist)
    reps = [0,20]#make actual hist for comparison 
    new_res = []
    for j in range(BVhist2.shape[2]):
        params = np.copy(BVhist2[:,:,j])
        inps, args = ready_make_out_all_spikes_par(reps, inp_indices, inp_spikes, num_units, weight_multi_3d, duration, params)
        new_iterable = ([x, args] for x in inps)
        if __name__ == '__main__':
            with multiprocessing.Pool(n_cores) as p:
                results = p.map(make_out_all_spikes_par, new_iterable)
        oas = np.ndarray([num_units, len(inp_indices[0]), reps[1]-reps[0]], dtype='object')
        for i in range(len(results)):
            r = results[i]
            oas[:,r[2],r[3]] = r[0]
        out_all_spikes = oas.tolist()
        
        RMSEneur, correlation, correlation2 = score_run(actual_hist, num_units, out_all_spikes, events, reps, nbins)
        new_res.append((RMSEneur, params[:,:5], correlation, correlation2))
        print('iteration ' +str(j+1)+'/'+str(BVhist2.shape[2])  +' done')
        print('run time:' +str(np.round(wall_time()-t00, decimals=2)))

    BestValues = CreateBestValues(new_res,Av_FR)
    
    print('finished creating weight scales and threshold offsets')
    print('run time:' +str(np.round(wall_time()-t00, decimals=2)))

    if SaveSandT:
        with open(scalethreshFile_s, 'wb') as f:
         	np.save(f, BestValues)
        with open(BVhist_s, "wb") as fp:   #Pickling
            pickle.dump(BVhist, fp)
        with open(PrevPar_s, "wb") as fp:   #Pickling
            pickle.dump(PrevPar, fp)
else:
    #Read Best Values from server batch-    Needed to optimize testing phase
    with open(scalethreshFile, 'rb') as f:
        BestValues = np.load(f)#0-2 are scale factors for first 3 weights, 3 is threshold for output neuron, 4 is minimum error achieved
    
    
#%% Testing of network--- production version multi-unit with specific weight steps and thresholds
#param comes from the optimization routine where param[0:2] ar the weight_steps for the three epochs and param[3] is the optimal threshold
params = np.copy(BestValues)
if CreateOutputs:
    rep_start = 0
    rep_end = len(inp_indices[0][0])
    inps, args = ready_make_out_all_spikes_par([rep_start, rep_end], inp_indices, inp_spikes, num_units, weight_multi_3d, duration, params)

    new_iterable = ([x, args] for x in inps)
    if __name__ == '__main__':
        with multiprocessing.Pool(n_cores) as p:
            results = p.map(make_out_all_spikes_par, new_iterable)
    oas = np.ndarray([num_units, len(inp_indices[0]), rep_end-rep_start], dtype='object')
    oap = np.ndarray([len(inp_indices[0]), rep_end-rep_start], dtype='object')
    for i in range(len(results)):
        r = results[i]
        oas[:,r[2],r[3]] = r[0]
        oap[r[2],r[3]] = r[1]
            
    out_all_spikes = oas.tolist()
    print('finished creating outputs')
    print('run time:' +str(np.round(wall_time()-t00, decimals=2)))

    if SaveOutputs:
        with open(OspkFile_s, 'wb') as f:
         	np.save(f, oas)
        with open(OpotFile_s, 'wb') as f:
         	np.save(f, oap)
else:
    with open(OspkFile, 'rb') as f:
            oas = np.load(f,allow_pickle=True)
    out_all_spikes = oas.tolist()

