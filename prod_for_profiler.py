# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:59:08 2023

@author: Andrew
"""

import os
os.chdir('C:/Users/BAH150/.spyder-py3/Brian2/Brady')
CodeDir = os.getcwd()
os.chdir(CodeDir)

import matplotlib.pyplot as plt
from matplotlib import rc   #this is needed for CorelDraw to read the fonts correctly

from brian2 import NeuronGroup, run, ms, defaultclock, SpikeMonitor, StateMonitor,start_scope,TimedArray,PopulationRateMonitor,PoissonGroup,Hz,second, Network
from brian2 import SpikeGeneratorGroup,mA,second, mV, volt, Synapses, collect

import math
import scipy.io as sio
from scipy.io import loadmat, savemat

import numpy as np
import pickle

from Libs.Helper_Functions import simple_regress, smooth, make_histos, RMSE, make_norm_histos, magnitude,spike_cause_FR_W
from Libs.Helper_Functions import plthist, spike_cause_count_v4,spike_cause_pot,make_norm_histos_nbins,score_run,bin_frac2,spike_cause_FR_W_all

#%% load in plot stuff
num_neurons= 90
colors= sio.loadmat(CodeDir+'/Data/rgbColorMap.mat')["rgbColors"]
col_reordered = np.roll(colors,-1,axis=(0))
colors90 = np.zeros([num_neurons,4])
for i in range(colors90.shape[1]):
    colors90[:,i] = np.interp(np.arange(num_neurons), np.linspace(0,num_neurons,colors.shape[0]), col_reordered[:,i])
colors90 = np.roll(colors90,1,axis=(0))
#%% which nhp
Monk = 'N'
# Monk = 'C'

#%% set up loaded in files
if Monk == 'N':
    filesuffix = '_03-07-2024-13-30-06.npy'
    spkstruct = sio.loadmat(CodeDir+'/Data/MonkN359Selected.mat')
if Monk == 'C':    
    filesuffix = '_28-06-2024-15-09-50.npy'
    spkstruct = sio.loadmat(CodeDir+'/Data/MonkCDataSelected.mat')
    
indFile = CodeDir+'/Data/Monk' + Monk + '_input_indices' + filesuffix[:-3]+'pickle'
spkFile = CodeDir+'/Data/Monk' + Monk + '_input_spikes' + filesuffix[:-3]+'pickle'
wFile = CodeDir+'/Data/Monk' + Monk + '_Weights' + filesuffix
offsetFile = CodeDir+'/Data/Monk' + Monk + '_W_offset' + filesuffix
scalethreshFile = CodeDir+'/Data/Monk' + Monk + '_RMSE_Scale_Thresh' + filesuffix
OspkFile = CodeDir+'/Data/Monk' + Monk + '_output_spikes' + filesuffix
OpotFile = CodeDir+'/Data/Monk' + Monk + '_output_potential' + filesuffix
    

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
rep_cnt_temp = spkstruct['rep_cnt']    
rcl =rep_cnt_temp.tolist()
rca= np.array(rcl)
rep_cnt = rca.flatten()     
speed_all = spkstruct['speed_out']
speed_times = spkstruct['speed_time']
gauss_center= spkstruct['gauss_mu'].flatten()
gauss_sigma= spkstruct['gauss_sigma'].flatten()
event_landmarks= spkstruct['landmarks'].flatten()

#%% inp_indices and inp_spikes
#Read input indices array and input spike array
with open(spkFile, 'rb') as file:
    inp_spikes = pickle.load(file)
with open(indFile, 'rb') as file:
    inp_indices = pickle.load(file)
    
inp_spikes_np = np.array(inp_spikes,dtype=object)
for i in range(inp_spikes_np.shape[0]):
    for ii in range(inp_spikes_np.shape[1]):
        for iii in range(inp_spikes_np.shape[2]):
            inp_spikes_np[i,ii,iii] = inp_spikes_np[i,ii,iii]/second
inp_indices_np = np.array(inp_indices,dtype=object)

#%% Weights
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
#Read Best Values from server batch-    Needed to optimize testing phase
with open(scalethreshFile, 'rb') as f:
    BestValues = np.load(f)#0-2 are scale factors for first 3 weights, 3 is threshold for output neuron, 4 is minimum error achieved

with open('PrevPar', 'rb') as file:
    PrevPar = pickle.load(file)
    
with open('BVhist', 'rb') as file:
    BVhist = pickle.load(file)
BVhistnp = np.array(BVhist)
    
#%% Output spikes
with open(OspkFile, 'rb') as f:
        oas = np.load(f,allow_pickle=True)
out_all_spikes = oas.tolist()     
with open(OpotFile, 'rb') as f:
        oap = np.load(f,allow_pickle=True)
#%%%%G) the stats part.
bintimes = np.empty(events[:,:,0].shape,dtype='object')
histin = np.empty([events.shape[0],events.shape[1],weight_multi_3d.shape[0],weight_multi_3d.shape[1]],dtype='object')
for target in range(events.shape[0]):
    print(' target:' + str(target))
    for rep in range(events.shape[1]):
        a_b_time = events[target,rep,5]
        num_bins= int(np.floor((a_b_time/0.02)+.0001))#sets the number of bins
        bin_width = a_b_time/num_bins#if it was not exact, makes bin_width correct.
        bintimes[target,rep] = np.linspace(bin_width/2, a_b_time-bin_width/2, num_bins)
        for inp_group in range(weight_multi_3d.shape[0]):
            for inp in range(weight_multi_3d.shape[1]):
                spikes = (inp_spikes[inp_group][target][rep][inp_indices[inp_group][target][rep]==inp]/ms)/1000                
                spikes = np.concatenate([spikes[0]-np.diff(spikes[0:2]), spikes, spikes[-1]+np.diff(spikes[-2:])])#add 1 spike before and after time limits
                histin[target,rep,inp_group,inp] = bin_frac2(spikes,0,a_b_time, bin_width)


# with open(CodeDir+'/Data/Monk' + Monk + '_temp_bintimes.npy', 'wb') as f:
#  	np.save(f, bintimes)
# with open(CodeDir+'/Data/Monk' + Monk + '_temp_histin.npy', 'wb') as f:
#  	np.save(f, histin)
# with open(CodeDir+'/Data/Monk' + Monk + '_temp_bintimes.npy', 'rb') as f:
#         bintimes = np.load(f,allow_pickle=True)
# with open(CodeDir+'/Data/Monk' + Monk + '_temp_histin.npy', 'rb') as f:
#         histin = np.load(f,allow_pickle=True)
#%%%%G) the stats part.

reps = [0, events.shape[1]]
winsiz = 0.1
infoint8All = np.empty([0,5],dtype=('int8'))
FRsAll = np.empty(0)

for unit in range(num_units):
    pred_spikes = out_all_spikes[unit]
    for t in range(events.shape[0]):
        (infoint8, FRs) = spike_cause_FR_W_all(t,events,pred_spikes,inp_spikes,inp_indices, winsiz,bintimes,histin)
        if FRs.size>0:
            infoint8=np.c_[infoint8,unit*np.ones_like(infoint8[:,0])]#[imp_g, inp_unit,rep, target,out_unit]
            infoint8All = np.concatenate((infoint8All,np.array(infoint8)))
            FRsAll = np.concatenate((FRsAll,FRs))
        
FRsAll = np.c_[FRsAll, np.zeros(FRsAll.shape[0])]
for s in range(infoint8All.shape[0]):
    FRsAll[s,1] = weight_multi_3d[infoint8All[s,[0]], infoint8All[s,[1]], infoint8All[s,[4]]]*BestValues[infoint8All[s,[4]],infoint8All[s,[0]]]

sdata = {'FRsAll':FRsAll, 'infoint8All':infoint8All}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_triggerFRW' + filesuffix[:-3]+'mat',sdata)





















