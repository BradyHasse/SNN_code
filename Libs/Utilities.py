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

from Libs.Helper_Functions import simple_regress, smooth, make_histos, RMSE, make_norm_histos, magnitude
from Libs.Helper_Functions import plthist, spike_cause_count_v4,spike_cause_pot,make_norm_histos_nbins,score_run,bin_frac2
from Libs.Helper_Functions import spike_cause_FR_W, spike_cause_base,spike_cause_FR_W_all

#%% which nhp
# Monk = 'N'
Monk = 'C'

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
duration = np.ceil(events[:, :, 5]*10000, )/10#reward time in ms to ceil 0.1ms limit

#%% inp_indices and inp_spikes
#Read input indices array and input spike array
with open(spkFile, 'rb') as file:
    inp_spikes = pickle.load(file)
with open(indFile, 'rb') as file:
    inp_indices = pickle.load(file)
    
#%% save inputs in new formats
inp_spikes_np = np.array(inp_spikes,dtype=object)
for i in range(inp_spikes_np.shape[0]):
    for ii in range(inp_spikes_np.shape[1]):
        for iii in range(inp_spikes_np.shape[2]):
            inp_spikes_np[i,ii,iii] = inp_spikes_np[i,ii,iii]/second
inp_indices_np = np.array(inp_indices,dtype=object)
sdata = {'inp_spikes':inp_indices_np,'inp_indices':inp_indices_np}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_inputs'+filesuffix[:-3]+'mat',sdata)
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
#%% Done loading in data       
        
#%% load in colors for plots
num_neurons= 90
colors= sio.loadmat(CodeDir+'/Data/rgbColorMap.mat')["rgbColors"]
col_reordered = np.roll(colors,-1,axis=(0))
colors90 = np.zeros([num_neurons,4])
for i in range(colors90.shape[1]):
    colors90[:,i] = np.interp(np.arange(num_neurons), np.linspace(0,num_neurons,colors.shape[0]), col_reordered[:,i])
colors90 = np.roll(colors90,1,axis=(0))
        
#%% calculate predicted-actual correlation for all units

modscore = []
reps = [20,40]#make actual hist for comparison 
actual_spikes= r_allL[0]#first unit to init
[actual_hist,d,e,nbins] =  make_norm_histos_nbins(actual_spikes,events,reps,4,o_binwidth=0.005)
actual_hist = np.zeros([num_units, actual_hist.shape[0], actual_hist.shape[1]])
for u in range(num_units):
    actual_spikes= r_allL[u]
    [actual_hist[u,:,:],d,e,nbins] = make_norm_histos_nbins(actual_spikes,events,reps,4,o_binwidth=0.005)
    r2scores = np.corrcoef(np.squeeze(actual_hist[u,:,:]).T)**2#to see if unit was modulated.
    modscore.append(np.mean(np.append(np.diag(r2scores,k=1), r2scores[-1,0])))
    
modscore = np.array(modscore)    
RMSEneur, correlation, correlation2 = score_run(actual_hist, num_units, oas[:,:,reps[0]:reps[1]].tolist(), events, reps, nbins)

sdata = {'correlation':correlation2,'modscore':modscore}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_r-score_'+filesuffix[:-3]+'mat',sdata)


#%% Prepare matlab files for figures.

#%%%Figure 1
#%%%%D)
reps = [0, len(inp_indices[0][0])]
[histo,xax_labels,mean_ev] =  make_norm_histos(r_allL[0],events,reps,4,o_binwidth=0.005)
histo_all = np.zeros([histo.shape[0], histo.shape[1], num_units])
for unit in range(num_units):#takes a few seconds
    [histo_all[:,:,unit],xax_labels,mean_ev] =  make_norm_histos(r_allL[unit],events,reps,4,o_binwidth=0.005)
    

sdata = {'histo_all':histo_all, 'xax_labels':xax_labels, 'mean_ev':mean_ev}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_histo_all_actual'+filesuffix[:-3]+'mat',sdata)

#%%%Figure 3
if (Monk=='C'):
    unit = 58
else:
    unit = 47
# example input and output spikes to show how weights are trained.
outexample = np.array(oas[unit,0,0])
inexample = inp_spikes_np[1,0,0]
inexampleind = inp_indices_np[1,0,0]
inexample = inexample[inexampleind == 0]

sdata = {'outexample':outexample, 'inexample':inexample}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_example_weight_training'+filesuffix[:-3]+'mat',sdata)

#%%%%A)
sspikes = []
inp_spike_number = 45
reps= [0,20]
for epoch in range(4):
    star = []
    for dtarget in range(16):
        rspks = []
        for drep in range(reps[0], reps[1]):
                rspks.append(inp_spikes[epoch][dtarget][drep][inp_indices[epoch][dtarget][drep]==inp_spike_number])
                # rspks.append(inp_spikes2[epoch][dtarget][drep][inp_indices2[epoch][dtarget][drep]==inp_spike_number])
        star.append(rspks)
    sspikes.append(star)
    

[histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[0],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4,o_binwidth=0.005)
histo = np.expand_dims(histo,2)
for inp_group in range(1,4): 
        histo =  np.append(histo, np.expand_dims(make_norm_histos(sspikes[inp_group],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4,o_binwidth=0.005)[0],2), axis=2)

sdata = {'histoin':histo, 'xax_labelsin':xax_labels, 'mean_evin':mean_ev, }
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_input_FR'+filesuffix[:-3]+'mat',sdata)

#%%%%B)
weights = weight_multi_3d[:,:,unit]
weights = np.multiply(np.tile(np.expand_dims(BestValues[unit,:4],1),[1,num_neurons]),weights)
sdata = {'weights':weights}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_weights_U'+str(unit)+'_'+filesuffix[:-3]+'mat',sdata)
weights_all = []
for u in range(num_units):
    weights = weight_multi_3d[:,:,u]
    weights_all.append(np.multiply(np.tile(np.expand_dims(BestValues[u,:4],1),[1,num_neurons]),weights))
    
sdata = {'weights':np.array(weights_all)}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_weights_all'+filesuffix[:-3]+'mat',sdata)
#%%%%C)
#done in figure 1
#%%%Figure 4
#%%%%A)
pot_snips, mev, centers = spike_cause_pot(events,oas,oap,gauss_center)
winsiz = 200#20ms
epoch = 1
testmat3 = np.zeros([0,winsiz])
target = 4
testmat = np.zeros([0,winsiz])
for rep in range(oas.shape[2]):
    testmat = np.append(testmat, pot_snips[unit,target,rep,epoch],axis = 0)
testmat3 = np.append(testmat3, testmat,axis = 0)

sdata = {'mev':mev, 'centers':centers, 'sta_pot':testmat3,'threshold':BestValues[unit,4]}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_STA_pot_U'+str(unit)+'_'+filesuffix[:-3]+'mat',sdata)
#%%%%B)
reps = [0, len(inp_indices[0][0])]
[histo,xax_labels,mean_ev] =  make_norm_histos(out_all_spikes[unit],events,reps,4,o_binwidth=0.005)

sdata = {'histo':histo, 'xax_labels':xax_labels, 'mean_ev':mean_ev}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_plthist_U'+str(unit)+'_'+filesuffix[:-3]+'mat',sdata)
    
#%%%%C-F)
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


pred_spikes = out_all_spikes[unit]
reps = [0, events.shape[1]]
winsizs = [0.1, 20]
epochs = [1,2,3]

EAccmeg = [[[]  for w in range(len(winsizs))] for e in range(len(epochs))]
EPermeg = [[[]  for w in range(len(winsizs))] for e in range(len(epochs))]
EFRsmeg = [[[]  for w in range(len(winsizs))] for e in range(len(epochs))]
out_sample_mat = [[[0  for w in range(len(winsizs))] for t in range(events.shape[0])] for e in range(len(epochs))]

for w in range(len(winsizs)):
    winsiz = winsizs[w]
    for t in range(events.shape[0]):
        for e in range(len(epochs)):
            (accum, percents, out_sample,FRs) = spike_cause_FR_W(t,reps,events,pred_spikes,inp_spikes,inp_indices,epochs[e],gauss_center, winsiz,bintimes,histin)
            EAccmeg[e][w].append(accum)
            EPermeg[e][w].append(percents)
            EFRsmeg[e][w].append(FRs)
            for i in range(len(out_sample)):
                out_sample_mat[e][t][w] = out_sample_mat[e][t][w]+np.size(out_sample[i])




sdata = {'EPermeg1':EPermeg[0], 'EPermeg2':EPermeg[1], 'EPermeg3':EPermeg[2]}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_EPermeg1_U'+str(unit)+'_'+str(np.round(winsiz,decimals=1))+'ms' + filesuffix[:-3]+'mat',sdata)

sdata = {'EFRsmeg1':EFRsmeg[0], 'EFRsmeg2':EFRsmeg[1], 'EFRsmeg3':EFRsmeg[2]}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_EFRsmeg1_U'+str(unit)+'_'+str(np.round(winsiz,decimals=1))+'ms' + filesuffix[:-3]+'mat',sdata)

sdata = {'EAccmeg1':EAccmeg[0], 'EAccmeg2':EAccmeg[1], 'EAccmeg3':EAccmeg[2]}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_EAccmeg1_U'+str(unit)+'_'+str(np.round(winsiz,decimals=1))+'ms' + filesuffix[:-3]+'mat',sdata)

sdata = {'out_sample_mat1':out_sample_mat[0], 'out_sample_mat2':out_sample_mat[1], 'out_sample_mat3':out_sample_mat[2]}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_out_sample_mat_U'+str(unit)+'_'+str(np.round(winsiz,decimals=1))+'ms' + filesuffix[:-3]+'mat',sdata)
#%%%%G) the stats part.
reps = [0, events.shape[1]]
winsiz = [0.1, 20]
epochs = [1,2,3]
EAccmeg = [[[]  for w in range(len(winsiz))] for e in range(len(epochs))]
EPermeg = [[[]  for w in range(len(winsiz))] for e in range(len(epochs))]

for w in range(len(winsiz)):
    for t in range(events.shape[0]):
        for e in range(len(epochs)):
            (accum, percents) = spike_cause_base(events,inp_spikes,inp_indices,t,epochs[e],gauss_center,winsiz[w])
            EAccmeg[e][w].append(accum)
            EPermeg[e][w].append(percents)
            
EAccmeg = np.array(EAccmeg)
EPermeg = np.array(EPermeg)
sdata = {'Counts':EAccmeg, 'Percents':EPermeg}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_input_spike_counts' + filesuffix[:-3]+'mat',sdata)

reps = [0, events.shape[1]]
winsiz = 20 #repeated for 0.1 and 20 to do the trigger and buildup. Also changed the save name.
epochs = [1,2,3]
EAccmeg = [[[]  for w in range(num_units)] for e in range(len(epochs))]
EPermeg = [[[]  for w in range(num_units)] for e in range(len(epochs))]
EFRsmeg = [[[]  for w in range(num_units)] for e in range(len(epochs))]

out_sample_mat = [[[0  for w in range(num_units)] for t in range(events.shape[0])] for e in range(len(epochs))]
for unit in range(num_units):#takes a few seconds
    pred_spikes = out_all_spikes[unit]

    for t in range(events.shape[0]):
        for e in range(len(epochs)):
            (accum, percents, out_sample,FRs) = spike_cause_FR_W(t,reps,events,pred_spikes,inp_spikes,inp_indices,epochs[e],gauss_center, winsiz,bintimes,histin)
            EAccmeg[e][unit].append(accum)
            EPermeg[e][unit].append(percents)
            # EFRsmeg[e][unit].append(FRs)
            for i in range(len(out_sample)):
                out_sample_mat[e][t][unit] = out_sample_mat[e][t][unit]+np.size(out_sample[i])
    
sdata = {'EPermeg1':EPermeg[0], 'EPermeg2':EPermeg[1], 'EPermeg3':EPermeg[2]}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_EPermeg1_All_Units_'+str(np.round(winsiz,decimals=1))+'ms' + filesuffix[:-3]+'mat',sdata)

# sdata = {'EFRsmeg1':EFRsmeg[0], 'EFRsmeg2':EFRsmeg[1], 'EFRsmeg3':EFRsmeg[2]}
# sio.savemat(CodeDir+'/Data/Monk'+Monk+'_EFRsmeg1_All_Units_'+str(np.round(winsiz,decimals=1))+'ms' + filesuffix[:-3]+'mat',sdata)

sdata = {'EAccmeg1':EAccmeg[0], 'EAccmeg2':EAccmeg[1], 'EAccmeg3':EAccmeg[2]}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_EAccmeg1_All_Units_'+str(np.round(winsiz,decimals=1))+'ms' + filesuffix[:-3]+'mat',sdata)

sdata = {'out_sample_mat1':out_sample_mat[0], 'out_sample_mat2':out_sample_mat[1], 'out_sample_mat3':out_sample_mat[2]}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_out_sample_mat_All_Units_'+str(np.round(winsiz,decimals=1))+'ms' + filesuffix[:-3]+'mat',sdata)

reps = [0, events.shape[1]]
winsiz = 20
epochs = [1,2,3]
EAccmeg = [[[]  for w in range(num_units)] for e in range(len(epochs))]
EPermeg = [[[]  for w in range(num_units)] for e in range(len(epochs))]
EFRsmeg = [[[]  for w in range(num_units)] for e in range(len(epochs))]

out_sample_mat = [[[0  for w in range(num_units)] for t in range(events.shape[0])] for e in range(len(epochs))]
for unit in range(num_units):#takes a few seconds
    pred_spikes = out_all_spikes[unit]

    for t in range(events.shape[0]):
        for e in range(len(epochs)):
            (accum, percents, out_sample,FRs) = spike_cause_FR_W(t,reps,events,pred_spikes,inp_spikes,inp_indices,epochs[e],gauss_center, winsiz,bintimes,histin)
            EAccmeg[e][unit].append(accum)
            EPermeg[e][unit].append(percents)
            # EFRsmeg[e][unit].append(FRs)
            for i in range(len(out_sample)):
                out_sample_mat[e][t][unit] = out_sample_mat[e][t][unit]+np.size(out_sample[i])
                
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
            


#%%%Figure 5
#%%%%A)
reps = [0, len(inp_indices[0][0])]
[histo,xax_labels,mean_ev] =  make_norm_histos(out_all_spikes[0],events,reps,4,o_binwidth=0.005)
histo_all = np.zeros([histo.shape[0], histo.shape[1], num_units])
for unit in range(num_units):#takes a few seconds
    [histo_all[:,:,unit],xax_labels,mean_ev] =  make_norm_histos(out_all_spikes[unit],events,reps,4,o_binwidth=0.005)
    

sdata = {'histo_all':histo_all, 'xax_labels':xax_labels, 'mean_ev':mean_ev}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'_histo_all'+filesuffix[:-3]+'mat',sdata)



#%% plot histograms 
# reps = [26, 46]
reps = [0, len(inp_indices[0][0])]
for unit in range(num_units):
    plthist(out_all_spikes[unit], events, reps, 4)
    plt.title('new predicted ' + str(unit) + ' r score ' + str(np.round(correlation2[unit], decimals=2)))
    actual_spikes= r_allL[unit]
    plthist(actual_spikes,events,reps,4)
    plt.title('actual ' + str(unit))
    plt.show()

#%% Build a raster for one epoch and one target and one rep of all input units 
target = 0
rep = 11
# inp_group = 3#0-3
for inp_group in range(4):
    plt.figure()
    ax= plt.gca()
    ax.set_prop_cycle(plt.cycler('color', colors90)) 
    ttimes = []  
    for tt in range(90):
        epochs = []
        #next line for 1 input group
        epochs =inp_spikes[inp_group][target][rep][inp_indices[inp_group][target][rep]==tt] 
        #next line for all input groups
        # epochs= np.concatenate((inp_spikes[0][target][rep][inp_indices[0][target][rep]==tt],inp_spikes[1][target][rep][inp_indices[1][target][rep]==tt],inp_spikes[2][target][rep][inp_indices[2][target][rep]==tt],inp_spikes[3][target][rep][inp_indices[3][target][rep]==tt]), axis = 0)
    
        ttimes.append(epochs)
        num_spikes = len(ttimes[tt]) 
        y = np.ones(num_spikes)*(tt+1)
        plt.plot(ttimes[tt],y,'.')
    plt.show
    
    
    
    

#%% Display input spike rates for each epoch,  A specific input can be specified

rc("pdf", fonttype=42)
sspikes = []
inp_spike_number = 45
mode = 'predicted'
reps= [0,20]
for epoch in range(4):
    star = []
    for dtarget in range(16):
        rspks = []
        for drep in range(reps[0], reps[1]):
                rspks.append(inp_spikes[epoch][dtarget][drep][inp_indices[epoch][dtarget][drep]==inp_spike_number])
                # rspks.append(inp_spikes2[epoch][dtarget][drep][inp_indices2[epoch][dtarget][drep]==inp_spike_number])
        star.append(rspks)
    sspikes.append(star)
    
for inp_group in range(4):            
    plt.figure()
       
    ax= plt.gca()
    ax.set_prop_cycle(plt.cycler('color', colors)) 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if inp_group == 0:
        [histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[0],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4)
        plt.plot(histo, lw = 2)
    if inp_group == 1:
        [histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[1],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4)
        plt.plot(histo,lw = 2)
    if inp_group == 2:
        [histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[2],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4)
        plt.plot(histo,lw = 2)
    if inp_group == 3:
        [histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[3],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4)
        plt.plot(histo,lw = 2)
    
    plt.ylabel ('Firing Rate   ')
    plt.xlabel ('Time (ms)')
    ymax =99
    plt.axis((0,35,0,ymax))
    ylabels = np.arange(0,ymax+1, 5)
    plt.yticks(ylabels) 
    xlabels= np.rint(xax_labels[0:len(xax_labels)+1:5]*1000).astype(int)
    xtplaces= np.arange(0,len(xax_labels),5)
    plt.xticks(ticks=xtplaces,labels=xlabels)
    mvmt_onset_bin = ((mean_ev[6]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1]
    peak_speed_bin= ((mean_ev[11]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1]
    target_acq_bin= ((mean_ev[9]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1] 
    plt.plot(mvmt_onset_bin,.5,'*')
    plt.plot(peak_speed_bin,.5,'*')
    plt.plot(target_acq_bin,.5,'*')
    plt.show()
# plt.savefig('Input spike rates 3 epochs',format = 'pdf', dpi= 600)  

#%%quick weight plot
unit = 58
for unit in range(num_units):
    w1 = weight1_multi[range(unit,len(weight1_multi),num_units)]  #unwrap vector
    w2 = weight2_multi[range(unit,len(weight2_multi),num_units)]  #unwrap vector
    w3 = weight3_multi[range(unit,len(weight3_multi),num_units)]  #unwrap vector
    w4 = weight4_multi[range(unit,len(weight4_multi),num_units)]  #unwrap vector
    f1 = plt.figure()
    plt.plot(w1*BestValues[unit,0], label='w1')
    plt.plot(w2*BestValues[unit,1], label='w2')
    plt.plot(w3*BestValues[unit,2], label='w3')   
    plt.plot(w4*BestValues[unit,3], label='w4')
    plt.title(str(unit))
    plt.legend(loc='lower right')
    
#%% Looking at output unit potentials 
#oap is a targets x reps ndarray. inside each cell is a num_units x duration (0.1ms res) array (float 64)


pot_snips, mev, centers = spike_cause_pot(events,oas,oap,gauss_center)
winsiz = 200#20ms

unit = 58

testmat2 = np.zeros([0,winsiz])
# for epoch in range(3):
for epoch in range(1,2):
    testmat3 = np.zeros([0,winsiz])
    # for target in range(oas.shape[1]):
    for target in range(4,5):
        testmat = np.zeros([0,winsiz])
        for rep in range(oas.shape[2]):
            testmat = np.append(testmat, pot_snips[unit,target,rep,epoch],axis = 0)
        plt.plot(testmat[:,:].T,c=colors[target,:])
        testmat2 = np.append(testmat2, testmat,axis = 0)
        testmat3 = np.append(testmat3, testmat,axis = 0)
    plt.plot(np.mean(testmat3[:,:].T, 1),c='k',linewidth=3)
    plt.axhline(y = BestValues[unit, 4], color = 'k', linestyle = ':') 
    plt.show()
     #Find the input spikes in an interval before each ouput spike within a specified epoch
     #epoch is based on 1 indexing - valid epochs are 1, 2, 3
 
    plt.plot(testmat2[:,-1])   
    plt.axhline(y = BestValues[unit, 4], color = 'k', linestyle = ':') 
    plt.plot(testmat2[:,-2])
    plt.show()

#%% Self contained block. Only need to run the initial lines. 
# run this section before the next if you want to generate inputs with speed set to 0
import pickle
import numpy as np
from brian2 import second
from random import randint
import multiprocessing
n_cores = multiprocessing.cpu_count()-2 #number of core for parallel processing
# Import helper functions from libraries
 

from Libs.Input_generation import make_individual_input_spikes_par


IND_FILE_S = os.path.join(CodeDir, 'Data', f'Monk{Monk}', f'Monk{Monk}_input_indices_no-speed{filesuffix[:-4]}.pickle')
SPK_FILE_S = os.path.join(CodeDir, 'Data', f'Monk{Monk}', f'Monk{Monk}_input_spikes_no-speed{filesuffix[:-4]}.pickle')

#Generate input indices array and input spike array
targ_rep = []
max_speed = 0.05

max_speeds = []
for i in range(speed_all.shape[0]):
    for j in range(speed_all.shape[0]):
        max_speeds.append(np.max(speed_all[i,j]))
max_speed = np.max(np.array(max_speeds))
for targ in range(rep_cnt.shape[0]):
    for rep in range(rep_cnt[targ]):
        # speed_all[targ, rep] = np.zeros(speed_all[targ, rep].shape)+0.001
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

with open(SPK_FILE_S, 'wb') as file:
    pickle.dump(inp_spikes, file)
with open(IND_FILE_S, 'wb') as file:
    pickle.dump(inp_indices, file)
    
# save inputs in new formats
inp_spikes_np = np.array(inp_spikes,dtype=object)
for i in range(inp_spikes_np.shape[0]):
    for ii in range(inp_spikes_np.shape[1]):
        for iii in range(inp_spikes_np.shape[2]):
            inp_spikes_np[i,ii,iii] = inp_spikes_np[i,ii,iii]/second
inp_indices_np = np.array(inp_indices,dtype=object)
sdata = {'inp_spikes':inp_indices_np,'inp_indices':inp_indices_np}
sio.savemat(CodeDir+'/Data/Monk'+Monk+'/_no-speed_inputs'+filesuffix[:-3]+'mat',sdata)


if (Monk=='C'):
    unit = 58
else:
    unit = 47

sspikes = []
inp_spike_number = 45
reps= [0,20]
for epoch in range(4):
    star = []
    for dtarget in range(16):
        rspks = []
        for drep in range(reps[0], reps[1]):
                rspks.append(inp_spikes[epoch][dtarget][drep][inp_indices[epoch][dtarget][drep]==inp_spike_number])
                # rspks.append(inp_spikes2[epoch][dtarget][drep][inp_indices2[epoch][dtarget][drep]==inp_spike_number])
        star.append(rspks)
    sspikes.append(star)
    
[histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[0],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4,o_binwidth=0.005)
histo = np.expand_dims(histo,2)
for inp_group in range(1,4): 
        histo =  np.append(histo, np.expand_dims(make_norm_histos(sspikes[inp_group],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4,o_binwidth=0.005)[0],2), axis=2)

sdata = {'histoin':histo, 'xax_labelsin':xax_labels, 'mean_evin':mean_ev, }
sio.savemat(CodeDir+'/Data/Monk'+Monk+'/_no-speed_input_FR'+filesuffix[:-3]+'mat',sdata)


#%% Self contained block. Only need to run the initial lines. 
# Create output file without input groups spiking.

from Libs.Input_generation import  make_out_all_spikes_par
from Libs.Helper_Functions import  ready_make_out_all_spikes_par
from brian2 import second
import copy
import multiprocessing
n_cores = multiprocessing.cpu_count()-2 #number of core for parallel processing

rep_start = 0
rep_end = len(inp_indices[0][0])
for in_group in range(len(inp_indices)):
    OSP_FILE_S = os.path.join(CodeDir, 'Data', f'Monk{Monk}', f'inputgroup{in_group+1}_no-speed_eliminated_output_spikes{filesuffix}')
    params = np.copy(BestValues)
    copied_indices = copy.deepcopy(inp_indices)
    copied_spikes = copy.deepcopy(inp_spikes)
    for target in range(len(copied_indices[in_group])):
        for rep in range(len(copied_indices[in_group][target])):
            copied_indices[in_group][target][rep] = np.array([]).astype(np.int32)
            copied_spikes[in_group][target][rep] = np.array([]).astype(np.int32)*second

    inps, args = ready_make_out_all_spikes_par(
        [rep_start, rep_end], copied_indices, copied_spikes, num_units, 
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
            
    with open(OSP_FILE_S, 'wb') as f:
     	np.save(f, oas)
         
reps = [0, len(inp_indices[0][0])]         
for in_group in range(len(inp_indices)):
    OspkFile = os.path.join(CodeDir, 'Data', f'Monk{Monk}', f'inputgroup{in_group+1}_no-speed_eliminated_output_spikes{filesuffix}')
    histo_File = os.path.join(CodeDir, 'Data', f'Monk{Monk}', f'inputgroup{in_group+1}_no-speed_eliminated_output_hist{filesuffix[:-4]}.mat')

    with open(OspkFile, 'rb') as f:
            oas = np.load(f,allow_pickle=True).tolist()
            

    [histo,xax_labels,mean_ev] =  make_norm_histos(oas[0],events,reps,4,o_binwidth=0.005)
    histo_all = np.zeros([histo.shape[0], histo.shape[1], num_units])
    for unit in range(num_units):#takes a few seconds
        [histo_all[:,:,unit],xax_labels,mean_ev] =  make_norm_histos(oas[unit],events,reps,4,o_binwidth=0.005)
        
    
    sdata = {'histo_all':histo_all, 'xax_labels':xax_labels, 'mean_ev':mean_ev}
    sio.savemat(histo_File,sdata)
    