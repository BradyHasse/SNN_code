# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:36:48 2024

@author: BAH150
"""



import os
CodeDir = os.getcwd()#'C:/Users/BAH150/.spyder-py3/Brian2/Brady'
os.chdir(CodeDir)

from brian2 import second

import multiprocessing
n_cores = 4 #number of core for parallel processing
import sys
from time import time as wall_time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# from Libs.Histograms import * #change this to only get the ones we need from helper functions
from Libs.Helper_Functions import make_norm_histos, simple_regress, magnitude, plthist
from Libs.Input_generation import make_individual_input_spikes_par, make_offset_weights_par, make_out_all_spikes_par, par_w_step
#%% which nhp
# Monk = 'N'
Monk = 'C'
#%% set up loaded in files
if Monk == 'N':
    spkstruct = sio.loadmat(CodeDir+'/Data/MonkN359Selected.mat')
    indFile = CodeDir+'/Data/MonkN_input_indices_18-03-2024-12-41-59.npy'
    spkFile = CodeDir+'/Data/MonkN_input_spikes_18-03-2024-12-41-59.npy'
    wFile = CodeDir+'/Data/MonkN_Weights_18-03-2024-12-41-59.npy'
    offsetFile = CodeDir+'/Data/MonkN_W_offset_18-03-2024-12-41-59.npy'
    scalethreshFile = CodeDir+'/Data/MonkN_RMSE_Scale_Thresh_18-03-2024-12-41-59.npy'
    OspkFile = CodeDir+'/Data/MonkN_output_spikes_18-03-2024-12-41-59.npy'
    
if Monk == 'C':    
    spkstruct = sio.loadmat(CodeDir+'/Data/MonkCDataSelected.mat')
    indFile = CodeDir+'/Data/MonkC_input_indices_18-03-2024-12-10-19.npy'
    spkFile = CodeDir+'/Data/MonkC_input_spikes_18-03-2024-12-10-19.npy'
    wFile = CodeDir+'/Data/MonkC_Weights_18-03-2024-12-10-19.npy'
    offsetFile = CodeDir+'/Data/MonkC_W_offset_18-03-2024-12-10-19.npy'
    scalethreshFile = CodeDir+'/Data/MonkC_RMSE_Scale_Thresh_18-03-2024-12-10-19.npy'
    OspkFile = CodeDir+'/Data/MonkC_output_spikes_18-03-2024-12-10-19.npy'
    
dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
spkFile_s = CodeDir+'/Data/Monk' + Monk + '_input_spikes_' +dt_string + '.npy'
indFile_s = CodeDir+'/Data/Monk' + Monk + '_input_indices_' +dt_string + '.npy'
wFile_s = CodeDir+'/Data/Monk' + Monk + '_Weights_' +dt_string + '.npy'
offsetFile_s = CodeDir+'/Data/Monk' + Monk + '_W_offset_' +dt_string + '.npy'
scalethreshFile_s = CodeDir+'/Data/Monk' + Monk + '_RMSE_Scale_Thresh_' +dt_string + '.npy'
OspkFile_s = CodeDir+'/Data/Monk' + Monk + '_output_spikes_' +dt_string + '.npy'

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
rc = spkstruct['rep_cnt']    
rcl =rc.tolist()
rca= np.array(rcl)
rep_cnt = rca.flatten()     
speed_all = spkstruct['speed_out']
speed_times = spkstruct['speed_time']
gauss_center= spkstruct['gauss_mu'].flatten()
gauss_sigma= spkstruct['gauss_sigma'].flatten()
event_landmarks= spkstruct['landmarks'].flatten()
duration = 1000
num_neurons= 90

#%%

with open('C:/Users/BAH150/.spyder-py3/Brian2/Brady/Original/Data/MonkC_input_indices[July30].npy', 'rb') as f:
    num_sources= int(np.load(f))
    num_targets= int(np.load(f))
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
            
with open('C:/Users/BAH150/.spyder-py3/Brian2/Brady/Original/Data/MonkC_input_spikes[July30].npy', 'rb') as f:

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
                inp_spikes[i][j][k]=g_tmp * second
   
repn = np.arange(nrep,0,-1)
repn = np.delete(repn, (repn>21) * (repn<42))-1
for i in range(num_sources):#delete the unused reps
    for j in range(num_targets):
        for k in repn.tolist():
            del inp_spikes[i][j][k]
            del inp_indices[i][j][k]
   
#andys orgininal
with open('C:/Users/BAH150/.spyder-py3/Brian2/Brady/Original/Data/MonkC_all_predicted_spikes[Oct17].mat', 'rb') as f:
        oas = sio.loadmat(f)
oas = oas["predicted_spikes"]
out_all_spikes = oas.tolist()

#%%

def spike_cause_count_v3(target,reps,events,pred_spikes,inp_spikes,inp_indices,epoch,gauss_center):
    #Find the input spikes in an interval before each ouput spike within a specified epoch

     width = np.zeros(3)
     center = np.zeros(3)
     width[0]= 60
     width[1]= 60 #was 200, narrow was 60,wide = 150
     width[2]= 60
     percents0 = []
     percents1 = []
     percents2 = []
     percents3 = []
   
     mev = np.mean(events,1)
     mtarget_show = mev[target][2]
     mstart_movement = mev[target][6]
     mpk_speed_time= mev[target][11]
     mreward = mev[target][5]
     mend_movement = mev[target][9]
     mcenter_target = mev[target][8]
     int1 = (mstart_movement-gauss_center[0])*1000
     int2 = (mpk_speed_time-gauss_center[1])*1000
     int3 = (mend_movement-gauss_center[2])*1000
     num_inp = 90
     try:
         length=len(reps)
         if length== 2:
             num_reps= reps[1]-reps[0]
             r_start = reps[0]
             r_end = reps[1]
         else:
             num_reps = reps[target]
             print('Target = ',target, 'Num Reps=',num_reps)
             r_start= 0
             r_end = num_reps
     except:
         num_reps= reps
         r_start= 0
         r_end = reps
     
     t_events = []
     for irep in range(r_start,r_end):
         try:
             len(events[0][0])
             t_events.append(events[target][irep])
         except:
             print('Problem with events')

       
    
   
     repcounter = 0
    
     accum0 = np.zeros(90)
     accum1 = np.zeros (90)
     accum2 = np.zeros (90)
     accum3 = np.zeros (90)
    
     win0_accum = np.zeros(90)
     win1_accum = np.zeros(90)
     win2_accum = np.zeros(90)
     win3_accum = np.zeros(90)

     out_sample= []
     hdex = 0
     accum_counter= 0
     total_input_accum = 0

     winsiz = 20 #ms   Window before ouput spikes for ramp_up causality
     for rep in range(r_start,r_end):
    
            counter = []
            ee = t_events[repcounter] * 1000
            ev = np.array(ee)
            target_show = int(ev[2])
            start_movement = int(ev[6])
            pk_speed= int(ev[11])
            reward = int(ev[5])
            end_movement = int(ev[9])                                

            center[0] = start_movement-int1
            center[1] = pk_speed -int2 
            center[2] = end_movement-int3

           
            epoch_start= center[(epoch-1)]-(width[(epoch-1)])
            epoch_end= center[(epoch-1)] +(width[(epoch-1)])
    
            

            pspikes = np.squeeze(np.array(pred_spikes[target][repcounter],dtype=float)*1000)  #Note this is for using predictions from one particular unit (e.g. 57)
            # pspikes = np.array(pred_spikes[target][repcounter],dtype=float)*1000  #Note this is for using predictions from one particular unit (e.g. 57)
          
            num_pspk = len(pspikes)  #number of predicted spikes in the sample
       
            g0 = np.array(inp_spikes[0][target][rep],dtype=float) *1000
            g1 = np.array(inp_spikes[1][target][rep],dtype=float) *1000
            g2 = np.array(inp_spikes[2][target][rep],dtype=float) *1000
            g3 = np.array(inp_spikes[3][target][rep],dtype=float) *1000
            
 
            
           
            print('\r Rep = ',rep, end = ' ' )
            for ps in range(num_pspk):   # cycle through all the predicted spikes

                t_point = pspikes[ps]  #look at each spike of the modeled unit
                if t_point > epoch_start and t_point <= epoch_end:   #find the predicted spikes in the epoch of interest
                    min0 = winsiz +1
                    min1 = winsiz +1
                    min2 = winsiz +1
                    min3 = winsiz +1
                    accum_counter +=1
                    counter.append(t_point)
                    #print ('number of output units in epoch=',accum_counter)
                    
                    prev_step0 =  g0[list(xx < .1001 and xx> 0 for xx in t_point-g0)]                       
                    num_hits = len(prev_step0)
                    if num_hits > 0:
                     #   print('inp0 hits = ',num_hits)
                        hdex = list(xx== prev_step0[0] for xx in g0).index(True)                            
                        for hit in range(num_hits):
                            unit_number = inp_indices[0][target][rep][hdex]
                            accum0[unit_number]+=1
                            total_input_accum+=1
                            hdex +=1
                   
                    win_step0 =  g0[list(xx < winsiz and xx> 0 for xx in t_point-g0)]                  
                    min0 = np.min(t_point-win_step0)
                  
                    if min0 <winsiz : 
                        inputs0_window = []
                        diff0 = g0[list (xx > 0 for xx in t_point-g0)]                        
                        win_beg= list(xx<winsiz for xx in t_point-diff0).index(True)  #index of first input spike within winsiz ms of the output spike
                        win_end =len(diff0)-1
                        inputs0_window = inp_indices[0][target][rep][win_beg:win_end]
                        for w in range (len(inputs0_window)):
                            win0_accum[inputs0_window[w]]+=1
                    
                    
                  
                    
                    prev_step1 =  g1[list(xx < .1001 and xx> 0 for xx in t_point-g1)]                       
                    num_hits = len(prev_step1)
                    if num_hits > 0:
                      #  print('inp1 hits = ',num_hits)
                        hdex = list(xx== prev_step1[0] for xx in g1).index(True)                            
                        for hit in range(num_hits):
                            unit_number = inp_indices[1][target][rep][hdex]
                            accum1[unit_number]+=1
                            total_input_accum+=1
                            hdex +=1
                   
                    win_step1 =  g1[list(xx < winsiz and xx> 0 for xx in t_point-g1)]                  
                    min1 = np.min(t_point-win_step1)
                  
                    if min1 <winsiz : 
                        inputs1_window = []
                        diff1 = g1[list (xx > 0 for xx in t_point-g1)]                        
                        win_beg= list(xx<winsiz for xx in t_point-diff1).index(True)  #index of first input spike within winsiz ms of the output spike
                        win_end =len(diff1)-1
                        inputs1_window = inp_indices[1][target][rep][win_beg:win_end]
                        for w in range (len(inputs1_window)):
                            win1_accum[inputs1_window[w]]+=1
                    
                     
                    prev_step2 =  g2[list(xx < .1001 and xx> 0 for xx in t_point-g2)]                       
                    num_hits = len(prev_step2)
                    if num_hits > 0:
                       # print('inp2 hits = ',num_hits)
                        hdex = list(xx== prev_step2[0] for xx in g2).index(True)                            
                        for hit in range(num_hits):
                            unit_number = inp_indices[2][target][rep][hdex]
                            accum2[unit_number]+=1
                            total_input_accum += 1
                            hdex +=1
                   
                    win_step2 =  g2[list(xx < winsiz and xx> 0 for xx in t_point-g2)]                  
                    min2 = np.min(t_point-win_step2)
                  
                    if min2 <winsiz : 
                        inputs2_window = []
                        diff2 = g2[list (xx > 0 for xx in t_point-g2)]                        
                        win_beg= list(xx<winsiz for xx in t_point-diff2).index(True)  #index of first input spike within winsiz ms of the output spike
                        win_end =len(diff2)-1
                        inputs2_window = inp_indices[2][target][rep][win_beg:win_end]
                        for w in range (len(inputs2_window)):
                            win2_accum[inputs2_window[w]]+=1
                   
                    prev_step3 =  g3[list(xx < .1001 and xx> 0 for xx in t_point-g3)]                       
                    num_hits = len(prev_step3)
                    if num_hits > 0:
                        #print('inp3 hits = ',num_hits)
                        hdex = list(xx== prev_step3[0] for xx in g3).index(True)                            
                        for hit in range(num_hits):
                            unit_number = inp_indices[3][target][rep][hdex]
                            accum3[unit_number]+=1
                            total_input_accum+=1
                            hdex +=1
                   
                    win_step3 =  g3[list(xx < winsiz and xx> 0 for xx in t_point-g3)]                  
                    min3 = np.min(t_point-win_step3)
                  
                    if min3 <winsiz : 
                        inputs3_window = []
                        diff3 = g3[list (xx > 0 for xx in t_point-g3)]                        
                        win_beg= list(xx<winsiz for xx in t_point-diff3).index(True) 
                        win_end =len(diff3)-1
                        inputs3_window = inp_indices[3][target][rep][win_beg:win_end]
                        for w in range (len(inputs3_window)):
                            win3_accum[inputs3_window[w]]+=1 
                            
                
                   
            repcounter += 1
            out_sample.append(counter)
  
           
            #reps ends here
   
    
   
    
     print('Spk_Accum = ',accum_counter)  # number of output spikes in window
    
 
     total_win_count= np.sum(win0_accum)+np.sum(win1_accum)+np.sum(win2_accum)+np.sum(win3_accum)  # number of input spikes in pre-output spike window
     
     
     if accum_counter> 0:
  
           percents0 =  accum0/total_input_accum
           percents1 =  accum1/total_input_accum
           percents2 =  accum2/total_input_accum
           percents3 =  accum3/total_input_accum
     
           win0_percents = win0_accum/total_win_count
           win1_percents = win1_accum/total_win_count
           win2_percents = win2_accum/total_win_count
           win3_percents = win3_accum/total_win_count
         
           
    
     return(accum0,accum1,accum2,accum3,percents0,percents1,percents2,percents3,out_sample,win0_percents,win1_percents,win2_percents,win3_percents)

#%%
def spike_cause_count_v4(target,reps,events,pred_spikes,inp_spikes,inp_indices,epoch,gauss_center, winsiz):
     #Find the input spikes in an interval before each ouput spike within a specified epoch
     #epoch is based on 1 indexing - valid epochs are 1, 2, 3
     
     reps_s = np.array([len(pred_spikes[0]), len(inp_spikes[0][0]), len(inp_indices[0][0])])
     
     if ~np.all(reps_s == events.shape[1]):
         raise Exception("#reps mismatch between ""in_array"" and ""events")
     
     if reps[0] > reps[1]:
         reps = np.flip(reps)
     if reps[1] > events.shape[1]:
         raise Exception("reps requested out of range for in_array")
     
     numreps = int(np.diff(reps))
     pred_spikes_1 = pred_spikes[target][reps[0]:reps[1]]
     width = 60
   
     eventInds = np.array([6, 11, 9])#start_movement (6) pk_speed (11) end_movement (9)

     mev = np.mean(events[np.ix_([target],np.arange(events.shape[1]),eventInds)],1)
     ints = (mev-gauss_center)*1000
     
     t_events = events[np.ix_([target],np.arange(reps[0],reps[1]),eventInds)]
     ev = np.round(t_events*1000)
     ev = ev.astype(int)
     
     centers = np.squeeze(ev - np.tile(ints, [numreps, 1]))
     
     accum = np.zeros([90,len(inp_spikes)])
     
     out_sample= []
     for rep in range(numreps):
         pred_spikes_2 = np.round((np.array(pred_spikes_1[rep])*1000)*10)/10 #0.1ms resolution in simulator
         pred_spikes_2 = pred_spikes_2[(pred_spikes_2 > centers[rep, epoch-1]-width) & (pred_spikes_2 <= centers[rep, epoch-1]+width)]
         out_sample.append(pred_spikes_2)  
         for imp_g in range(len(inp_spikes)):#for all input groups
             inp_spikes_1 = np.round((np.array(inp_spikes[imp_g][target][rep+reps[0]])*1000)*10)/10 #0.1ms resolution in simulator
             inp_indices_1 = np.array(inp_indices[imp_g][target][rep+reps[0]])
             for p_spike in range(pred_spikes_2.size):
                 inp_spikes_2 = (inp_spikes_1 < pred_spikes_2[p_spike]) & (inp_spikes_1 >= pred_spikes_2[p_spike]-winsiz)
                 [imp_ind, imp_c] = np.unique(inp_indices_1[inp_spikes_2], return_counts=True)
                 accum[imp_ind,imp_g] = accum[imp_ind,imp_g] + imp_c
                 
     percents = accum/np.sum(accum)
             
     return(accum, percents, out_sample)
#%%
out_all_spikes = oas[:,:,21:41].tolist()#if getting outputs from Brady's run

#unit58 all reps andy
with open('C:/Users/BAH150/.spyder-py3/Brian2/Brady/Original/Data/U58_pred_spikes_all_reps[Nov20].npy', 'rb') as f:
        pred_spikes = np.load(f,allow_pickle=True)
pred_spikes = pred_spikes.tolist()




pred_spikes = out_all_spikes[58]
predicted_spikes = pred_spikes
eventsc = np.copy(events)#events[:,21:41] 
reps =[0, 46]# [0,20]
EPer1_all = []
EPer2_all = []
EPer3_all = []
for t in range(16):
    (accum, percents, out_sample) = spike_cause_count_v4(t,reps,eventsc,pred_spikes,inp_spikes,inp_indices,1,gauss_center,20) # 0.2 for the last spike, 20 ms for lead up
    EPer1_all.append(percents)
    (accum, percents, out_sample) = spike_cause_count_v4(t,reps,eventsc,pred_spikes,inp_spikes,inp_indices,2,gauss_center,20) 
    EPer2_all.append(percents)
    (accum, percents, out_sample) = spike_cause_count_v4(t,reps,eventsc,pred_spikes,inp_spikes,inp_indices,3,gauss_center,20) 
    EPer3_all.append(percents)
sdata = {'EPer1_all_accum':EPer1_all, 'EPer2_all_accum':EPer2_all, 'EPer3_all_accum':EPer3_all}
sio.savemat(CodeDir+'/Data/MonkC_causal_accum_allepochs[Feb-20-2024].mat',sdata)



EP = []
for epoch in range(1,4):
    EPer3_0 = []
    EPer3_1 = []
    EPer3_2 = []
    EPer3_3 = []
    for t in range(16):
       (accum0,accum1,accum2,accum3,percents0,percents1,percents2,percents3,out_sample,win0_percents,win1_percents,win2_percents,win3_percents)= spike_cause_count_v3(t,reps,events,predicted_spikes,inp_spikes,inp_indices,epoch,gauss_center) 
       EPer3_0.append(win0_percents)
       EPer3_1.append(win1_percents)
       EPer3_2.append(win2_percents)
       EPer3_3.append(win3_percents)
    
    EP0 = np.array(EPer3_0)
    EP1 = np.array(EPer3_1)
    EP2 = np.array(EPer3_2)
    EP3 = np.array(EPer3_3)
    EP.append(np.concatenate((EP0[...,None], EP1[...,None], EP2[...,None], EP3[...,None]), axis=2))
sdata = {'EPer1_all_accum':EP[0], 'EPer2_all_accum':EP[1], 'EPer3_all_accum':EP[2]}
sio.savemat(CodeDir+'/Data/MonkC_causal_accum_allepochs[Mar-21-2024].mat',sdata)

   
unit = 58
plthist(out_all_spikes[unit], eventsc, reps, 4)
plt.title(unit)
plt.show()
   


make_norm_histos(pred_spikes, eventsc, reps, 4)
plthist(pred_spikes, eventsc, reps, 4)
in_array = out_all_spikes[unit]
events = eventsc
numsegs = 4
   
plt.plot(BestValues2[:,-1])
plt.plot(BestValues[:,-1])
plt.title('rmse')
   
plt.plot(correlation2)
plt.plot(correlation3)
plt.title('r')


#%% load in results from large grid search

import pickle
with open('tuple.pickle', 'wb') as file:
    pickle.dump(name_of_tuple, file)
    
with open('PrevPar', 'rb') as file:
    PrevPar = pickle.load(file)
    
with open('BVhist', 'rb') as file:
    BVhist = pickle.load(file)
BVhistnp = np.array(BVhist)

RMSEneur = np.zeros([BVhist[0].shape[0], len(BVhist)])
r_score = np.zeros(RMSEneur.shape)
fitness = np.zeros(RMSEneur.shape)
for i in range(len(BVhist)):  
    RMSEneur[:,i] = BVhist[i][:,5]
    r_score[:,i] = BVhist[i][:,6]
    fitness[:,i] = BVhist[i][:,7]
    
plt.plot(np.mean(RMSEneur,axis=0))
plt.plot(np.mean(r_score,axis=0))
plt.plot(np.mean(fitness,axis=0))
   
prevall = np.zeros([len(PrevPar),PrevPar[0][1]. shape[0],PrevPar[0][1]. shape[1]]) 
for i in range(len(PrevPar)):
    prevall[i,:,:] = PrevPar[i][1]

for i in range(prevall. shape[2]):
    for unit in range(prevall. shape[1]):
        
        plt.plot(prevall[:, unit, i])
    #plt. title(unit) 
    plt. show()

sdata = {'RMSEneur':RMSEneur[58,:], 'r_score':r_score[58,:], 'w_steps_np':w_steps_np}
sio.savemat(CodeDir+'/Data/MonkC_Unit58_stepthesh_results].mat',sdata)
  
sdata = {'BestValues':BestValues}
sio.savemat(CodeDir+'/Data/BestValues.mat',sdata)
