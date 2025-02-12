# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:53:21 2024

@author: BAH150
"""

import numpy as np
 #(accum0,accum1,accum2,accum3,percents0,percents1,percents2,percents3,out_spikes,win0_accum,win1_accum,win2_accum,win3_accum) = ...
 #spike_cause_count_v3(t,reps,events,predicted_spikes,inp_spikes,inp_indices,1,gauss_center) 
# target = t
# predicted_spikes = out_all_spikes[57]
# pred_spikes = out_all_spikes[57]
# epoch = 1


#%% spike_cause_count_v4
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
     ev = np.round(t_events*1000)#comment out to make it match Andys
     ev = ev.astype(int)
     
     centers = np.squeeze(ev - np.tile(ints, [numreps, 1]))
     
     accum = np.zeros([90,len(inp_spikes)])
     
     out_sample= []
     for rep in range(numreps):
         pred_spikes_2 = np.round((np.array(pred_spikes_1[rep])*1000)*10)/10 #0.1ms resolution in simulator
         pred_spikes_2 = pred_spikes_2[(pred_spikes_2 > centers[rep, epoch-1]-width) & (pred_spikes_2 <= centers[rep, epoch-1]+width)]
         out_sample.append(pred_spikes_2)  
         for imp_g in range(len(inp_spikes)):
             inp_spikes_1 = np.round((np.array(inp_spikes[imp_g][target][rep+reps[0]])*1000)*10)/10 #0.1ms resolution in simulator
             inp_indices_1 = np.array(inp_indices[imp_g][target][rep+reps[0]])
             for p_spike in range(pred_spikes_2.size):
                 inp_spikes_2 = (inp_spikes_1 < pred_spikes_2[p_spike]) & (inp_spikes_1 >= pred_spikes_2[p_spike]-winsiz)
                 [imp_ind, imp_c] = np.unique(inp_indices_1[inp_spikes_2], return_counts=True)
                 accum[imp_ind,imp_g] = accum[imp_ind,imp_g] + imp_c
                 
     percents = accum/np.sum(accum)
             
     return(accum, percents, out_sample)