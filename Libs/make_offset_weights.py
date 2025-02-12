# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:06:12 2024

@author: BAH150
"""
from brian2 import SpikeGeneratorGroup,mA,second, mV, volt, Synapses, collect
from brian2 import NeuronGroup, run, ms, defaultclock, SpikeMonitor, StateMonitor,start_scope,TimedArray,PopulationRateMonitor,PoissonGroup,Hz,second, Network
import numpy as np

#%% make_offset_weights
def make_offset_weights(r_all, final_offsets, inp_spikes, inp_indices, reps, duration):
    num_units = len(r_all)
    
    num_neurons = 90  #number of input neurons
    num_inputs = num_neurons
    taup = 20*ms
    
    Apre = 0.01
    Apost= 0.01
    
    w_offsets = np.zeros([len(inp_spikes), final_offsets.shape[0]*num_neurons])
    for epoch in range(final_offsets.shape[1]):
        w_offsets[epoch,:] = np.squeeze(np.tile(final_offsets[:,epoch], [num_neurons,1]))
        
    W_previous = [.002, .002, .002, .002]#initial weights
    
    start_scope() 
    
    for rep in range(reps):   #Train on the first 20 reps
        for target in range(r_all.shape[1]):
            out_st_array = []
            out_ind_array = []
            for u in range(num_units):  #build arrays of actual units as output
                rr = r_all[u][target][rep]
                st = []
                shaperr = rr.shape  #Check for empty array
                if(shaperr[0] == 1):
    	            rr = np.transpose(rr)
                for jj in range(len(rr)):
                    if rr[jj]>=0:
                        spktime = np.float64(rr[jj] * 1000)
                        st.append(spktime)
                out_st_array= np.concatenate((out_st_array,st),axis=0)  #build packed indices and spikes for actual spikes
                indices= np.ones(len(st),int) * u
                out_ind_array= np.concatenate((out_ind_array,indices),axis= 0)   
     
            # Make spike generators for the four input groups (3 gassians and one speed input)
            # inp_indices and inp_spikes come from "make_individual_input_spikes"
            out_group = SpikeGeneratorGroup(num_units,out_ind_array,out_st_array *ms)
            inp_group = []
            Syns = []
            
            for i in range(len(inp_indices)):
                inp_group.append(SpikeGeneratorGroup(num_inputs,inp_indices[i][target][rep],inp_spikes[i][target][rep]))
                Syns.append(Synapses(inp_group[i], out_group,
                    '''
                    w : 1
                    w_offset : 1
                    
                    dapre/dt = -apre/taup : 1 (event-driven)
                    dapost/dt = -apost/taup : 1 (event-driven)
                    ''',
                    on_pre='''
                    apre += Apre
                    w = w+apost-w_offset
                    ''',
                    on_post='''
                    apost += Apost
                    w = w+apre-w_offset
                    '''))
            
                Syns[i].connect()
                Syns[i].w = W_previous[i]
                Syns[i].w_offset = w_offsets[i,:]
            S1 = Syns[0]
            S2 = Syns[1]
            S3 = Syns[2]
            S4 = Syns[3]
            inp_group1 = inp_group[0]
            inp_group2 = inp_group[1]
            inp_group3 = inp_group[2]
            inp_group4 = inp_group[3]
            run(duration*ms)
            Syns[0] = S1
            Syns[1] = S2
            Syns[2] = S3
            Syns[3] = S4
            for i in range(len(inp_indices)):
                W_previous[i] = Syns[i].w
        
        print('Rep ',rep)    
        
    weight_multi = np.zeros([len(inp_spikes),num_inputs*r_all.shape[0]])
    for i in range(len(inp_spikes)):
        weight_multi[i,:]= Syns[i].w

    print('Finished')    
    return(weight_multi)