# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:17:23 2024

@author: BAH150
"""
import matplotlib.pyplot as plt
from brian2 import SpikeGeneratorGroup,mA,second, mV, volt, Synapses, collect
from brian2 import NeuronGroup, run, ms, defaultclock, SpikeMonitor, StateMonitor,start_scope,TimedArray,PopulationRateMonitor,PoissonGroup,Hz,second, Network
import numpy as np

import scipy.io as sio
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter
from random import randint
from Libs.Helper_Functions import simple_regress

#%%
def make_out_all_spikes(r_all, weight_multi_3d, inp_spikes, inp_indices, params, duration):

    start_scope()
    
    
    num_neurons = 90
    num_units = len(r_all)
    num_directions=16  
    buf_size = num_neurons * num_units
    params[:,3] = 1
    step = np.transpose(params[:,:-1])
    step = np.tile(numpy.expand_dims(step),[1, 1, num_neurons])
    weight_multi_3d
    

    w1m = np.zeros(len(weight1_multi))
    w2m = np.zeros(len(weight2_multi))          
    w3m = np.zeros(len(weight3_multi))
    w1 = np.zeros(len(weight1_multi))
    w2 = np.zeros(len(weight2_multi))          
    w3 = np.zeros(len(weight3_multi))
    
    slope = np.zeros(num_units)
     
    w1m = weight1_multi    
    w2m = weight2_multi     
    w3m = weight3_multi
    w4m = weight4_multi
    uth = np.zeros(num_units)
    
    step = np.zeros((num_units,3))
    for uni in range(num_units):
        step[uni][0]= params[uni][0]
        step[uni][1]= params[uni][1]
        step[uni][2]= params[uni][2]
        uth[uni] =  params[uni][3]
    
    w_array= []
    
    
    
    
    for un in range (num_units):
    
        weight_multi_3d
        w1w= w1m[range(un,len(w1m),num_units)]*step[un][0]  #unwrap vector
        w2w= w2m[range(un,len(w2m),num_units)]*step[un][1]  #unwrap vector
        w3w= w3m[range(un,len(w3m),num_units)]*step[un][2]  #unwrap vector
        w4w = w4m[range(un,len(w4m),num_units)]  #unwrap vector
        
        ndex = 0
        odex = 0
        pdex = 0
           
        for i in range(un,len(weight1_multi),num_units):  #wrap back into vector
            w1[i] = w1w[ndex]
            w2[i] = w2w[ndex]
            w3[i] = w3w[ndex]
            ndex = ndex+1
    
    
    
    
    SpkMon=[]
    StMon= []
    SpkCnt=[] 
    th_in = uth
    #num_reps = 3
    
    
    H_eqs= '''
    dv/dt = -(v)/tau :1
    tau:second
    th :1
    '''
    
    out_all_potentials= []
    out_tmp_spikes=[] 
    out_ind_spikes=[]
    
      
    
    r_counter = np.zeros(num_directions,dtype = int)
    ndex = 0
    for direction in range(num_directions):
        SpkMonr=[] 
        SpkCntr=[]
        StMonr= []
        inp1_group=[]
        inp2_group=[]
        inp3_group=[]
        inp4_group=[]
        
        
        rep_start = 0 
        rep_end = len(inp_indices[0][0])#do all of the reps so events and out_all_spikes have the corresponding data.
        for rep in range(rep_start,rep_end): 
            r_counter[ndex] +=1
    
            start_scope()
            
            inp1_group = SpikeGeneratorGroup(num_neurons,inp_indices[0][ndex][rep],inp_spikes[0][ndex][rep])
            inp2_group = SpikeGeneratorGroup(num_neurons,inp_indices[1][ndex][rep],inp_spikes[1][ndex][rep])
            inp3_group = SpikeGeneratorGroup(num_neurons,inp_indices[2][ndex][rep],inp_spikes[2][ndex][rep])
            inp4_group = SpikeGeneratorGroup(num_neurons,inp_indices[3][ndex][rep],inp_spikes[3][ndex][rep])
                  
            H= NeuronGroup(num_units,H_eqs,threshold = 'v > th', reset = 'v = -10 ',  method = 'exact')   #.08,-.3 #  0 -1 #-.006 -.5
            #H= NeuronGroup(num_units,H_eqs,threshold = 'v > 0.1', reset = 'v = -10 ',  method = 'exact')   #.08,-.3 #  0 -1 #-.006 -.5
    
            H.tau= 10*ms
            ##H.Voff = 0 #-.1
            H.th = th_in
            #H.th = uth
            
            
            SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += w')  #.1
            SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += w') #.1
            SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += w') #.1
            SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += 0.001*w') #.05
            
            SS1.connect()
            SS2.connect()
            SS3.connect()
            SS4.connect()
            
            SM=SpikeMonitor(H)
            M = StateMonitor(H, variables=True, record=True)
            
           
      
            SS1.w= w1
            SS2.w= w2
            SS3.w= w3
            SS4.w= w4m  
            
            
            run(duration*ms)
            unit_spikes = []
            for ui in range(num_units):
                unit_spikes.append(SM.t[SM.i==ui])
            SpkMonr.append(unit_spikes) 
            SpkCntr.append(SM.count)
            StMonr.append(M.v)
        out_tmp_spikes.append(SpkMonr)    #out_tmp_indices.append(SpkIndices)
        SpkCnt.append(SpkCntr)
        out_all_potentials.append(StMonr)
        ndex +=1
    
    
        print('Finished Direction ', ndex)
    out_all_spikes = []  
    out_all_spikes = [[[out_tmp_spikes[i][k][j] for k in range(r_counter[i])] for i in range(num_directions)] for j in range(num_units)]
    return out_all_spikes
