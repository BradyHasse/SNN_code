# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:53:41 2024

@author: BAH150
"""

#%%
import numpy as np




#%%

def simrun2(offset,inp_indices,inp_spikes,r_all,unit,epoch,num_reps):
    weightchange = np.zeros([num_reps, 16,90])
    Apre = 0.01
    Apost= 0.01
    taup = 20 #*ms
    Apre = Apre/np.exp(0.1/taup)
    # Apost = Apre/np.exp(0.1/taup)
    duration = 1000
    num_inputs = 90
    apreTC = Apre/np.exp(np.arange(0,1000,0.1)/taup);#1000 ms in 0.1ms steps
    apostTC = Apost/np.exp(np.arange(0,1000,0.1)/taup);
    W_previous = .002   #initial weights
    inp_voltage = np.zeros([num_inputs, duration*10])
    out_voltage = np.zeros([1, duration*10])
    new_wt = np.zeros([num_inputs]) + W_previous
    
    target = 0#this should loop
    rep =  0#this should loop
    for target in range(r_all.shape[1]):
        for rep in range(num_reps):
            inp_spikes1 = np.array(inp_spikes[epoch-1][target][rep])*1000#spike times in ms. 0.1ms resolution
            inp_indices1 = np.array(inp_indices[epoch-1][target][rep])#unit number, corresponding to inp_spikes1
            inp_spikes1 = np.round(inp_spikes1*10).astype(int)#convert from ms to index of timestep
            inp_indices1 = inp_indices1[inp_spikes1<=duration*10]#remove spikes outside of duration
            inp_spikes1 = inp_spikes1[inp_spikes1<=duration*10]#remove spikes outside of duration
            #test
            inp_indtest = inp_spikes1[inp_indices1==1]
            #test
            r_all1 = np.squeeze(r_all[unit,target,rep])*1000#output spikes for the unit.
            r_all1 = np.round(r_all1*10).astype(int)#convert from ms to index of timestep
            r_all1 = r_all1[r_all1<=duration*10]#remove spikes outside of duration
        
            inp_voltage = inp_voltage*0
            for imps in range(np.size(inp_spikes1)):
                inp_voltage[inp_indices1[imps], inp_spikes1[imps]+1:] = inp_voltage[inp_indices1[imps], inp_spikes1[imps]+1:] + apreTC[0:-(inp_spikes1[imps]+1)]
            inp_voltage = inp_voltage - offset
        
            out_voltage = out_voltage*0
            for outs in range(np.size(r_all1)):
                out_voltage[0, r_all1[outs]:] = out_voltage[0, r_all1[outs]:] + apostTC[0:-r_all1[outs]]
            out_voltage = out_voltage - offset
        
            new_wt_tmp = inp_voltage[:,r_all1]
            new_wt = np.sum(new_wt_tmp,1)+new_wt
            
            for imps in range(np.size(inp_spikes1)): 
                new_wt[inp_indices1[imps]] = new_wt[inp_indices1[imps]] + out_voltage[0,inp_spikes1[imps]]
            weightchange[rep,target,:] = new_wt
    mean = np.mean(new_wt)
    return(mean,new_wt)   





#%% simrun
def simrun(offset,inp_indices,inp_spikes,r_all,unit,epoch,num_reps):
# This should be more efficient than previous versions in that it trains for a specific output unit
#8/1/23 
# 12/25/22    
#offset is the voltage offset for making the weights
#inp_indices are the indices (90 input units) for the input units which come packed
#inp_spikes are the spike times corresponding to the indices
#r_all are the spike occurrences for the actual spikes (65 units)
#unit is the individual actual unit for which the weights are being learned
#epoch is one of the three epochs (numbered 1-3)
#num_reps are the number of repetitions to be used in the weight calculation


   start_scope()
   print('start offset = %7.6f' % (offset))
   ndex = 0
   num_inputs = 90
   duration = 1000
   weightchange = np.zeros([num_reps, 16,90])

     
   W_previous = .002   #initial weights
   taup = 20*ms

   Apre = 0.01
   Apost= 0.01

   #rep = 0
   #print('Number of Reps = ', num_reps,'\n')
   for rep in range(num_reps):
       for target in range(16):
           start_scope()
           
           
           
           out_st_array = []
           out_ind_array = []
           spktime= np.float64(r_all[unit][target][rep])*1000           
           out_st_array= np.array(spktime).flatten()
           out_ind_array= np.zeros(len(out_st_array),int)
        
           out_group = SpikeGeneratorGroup(1,out_ind_array,out_st_array *ms)
           if epoch== 1 :
               #print('Using epoch 1')
               ip = SpikeGeneratorGroup(num_inputs,inp_indices[0][target][rep],inp_spikes[0][target][rep],sorted='true')               
           elif epoch== 2:
               #print('Using epoch 2')
               ip = SpikeGeneratorGroup(num_inputs,inp_indices[1][target][rep],inp_spikes[1][target][rep],sorted='true')  
           elif epoch ==3:
               #print('Using epoch 3')
               ip = SpikeGeneratorGroup(num_inputs,inp_indices[2][target][rep],inp_spikes[2][target][rep],sorted='true')
           else:
                print('Epoch not specified correctly')
                
           SSS = Synapses(ip, out_group,           
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
              ''')
     
     
           SSS.connect() 
           SSS.w = W_previous                
           SSS.w_offset = offset
    
           run(duration*ms)
           
           W_previous = SSS.w 
           weightchange[rep,target,:] = np.array(W_previous)
           
           ndex +=1
           #print('Target ',target)
       #print ('Rep = ',rep)
   new_wt = SSS.w 
   mean = np.mean(new_wt)
   print ('Returning Mean 1 %4.3f' % (mean))

   return(mean, new_wt)   