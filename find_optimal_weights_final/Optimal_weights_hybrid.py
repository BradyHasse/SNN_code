# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:09:01 2023

@author: Andrew
"""


import numpy as np
from os.path import exists
from os import makedirs
from brian2 import TimedArray, Synapses, defaultclock,SpikeMonitor,\
                    StateMonitor,NeuronGroup,amp,ms,run,mA,start_scope,\
                        SpikeGeneratorGroup,Hz,PoissonGroup,second,Network,\
                            volt,mV
#import matplotlib.pyplot as plt                       
#import matlab as ml
import scipy.io as sio
from scipy.io import loadmat, savemat 
from scipy.signal import savgol_filter
from random import randint, random, randrange     

#%% bin_frac2
def bin_frac2(spikes,a_time,a_b_time, bin_width): 
    '''    
    % Calculates the spike rate using fractional intervals
    
    % rates= bin_frac2(spikes,a_time,a_b_time, bin_width)
    % takes in time of spikes as a vector (typically in (s)) - spikes, a_time, a_b_time, and bin_width are all the same units of time
    % a_time is the starting time of the analysis period
    % a_b_time is the ending time of the analysis period
    % bin_width is the width of each bin in the analysis period
    % rates is the spiking freqency in each of the bins
    
    % example 
    % spikes = [0.0228,0.0461,0.0689,0.1145,0.1375,0.1606,0.2046,0.2294,0.2523,0.2752,0.2982,0.3212,0.344,0.3669,0.3898,0.4128,0.4358,0.4779,0.5045,0.5356,0.5691,0.5962,0.6289,0.6601,0.6879,0.7283,0.7567,0.7917,0.8185,0.848,0.8846,0.9111,0.9339,0.9627,1.0007,1.0249,1.0474,1.0783,1.1165,1.1456,1.1697,1.1932,1.2261,1.2566,1.2725,1.2838,1.3019,1.3158,1.3292,1.343,1.3641,1.3861,1.4805,1.5038,1.5246,1.5435,1.5663,1.5967,1.6253,1.6486,1.6715,1.6947,1.7181,1.7645,1.788,1.811,1.8342,1.8576,1.9041,1.9508,1.9973];
    % a_time = 1.141;
    % a_b_time = 1.328;
    % bin_width = (1.328-1.141)/11;%will give 11 bins
    % rates= bin_frac2(spikes,a_time,a_b_time, bin_width);
    
    % bin_frac by Tony Reina
    % updated to bin_frac2 by Brady Hasse Jan 16, 2023, then converted from matlab to python.
    
    '''
    #import numpy as np
    
    num_bins= int(np.floor(((a_b_time-a_time)/bin_width)+.0001))#sets the number of bins
    bin_width = (a_b_time-a_time)/num_bins#if it was not exact, makes bin_width correct.
    spikes=np.unique(np.sort(spikes.flatten()))#sort the spikes in order and remove any duplicates.
    if spikes.size == 0:
        rates = np.zeros(num_bins) #if no spikes, the rate is 0
    else:
        spikes = np.hstack((np.min(spikes) - 500, spikes, np.max(spikes) + 500)) #add spikes far in the past and future.
        
        spikes = spikes - a_time
        a_b_time = a_b_time - a_time
        a_time=0
        
        spikes = spikes*(num_bins/a_b_time)
        a_b_time = num_bins
        
        ind_s = np.nonzero(spikes<a_time)
        ind_s = ind_s[0][-1]#first spike index
        ind_e = np.nonzero(spikes>a_b_time)
        ind_e = ind_e[0][0]+1#final spike index
        
        spikes = spikes[ind_s:ind_e]#from spike preceeding a_time to spike after a_b_time
        dspikes = np.diff(spikes)#intervals between spikes
        intervals = np.zeros(num_bins)
        for i in  range(dspikes.size):#assign each interval between spikes to bins.
            j = np.arange(np.floor(spikes[i])+1, np.floor(spikes[i+1])+0.1)#added 0.1 to get last number included
            divs = np.sort(np.hstack((j, spikes[np.array([i,i+1])])))#time of two spikes creating dspikes and the whole numbers  between them
            int_share = np.diff(divs)*(1/dspikes[i])#each full unit is worth (1/dspikes(i))
            int_inds = np.ceil(divs[1:])-1#indcies of bin in interval where int_shares get assigned to
            
            rm_inds = (int_inds<0) + (int_inds>(a_b_time-1)) + (int_share==0)#indicies to remove because they belong to unassined bins
            int_share = int_share[~rm_inds]#remove those shares
            int_inds = int_inds[~rm_inds]#remove those shares
            intervals[(np.rint(int_inds)).astype(int)] = intervals[(np.rint(int_inds)).astype(int)] + int_share#assign portion of spike to the correct bin(s)

        rates = intervals/bin_width
    return rates
 
#%% make_histos
def make_histos(in_array,ev,mode):    
#simple one-trial histogram aligned to movement onset 
# events need to be specified events[target][rep]  
     start_scope()
     

     num_targets = 1
     num_reps = 1
 
     #total_bins = int(np.ceil(nbins1+nbins2+nbins3))
     #max_num_reps = np.max(reps)
     stimes = []
         
     if mode == 'predicted':
                 
                 rr= in_array
                 stimes = np.array(rr,dtype=float)
                    
     else:                       #Actual
                 rr = (np.squeeze(in_array))
                 stimes= rr

     target_show = ev[2]
     start_movement = ev[6]
     pk_speed= ev[11]
     reward = ev[5]
     end_movement = ev[9]
     delta1 = end_movement-target_show                
     binwidth1 = float(.02)
     rates= []
     rates = bin_frac2(stimes,target_show,reward,binwidth1)
     #rates= eng.bin_frac(ml.double(stimes),ml.double(target_show),ml.double(reward),binwidth1) 
     #sm_rates =np.array(eng.smooth(ml.double(rates),ml.double(10),'sgolay'))
     sm_rates = smooth(rates,10)
     return(sm_rates)

#%% ABSERROR
def ABSERROR(v1, v2):
    err = v1 - v2
    return( np.mean(abs(v1-v2)) )

#%% RMSE
def RMSE(v1,v2):
    err = v1-v2
    rmse = np.mean(err**2) **.5
    return(rmse)

#%% make_weights
def make_weights(weights1_multi,weights2_multi,weights3_multi,weights4_multi):
    weights= []
    weights.append(weights1_multi)
    weights.append(weights2_multi)
    weights.append(weights3_multi)
    weights.append(weights4_multi)
    return(weights)

#%% get_rmse_v2
def get_rmse_v2(args): #w_step,unit,threshold,weights,inp_indices,inp_spikes,actual,events,duration, threshold):
	start_scope()

	#unit = 0

	num_inputs = 90

	w_step = args[0]
#	threshold_ = args[1]
	unit = args[1][0]
	weights = args[1][1]
	inp_indices = args[1][2]
	inp_spikes = args[1][3]
	actual = args[1][4]
	events = args[1][5]
	duration = args[1][6]
	outputDir = args[1][7]
	
	print(f'WStep {w_step}')
	
    
	threshold_ = w_step[3] #args[1][7]
	
#	threshold_ = 0.3

	#if slopesW4MonkN[unit] < 0.12:
	#	threshold_ = 0.02

	num_units = len(actual)
	num_directions = 16
	dir_range = np.linspace(0, 2*np.pi, num_directions)

	print(f'Unit# {unit}: Step: {w_step[0]}_{w_step[1]}_{w_step[2]}, Th: {threshold_}')
	#print(f'OutputDir: {outputDir}')
	if not exists(outputDir):
		print('Dir does not exist. Creating one.')
		makedirs(outputDir)
	if(exists(outputDir + f'/RMSE_Unit_{unit}_wStep0_{w_step[0]}_wStep1_{w_step[1]}_wStep2_{w_step[2]}_th_{threshold_}.npz')):
		print("Already ran before. Returning")
		return
		

	w1m = weights[0]
	w2m = weights[1]
	w3m = weights[2]
	w4m = weights[3]

	#find max weights
	# w_max = np.zeros(num_units)
	# for un in range (num_units):
	w1w = w1m[range(unit,len(w1m),num_units)]  #unwrap vector
	w2w = w2m[range(unit,len(w2m),num_units)]  #unwrap vector
	w3w = w3m[range(unit,len(w3m),num_units)]  #unwrap vector
	w4w = w4m[range(unit,len(w4m),num_units)]  #unwrap vector

	w1s = smooth(w1w,10)
	w2s = smooth(w2w,10)
	w3s = smooth(w3w,10)

	w1c = w1w-np.mean(w1s)
	w2c = w2w-np.mean(w2s)
	w3c = w3w-np.mean(w3s)

	H_eqs='''
	dv/dt = -v/tau :1
	tau:second
	v_th : 1
	'''

#	v_th = threshold_

	rmse_accum = []

	ndex = 0

	for direction in dir_range:
		inp1_group = []
		inp2_group = []
		inp3_group = []
		inp4_group = []

		for rep in range(19, 20):
			start_scope()
			inp1_group = SpikeGeneratorGroup(num_inputs,inp_indices[0][ndex][rep],inp_spikes[0][ndex][rep])
			inp2_group = SpikeGeneratorGroup(num_inputs,inp_indices[1][ndex][rep],inp_spikes[1][ndex][rep])
			inp3_group = SpikeGeneratorGroup(num_inputs,inp_indices[2][ndex][rep],inp_spikes[2][ndex][rep])
			inp4_group = SpikeGeneratorGroup(num_inputs,inp_indices[3][ndex][rep],inp_spikes[3][ndex][rep])

			#print(threshold_)
			H= NeuronGroup(1,H_eqs,threshold = 'v >  v_th  ', reset = 'v = -10',  method = 'exact')
			H.tau= 10*ms
			#print(threshold_)
			H.v_th = threshold_

			ws0 = w_step[0]
			ws1 = w_step[1]
			ws2 = w_step[2]

			SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += ws0*w')  
			SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += ws1*w')
			SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += ws2*w')
			SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += .001*w')

			SS1.connect(i=np.arange(num_inputs),j=0)
			SS2.connect(i=np.arange(num_inputs),j=0)
			SS3.connect(i=np.arange(num_inputs),j=0)
			SS4.connect(i=np.arange(num_inputs),j=0)

			SM=SpikeMonitor(H)
			#M = StateMonitor(H, variables=True, record=True)
			#PMR= PopulationRateMonitor(H)

			SS1.w= w1c
			SS2.w= w2c
			SS3.w= w3c
			SS4.w= w4w

			run(duration*ms)

			 # Calculate RMSE
			rmse = 100
			if len(actual[unit][ndex][rep]) != 0:
				ar = make_histos(actual[unit][ndex][rep],events[ndex][rep],'actual')
				if SM.t != []:
					pr = make_histos(SM.t,events[ndex][rep],'predicted')
				else:
					pr = np.ones(len(ar))*100
				rmse = RMSE(ar[5:len(ar)-5],pr[5:len(ar)-5])
			#print ('RMSE=', rmse)
			rmse_accum.append(rmse)

		ndex += 1

		tempError = np.mean(rmse_accum)

		#print(f'Unit {unit}::: direction {ndex}:: threshold {threshold_}:step{w_step} = error {tempError}')
	print(f'RMSE_Unit: {unit}, w0: {w_step[0]}, w1: {w_step[1]}, w2 {w_step[2]}, th: {threshold_}, Mean error: {np.mean(rmse_accum)}')
	np.savez(outputDir + f'/RMSE_Unit_{unit}_wStep0_{w_step[0]}_wStep1_{w_step[1]}_wStep2_{w_step[2]}_th_{threshold_}.npz', rmse_accum, np.mean(rmse_accum))

#	np.savez(f'RMSE_TH_MonkN_Aug7_Results_15_20/RMSE_Unit_{unit}_wStep_{w_step}_duration_{duration}_threshold_{threshold_}.npz', rmse_accum, np.mean(rmse_accum).tolist())

	return
#	return(np.mean(rmse_accum))

#%% get_rmse
def get_rmse(args): #unit,weights,inp_indices,inp_spikes,actual,events,duration,w_step):


       start_scope()

       unit = args[1]
       w_step = args[0]
       weights = args[2][0]
       inp_indices = args[2][1]
       inp_spikes = args[2][2]
       actual = args[2][3]
       events = args[2][4]
       duration = args[2][5]
#       w_step = args[2][6]

       print(unit)

       #unit = 0  

       if(exists(f'RMSE_Results_wsteps_epoch/RMSE_Unit_{unit}_wstep0_{w_step[0]}_wstep1_wstep2_{w_step[2]}_th_threshold_.npz')):
              print("Already ran before. Returning")
              return


       num_inputs = 90
       num_units = len(actual)
       num_directions=16
       dir_range= np.linspace(0,2*np.pi,num_directions)

        
       w1m = weights[0]     
       w2m = weights[1]
       w3m = weights[2]
       w4m = weights[3]

       #find max weights
       # w_max = np.zeros(num_units)
       # for un in range (num_units):
       w1w = w1m[range(unit,len(w1m),num_units)]  #unwrap vector
       w2w = w2m[range(unit,len(w2m),num_units)]  #unwrap vector
       w3w = w3m[range(unit,len(w3m),num_units)]  #unwrap vector
       w4w = w4m[range(unit,len(w4m),num_units)]  #unwrap vector
       
       w1s = smooth(w1w,10)
       w2s = smooth(w2w,10)
       w3s = smooth(w3w,10)

       w1c = w1w-np.mean(w1s)
       w2c = w2w- np.mean(w2s)
       w3c = w3w-np.mean(w3s)

       H_eqs= '''
       dv/dt = -v/tau :1
       tau:second
       '''



       rmse_accum= []



       ndex = 0
       for direction in dir_range:

           inp1_group=[]
           inp2_group=[]
           inp3_group=[]
           inp4_group=[]

           
           
        
           for rep in range(0,20):
               start_scope()
               inp1_group = SpikeGeneratorGroup(num_inputs,inp_indices[0][ndex][rep],inp_spikes[0][ndex][rep])
               inp2_group = SpikeGeneratorGroup(num_inputs,inp_indices[1][ndex][rep],inp_spikes[1][ndex][rep])
               inp3_group = SpikeGeneratorGroup(num_inputs,inp_indices[2][ndex][rep],inp_spikes[2][ndex][rep])
               inp4_group = SpikeGeneratorGroup(num_inputs,inp_indices[3][ndex][rep],inp_spikes[3][ndex][rep])

           
               H= NeuronGroup(1,H_eqs,threshold = 'v > .4   ', reset = 'v = -10',  method = 'exact')
        
               H.tau= 10*ms

               
               

               SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += w_step*w')  
               SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += w_step*w') 
               SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += w_step*w') 
               SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += .001*w') 

               
              
               
               SS1.connect(i=np.arange(num_inputs),j=0)
               SS2.connect(i=np.arange(num_inputs),j=0)
               SS3.connect(i=np.arange(num_inputs),j=0)
               SS4.connect(i=np.arange(num_inputs),j=0)
               
         
               
               SM=SpikeMonitor(H)
               #M = StateMonitor(H, variables=True, record=True)
               #PMR= PopulationRateMonitor(H)
              
              

               SS1.w= w1c
               SS2.w= w2c
               SS3.w= w3c
               SS4.w= w4w
               
               
               run(duration*ms)
        
               # Calculate RMSE
               ar = make_histos(actual[unit][ndex][rep],events[ndex][rep],'actual')
               pr = make_histos(SM.t,events[ndex][rep],'predicted')
               rmse = RMSE(ar[5:len(ar)-5],pr[5:len(ar)-5])
               #print ('RMSE=', rmse)
               rmse_accum.append(rmse)
               
         
           ndex +=1

           #temp_w_step = w_step*10000


#           print(f'Finished Direction{ndex} for unit {unit} and wstep {w_step}')
#           print ('Mean rmse=', np.mean(rmse_accum))

#       temp_w_step = w_step*10000

       np.savez(f'RMSE_Results_wsteps_epoch/RMSE_Unit_{unit}_wStep_{w_step}_duration_{duration}.npz', rmse_accum, np.mean(rmse_accum).tolist())
        
       return(np.mean(rmse_accum))

#%% smooth       
def smooth(in_array,span):
    try:
        s_array = savgol_filter(in_array,span,2)
    except:
        s_array = in_array
    return(s_array)

#%%  prototype of optimal weight step

#w_steps = [.0025] #, .005, .015, .02, .025, .03, .035]
#thresholds = [0.0]

#rmse_accum= []

#unit = 0

#for ws in range(len(w_steps)):
#	for th in range(len(thresholds)):
#	    mrmse= get_rmse_v2(unit,weights,inp_indices,inp_spikes,r_all,events,1000,w_steps[ws], thresholds[th])
#	    print ('RMSE for',w_steps[ws], '=', mrmse)
#	    rmse_accum.append(mrmse)

#best_weight_step= w_steps[rmse_accum.index(np.min(rmse_accum))]
