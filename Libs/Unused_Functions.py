# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:17:15 2024

@author: BAH150
"""
from os import makedirs
from os.path import exists

import matplotlib.pyplot as plt
from brian2 import SpikeGeneratorGroup,mA,second, mV, volt, Synapses, collect
from brian2 import NeuronGroup, run, ms, defaultclock, SpikeMonitor, StateMonitor,start_scope,TimedArray,PopulationRateMonitor,PoissonGroup,Hz,second, Network
import numpy as np

import scipy.io as sio
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter
from random import randint
from Libs.Helper_Functions import simple_regress, smooth, make_histos, RMSE, make_norm_histos, magnitude
from sklearn.gaussian_process import GaussianProcessRegressor


#%% auto_offset_new
def auto_offset_new(unit,epoch,num_reps,inp_indices,inp_spikes, r_all):    
    start_scope()
    
    #unit = 57
    print('Output unit =\n\n', unit)
  
    os = np.zeros(2)
    os[0] = .002
    os[1] = .0025
    #os[2] = .003
    
    new_mean = np.zeros(2)
    
    for mf in range(2):
        offset = os[mf]
        (mean_ret,new_wt)=simrun(offset,inp_indices,inp_spikes,r_all,unit,epoch,num_reps)
        new_mean[mf]= mean_ret
    (b0,slope)= simple_regress(os,new_mean)
    calc_offset = -b0/slope    
    return(calc_offset)

#%% auto_offset_new2
def auto_offset_new2(unit,epoch,num_reps,inp_indices,inp_spikes, r_all):  
    
    os = np.array([0.002, 0.0025])
    new_mean = np.zeros(2)
    
    for mf in range(2):
        offset = os[mf]
        (mean_ret,new_wt)=simrun2(offset,inp_indices,inp_spikes,r_all,unit,epoch,num_reps)
        new_mean[mf]= mean_ret
        
    (b0,slope)= simple_regress(os,new_mean)
    calc_offset = -b0/slope    
    return(calc_offset)
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
               ip = SpikeGeneratorGroup(num_inputs,inp_indices[0][target][rep],inp_spikes[0][target][rep],sorted='true')               
           elif epoch== 2:
               ip = SpikeGeneratorGroup(num_inputs,inp_indices[1][target][rep],inp_spikes[1][target][rep],sorted='true')  
           elif epoch ==3:
               ip = SpikeGeneratorGroup(num_inputs,inp_indices[2][target][rep],inp_spikes[2][target][rep],sorted='true')
           # elif epoch ==4:
           #     ip = SpikeGeneratorGroup(num_inputs,inp_indices[3][target][rep],inp_spikes[3][target][rep],sorted='true')
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
           ndex +=1
           #print('Target ',target)
       #print ('Rep = ',rep)
   new_wt = SSS.w 
   mean = np.mean(new_wt)
   print ('Returning Mean 1 %4.3f' % (mean))

   return(mean,new_wt)   

#%% simrun2
def simrun2(offset,inp_indices,inp_spikes,r_all,unit,epoch,num_reps): #50x faster, slightly differnt values.
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

            r_all1 = np.squeeze(r_all[unit,target,rep])*1000#output spikes for the unit.
            r_all1 = np.round(r_all1*10).astype(int)#convert from ms to index of timestep
            r_all1 = r_all1[r_all1<duration*10]#remove spikes outside of duration
        
            inp_voltage = inp_voltage*0
            for imps in range(np.size(inp_spikes1)):
                inp_voltage[inp_indices1[imps], inp_spikes1[imps]+1:] = inp_voltage[inp_indices1[imps], inp_spikes1[imps]+1:] + apreTC[0:apreTC.size-(inp_spikes1[imps]+1)]
            inp_voltage = inp_voltage - offset
        
            out_voltage = out_voltage*0
            for outs in range(np.size(r_all1)):
                out_voltage[0, r_all1[outs]:] = out_voltage[0, r_all1[outs]:] + apostTC[0:apostTC.size-r_all1[outs]]
            out_voltage = out_voltage - offset
        
            new_wt_tmp = inp_voltage[:,r_all1]
            new_wt = np.sum(new_wt_tmp,1)+new_wt
            
            for imps in range(np.size(inp_spikes1)): 
                new_wt[inp_indices1[imps]] = new_wt[inp_indices1[imps]] + out_voltage[0,inp_spikes1[imps]]
            weightchange[rep,target,:] = new_wt
    mean = np.mean(new_wt)
    return(mean,new_wt)   


#%% Scale for weights and threshold for output units. try new way. Mess with running multiple guesses per iteration
# if CreateSandT:
    #make parameter search space
#     w_steps_ = []
    
#     steps = np.linspace(0.03, 0.09, num=8)
#     num_steps = len(steps)
#     thresholds = np.linspace(0.1, 0.6, 5)
    
    
#     # steps = np.linspace(0.03, .5, num=20)
#     # thresholds = np.linspace(0, .5, num=6)#lower then 0 will fire without input, higher than 0.5 often leads to no output spikes.
#     # num_steps = len(steps)
#     for th in thresholds:
#      	for ws1 in range(num_steps):
#      	    for ws2 in range(num_steps):
#      	        for ws3 in range(num_steps):
#      	            temp = np.zeros(4)
#      	            temp[0] = steps[ws1]
#      	            temp[1] = steps[ws2]
#      	            temp[2] = steps[ws3]
#      	            temp[3] = th
#      	            w_steps_.append(temp)
#     w_steps = w_steps_ #[w_steps_[0], w_steps_[1794]]
#     w_steps_np = np.array(w_steps)
    
#     reps = [19,20]#make actual hist for comparison 
#     actual_spikes= r_allL[0]#first unit to init
#     [actual_hist,d,e] =  make_norm_histos(actual_spikes,events,reps,4)
#     actual_hist = np.zeros([num_units, actual_hist.shape[0], actual_hist.shape[1]])
#     for u in range(num_units):
#         actual_spikes= r_allL[u]
#         [actual_hist[u,:,:],d,e] = make_norm_histos(actual_spikes,events,reps,4)
        
#     iterations = 40
#     n_cores = 4
#     skipfrac = 0.95#what proportion of the iterations do we skip at the begining and do a random sample instead?
#     FstIter = round(iterations*skipfrac)
#     EffIter = iterations-FstIter
#     iterTimes = np.zeros(EffIter+1)
#     WSInd_hist = np.int_(np.zeros([num_units, iterations*n_cores]))
#     for unit in range(num_units):
#         randinds = np.random.choice(range(w_steps_np.shape[0]), FstIter*n_cores, replace=False)
#         WSInd_hist[unit,:FstIter*n_cores] = randinds
#     w_steps_ = np.copy(np.transpose(w_steps_np[WSInd_hist[:,0:FstIter*n_cores],:],[0,2,1]))#first set of weights to try
        
#     w_steps_hist = np.zeros([num_units,w_steps_np.shape[1], iterations*n_cores])
#     RMSEneur = np.zeros([num_units, iterations*n_cores])
#     rRMSEneur = np.zeros([num_units, 2])
#     Corrneur = np.zeros([num_units, iterations*n_cores])
#     T0 = wall_time()
#     tp2 = 0
#     tp3 = 0
#     for r in range(EffIter+1):
#         w_steps = list(np.transpose(w_steps_,[2,0,1]))
#         args = [num_units, weight_multi_3d, inp_spikes, inp_indices, reps, duration, events, actual_hist]
#         new_iterable = ([x, args] for x in w_steps)
#         if __name__ == '__main__':
#             with multiprocessing.Pool(n_cores) as p:
#                 results = p.map(par_w_step, new_iterable)
#         for i in range(len(results)):  
#             w_steps_hist[:,:,tp2] = np.copy(results[i][1])
#             RMSEneur[:,tp2] = results[i][0]
#             # Corrneur[:,tp2] = results[i][2]
#             tp2 = tp2+1
            
#         if ~tp3:
#             rRMSEneur[:,0] = np.min(RMSEneur[:,:tp2],axis=1)
#             rRMSEneur[:,1] = np.max(RMSEneur[:,:tp2],axis=1)
#         rt0 = tp3*(FstIter-1)*n_cores + r*n_cores
#         RMSEneurt = np.divide(RMSEneur[:,:(tp2)]-np.transpose(np.tile(rRMSEneur[:,0],[tp2,1])), np.tile(np.diff(rRMSEneur),[1,tp2]))
#         # yL = list(RMSEneurt - Corrneur[:,:(tp2)])
#         yL = list(RMSEneurt)
#         XL = list(np.transpose(w_steps_hist[:,:,:(tp2)], [0,2,1]))
#         WSInd_hist_L = list(WSInd_hist)
#         u_s = list(np.arange(num_units))
#         w_steps = list(np.transpose(w_steps_,[0,2,1]))
#         gpr_input = [list(x) for x in zip(yL, XL, WSInd_hist_L, u_s, w_steps)]
        
#         args = [rt0, w_steps_np, EffIter, r, n_cores]
#         new_iterable = ([x, args] for x in gpr_input)
#         if __name__ == '__main__':
#             with multiprocessing.Pool(n_cores) as p:
#                 results = p.map(par_new_w_step, new_iterable)
#         w_steps_ = np.zeros([num_units, w_steps_np.shape[1], n_cores])
#         for u in range(len(results)):
#             rs = results[u]
#             WSInd_hist[rs[0], :] = rs[1]
#             w_steps_[u,:,:] = rs[2]
#         tp3 = 1
#         iterTimes[r] = wall_time()-T0
#         print(['iteration', r])        
#         print('elapsed time: ')
#         print(iterTimes[r])
#         print('average time: ')
#         print(iterTimes[r]/(r+1))
    
#     RMSEneur_min = np.min(RMSEneur, axis=1)
#     RMSEneur_argmin = np.argmin(RMSEneur, axis=1)
    
#     BestValues = w_steps_hist[np.arange(w_steps_hist.shape[0]),:,RMSEneur_argmin]
#     BestValues = np.hstack([BestValues, np.expand_dims(RMSEneur_min,1)])

#     if SaveSandT:
#         with open(scalethreshFile_s, 'wb') as f:
#          	np.save(f, BestValues)
#         # with open('RMSEneur.py', 'wb') as f:
#         # 	np.save(f, RMSEneur)
#         # with open('rRMSEneur.py', 'wb') as f:
#         # 	np.save(f, rRMSEneur)
#         # with open('Corrneur.py', 'wb') as f:
#         # 	np.save(f, Corrneur)
#         # with open('w_steps_hist.py', 'wb') as f:
#         # 	np.save(f, w_steps_hist)
# else:
#     #Read Best Values from server batch-    Needed to optimize testing phase
#     with open(scalethreshFile, 'rb') as f:
#         BestValues = np.load(f)#0-2 are scale factors for first 3 weights, 3 is threshold for output neuron, 4 is minimum error achieved
    
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

	w1s = smooth(w1w,9)
	w2s = smooth(w2w,9)
	w3s = smooth(w3w,9)

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
       
       w1s = smooth(w1w,9)
       w2s = smooth(w2w,9)
       w3s = smooth(w3w,9)

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

def par_new_w_step(args):
    y =             args[0][0]
    X =             args[0][1]
    WSInd_hist =    args[0][2]
    u =             args[0][3]
    w_steps_ =      args[0][4]
    
    rt0 =           args[1][0]
    w_steps_np =    args[1][1]
    EffIter =       args[1][2]
    r =             args[1][3]
    n_cores =       args[1][4]
    
    gpr = GaussianProcessRegressor(kernel=None,random_state=0).fit(X, y)
    for tp in range(w_steps_.shape[0]):
        rt = rt0 + tp
        WSInd_hist[rt] = np.where(~(w_steps_np - w_steps_[tp,:]).any(axis=1))[0]
        
    w_steps_np_copy = np.delete(w_steps_np, WSInd_hist[:(1+rt)], axis = 0)
    exptErrStd = gpr.predict(w_steps_np_copy, return_std=True)
    idx = np.argpartition(exptErrStd[0] - 10*(EffIter-r-2)*exptErrStd[1], n_cores)
    w_steps_ = np.transpose(w_steps_np_copy[idx[:n_cores],:])
    return u, WSInd_hist, w_steps_

#%% make_individual_input_spikes   
def make_individual_input_spikes(num_targets,rep_in,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks):
            #make_individual_input_spikes(16,rep_cnt,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks)

    ## This is the master program for making input spikes-  calls the other input generation routines

    dir_range= np.linspace(0,2*np.pi-(2*np.pi)/num_targets,num_targets)
    num_neurons = 90
    speed_gain = 2  #4.5
    speed_lag = 50 #71
    inp_groups = 4
    
    out_inp_indices = np.ndarray([inp_groups, num_targets, np.max(rep_in)], dtype='object')
    out_inp_spikes = np.ndarray([inp_groups, num_targets, np.max(rep_in)], dtype='object')

    mev = np.mean(events,1)
    for target in range(num_targets):
        direction = dir_range[target]

        mtarget_show = mev[target][2]
        mstart_movement = mev[target][6]
        mpk_speed_time= mev[target][11]
        mreward = mev[target][5]
        mend_movement = mev[target][9]
        menter_target = mev[target][8]
        int1 = (mstart_movement-gauss_center[0])*1000
        int2 = (mpk_speed_time-gauss_center[1])*1000
        int3 = (mend_movement-gauss_center[2])*1000
        try:
            len(rep_in)
            num_reps = rep_in[target]
        except:
            num_reps= rep_in
        print('Reps for target ',target,' =' ,num_reps )    
        for rep in range(num_reps): 
            ev = np.round(events[target][rep] * 1000)
            sp_times= speed_times[target][rep].flatten() *1000 
           
            speed= speed_all[target][rep].flatten()
            speed*= speed_gain
            speed = speed[np.logical_and(sp_times>=ev[6], sp_times<=ev[9])]
            sp_times_slice = sp_times[np.logical_and(sp_times>=ev[6], sp_times<=ev[9])]
              
            center1 = ev[6]-int1 #start_movement
            center2 = ev[11] -int2 #pk_speed_time
            center3 = ev[9]-int3  #end_movement

            width1 = gauss_sigma[0]*300    # original *1000
            width2 = gauss_sigma[1]*500
            width3 = gauss_sigma[2]*300

            out_inp_indices[0,target,rep],out_inp_spikes[0,target,rep] = make_input_spikes(duration,direction,num_neurons,center1*10,width1*10,speed,sp_times_slice,speed_lag) #.35 worked well but background was too high for training
            out_inp_indices[1,target,rep],out_inp_spikes[1,target,rep] = make_input_spikes(duration,direction,num_neurons,center2*10,width2*10,speed,sp_times_slice,speed_lag) #gain = 3
            out_inp_indices[2,target,rep],out_inp_spikes[2,target,rep] = make_input_spikes(duration,direction,num_neurons,center3*10,width3*10,speed,sp_times_slice,speed_lag)
            out_inp_indices[3,target,rep],out_inp_spikes[3,target,rep] = make_input_spikes_speed(duration,num_neurons,speed,sp_times_slice,speed_lag)

        
        print('Finished target ',target)
        print ('Number of spikes ',len(out_inp_indices[0,target,rep]))

    out_indices = np.ndarray.tolist(out_inp_indices)
    out_spikes = np.ndarray.tolist(out_inp_spikes)
    return(out_indices, out_spikes)


#%% make_offset_weights
# def make_offset_weights(r_all, final_offsets, inp_spikes, inp_indices, reps, duration):
#     num_units = len(r_all)
    
#     num_neurons = 90  #number of input neurons
#     num_inputs = num_neurons
#     taup = 20*ms
    
#     Apre = 0.01
#     Apost= 0.01
    
#     w_offsets = np.zeros([len(inp_spikes), final_offsets.shape[0]*num_neurons])
#     for epoch in range(final_offsets.shape[1]):
#         w_offsets[epoch,:] = np.squeeze(np.tile(final_offsets[:,epoch], [num_neurons,1]))
        
#     W_previous = [.002, .002, .002, .002]#initial weights
    
#     start_scope() 
    
#     for rep in range(reps):   #Train on the first 20 reps
#         for target in range(r_all.shape[1]):
#             out_st_array = []
#             out_ind_array = []
#             for u in range(num_units):  #build arrays of actual units as output
#                 rr = r_all[u][target][rep]
#                 st = []
#                 shaperr = rr.shape  #Check for empty array
#                 if(shaperr[0] == 1):
#     	            rr = np.transpose(rr)
#                 for jj in range(len(rr)):
#                     if rr[jj]>=0:
#                         spktime = np.float64(rr[jj] * 1000)
#                         st.append(spktime)
#                 out_st_array= np.concatenate((out_st_array,st),axis=0)  #build packed indices and spikes for actual spikes
#                 indices= np.ones(len(st),int) * u
#                 out_ind_array= np.concatenate((out_ind_array,indices),axis= 0)   
     
#             # Make spike generators for the four input groups (3 gassians and one speed input)
#             # inp_indices and inp_spikes come from "make_individual_input_spikes"
#             out_group = SpikeGeneratorGroup(num_units,out_ind_array,out_st_array *ms)
#             inp_group = []
#             Syns = []
            
#             for i in range(len(inp_indices)):
#                 inp_group.append(SpikeGeneratorGroup(num_inputs,inp_indices[i][target][rep],inp_spikes[i][target][rep]))
#                 Syns.append(Synapses(inp_group[i], out_group,
#                     '''
#                     w : 1
#                     w_offset : 1
                    
#                     dapre/dt = -apre/taup : 1 (event-driven)
#                     dapost/dt = -apost/taup : 1 (event-driven)
#                     ''',
#                     on_pre='''
#                     apre += Apre
#                     w = w+apost-w_offset
#                     ''',
#                     on_post='''
#                     apost += Apost
#                     w = w+apre-w_offset
#                     '''))
            
#                 Syns[i].connect()
#                 Syns[i].w = W_previous[i]
#                 Syns[i].w_offset = w_offsets[i,:]
#             S1 = Syns[0]
#             S2 = Syns[1]
#             S3 = Syns[2]
#             S4 = Syns[3]
#             inp_group1 = inp_group[0]
#             inp_group2 = inp_group[1]
#             inp_group3 = inp_group[2]
#             inp_group4 = inp_group[3]
#             run(duration*ms)
#             Syns[0] = S1
#             Syns[1] = S2
#             Syns[2] = S3
#             Syns[3] = S4
#             for i in range(len(inp_indices)):
#                 W_previous[i] = Syns[i].w
        
#         print('Rep ',rep)    
        
#     weight_multi = np.zeros([len(inp_spikes),num_inputs*r_all.shape[0]])
#     for i in range(len(inp_spikes)):
#         weight_multi[i,:]= Syns[i].w

#     print('Finished')    
#     return(weight_multi)