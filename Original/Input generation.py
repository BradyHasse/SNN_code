# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:17:26 2023
Routines for generating SNN input consisting of three gaussian rate profiles and one based on speed
Typical usage:
[inp_indices,inp_spikes]=make_individual_input_spikes(16,rep_cnt,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks)

@author: Andrew
"""
#%%   Initialize the enviornment-  Make sure you are in the correct directory
import matplotlib.pyplot as plt
from brian2 import SpikeGeneratorGroup,mA,second
from brian2 import NeuronGroup, run, ms, defaultclock, SpikeMonitor, StateMonitor,start_scope,TimedArray,PopulationRateMonitor,PoissonGroup,Hz,second, Network

import numpy as np
import matplotlib.pyplot as plt                       
import matlab as ml
import matlab.engine as mat
eng= mat.start_matlab()
mw = eng.workspace
eng.addpath('C:/Users/Andrew/Documents/MATLAB/work/Programs')
from Libs import *
import scipy.io as sio
from scipy.io import loadmat, savemat
from scipy.signal import savgol_filter


   
def make_individual_input_spikes(num_targets,rep_in,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks):
## This is the master program for making input spikes-  calls the other input generation routines

    dir_range= np.linspace(0,2*np.pi,num_targets)
    num_neurons= 90
    speed_gain = 2  #4.5
    speed_lag = 50 #71
    
    out1_inp_indices= []
    out2_inp_indices= []
    out3_inp_indices= []
    out4_inp_indices= []
    out1_inp_spikes= []
    out2_inp_spikes= []
    out3_inp_spikes= []  
    out4_inp_spikes= []  

       
        
    mev = np.mean(events,1)
    for target in range(num_targets):
        direction = dir_range[target]
        tmp1_inp_indices = []
        tmp2_inp_indices = []
        tmp3_inp_indices = []
        tmp4_inp_indices = []
        tmp1_inp_spikes = []
        tmp2_inp_spikes = []
        tmp3_inp_spikes = []
        tmp4_inp_spikes = []
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
            ev = events[target][rep] * 1000
            sp_times= speed_times[target][rep].flatten() *1000 
            #speed_width = (sp_times[-1]-sp_times[0])/10    #move_end - move_start
           
            speed= speed_all[target][rep].flatten()
            speed*= speed_gain
            a= np.where(sp_times>=ev[6])[0][0]
            b= np.where(sp_times>=ev[9])[0][0]
            sp_times_slice = sp_times[a:b]   #only include the speed profile
            speed = speed[a:b]
            target_show = ev[2]
            start_movement = ev[6]
            pk_speed_time= ev[11]
            reward = ev[5]
            end_movement = ev[9]
            enter_target = ev[8]
            speed_width = (sp_times_slice[-1]-sp_times_slice[0])
            #center1 = target_show + 200
            center1 = start_movement-int1
            center2 = pk_speed_time -int2
            #center3_tmp = pk_speed_time + 120 
            center3 = end_movement-int3  #was 80
            #print('Events',ev)
            #print('Centers', center3,center3_tmp)
            # width1= (start_movement-target_show)*.6 #was .8
            # width2 = speed_width *.8
            # width3= (enter_target-pk_speed_time)*1.5
            # width1 = gauss_sigma[0]*1000    # original *1000
            # width2 = gauss_sigma[1]*1000
            # width3 = gauss_sigma[2]*1000
            width1 = gauss_sigma[0]*300    # original *1000
            width2 = gauss_sigma[1]*500
            width3 = gauss_sigma[2]*300
            
            
            # if width3 < 0:
            #     print('Problem in make_ind with width3, enter target =  ', enter_target, 'end of speed= ', end_movement)
            #     print('Rep= ',rep, 'Target= ', target)
            #     width3= (end_movement-pk_speed_time)*3.5
            #print('Widths',width1, width2,width3)
            
            
            #if center3 > 1800:
            #    print ('Problem with third epoch center3=',center3, ' Target=', target,' Rep=',rep)
           
            inps_ind1,inps_spikes1 = make_input_spikes(duration,direction,num_neurons,center1*10,width1*10,speed,sp_times_slice,speed_lag) #.35 worked well but background was too high for training
            inps_ind2,inps_spikes2 = make_input_spikes(duration,direction,num_neurons,center2*10,width2*10,speed,sp_times_slice,speed_lag) #gain = 3
            inps_ind3,inps_spikes3 = make_input_spikes(duration,direction,num_neurons,center3*10,width3*10,speed,sp_times_slice,speed_lag)
            inps_ind4,inps_spikes4 =make_input_spikes_speed(duration,num_neurons,speed,sp_times_slice,speed_lag)
            tmp1_inp_indices.append(inps_ind1)
            tmp2_inp_indices.append(inps_ind2)
            tmp3_inp_indices.append(inps_ind3)
            tmp4_inp_indices.append(inps_ind4)
            tmp1_inp_spikes.append(inps_spikes1)
            tmp2_inp_spikes.append(inps_spikes2)
            tmp3_inp_spikes.append(inps_spikes3)
            tmp4_inp_spikes.append(inps_spikes4)
            
        out1_inp_indices.append(tmp1_inp_indices)
        out2_inp_indices.append(tmp2_inp_indices)
        out3_inp_indices.append(tmp3_inp_indices)
        out4_inp_indices.append(tmp4_inp_indices)
        out1_inp_spikes.append(tmp1_inp_spikes)
        out2_inp_spikes.append(tmp2_inp_spikes)
        out3_inp_spikes.append(tmp3_inp_spikes)
        out4_inp_spikes.append(tmp4_inp_spikes)
        
        print('Finished target ',target)
        print ('Number of spikes ',len(out1_inp_indices[target][0]))
    out_indices= []
    out_spikes= []
    out_indices.append(out1_inp_indices)
    out_indices.append(out2_inp_indices)
    out_indices.append(out3_inp_indices)
    out_indices.append(out4_inp_indices)
    out_spikes.append(out1_inp_spikes)
    out_spikes.append(out2_inp_spikes)
    out_spikes.append(out3_inp_spikes)
    out_spikes.append(out4_inp_spikes)
    
    return(out_indices, out_spikes)

def make_individual_input_spikes(num_targets,rep_in,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks):
    dir_range= np.linspace(0,2*np.pi,num_targets)
    num_neurons= 90
    speed_gain = 2  #4.5
    speed_lag = 50 #71
    
    out1_inp_indices= []
    out2_inp_indices= []
    out3_inp_indices= []
    out4_inp_indices= []
    out1_inp_spikes= []
    out2_inp_spikes= []
    out3_inp_spikes= []  
    out4_inp_spikes= []  

       
        
    mev = np.mean(events,1)
    for target in range(num_targets):
        direction = dir_range[target]
        tmp1_inp_indices = []
        tmp2_inp_indices = []
        tmp3_inp_indices = []
        tmp4_inp_indices = []
        tmp1_inp_spikes = []
        tmp2_inp_spikes = []
        tmp3_inp_spikes = []
        tmp4_inp_spikes = []
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
            ev = events[target][rep] * 1000
            sp_times= speed_times[target][rep].flatten() *1000 
            #speed_width = (sp_times[-1]-sp_times[0])/10    #move_end - move_start
           
            speed= speed_all[target][rep].flatten()
            speed*= speed_gain
            a= np.where(sp_times>=ev[6])[0][0]
            b= np.where(sp_times>=ev[9])[0][0]
            sp_times_slice = sp_times[a:b]   #only include the speed profile
            speed = speed[a:b]
            target_show = ev[2]
            start_movement = ev[6]
            pk_speed_time= ev[11]
            reward = ev[5]
            end_movement = ev[9]
            enter_target = ev[8]
            speed_width = (sp_times_slice[-1]-sp_times_slice[0])
            #center1 = target_show + 200
            center1 = start_movement-int1
            center2 = pk_speed_time -int2
            #center3_tmp = pk_speed_time + 120 
            center3 = end_movement-int3  #was 80
            #print('Events',ev)
            #print('Centers', center3,center3_tmp)
            # width1= (start_movement-target_show)*.6 #was .8
            # width2 = speed_width *.8
            # width3= (enter_target-pk_speed_time)*1.5
            # width1 = gauss_sigma[0]*1000    # original *1000
            # width2 = gauss_sigma[1]*1000
            # width3 = gauss_sigma[2]*1000
            width1 = gauss_sigma[0]*300    # original *1000
            width2 = gauss_sigma[1]*500
            width3 = gauss_sigma[2]*300
            
            
            # if width3 < 0:
            #     print('Problem in make_ind with width3, enter target =  ', enter_target, 'end of speed= ', end_movement)
            #     print('Rep= ',rep, 'Target= ', target)
            #     width3= (end_movement-pk_speed_time)*3.5
            #print('Widths',width1, width2,width3)
            
            
            #if center3 > 1800:
            #    print ('Problem with third epoch center3=',center3, ' Target=', target,' Rep=',rep)
           
            inps_ind1,inps_spikes1 = make_input_spikes(duration,direction,num_neurons,center1*10,width1*10,speed,sp_times_slice,speed_lag) #.35 worked well but background was too high for training
            inps_ind2,inps_spikes2 = make_input_spikes(duration,direction,num_neurons,center2*10,width2*10,speed,sp_times_slice,speed_lag) #gain = 3
            inps_ind3,inps_spikes3 = make_input_spikes(duration,direction,num_neurons,center3*10,width3*10,speed,sp_times_slice,speed_lag)
            inps_ind4,inps_spikes4 =make_input_spikes_speed(duration,num_neurons,speed,sp_times_slice,speed_lag)
            tmp1_inp_indices.append(inps_ind1)
            tmp2_inp_indices.append(inps_ind2)
            tmp3_inp_indices.append(inps_ind3)
            tmp4_inp_indices.append(inps_ind4)
            tmp1_inp_spikes.append(inps_spikes1)
            tmp2_inp_spikes.append(inps_spikes2)
            tmp3_inp_spikes.append(inps_spikes3)
            tmp4_inp_spikes.append(inps_spikes4)
            
        out1_inp_indices.append(tmp1_inp_indices)
        out2_inp_indices.append(tmp2_inp_indices)
        out3_inp_indices.append(tmp3_inp_indices)
        out4_inp_indices.append(tmp4_inp_indices)
        out1_inp_spikes.append(tmp1_inp_spikes)
        out2_inp_spikes.append(tmp2_inp_spikes)
        out3_inp_spikes.append(tmp3_inp_spikes)
        out4_inp_spikes.append(tmp4_inp_spikes)
        
        print('Finished target ',target)
        print ('Number of spikes ',len(out1_inp_indices[target][0]))
    out_indices= []
    out_spikes= []
    out_indices.append(out1_inp_indices)
    out_indices.append(out2_inp_indices)
    out_indices.append(out3_inp_indices)
    out_indices.append(out4_inp_indices)
    out_spikes.append(out1_inp_spikes)
    out_spikes.append(out2_inp_spikes)
    out_spikes.append(out3_inp_spikes)
    out_spikes.append(out4_inp_spikes)
    
    return(out_indices, out_spikes)

def make_input_spikes(duration,direction,num_neurons,center,width,speed,speed_points,speed_lag):
#Uses Hongwei's equations for adding noise
# Make a set of num_neurons, each with a different prefered direction, with firing rates dictated by gaussian input 
    #print('direction',direction,'gaincenter',center,'width',width, 'speed_center',speed_center)
    start_scope()
 
    
    vr = -70*mV 	# resting potential level
    ##vt = -55*mV		# threshold for firing of action potential #-55
    #vt = -60*mV
    vt= .35 * volt
    
    
    n_sigma = 0.09*(vt-vr)	# 0.05; sigma of noise added to membrane potential
    
    
    
    inputs =  gaussian_input_speed(duration,direction,num_neurons,center,width,speed,speed_points,speed_lag)
    #inputs =  gaussian_input_speed(duration,direction,num_neurons,center,width,speed_lag)
    trans_inputs = np.transpose(inputs)
    # v_drive = TimedArray(trans_inputs*1.5*(vt-vr) + .86*(vt-vr), dt=defaultclock.dt)#increased gain from .3 to .6  and offset was .7
    ## v_drive = TimedArray(trans_inputs*.7*(vt-vr) + .8*(vt-vr), dt=defaultclock.dt)# gain was 1  and offset was .86
    v_drive = TimedArray(trans_inputs*.7*(vt-vr) + 1.0*(vt-vr), dt=defaultclock.dt)# gain was 1  and offset was .86

    
    
    # eqs defines the differential equation(s) for membrane potential
    #     potential is locked (no change) during refractory period
    # xi: noise added to membrane potential; otherwise spikes are sync-ed across neurons
    # v_drive: a gaussian shaped input voltage that serves as inputs
    #          amplitude of gaussian goes from negative to positive across neurons
    # i:  neuron index, 0 ~ N-1
    
   
    eqs = '''
     ##dv/dt = (v_drive(t,i)+vr-v)/tau*int(not_refractory)+n_sigma*((tau*.5)**-0.5)*xi*int(not_refractory) : volt
     #dv/dt = (v_drive(t,i)+vr-v)/tau*int(not_refractory)+n_sigma*xi*tau**-0.5*int(not_refractory) : volt
    
    #dv/dt = (v_drive(t,i)+vr-v)/tau+n_sigma*xi*tau**-0.5 : volt

    dv/dt = (v_drive(t,i)+vr -v) / tau : volt
    tau:second 
    '''
   
    
    
    # reset potential to this value after each spike
    reset = '''
    v = vr
    '''
    all_indices=[]
    all_occ=[]  
    G = NeuronGroup(num_neurons, eqs, threshold='v>vt', reset=reset, refractory=10*ms, method='euler')
    
    
    # randomize initial membrane potential values
    G.v = 'rand()*(vt-vr)+vr'
    G.tau= 10 *ms

    SM = SpikeMonitor(G,record= True)
    
    
    
    # run the simulation for specified duration
    run(duration*ms)


    
    all_indices=SM.i;
    all_occ= SM.t
    #all_count=SM.count;

    return(all_indices,all_occ) 
    