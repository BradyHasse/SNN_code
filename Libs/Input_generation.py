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
from brian2 import SpikeGeneratorGroup,mA,second, mV, volt, Synapses, collect, devices
from brian2 import NeuronGroup, run, ms, defaultclock, SpikeMonitor, StateMonitor,start_scope,TimedArray,PopulationRateMonitor,PoissonGroup,Hz,second, Network
import numpy as np

from Libs.Helper_Functions import simple_regress, smooth, make_histos, RMSE, make_norm_histos, magnitude, make_norm_histos_nbins, score_run


#%% make_individual_input_spikes   
def make_individual_input_spikes_par(args):
    # This is the master program for making input spikes-  calls the other input generation routines
    target = args[0][0]
    rep = args[0][1]
    seed = args[0][2]
    
    num_targets = args[1][0]
    rep_in = args[1][1]
    events = args[1][2]
    duration = args[1][3][target, rep]
    speed_all = args[1][4]
    speed_times = args[1][5]
    gauss_center = args[1][6]
    gauss_sigma = args[1][7]
    event_landmarks = args[1][8]
    maxspeed = args[1][9]
    
    

    dir_range= np.linspace(0,2*np.pi-(2*np.pi)/num_targets,num_targets)
    num_neurons = 90
    # speed_gain = 2   #changing to default
    speed_gain = 1  #4.5
    speed_lag = 50 #71
    inp_groups = 4
    
    out_inp_indices = np.ndarray(inp_groups, dtype='object')
    out_inp_spikes = np.ndarray(inp_groups, dtype='object')

    mev = np.mean(events,1)
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
    ev = np.round(events[target][rep] * 1000)
    sp_times= speed_times[target][rep].flatten() *1000 
   
    speed= speed_all[target][rep].flatten()
    speed*= speed_gain
    sp_times_slice = sp_times
    speed_s = speed[np.logical_and(sp_times>=ev[6], sp_times<=ev[9])]
    sp_times_slice_s = sp_times[np.logical_and(sp_times>=ev[6], sp_times<=ev[9])]
      
    center1 = ev[6]-int1 #start_movement
    center2 = ev[11] -int2 #pk_speed_time
    center3 = ev[9]-int3  #end_movement

    # width1 = gauss_sigma[0]*300   #changing to default width 
    # width2 = gauss_sigma[1]*500#changing to default width
    # width3 = gauss_sigma[2]*300#changing to default width
    width1 = gauss_sigma[0]*1000   
    width2 = gauss_sigma[1]*1000
    width3 = gauss_sigma[2]*1000

    out_inp_indices_,out_inp_spikes_ = make_input_spikes(duration,direction,num_neurons,center1*10,width1*10,speed,sp_times_slice,speed_lag,seed,maxspeed) #.35 worked well but background was too high for training
    out_inp_indices[0] = np.copy(out_inp_indices_)
    out_inp_spikes[0] = np.copy(out_inp_spikes_/second)
    
    out_inp_indices_,out_inp_spikes_ = make_input_spikes(duration,direction,num_neurons,center2*10,width2*10,speed,sp_times_slice,speed_lag,seed+1,maxspeed) #gain = 3
    out_inp_indices[1] = np.copy(out_inp_indices_)
    out_inp_spikes[1] = np.copy(out_inp_spikes_/second)
    
    out_inp_indices_,out_inp_spikes_ = make_input_spikes(duration,direction,num_neurons,center3*10,width3*10,speed,sp_times_slice,speed_lag,seed+2,maxspeed)
    out_inp_indices[2] = np.copy(out_inp_indices_)
    out_inp_spikes[2] = np.copy(out_inp_spikes_/second)
    
    out_inp_indices_,out_inp_spikes_ = make_input_spikes_speed(duration,num_neurons,speed_s,sp_times_slice_s,speed_lag,seed+3,maxspeed)
    out_inp_indices[3] = np.copy(out_inp_indices_)
    out_inp_spikes[3] = np.copy(out_inp_spikes_/second)
    
    # print('Finished target ',target,' rep ' ,rep ) 

    return(out_inp_indices, out_inp_spikes, target, rep)

#%% make_input_spikes
def make_input_spikes(duration,direction,num_neurons,center,width,speed,speed_points,speed_lag,seed,maxspeed):
#Uses Hongwei's equations for adding noise
# Make a set of num_neurons, each with a different prefered direction, with firing rates dictated by gaussian input 
    #print('direction',direction,'gaincenter',center,'width',width, 'speed_center',speed_center)
    start_scope()
    devices.device.seed(seed=seed)
    
    vr = -70*mV#-70*mV 	# resting potential level
    vt = -55*mV#0.35*volt# threshold for firing of action potential
    baseFR = 35
    n_sigma = 0.15*(vt-vr)	# 0.05; sigma of noise added to membrane potential
    
    inputs =  gaussian_input_speed(duration,direction,num_neurons,center,width,speed,speed_points,speed_lag,seed,maxspeed)
    trans_inputs = np.transpose(inputs)
    trans_inputs = ((trans_inputs*(vt-vr))+0.50*(vt-vr))/(50/baseFR)
    v_drive = TimedArray(trans_inputs, dt=defaultclock.dt)# gain was 1  and offset was .86

    # eqs defines the differential equation(s) for membrane potential
    #     potential is locked (no change) during refractory period
    # xi: noise added to membrane potential; otherwise spikes are sync-ed across neurons
    # v_drive: a gaussian shaped input voltage that serves as inputs
    #          amplitude of gaussian goes from negative to positive across neurons
    # i:  neuron index, 0 ~ N-1
    
    eqs = '''
    dv/dt = (v_drive(t,i))/tau + n_sigma*xi*(tau**-0.5): volt
    tau:second 
    '''
    # reset potential to this value after each spike
    reset = '''
    v = vr
    '''

    G = NeuronGroup(num_neurons, eqs, threshold='v>vt', reset=reset, refractory=10*ms, method='euler')
    
    # randomize initial membrane potential values
    G.v = 'rand()*(vt-vr)+vr'
    G.tau= 10 *ms

    SM = SpikeMonitor(G,record= True)
    
    # run the simulation for specified duration
    run(duration*ms)

    all_indices=SM.i;
    all_occ= SM.t
    
    
    # plt.figure()
    # ax= plt.gca()
    # ax.set_prop_cycle(plt.cycler('color', colors90)) 
    # plt.plot(trans_inputs)
    # plt.show()
    
    # reps = [0,1]  
    # events = events[:,reps[0]:reps[1],:]
    # numsegs = 4
    # events = events[0,reps[0]:reps[1],:]
    # events = np.tile(events, [90,1])
    # events = np.expand_dims(events,1)
    
    # inds = np.array(all_indices)#this block is used to check what inputs look like.
    # occs = np.array(all_occ)
    
    # plt.figure()
    # ax= plt.gca()
    # ax.set_prop_cycle(plt.cycler('color', colors90)) 
    # for ind in range(num_neurons):
    #     ttimes = occs[inds==ind]
    #     num_spikes = len(ttimes) 
    #     y = np.ones(num_spikes)*(ind+1)
    #     plt.plot(ttimes,y,'.')
    #     plt.title(len(occs))
    # plt.show()
        
    # occs2 = []   
    # for ind in range(num_neurons): 
    #     occs2.append([occs[inds==ind]])
    # occs3 = np.array(occs2,dtype=object)
    # in_array = occs3  
    # [histo,xax_labels,mean_ev] =  make_norm_histos(occs3,events,reps,4)
    # plt.figure()
    # ax= plt.gca()
    # ax.set_prop_cycle(plt.cycler('color', colors90)) 
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # for i in range(90):
    #     plt.plot(histo[:,i],lw = 2)
    
    return(all_indices,all_occ) 
#%% make_input_spikes_speed
def make_input_spikes_speed(duration,num_neurons,speed,speed_points,speed_lag,seed,maxspeed):
    #New input class for non-directional component
    #Matched to make_input_spikes
    start_scope()
    devices.device.seed(seed=seed)
    
    vr = -70*mV#-70*mV 	# resting potential level
    vt = -55*mV#0.35*volt# threshold for firing of action potential
    baseFR = 35
    n_sigma = 0.10*(vt-vr)	# 0.05; sigma of noise added to membrane potential
    
    inputs =  make_ndd(duration,num_neurons,speed,speed_points,speed_lag,maxspeed)
    trans_inputs = np.transpose(inputs)
    trans_inputs = ((trans_inputs*(vt-vr))+0.50*(vt-vr))/(50/baseFR)
    v_drive = TimedArray(trans_inputs, dt=defaultclock.dt)# gain was 1  and offset was .86
    
    eqs = '''
    dv/dt = (v_drive(t,i))/tau + n_sigma*xi*(tau**-0.5): volt
    tau:second 
    '''
    reset = '''
    v = vr
    '''
    
    G = NeuronGroup(num_neurons, eqs, threshold='v>vt', reset= reset,refractory=10*ms,  method='euler')
    
    G.tau= 10 *ms
    G.v = 'rand()*(vt-vr)+vr'
    
    SM = SpikeMonitor(G,record= True)

    # run the simulation for specified duration
    run(duration*ms)

    all_indices=SM.i;
    all_occ= SM.t
    
    
    # plt.figure()
    # ax= plt.gca()
    # ax.set_prop_cycle(plt.cycler('color', colors90)) 
    # plt.plot(trans_inputs)
    # plt.show()
    
    # reps = [0,1]  
    # events = events[:,reps[0]:reps[1],:]
    # numsegs = 4
    # events = events[0,reps[0]:reps[1],:]
    # events = np.tile(events, [90,1])
    # events = np.expand_dims(events,1)
    
    # inds = np.array(all_indices)#this block is used to check what inputs look like.
    # occs = np.array(all_occ)
    
    # plt.figure()
    # ax= plt.gca()
    # ax.set_prop_cycle(plt.cycler('color', colors90)) 
    # for ind in range(num_neurons):
    #     ttimes = occs[inds==ind]
    #     num_spikes = len(ttimes) 
    #     y = np.ones(num_spikes)*(ind+1)
    #     plt.plot(ttimes,y,'.')
    #     plt.title(len(occs))
    # plt.show()
        
    # occs2 = []   
    # for ind in range(num_neurons): 
    #     occs2.append([occs[inds==ind]])
    # occs3 = np.array(occs2,dtype=object)
    # in_array = occs3  
    # [histo,xax_labels,mean_ev] =  make_norm_histos(occs3,events,reps,4)
    # plt.figure()
    # ax= plt.gca()
    # ax.set_prop_cycle(plt.cycler('color', colors90)) 
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # for i in range(90):
    #     plt.plot(histo[:,i],lw = 2)


    return(all_indices,all_occ)

#%% gaussian_input_speed
def gaussian_input_speed (duration,direction,num_neurons,center,gauss_sigma,speed_profile,speed_points,speed_lag,seed,maxspeed):
        
    #Uses Eq 1 from Moran and Schwartz
    speed_lag *= 10  #  lead time for non-directional component Moran and Schwartz x10 to match Brain2 default clock.
    num_pts = round(duration*ms/defaultclock.dt)#time in ms, 0.1ms per timestep (defaultclock.dt) - number of timepoints under Brain2 clock
    pd_range= np.linspace(0,2*np.pi-(2*np.pi)/num_neurons,num_neurons) #preferred direction of input neurons
    Drive = np.zeros([num_pts])  #The input driver
    sigma_gain = 1   #profile width #changing to default width
    speed_gain = 1/maxspeed


    # Make a Gaussian profile for the input signal
    # Make a Gaussian speed profile for the middle 8/10 of the trial
    #f(x) = a*exp-((x-b)**2/2c**2
    # x is an array of points
    # a is the amplitude
    # b is the center of the distribution
    # c is the standard deviation
    # width = 4*gauss_sigma * sigma_gain #changing to default width
    width = gauss_sigma * sigma_gain
    buff = [round(center-width), round(center+ width)]
    if buff[1] > duration *10:
        buff[1] = duration*10
        print('warn')
        if buff[1]> len(Drive):
            print('End condition detected, length of Gauss= ',len(Drive), 'End=',buff[1], 'Center=',center, 'Width=',width)
    b = center
    sigma = gauss_sigma * sigma_gain
    a = 0.1
    bufftmp = np.arange(buff[0], buff[1])
    Gauss_tmp= a*np.exp( -(np.power((bufftmp-b),2) / (2*np.power(sigma,2) )))
    #make an amplitude for the Gaussian using cosine model and equally spaced pds
    half_amp = 5 # was 5
    b0 = 0
    Gauss_mat = np.tile(Gauss_tmp,[num_neurons,1])
    current_amp = half_amp*np.cos(pd_range-direction)
    amp_mat = np.transpose(np.tile(current_amp,[Gauss_tmp.shape[0],1]))
    profile = np.multiply(amp_mat,Gauss_mat) + b0#gaussian profile multiplied by current amp
        
    
    # Make a speed profile
    # speed = np.zeros(num_pts)
    sp_time_points = np.array(np.round(speed_points*10),dtype= int)
    xx = np.arange(sp_time_points[0],sp_time_points[-1])
    sp_buffer = np.interp(xx,sp_time_points,speed_profile)
    speed = sp_buffer
    # speed[xx-int(speed_lag)] = sp_buffer
    
    
    b_offset = .001   #mean level of noise pulse
    fracdrive = 0.3#what proportion of the drive is random added to
    drivecng = 100#how many ms*10 does the random drive change.
    np.random.seed(seed=seed)
    Rjitt = np.random.uniform (-.05,.05, [num_neurons, round(num_pts/drivecng)])+b_offset
    Rjitt = np.expand_dims(Rjitt,2)
    Rjitt = np.tile(Rjitt, [1,1,round(drivecng*fracdrive)])
    Drive = np.zeros([num_neurons, Rjitt.shape[1], drivecng])+0.0001  #The input driver
    Drive[:,:,:round(drivecng*fracdrive)]  = Rjitt
    Drive2 = np.reshape(Drive, [num_neurons,-1])
    rows, column_indices = np.ogrid[:Drive2.shape[0], :Drive2.shape[1]]
    r = np.random.randint(0,drivecng*2, num_neurons)
    column_indices = column_indices - r[:,np.newaxis]
    Drive3 = Drive2[rows, column_indices]
    
    # speed2 = np.tile(1+ speed_gain*speed[bufftmp], [num_neurons,1])
    speed2 = np.tile(1+ speed_gain*speed[bufftmp+speed_lag], [num_neurons,1])
    profile2 = np.multiply(profile, speed2)
    Drive3[:,bufftmp] += profile2
    argument = Drive3
    # block is for testing stuff
    # plt.figure()
    # ax= plt.gca()
    # ax.set_prop_cycle(plt.cycler('color', colors90))
    # plt.plot(speed2.T)
    # plt.axis((0,10000,-.7,.7))
    return(argument)
    

#%% make_ndd
def make_ndd(duration,num_neurons,speed_profile,speed_points,speed_lag,maxspeed):
     #makes the driver for the non-directional input neurons
     speed_gain = 1/maxspeed
     speed_profile = speed_profile * speed_gain
     #vr is the background level
     num_pts = round(duration*ms/defaultclock.dt)#time in ms, 0.1ms per timestep (defaultclock.dt) - number of timepoints under Brain2 clock

     coeffs= np.linspace(-.5,1,num_neurons)#speed coeffs
     
     speed_lag *= 10  #  lead time for non-directional component Moran and Schwartz, *10 because number of timepoints under Brain2 clock

     speed = np.zeros(num_pts)
     sp_time_points = np.array(np.round(speed_points*10),dtype= int)
     xx = np.arange(sp_time_points[0],sp_time_points[-1])
     sp_buffer = np.interp(xx,sp_time_points,speed_profile)
     speed[xx-speed_lag] = sp_buffer
     
     outbuf= []
     for unit in range(num_neurons):
         outbuf.append(speed*coeffs[unit])
     return(outbuf)

#%% make_offset_weights
def make_offset_weights_par(args):
    
    r_all = args[0][0]
    inp_indices = args[0][1]
    inp_spikes = args[0][2]
    target = args[0][3]
    rep = args[0][4]
    duration = args[0][5][target, rep] 
    
    
    final_offsets = args[1]
    
    num_units = len(r_all)#number of output units
    num_neurons = 90  #number of input neurons
    num_inputs = num_neurons
    taup = 20*ms
    
    Apre = 0.01
    Apost= 0.01
    
    w_offsets = np.zeros([len(inp_spikes), final_offsets.shape[0]*num_neurons])
    for epoch in range(final_offsets.shape[1]):
        w_offsets[epoch,:] = np.squeeze(np.tile(final_offsets[:,epoch], [num_neurons,1]))
        
    W_previous = [.000, .000, .000, .000]#initial weights
    
    start_scope() 
    
    out_st_array = []
    out_ind_array = []
    for u in range(num_units):  #build arrays of actual units as output
        rr = r_all[u]
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
        inp_group.append(SpikeGeneratorGroup(num_inputs,inp_indices[i],inp_spikes[i]))
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
    W_previous = np.array(W_previous)
    
    # print('Rep ',rep)    
        
    # weight_multi = np.zeros([len(inp_spikes),num_inputs*r_all.shape[0]])
    # for i in range(len(inp_spikes)):
    #     weight_multi[i,:]= Syns[i].w

    # print('Finished')    
    return(W_previous, target, rep)


#%% par_w_step
def par_w_step(args):
    #args = [num_units, weight_multi_3d, inp_spikes, inp_indices, reps, duration, events, actual_hist]
    w_step = args[0]
    num_units = args[1][0]
    weight_multi_3d = args[1][1]
    inp_spikes = args[1][2]
    inp_indices = args[1][3]
    reps = args[1][4]
    duration = args[1][5]
    events = args[1][6]
    actual_hist = args[1][7]
    nbins = args[1][8]
    
    out_all_spikes_ = make_out_all_spikes([num_units, weight_multi_3d, inp_spikes, inp_indices, reps, duration, np.copy(w_step)])
    RMSEneur, correlation, correlation2 = score_run(actual_hist, num_units, out_all_spikes_, events, reps, nbins)
    return RMSEneur, w_step, correlation, correlation2

#%% make_out_all_spikes
def make_out_all_spikes(args):
    #num_units, weight_multi_3d, inp_spikes, inp_indices, params, duration
    
    num_units = args[0]
    weight_multi_3d = args[1]
    inp_spikes = args[2]
    inp_indices = args[3]
    reps = args[4]
    duration = args[5]
    params = np.copy(args[6])
    if len(args)>7:
        rtn_OAP = True
    else:
        rtn_OAP = False
    
    
    start_scope()
    
    num_neurons = 90
    num_directions=16  
    buf_size = num_neurons * num_units
    
    uth = np.zeros(num_units)
    for uni in range(num_units):
        uth[uni] =  params[uni][4]
    th_in = uth
    step = np.transpose(params[:,0:4])
    # step[-1,:] = 1
    step = np.tile(np.expand_dims(step, axis=2),[1, 1, num_neurons])
    step = np.moveaxis(step, 1,2)
    weight_multi_3ds = np.multiply(weight_multi_3d, step)#scaled weights
    weight_multi_3dw = np.reshape(np.transpose(weight_multi_3ds, [1,2,0]), [buf_size,-1])#
    #equivilent to w1

    SpkCnt=[] 
    
    H_eqs= '''
    dv/dt = -(v)/tau :1
    tau:second
    th :1
    '''
    
    out_all_potentials= []
    out_tmp_spikes=[] 
    
    r_counter = np.zeros(num_directions,dtype = int)
    ndex = 0
    for target in range(num_directions):
        SpkMonr=[] 
        SpkCntr=[]
        StMonr= []        
        
        
        for rep in range(reps[0],reps[1]): 
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
            
            SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += w')  #.1
            SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += w') #.1
            SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += w') #.1
            SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += w') #.05
            
            SS1.connect()
            SS2.connect()
            SS3.connect()
            SS4.connect()
            
            SM=SpikeMonitor(H)
            if rtn_OAP:
                M = StateMonitor(H, variables=True, record=True)
            
            SS1.w= weight_multi_3dw[:,0]
            SS2.w= weight_multi_3dw[:,1]
            SS3.w= weight_multi_3dw[:,2]
            SS4.w= weight_multi_3dw[:,3]  
            
            run(duration[target, rep]*ms)
            unit_spikes = []
            for ui in range(num_units):
                unit_spikes.append(SM.t[SM.i==ui])
            SpkMonr.append(unit_spikes) 
            SpkCntr.append(SM.count)
            if rtn_OAP:
                StMonr.append(M.v)
        out_tmp_spikes.append(SpkMonr)
        SpkCnt.append(SpkCntr)
        if rtn_OAP:
            out_all_potentials.append(StMonr)
        ndex +=1
        if (reps[1]-reps[0]) > 1:
            print('Finished Direction ', ndex)
    out_all_spikes = []  
    out_all_spikes = [[[out_tmp_spikes[i][k][j] for k in range(r_counter[i])] for i in range(num_directions)] for j in range(num_units)]
    if rtn_OAP:
        return out_all_spikes, out_all_potentials
    else:
        return out_all_spikes
#%% make_out_all_spikes_par
def make_out_all_spikes_par(args):

    inp_indices = args[0][0]
    inp_spikes = args[0][1]
    target = args[0][2]
    rep = args[0][3]
    
    num_units = args[1][0]#output units
    weight_multi_3d = args[1][1]
    duration = args[1][2][target, rep]
    params = np.copy(args[1][3])
    
    start_scope()
    
    num_neurons = 90
    buf_size = num_neurons * num_units
    
    uth = np.zeros(num_units)
    for uni in range(num_units):
        uth[uni] =  np.copy(params[uni][4])
    th_in = uth
    step = np.transpose(params[:,0:4])
    # step[-1,:] = 1
    step = np.tile(np.expand_dims(step, axis=2),[1, 1, num_neurons])
    step = np.moveaxis(step, 1,2)
    weight_multi_3ds = np.multiply(weight_multi_3d, step)#scaled weights
    weight_multi_3dw = np.reshape(np.transpose(weight_multi_3ds, [1,2,0]), [buf_size,-1])#
    #equivilent to w1

    
    H_eqs= '''
    dv/dt = -(v)/tau :1
    tau:second
    th :1
    '''
    
    start_scope()
    
    inp1_group = SpikeGeneratorGroup(num_neurons,inp_indices[0],inp_spikes[0])
    inp2_group = SpikeGeneratorGroup(num_neurons,inp_indices[1],inp_spikes[1])
    inp3_group = SpikeGeneratorGroup(num_neurons,inp_indices[2],inp_spikes[2])
    inp4_group = SpikeGeneratorGroup(num_neurons,inp_indices[3],inp_spikes[3])
          
    H= NeuronGroup(num_units,H_eqs,threshold = 'v > th', reset = 'v = -10 ',  method = 'exact')   #.08,-.3 #  0 -1 #-.006 -.5
    #H= NeuronGroup(num_units,H_eqs,threshold = 'v > 0.1', reset = 'v = -10 ',  method = 'exact')   #.08,-.3 #  0 -1 #-.006 -.5

    H.tau= 10*ms
    ##H.Voff = 0 #-.1
    H.th = th_in
    
    SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += w')  #.1
    SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += w') #.1
    SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += w') #.1
    SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += w') #.05
    
    SS1.connect()
    SS2.connect()
    SS3.connect()
    SS4.connect()
    
    M = StateMonitor(H, variables=True, record=True)
    SM=SpikeMonitor(H)
    
    SS1.w= weight_multi_3dw[:,0]
    SS2.w= weight_multi_3dw[:,1]
    SS3.w= weight_multi_3dw[:,2]
    SS4.w= weight_multi_3dw[:,3]  
    
    run(duration*ms)
    unit_spikes = []
    for ui in range(num_units):
        unit_spikes.append(SM.t[SM.i==ui])
    unit_pot = M.v
    
    return unit_spikes, unit_pot, target, rep





