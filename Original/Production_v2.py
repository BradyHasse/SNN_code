# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:54:31 2023

@author: Andrew
"""
#%%
import matplotlib.pyplot as plt
from brian2 import SpikeGeneratorGroup,mA,second
from brian2 import NeuronGroup, run, ms, defaultclock, SpikeMonitor, StateMonitor,start_scope,TimedArray,PopulationRateMonitor,PoissonGroup,Hz,second, Network

# from Libs import timed_input, make_temporal_input_spikes
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
#%%Get Hongwei's Data
#spkstruct = sio.loadmat('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkCDataSelected.mat')
spkstruct = sio.loadmat('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkN359Selected.mat')

all_units = spkstruct['spk_all']

r_all = all_units
r_raw= spkstruct['spk_raw']
events = spkstruct['events_out']
event_names = spkstruct['event_names']
#events[:,:,2]= 0;
mevents= np.mean(np.mean(events,0),0)*1000 
rc = spkstruct['rep_cnt']    
rcl =rc.tolist()
rca= np.array(rcl)
rep_cnt = rca.flatten()     
speed_all = spkstruct['speed_out']
#speed_all*=3   #3
speed_time = spkstruct['speed_time']
gauss_center= spkstruct['gauss_mu'].flatten()
gauss_sigma= spkstruct['gauss_sigma'].flatten()
event_landmarks= spkstruct['landmarks'].flatten()
#speed_time *= 1000
duration = 1000


    
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

#%%  use individual offset to train on all units from monkey C or N

   
#Get stored offsets that came from auto_off_new and simrun
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/final_offsets_MonkC_July30_20reps.npy', 'rb') as f:
with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/final_offsets_MonkN_July25_20reps.npy', 'rb') as f:

     final_offsets = np.load(f)
 
 

# spkstruct = sio.loadmat('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkCallspikes.mat')
# r_all = spkstruct['spk_all']
# num_units = len(np.squeeze(r_all[0]))
# events = spkstruct['events_out']
# mevents= np.mean(np.mean(events,0),0)*1000 
# rc = spkstruct['rep_cnt']    
# rcl =rc.tolist()
# rca= np.array(rcl)
# rep_cnt = rca.flatten()         
num_units = len(r_all)



num_neurons = 90  #number of input neurons
num_inputs = num_neurons
num_directions=16
dir_range= np.linspace(0,2*np.pi,num_directions) 
taup = 20*ms

Apre = 0.01
Apost= 0.01

buf_size = num_neurons * num_units

off1 = np.zeros(buf_size)
off2 = np.zeros(buf_size)
off3 = np.zeros(buf_size)
for un in range(num_units):
    for i in range(un,buf_size,num_units):  #wrap into vector
            off1[i] = final_offsets[un,0]
            off2[i] = final_offsets[un,1]
            off3[i] = final_offsets[un,2]


w_offset1 = off1
w_offset2 = off2
w_offset3 = off3
# w_offset1 = 0
# w_offset2 = 0
# w_offset3 = 0

      
W1_save = []            
W2_save = [] 
W3_save = []
W4_save = []

weight1_multi= []
weight2_multi= []
weight3_multi= []
# weight4_multi= {}
weight4_multi= []

W1_previous = .002
W2_previous=  .002
W3_previous = .002
W4_previous =.002   #initial weights






start_scope() 

for rep in range(20):
#for rep in range(rep_cnt[target]):
    for target in range(16):

        out_st_array = []
        out_ind_array = []
        for u in range(num_units):  #build arrays of actual units as output
            # r = np.squeeze(r_all[0][u])
            # rr=np.squeeze(r[target,rep])
            rr = r_all[u][target][rep]
            #rr=r[target][rep]
            st = []
            shaperr = rr.shape
            if(shaperr[0] == 1):
	            rr = np.transpose(rr)
            
            for jj in range(len(rr)):
                if rr[jj]>=0:
                    spktime = np.float64(rr[jj] * 1000)
                    st.append(spktime)
            # else:
            #         print('Found a single spike Target=',target,'Rep=',rep)
            #         if rr[0]>=0:
            #             spktime = np.float64(rr[0] * 1000)
            #             st.append(spktime)  
      
            out_st_array= np.concatenate((out_st_array,st),axis=0)
            indices= np.ones(len(st),int) * u
            out_ind_array= np.concatenate((out_ind_array,indices),axis= 0)
 
             
        
        # inp1_group = SpikeGeneratorGroup(num_inputs,inp_indices[0][target][rep],inp_spikes[0][target][rep])
        # inp2_group = SpikeGeneratorGroup(num_inputs,inp_indices[1][target][rep],inp_spikes[1][target][rep])
        # inp3_group = SpikeGeneratorGroup(num_inputs,inp_indices[2][target][rep],inp_spikes[2][target][rep])
        inp1_group = SpikeGeneratorGroup(num_inputs,inp_indices[0][target][rep],inp_spikes[0][target][rep])
        inp2_group = SpikeGeneratorGroup(num_inputs,inp_indices[1][target][rep],inp_spikes[1][target][rep])
        inp3_group = SpikeGeneratorGroup(num_inputs,inp_indices[2][target][rep],inp_spikes[2][target][rep]) 
        inp4_group = SpikeGeneratorGroup(num_inputs,inp_indices[3][target][rep],inp_spikes[3][target][rep]) 
        
        out_group = SpikeGeneratorGroup(num_units,out_ind_array,out_st_array *ms)
    
        S1= Synapses(inp1_group, out_group,
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
    
        S2= Synapses(inp2_group, out_group,
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
        
        S3= Synapses(inp3_group, out_group,
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
    
        S4= Synapses(inp4_group, out_group,
          '''
          w : 1
          
          dapre/dt = -apre/taup : 1 (event-driven)
          dapost/dt = -apost/taup : 1 (event-driven)
          ''',
          on_pre='''
          apre += Apre
          w = w+apost
          ''',
          on_post='''
          apost += Apost
          w = w+apre
          ''')      
        #S1.connect(i=np.arange(num_neurons),j=np.arange(num_units))
        # S2.connect(i=np.arange(num_neurons),j=np.arange(num_units))
        # S3.connect(i=np.arange(num_neurons),j=np.arange(num_units))
        S1.connect()
        S2.connect()
        S3.connect()
        S4.connect()
        
        S1.w = W1_previous
        S2.w = W2_previous 
        S3.w = W3_previous
        S4.w = W4_previous
        S1.w_offset = w_offset1
        S2.w_offset = w_offset2
        S3.w_offset = w_offset3
       
        #SM = SpikeMonitor(out_group)
        run(duration*ms)
        
        W1_previous = S1.w
        W2_previous = S2.w
        W3_previous = S3.w
        W4_previous = S4.w
        W1_save.append (W1_previous)
        W2_save.append (W2_previous)
        W3_save.append (W3_previous)
        W4_save.append (W4_previous)
 
        #print('Target ',target)
        
    print('Rep ',rep)    
    


weight1_multi= S1.w
weight2_multi= S2.w
weight3_multi= S3.w
weight4_multi= S4.w

    
print('Finished')     

#%%  for input 4 only :make_individual_input_spikes(num_targets,rep_in,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks):
dir_range= np.linspace(0,2*np.pi,num_targets)
num_neurons= 90
speed_gain = 2  #4.5
speed_lag = 71 #70


out4_inp_indices= []

out4_inp_spikes= []  
# Find the mean intervals between the peaks of each PCA component and one of the behavioral events    
int1 = (event_landmarks[0]-gauss_center[0])*1000
int2 = (event_landmarks[1]-gauss_center[1])*1000
int3 = (event_landmarks[2]-gauss_center[2])*1000

   
for target in range(num_targets):
    direction = dir_range[target]
    
    tmp4_inp_indices = []
    
    tmp4_inp_spikes = []

    try:
        len(rep_in)
        num_reps = rep_in[target]
    except:
        num_reps= rep_in
    #print('Reps for target ',target,' =' ,num_reps )    
    for rep in range(num_reps): 
        ev = events[target][rep] * 1000
        sp_times= speed_time[target][rep].flatten() *1000 
        #speed_width = (sp_times[-1]-sp_times[0])/10    #move_end - move_start
       
        speed= speed_all[target][rep].flatten()
        speed*= speed_gain
        a= np.where(sp_times>=ev[6])[0][0]
        if np.max(sp_times)>ev[9]:
            b= np.where(sp_times>=ev[9])[0][0]
        else:
            b = len(sp_times)-1
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
        width1 = gauss_sigma[0]*1000
        width2 = gauss_sigma[1]*1000
        width3 = gauss_sigma[2]*1000
        
        
        inps_ind4,inps_spikes4 =make_input_spikes_speed(duration,num_neurons,speed,sp_times_slice,speed_lag)
       
        tmp4_inp_indices.append(inps_ind4)
        
        tmp4_inp_spikes.append(inps_spikes4)
  
    out4_inp_indices.append(tmp4_inp_indices)
  
    out4_inp_spikes.append(tmp4_inp_spikes)
    
    print('Finished target ',target)
    print ('Number of spikes ',len(out4_inp_indices[target][0]))
    
#%% Testing of network--- production version multi-unit with specific events
start_scope()

#duration = 2000
num_neurons = 90
num_units = len(r_all)
num_directions=16  
dir_range= np.linspace(0,2*np.pi,num_directions)
xxx= range(0,num_neurons)
uth = np.zeros(num_units)
# m1 = []
# m2 = []
# m3 = []
# w1 = np.zeros(len(weight1_multi))  
# w2 = np.zeros(len(weight2_multi))
# w3 = np.zeros(len(weight3_multi))
w1m = np.zeros(len(weight1_multi))
w2m = np.zeros(len(weight2_multi))          
w3m = np.zeros(len(weight3_multi))
slope = np.zeros(num_units)
# w1m = weight1_multi     
# w2m = weight2_multi     
# w3m = weight3_multi  
w1m = weight1_multi    
w2m = weight2_multi     
w3m = weight3_multi
w4m = weight4_multi
#find max weights  
w_array= []
# w_max = np.zeros(num_units)
# w_max1= np.zeros(num_units)
# w_max2= np.zeros(num_units)
# w_max3= np.zeros(num_units)
# w_max4= np.zeros(num_units)
# w_min4= np.zeros(num_units)
w4_check= np.zeros((num_units,num_neurons))
for un in range (num_units):
#for un in range (34,35):    
    w1w = w1m[range(un,len(w1m),num_units)]  #unwrap vector
    w2w = w2m[range(un,len(w2m),num_units)]  #unwrap vector
    w3w = w3m[range(un,len(w3m),num_units)]  #unwrap vector
    w4w = w4m[range(un,len(w4m),num_units)]
    # w1s = np.array(eng.smooth(ml.double(w1w),ml.double(20),'sgolay'))
    # w2s = np.array(eng.smooth(ml.double(w2w),ml.double(20),'sgolay'))
    # w3s = np.array(eng.smooth(ml.double(w3w),ml.double(20),'sgolay'))
    # w_array.append(np.concatenate((w1s,w2s,w3s),axis=0))
    # w_max[un] = np.max(w_array)
    # w_max1[un] = np.max(w1s)
    # w_max2[un] = np.max(w2s)
    # w_max3[un] = np.max(w3s)
    # w_max4[un] = np.max(w4w)
    # w_min4[un] = np.min(w4w)
    #w4_check[un] = w4w 
    (slope[un],b)= np.polyfit(xxx,w4w,1)

# Set thresholds to account for output units with small weights 
buf_size = num_neurons * num_units
#th = np.zeros(num_units )

for un in range(num_units):
#for un in range(52,53):
    # w_min =  np.min([w_max1[un], w_max2[un], w_max3[un]])
    # if w_min < 0:
    #     th[un]= -1.5
    #     print('Unit', un, 'had a negative max weight', w_min)
    # elif np.max(w_array[un])>= .2 and np.max(w_array[un])<.75:
    #         th[un]= 0
    # elif np.max(w_array[un])>0 and np.max(w_array[un]) < .2:
    #       th[un]= 0
    
    
    # if np.max(w_array[un])<.5:
        

    # if w_min4[un]<.002:
    if slope[un]<.12: #.00025, .09, .11
        uth[un]= .005  #.065, .1,0, -.2 works well for Monk C
        print('Unit', un, 'had a low slope', slope[un])   
    else:
        uth[un]= .02  #.1,.2,.6   .4 works well for Monk C
    # if amp[un]<20 and slope[un] < .2 :
    #     th[un] = .5
    #     print('Unit', un, 'had a low amplitude and a low slope')
    # elif amp[un] >= 20 and (slope[un] < .25 or amp[un]<50):
    #     print ('Unit', un, 'gets an intermediate threshold')
    #     th[un]= 1
    # elif slope[un] < .25:
    #     th[un] = .3
    #     print ('Unit ',un, 'has only a low slope')
    # else:
    #     th[un]= 2
    #     print('Unit', un, 'had a high amplitude')
        
        

#    th[un]= .1

    # if un == 14:
    #     print('Threshold for unit',un, 'is', th[un])
    #     print('Max =',np.max(w_array[un])) 


     

SpkMon=[]
StMon= []
SpkCnt=[] 
#num_reps = 3


H_eqs= '''
##dv/dt = -(v+Voff)/tau :1
dv/dt = -(v)/tau :1
tau:second
##Voff :1
th :1
'''

out_all_potentials= []
out_tmp_spikes=[] 
out_ind_spikes=[]

  

r_counter = np.zeros(num_directions,dtype = int)
ndex = 0
for direction in dir_range:
    SpkMonr=[] 
    SpkCntr=[]
    StMonr= []
    inp1_group=[]
    inp2_group=[]
    inp3_group=[]
    inp4_group=[]
    
    rep_start = 21
    rep_end = 41
    # for rep in range(rep_cnt[ndex]):
    for rep in range(rep_start,rep_end): 
    #for rep in range(21,22):
        r_counter[ndex] +=1

        start_scope()
        
        inp1_group = SpikeGeneratorGroup(num_neurons,inp_indices[0][ndex][rep],inp_spikes[0][ndex][rep])
        inp2_group = SpikeGeneratorGroup(num_neurons,inp_indices[1][ndex][rep],inp_spikes[1][ndex][rep])
        inp3_group = SpikeGeneratorGroup(num_neurons,inp_indices[2][ndex][rep],inp_spikes[2][ndex][rep])
        inp4_group = SpikeGeneratorGroup(num_neurons,inp_indices[3][ndex][rep],inp_spikes[3][ndex][rep])
              
        H= NeuronGroup(num_units,H_eqs,threshold = 'v > th', reset = 'v = -10 ',  method = 'exact')   #.08,-.3 #  0 -1 #-.006 -.5
        #H= NeuronGroup(num_units,H_eqs,threshold = 'v > 0', reset = 'v = -10 ',  method = 'exact')   #.08,-.3 #  0 -1 #-.006 -.5

        H.tau= 10*ms
        ##H.Voff = 0 #-.1
        H.th = uth
        
        
        
        # SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += 0.01*(w*int(w<0)*int(v_post>=-1) + w*int(w>0))')
        # SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += 0.01*(w*int(w<0)*int(v_post>=-1) + w*int(w>0))')
        # SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += 0.01 *(w*int(w<0)*int(v_post>=-1) + w*int(w>0))')
        # SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += 0.03*w')  #  good for MonkC.1
        # SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += 0.03*w') #.1
        # SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += 0.03*w') #.1
        SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += 0.001*w') #.05
        SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += 0.06*w')  #.1
        SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += 0.06*w') #.1
        SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += 0.06*w') #.1
        
        # SS1.connect(i=np.arange(num_neurons),j=0)
        # SS2.connect(i=np.arange(num_neurons),j=0)
        # SS3.connect(i=np.arange(num_neurons),j=0)
        
        SS1.connect()
        SS2.connect()
        SS3.connect()
        SS4.connect()
        
        SM=SpikeMonitor(H)
        M = StateMonitor(H, variables=True, record=True)
        
       
       
        # SS1.w= m1
        # SS2.w= m2
        # SS3.w= m3
        SS1.w= w1m
        SS2.w= w2m
        SS3.w= w3m
        SS4.w= w4m  
        
        
        run(duration*ms)
        unit_spikes = []
        for ui in range(num_units):
            unit_spikes.append(SM.t[SM.i==ui])
        SpkMonr.append(unit_spikes) 
        SpkCntr.append(SM.count)
        StMonr.append(M.v)
    out_tmp_spikes.append(SpkMonr)
    #out_tmp_indices.append(SpkIndices)
    SpkCnt.append(SpkCntr)
    out_all_potentials.append(StMonr)
    ndex +=1


    print('Finished Direction ', ndex)
out_all_spikes = []  
out_all_spikes = [[[out_tmp_spikes[i][k][j] for k in range(r_counter[i])] for i in range(num_directions)] for j in range(num_units)]    
#out_list = [len(SpkMon[i]) for i in range(16)] 




# to here 1/16/24
#%% input-output correlation for one unit   
#this is just for ad hoc checking not for actual evaluation 

from Libs import magnitude
pred_hist,b,c =  make_norm_histos(SpkTimes,events,reps,'predicted')
print('Actual')
s_actual_spikes= np.squeeze(r_all[unit])
actual_hist,b,c =  make_norm_histos(s_actual_spikes,events,reps,'actual')
pred_hist = np.transpose(pred_hist)
#pred_hist = tmp_hist
actual_hist = np.transpose(actual_hist)
predicted = []
actual = []
coort = []
pred = []
act = []
asave = []
psave = []
num_targets = len(s_actual_spikes)
for target in range(num_targets):
        pred = pred_hist[target][5:25]
        act = actual_hist[target][5:25]
        predicted=np.concatenate((predicted,pred),axis = 0)
        actual=np.concatenate((actual,act),axis = 0)
        coort.append(np.dot(act,pred)/(magnitude(act)*magnitude(pred))  )
        asave.append(act)
        psave.append(pred)
    
correlation=np.dot(actual,predicted)/(magnitude(actual)*magnitude(predicted))        

print('r=',correlation)

#%% calculate predicted-actual correlation for all units
from Libs import magnitude
correlation=[]
for unit in range(num_units):
#for unit in range(59,60): 
    print("Unit", unit)
    [pred_hist,b,c] =  make_norm_histos_short(out_all_spikes[unit],events,reps,'predicted')
    #[pred_hist,b,c] =  make_norm_histos(SpkTimes,events,reps,'predicted')
    actual_spikes= np.squeeze(r_all[unit])
    [actual_hist,d,e] =  make_norm_histos_short(actual_spikes,events,reps,'actual')
    pred_hist = np.transpose(pred_hist)
    #pred_hist = tmp_hist
    actual_hist = np.transpose(actual_hist)
    predicted = []
    actual = [] 
    coort = []
    pred = []
    act = []
    asave = []
    psave = []
    num_bins = len(pred_hist[0])
    num_targets = len(actual_spikes)
    for target in range(num_targets):
            # pred = pred_hist[target][5:30]
            # act = actual_hist[target][5:30]
            # pred = pred_hist[target][5:-1]   #Use rates 100 ms after target onset to end of movement
            # act = actual_hist[target][5:-1]
            pred = pred_hist[target][5:num_bins-4] 
            act = actual_hist[target][5:num_bins-4]
            predicted=np.concatenate((predicted,pred),axis = 0)
            actual=np.concatenate((actual,act),axis = 0)
            mag_ap = magnitude(act)*magnitude(pred)
            if mag_ap == 0:
                coort.append(0)
            else:
                coort.append(np.dot(act,pred)/mag_ap)  
            asave.append(act)  
            psave.append(pred)
    mag_prod = magnitude(actual)*magnitude(predicted)
    if mag_prod != 0:   
        correlation.append(np.dot(actual,predicted)/mag_prod)
    else:
        correlation.append(0)

    print('Finished Unit ',unit, 'r= ',correlation[unit])        

correlation = np.array(correlation,dtype= float).flatten()        

def make_norm_histos_short(in_array,events,reps,mode):    
# Use first three intervals only.  Stopping at end of movement    
    start_scope()
    
    max_num_reps = 47
    
    interval1= []
    interval2= []
    interval3= []
    interval4= []
    bw1_accum=[]
    bw2_accum=[]
    bw3_accum=[]
    bw4_accum=[]
    nbins1_accum=[]
    nbins2_accum=[]
    nbins3_accum=[]
    nbins4_accum=[]
    ev_accum= []
    
    if len(in_array) != 16:
        num_targets = 1
    else:
        num_targets = len(in_array)
        
    for it in range(num_targets): 
        try:
            length=len(reps)
            if length== 2:
                num_reps= reps[1]-reps[0]
                r_start = reps[0]
                r_end = reps[1]
            else:
                num_reps = reps[it]
                r_start= 0
                r_end = num_reps
        except:
            num_reps= reps
            r_start= 0
            r_end = reps
            #print('IT',it,'rsingle\n')
        #print('Num Reps=',num_reps)    
        for irep in range(r_start,r_end):
            try:
                len(events[0][0])
                marker = events[it][irep]
            except:
                try:
                    len(events[0])
                    marker = events[irep]
                except:
                    marker = events
                #print('Markers=',marker)
            interval1.append(marker[6]-marker[2])  #accumulate across reps and targets
            #interval1.append(marker[6])
            interval2.append(marker[11]-marker[6])
            interval3.append(marker[9]-marker[11])
            interval4.append(marker[5]-marker[9])
    mint1= np.mean(interval1)
    mint2= np.mean(interval2)
    mint3= np.mean(interval3)
    mint4= np.mean(interval4)
    #calculate the average number of 20 ms bins in each interval
    nbins1 = np.round(mint1/.02)
    nbins2 = np.round(mint2/.02)
    nbins3 = np.round(mint3/.02)
    nbins4 = np.round(mint4/.02)
    # total_bins = int(np.ceil(nbins1+nbins2+nbins3+nbins4))
    total_bins = int(np.ceil(nbins1+nbins2+nbins3))
    max_num_reps = np.max(reps)
    srates=np.zeros((16,max_num_reps,total_bins))
    mean_trials= []
    
    for t in range(num_targets):
        try:
            length=len(reps)
            if length== 2:
                num_reps= reps[1]-reps[0]
                r_start = reps[0]
                r_end = reps[1]
            else:
                num_reps = reps[t]
                r_start= 0
                r_end = num_reps
        except:
            num_reps= reps
            r_start= 0
            r_end = reps
        stimes = []
        

              
        rcounter = 0   
        for rep in range(r_start,r_end):  # for rsingle, there is only 1 rep
            try:
                len(events[0][0])
                ev = events[t][rep]
            except:
                 try:
                     len(events[0])
                     ev = events[rep]
                     #print('rep=',rep, ev[6])
                 except:
                     ev = events
            
            if mode == 'predicted':
                if num_targets== 1 and num_reps>1:
                    rr= in_array[rcounter]
                else:
                    try:
                        len(in_array[0])
                        # rr= in_array[t][rep]
                        if np.size(in_array[t][rcounter])!=0:
                            rr= in_array[t][rcounter]
                        else:
                            rr = 0
                        #print('rep=',rcounter)
                    except:
                        print('one target')
                        rr= in_array
                    if num_reps== 1:
                        try:
                            len(in_array[0][0])
                        except:
                            if num_targets==1:
                                rr= in_array
                               
                            else:
                                rr= in_array[t]
                    
                stimes = np.array(rr,dtype=float)
                rcounter +=1
    
                    
                        
            else: 
                if np.size(in_array[t][rep])>1:
                    rr= in_array[t][rep]
                    stimes=np.squeeze(rr.tolist()) 
                else:
                    rr = 0
                    stimes= []
                #rr= in_array[t][rep]
                # stimes=np.squeeze(rr.tolist()) 
     
     
            target_show = ev[2]
            start_movement = ev[6]
            pk_speed= ev[11]
            reward = ev[5]
            end_movement = ev[9]
            #print('\nSpike Times',stimes,'\n', 'Start=',start_movement,'\nPeak=', pk_speed)
            delta1 = start_movement-target_show                
            binwidth1 = ml.double(np.float64(delta1/nbins1))
            rates1= []
            rates1= eng.bin_frac(ml.double(stimes),ml.double(target_show),ml.double(start_movement),binwidth1) 
    
            delta2 = pk_speed-start_movement       
            binwidth2 = ml.double(np.float64(delta2/nbins2))
            rates2= eng.bin_frac(ml.double(stimes),ml.double(start_movement),ml.double(pk_speed),binwidth2)
    
            delta3 = end_movement-pk_speed        
            binwidth3 = ml.double(np.float64(delta3/nbins3))
            rates3= eng.bin_frac(ml.double(stimes),ml.double(pk_speed),ml.double(end_movement),binwidth3)
    
            delta4 = reward-end_movement     
            binwidth4 = ml.double(np.float64(delta4/nbins4))
            rates4= eng.bin_frac(ml.double(stimes),ml.double(end_movement),ml.double(reward),binwidth4)
    
            #tmp = np.concatenate((rates1,rates2,rates3,rates4),axis = 0)
            tmp = np.concatenate((rates1,rates2,rates3),axis = 0)
            sr =np.array(eng.smooth(ml.double(tmp),ml.double(10),'sgolay'))
            
            for fill in range(len(sr)):          
                  srates[t][rep][fill]=sr[fill]
            bw1_accum.append(binwidth1)
            bw2_accum.append(binwidth2)
            bw3_accum.append(binwidth3)
            #bw4_accum.append(binwidth4)
            nbins1_accum.append(len(rates1))
            nbins2_accum.append(len(rates2))
            nbins3_accum.append(len(rates3))
            #nbins4_accum.append(len(rates4))
            ev_accum.append(ev)
                                     
    #print('nbins', nbins1,nbins2,nbins3,nbins4)
    bw1 = np.mean(bw1_accum)
    bw2 = np.mean(bw2_accum)
    bw3 = np.mean(bw3_accum)
    #bw4 = np.mean(bw4_accum)
    nbins1= int(np.round(np.mean(nbins1_accum)))
    nbins2= int(np.round(np.mean(nbins2_accum)))
    nbins3= int(np.round(np.mean(nbins3_accum)))
    #nbins4= int(np.round(np.mean(nbins4_accum)))
    #total_time_bins= int(nbins1+nbins2+nbins3+nbins4)
    total_time_bins= int(nbins1+nbins2+nbins3)
  
    timax = np.zeros(total_time_bins)
    bdex = 0
    #for intv in range(4):
    for intv in range(3):    
        if intv == 0:
            timax[0]= 0
            timax[1]= bw1
            bdex= 1
            for tint in range(2,nbins1):
                bdex+=1
                timax[bdex]=timax[bdex-1]+bw1
        if intv == 1:
            bstart= bdex+1
            bend= bstart+nbins2
            for bdex in range(bstart,bend):
                timax[bdex]= timax[bdex-1]+bw2
        if intv == 2:
            bstart= bdex+1
            bend= bstart+nbins3
       
            for bdex in range(bstart,bend):
                timax[bdex]= timax[bdex-1]+bw3
        # if intv == 3:
        #     bstart= bdex+1
        #     bend= bstart+nbins4
        #     for bdex in range(bstart,bend):
        #         timax[bdex]= timax[bdex-1]+bw4 
    mean_ev = np.mean(ev_accum,0)            
    sum_rates = np.sum(srates,1)
    for m in range(num_targets):
        mean_trials.append(sum_rates[m]/num_reps)   
    # mean_targets = np.mean(mean_trials)
    # centered_profiles = mean_trials-mean_targets # subtract off overall mean rate
    #mr = np.transpose(centered_profiles)
    mr = np.transpose(mean_trials)
    return(mr,timax,mean_ev)      


def make_norm_histos(in_array,events,reps,mode):    
# Use first three intervals only.  Stopping at end of movement    
    start_scope()
    
    max_num_reps = 47
    
    interval1= []
    interval2= []
    interval3= []
    interval4= []
    bw1_accum=[]
    bw2_accum=[]
    bw3_accum=[]
    bw4_accum=[]
    nbins1_accum=[]
    nbins2_accum=[]
    nbins3_accum=[]
    nbins4_accum=[]
    ev_accum= []
    
    if len(in_array) != 16:
        num_targets = 1
    else:
        num_targets = len(in_array)
        
    for it in range(num_targets): 
        try:
            length=len(reps)
            if length== 2:
                num_reps= reps[1]-reps[0]
                r_start = reps[0]
                r_end = reps[1]
            else:
                num_reps = reps[it]
                r_start= 0
                r_end = num_reps
        except:
            num_reps= reps
            r_start= 0
            r_end = reps
            #print('IT',it,'rsingle\n')
        #print('Num Reps=',num_reps)    
        for irep in range(r_start,r_end):
            try:
                len(events[0][0])
                marker = events[it][irep]
            except:
                try:
                    len(events[0])
                    marker = events[irep]
                except:
                    marker = events
                #print('Markers=',marker)
            interval1.append(marker[6]-marker[2])  #accumulate across reps and targets
            #interval1.append(marker[6])
            interval2.append(marker[11]-marker[6])
            interval3.append(marker[9]-marker[11])
            interval4.append(marker[5]-marker[9])
    mint1= np.mean(interval1)
    mint2= np.mean(interval2)
    mint3= np.mean(interval3)
    mint4= np.mean(interval4)
    #calculate the average number of 20 ms bins in each interval
    nbins1 = np.round(mint1/.02)
    nbins2 = np.round(mint2/.02)
    nbins3 = np.round(mint3/.02)
    nbins4 = np.round(mint4/.02)
    total_bins = int(np.ceil(nbins1+nbins2+nbins3+nbins4))
    max_num_reps = np.max(reps)
    srates=np.zeros((16,max_num_reps,total_bins))
    mean_trials= []
    
    for t in range(num_targets):
        try:
            length=len(reps)
            if length== 2:
                num_reps= reps[1]-reps[0]
                r_start = reps[0]
                r_end = reps[1]
            else:
                num_reps = reps[t]
                r_start= 0
                r_end = num_reps
        except:
            num_reps= reps
            r_start= 0
            r_end = reps
        stimes = []
        

              
        rcounter = 0   
        for rep in range(r_start,r_end):  # for rsingle, there is only 1 rep
            try:
                len(events[0][0])
                ev = events[t][rep]
            except:
                 try:
                     len(events[0])
                     ev = events[rep]
                     #print('rep=',rep, ev[6])
                 except:
                     ev = events
            
            if mode == 'predicted':
                if num_targets== 1 and num_reps>1:
                    rr= in_array[rcounter]
                else:
                    try:
                        len(in_array[0])
                        # rr= in_array[t][rep]
                        rr= in_array[t][rcounter]
                        #print('rep=',rcounter)
                    except:
                        print('one target')
                        rr= in_array
                    if num_reps== 1:
                        try:
                            len(in_array[0][0])
                        except:
                            if num_targets==1:
                                rr= in_array
                               
                            else:
                                rr= in_array[t]
                    
                stimes = np.array(rr,dtype=float)
                rcounter +=1
    
                    
                        
            else:  
                rr= in_array[t][rep]
                stimes=np.squeeze(rr.tolist()) 
     
     
            target_show = ev[2]
            start_movement = ev[6]
            pk_speed= ev[11]
            reward = ev[5]
            end_movement = ev[9]
            #print('\nSpike Times',stimes,'\n', 'Start=',start_movement,'\nPeak=', pk_speed)
            delta1 = start_movement-target_show                
            binwidth1 = ml.double(np.float64(delta1/nbins1))
            rates1= []
            rates1= eng.bin_frac(ml.double(stimes),ml.double(target_show),ml.double(start_movement),binwidth1) 
    
            delta2 = pk_speed-start_movement       
            binwidth2 = ml.double(np.float64(delta2/nbins2))
            rates2= eng.bin_frac(ml.double(stimes),ml.double(start_movement),ml.double(pk_speed),binwidth2)
    
            delta3 = end_movement-pk_speed        
            binwidth3 = ml.double(np.float64(delta3/nbins3))
            rates3= eng.bin_frac(ml.double(stimes),ml.double(pk_speed),ml.double(end_movement),binwidth3)
    
            delta4 = reward-end_movement     
            binwidth4 = ml.double(np.float64(delta4/nbins4))
            rates4= eng.bin_frac(ml.double(stimes),ml.double(end_movement),ml.double(reward),binwidth4)
    
            tmp = np.concatenate((rates1,rates2,rates3,rates4),axis = 0)
            sr =np.array(eng.smooth(ml.double(tmp),ml.double(10),'sgolay'))
            
            for fill in range(len(sr)):          
                  srates[t][rep][fill]=sr[fill]
            bw1_accum.append(binwidth1)
            bw2_accum.append(binwidth2)
            bw3_accum.append(binwidth3)
            bw4_accum.append(binwidth4)
            nbins1_accum.append(len(rates1))
            nbins2_accum.append(len(rates2))
            nbins3_accum.append(len(rates3))
            nbins4_accum.append(len(rates4))
            ev_accum.append(ev)
                                     
    #print('nbins', nbins1,nbins2,nbins3,nbins4)
    bw1 = np.mean(bw1_accum)
    bw2 = np.mean(bw2_accum)
    bw3 = np.mean(bw3_accum)
    bw4 = np.mean(bw4_accum)
    nbins1= int(np.round(np.mean(nbins1_accum)))
    nbins2= int(np.round(np.mean(nbins2_accum)))
    nbins3= int(np.round(np.mean(nbins3_accum)))
    nbins4= int(np.round(np.mean(nbins4_accum)))
    total_time_bins= int(nbins1+nbins2+nbins3+nbins4)

    timax = np.zeros(total_time_bins)
    bdex = 0
    for intv in range(4):
        
        if intv == 0:
            timax[0]= 0
            timax[1]= bw1
            bdex= 1
            for tint in range(2,nbins1):
                bdex+=1
                timax[bdex]=timax[bdex-1]+bw1
        if intv == 1:
            bstart= bdex+1
            bend= bstart+nbins2
            for bdex in range(bstart,bend):
                timax[bdex]= timax[bdex-1]+bw2
        if intv == 2:
            bstart= bdex+1
            bend= bstart+nbins3
            for bdex in range(bstart,bend):
                timax[bdex]= timax[bdex-1]+bw3
        if intv == 3:
            bstart= bdex+1
            bend= bstart+nbins4
            for bdex in range(bstart,bend):
                timax[bdex]= timax[bdex-1]+bw4 
    mean_ev = np.mean(ev_accum,0)            
    sum_rates = np.sum(srates,1)
    for m in range(num_targets):
        mean_trials.append(sum_rates[m]/num_reps)   
    # mean_targets = np.mean(mean_trials)
    # centered_profiles = mean_trials-mean_targets # subtract off overall mean rate
    #mr = np.transpose(centered_profiles)
    mr = np.transpose(mean_trials)
    return(mr,timax,mean_ev)     

def plthist(*args):
#Plots Contributions of input units using "Percents" from Probability of Contribution routine
# Takes 1-3 arguments-  First argument is the first profile, 2nd argument is either a second profile, or the name of a pdf file to be saved, 3rd argument is the name of the pdf for 2 profiles
   from matplotlib import rc   #this is needed for CorelDraw to read the fonts correctly
   rc("pdf", fonttype=42)
   custom_colormap= sio.loadmat('C:/Users/Andrew/Documents/MATLAB/work/Programs/rgbColorMap.mat')
   colors = custom_colormap["rgbColors"]
   vargs= []
   vargs = args
   num_args = len(vargs)
   plt.figure()
      
   
   ax= plt.gca()
   ax.set_prop_cycle(plt.cycler('color', colors)) 
   ax.spines['right'].set_visible(False)
   ax.spines['top'].set_visible(False)
   Sfig= False
   #print('Number of input arguments= ',num_args)
   
   
   rates= vargs[0]
   events = vargs[1]
   reps = vargs[2]
   mode = vargs[3] 
   num_targets= len(rates)
   # max_rate = np.max(np.max(rates))
   # if max_rate<=50:
   #     ymax = 50
   # else:
   #     ymax = max_rate+5
   ymax =71
   plt.axis((0,35,0,ymax))
   
   [histo,xax_labels,mean_ev] =  make_norm_histos(rates,events,reps,mode)
   
   #print('Number of arguments= ',num_args)
   
   
   
   
   
   if len(histo[0]) != num_targets:
       try:
           histo = np.transpose(histo)
       except:
           histo = histo
  
   
 
   
  
   
   if num_args== 4:
       plt.plot(histo,lw = 2)
   else:
       try:
           disp_target = int(vargs[4])
           #print('Target= ',disp_target,'\n',len(histo[:,disp_target]))
           plt.plot(histo[:,disp_target],'k')
       except:
           plt.plot(histo,lw = 2)
           if num_args ==5:
               out_file= vargs[4]
               if np.char.find(out_file,'.')< 1:
                   out_file = str(np.char.add(out_file, '.pdf'))  
               Sfig = True
               
           
   
   plt.ylabel ('Firing Rate   ')
   plt.xlabel ('Time (ms)')
   # xlabels= np.arange(0,380,60)
   # plt.xticks(xlabels)
   ylabels = np.arange(0,ymax+1, 5)
   plt.yticks(ylabels) 
   xlabels= np.rint(xax_labels[0:len(xax_labels)+1:5]*1000).astype(int)
   xtplaces= np.arange(0,len(xax_labels),5)
   #print('labels= ',xlabels,'/n xplaces= ',xtplaces)
   plt.xticks(ticks=xtplaces,labels=xlabels)
   mvmt_onset_bin = ((mean_ev[6]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1]
   #print ('movement_onset',(mean_ev[6]-mean_ev[2])*1000,'onset bin=',mvmt_onset_bin)
   peak_speed_bin= ((mean_ev[11]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1]
   target_acq_bin= ((mean_ev[9]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1] 
   plt.plot(mvmt_onset_bin,.5,'*')
   plt.plot(peak_speed_bin,.5,'*')
   plt.plot(target_acq_bin,.5,'*')

   if num_args==6:
  
     try:
         disp_target = int(vargs[5])
         plt.plot(histo[:,disp_target],'r',lw = 2)
     except:
         out_file= vargs[5]
         if np.char.find(out_file,'.')< 1:
             out_file = str(np.char.add(out_file, '.pdf'))
         Sfig = True
   
   if num_args== 7:
       try:
           disp_target = int(vargs[6])
           print('Target= ',disp_target,'\n',len(histo),len(histo[0]),len(histo[:,disp_target]))
           plt.plot(histo[:,disp_target],'b',lw =2 )
       except:
           out_file= vargs[6]
           if np.char.find(out_file,'.')< 1:
               out_file = str(np.char.add(out_file, '.pdf'))
           Sfig = True
   if num_args== 8:
       out_file= vargs[7]
       if np.char.find(out_file,'.')< 1:
           out_file = str(np.char.add(out_file, '.pdf'))
       Sfig = True
       
   if Sfig:
       plt.savefig(out_file,format = 'pdf', dpi= 600)  
        
  



#%%  Mini test     with specific events using one unit
# This is useful for experimenting with different parameters to see how they
# affect the predicted firing rates of a particular neuron.
# Also useful for making predictions using all reps.
# The weights and their coefficients can be found with the iterative routines and then
# plugged in manually here to match the predicted algorithm used for the batch routine
start_scope()
# from brian2 import Synapses
# from Libs import get_ml_data_file, make_real_input_spikes
unit = 58  
#duration = 1250   #was 2000
num_inputs = 90
#num_units = len(r_all[0])
num_units = len(r_all)
num_directions=16
dir_range= np.linspace(0,2*np.pi,num_directions)

 
w1m = weight1_multi     
w2m = weight2_multi
w3m = weight3_multi
w4m = weight4_multi


w1w = w1m[range(unit,len(w1m),num_units)]  #unwrap vector
w2w = w2m[range(unit,len(w2m),num_units)]  #unwrap vector
w3w = w3m[range(unit,len(w3m),num_units)]  #unwrap vector
w4w = w4m[range(unit,len(w4m),num_units)]  #unwrap vector
w1s = np.array(eng.smooth(ml.double(w1w),ml.double(10),'sgolay')).flatten()
w2s = np.array(eng.smooth(ml.double(w2w),ml.double(10),'sgolay')).flatten()
w3s = np.array(eng.smooth(ml.double(w3w),ml.double(10),'sgolay')).flatten()
#w4s = np.array(eng.smooth(ml.double(w4w),ml.double(5),'sgolay')).flatten()
# w_array = np.concatenate((w1s,w2s,w3s),axis=0)
# w_max = np.max(w_array)

# w1c = w1w-np.mean(w1s)
# w2c = w2w- np.mean(w2s)
# w3c = w3w-np.mean(w3s)

w1c = w1w-np.mean(w1s)   #mean center the weights manually
w2c = w2w- np.mean(w2s)
w3c = w3w-np.mean(w3s)




# Set thresholds to account for output units with small weights
# Note that the threshold can be found with the separate iterative routine 
buf_size = num_inputs * num_units
#th = np.zeros(num_units )

#for un in range(num_units):
# if w_max > .7:        
#     # th[un] =.5
#     th =.5
# else:
#     th = 0


SpkMon=[]
StMon= []
SpkCnt=[] 
SpkTimes= []
SpkPot= []
SpkRates= []


H_eqs= '''
dv/dt = -v/tau :1
tau:second
'''


SpkTimes=[] #output across directions for each unit
SpkCnt=[]
SpkPot= []
SpkIndices=[]



ndex = 0

for direction in dir_range:

    inp1_group=[]
    inp2_group=[]
    inp3_group=[]
    inp4_group=[]
    SpkMonr= []
    SpkIndicesr=[]
    SpkCntr= []
    StMonr= []
    RM = []
    #print ('Direction start', direction)

    
    
    for rep in range(rep_cnt[ndex]):    
    #for rep in range(21,41):
    #for rep in range(21,26):  
    #for rep in range(21,22):
        start_scope()
        #start = time.time()
        
     
        inp1_group = SpikeGeneratorGroup(num_inputs,inp_indices[0][ndex][rep],inp_spikes[0][ndex][rep])
        inp2_group = SpikeGeneratorGroup(num_inputs,inp_indices[1][ndex][rep],inp_spikes[1][ndex][rep])
        inp3_group = SpikeGeneratorGroup(num_inputs,inp_indices[2][ndex][rep],inp_spikes[2][ndex][rep])
        inp4_group = SpikeGeneratorGroup(num_inputs,inp_indices[3][ndex][rep],inp_spikes[3][ndex][rep])

        
        #H= NeuronGroup(1,H_eqs,threshold = 'v > 100000000', reset = 'v = -10 ',  method = 'exact')   #.08,-.3 #  0 -1 #-.006 -.5
        ###H= NeuronGroup(1,H_eqs,threshold = 'v > 0.3', reset = 'v = -10 ',  method = 'exact')  # good for no_ndd, no_noise
        H= NeuronGroup(1,H_eqs,threshold = 'v > .475 ', reset = 'v = -10',  method = 'exact')  #-.06
        #H= NeuronGroup(1,H_eqs,threshold = 'v > 2   ', reset = 'v = -10',  method = 'exact') #1.5
        ##H= NeuronGroup(1,H_eqs,threshold = 'v > .3   ', reset = 'v = -10',  method = 'exact')  #-.06

        H.tau= 10*ms
        
        
        
       
        
      
        SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += 0.081429*w')
        #SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += 0.0000001*w')
        SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += 0.03*w')
        #SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += 0.0000001*w')
        SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += 0.03*w')    
        #SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += 0.0000001*w')
        SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += .001*w')   #works for Monkey N on 7/25
        #SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += .0000001*w')
        
        SS1.connect(i=np.arange(num_inputs),j=0)
        SS2.connect(i=np.arange(num_inputs),j=0) 
        SS3.connect(i=np.arange(num_inputs),j=0)
        SS4.connect(i=np.arange(num_inputs),j=0)
      
        
        SM=SpikeMonitor(H)
        M = StateMonitor(H, variables=True, record=True)
       
       
       
       
        SS1.w= w1w
        SS2.w= w2w
        SS3.w= w3w
        SS4.w= w4w
        
        
        run(duration*ms)
        freq = np.zeros((duration,1))
        #timestep = 0.005
        #np.add.at(freq,(np.array(SM.t/(5*ms),dtype = int),SM.i),1)
        #freq /= timestep
        SpkMonr.append(SM.t) 
        SpkIndicesr.append(SM.i)
        SpkCntr.append(SM.count)
        StMonr.append(M.v)
        
        print('\r Rep = ',rep, end = ' ' )
      
    SpkTimes.append(SpkMonr)
    SpkIndices.append(SpkIndicesr)
    #out_tmp_indices.append(SpkIndices)
    SpkCnt.append(SpkCntr)
    SpkPot.append(StMonr)
    #SpkRates.append(RM)
    ndex +=1


    print('\n Finished Direction ', ndex)
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
    
    
    
    inputs =  gaussian_input_speed(duration,direction,num_neurons,center,width,speed,speed_points,speed_lag)# uses a speed-dependent gain 
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
def make_input_spikes_speed(duration,num_neurons,speed,speed_points,speed_lag):
    #Makes the non-directional input spike trains based on the speed profile
    
    
    start_scope()
    
    #vr = -70*mV 	# resting potential level
    # vt = -55*mV		# threshold for firing of action potential #-55
    # vt = -.05*volt
    # vr = -.12*volt
    # vr_in = -.12
    # vt = .07*volt
    # vr =  0*volt
    # vr_in = 0
    vt =.05001*volt
    vr =  .048*volt
    vr_in = .03
    
    # n_sigma = 0.09*(vt-vr)	# 0.05; sigma of noise added to membrane potential
    n_sigma = 5*(vt-vr)
    
    inputs =  make_ndd(duration,num_neurons,speed,speed_points,vr_in,speed_lag) 
    trans_inputs = np.transpose(inputs) *volt
    # v_drive = TimedArray(trans_inputs*1.5*(vt-vr) + .86*(vt-vr), dt=defaultclock.dt)#increased gain from .3 to .6  and offset was .7
    # v_drive = TimedArray(trans_inputs*.7*(vt-vr) + .8*(vt-vr), dt=defaultclock.dt)# gain was 1  and offset was .86
    v_drive = TimedArray(trans_inputs, dt=defaultclock.dt)
    
   
    eqs = '''
      dv/dt = (v_drive(t,i)-v)/tau*int(not_refractory)+n_sigma*((tau*.5)**-0.5)*xi*int(not_refractory) : volt
    tau:second 
    '''
    
    
    
    all_indices=[]
    all_occ=[]  
    # G = NeuronGroup(num_neurons, eqs, threshold='v>vt', reset= 'v= -10*volt', refractory=5*ms, method='euler')
    # G = NeuronGroup(num_neurons, eqs, threshold='v>vt', reset= 'v= -.001*volt',refractory=5*ms,  method='euler')
    G = NeuronGroup(num_neurons, eqs, threshold='v>vt', reset= 'v= -.2*volt',refractory=2*ms,  method='euler')
    
    
   
    G.v= vr
    G.tau= 10 *ms

    SM = SpikeMonitor(G,record= True)
    M = StateMonitor(G, variables=True, record=True)

    # run the simulation for specified duration
    run(duration*ms)


    
    all_indices=SM.i;
    all_occ= SM.t

    return(all_indices,all_occ) 
    
def make_ndd(duration,num_neurons,speed_profile,speed_points,vr,speed_lag):
    #makes the driver for the non-directional input neurons
 
     #vr is the background level
     
     t_recorded = np.arange(int(duration*ms/defaultclock.dt))*defaultclock.dt
     num_pts = len(t_recorded)

     coeffs= np.linspace(-.5,1,num_neurons)
     

     speed_lag *= 10  #  lead time for non-directional component Moran and Schwartz


        
     speed_all = np.ones(num_pts)*vr
     sp_time_points = np.array(np.round(speed_points*10),dtype= int)
     xx = np.arange(sp_time_points[0],sp_time_points[-1])
     sp_buffer = np.zeros(len(sp_time_points))
     sp_buffer = np.interp(xx,sp_time_points,speed_profile)
     outbuf= []
     
     ndex = 0  
     for stime in range(xx[0]-speed_lag,xx[-1]-speed_lag):
          speed_all[stime]=sp_buffer[ndex]
          ndex +=1
     
     for unit in range(num_neurons):
         product =  np.ones(num_pts)*vr
         speed_out= np.zeros(num_pts)
         for stime in range(xx[0]-speed_lag,xx[-1]-speed_lag):
             speed_out[stime] = speed_all[stime]*coeffs[unit]
         outbuf.append(product+speed_out) 
              

         
     return(outbuf)

    

def make_individual_input_spikes(num_targets,rep_in,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks):
# Makes the four input groups with 90 neurons in each group
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

            center3 = end_movement-int3  #was 80
           
           
            width1 = gauss_sigma[0]*300    # original *1000
            width2 = gauss_sigma[1]*500
            width3 = gauss_sigma[2]*300
            
            
            
            
            
           
           
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
def auto_offset_new(unit,epoch,num_reps,inp_indices,inp_spikes):    
    start_scope()
    
    num_units = 65 
    num_inputs = 90  
    mdex=np.zeros(1000)
    
    spkstruct = sio.loadmat('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkCallspikes.mat')
    r_all = spkstruct['spk_all']
    
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
   num_units = len(r_all)
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
           ndex +=1
           #print('Target ',target)
       #print ('Rep = ',rep)
   new_wt = SSS.w 
   mean = np.mean(new_wt)
   print ('Returning Mean 1 %4.3f' % (mean))

   return(mean,new_wt)     

#%% Testing of network--- production version multi-unit with specific weight steps and thresholds
#param comes from the optimization routine where param[0:2] ar the weight_steps for the three epochs and param[3] is the optimal threshold


start_scope()


num_neurons = 90
num_units = len(r_all)
num_directions=16  
dir_range= np.linspace(0,2*np.pi,num_directions)
xxx= range(0,num_neurons)

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
    
    
    
    # if amp[un]< 20:
    #     th[un]= 0
    #     step[un]=.03
    #     print('Unit', un, 'had a low amplitude', amp[un])        
    # else:
    #     th[un]= .6
    #     step[un]= .005
   
    w1w= w1m[range(un,len(w1m),num_units)]*step[un][0]  #unwrap vector
    w2w= w2m[range(un,len(w2m),num_units)]*step[un][1]  #unwrap vector
    w3w= w3m[range(un,len(w3m),num_units)]*step[un][2]  #unwrap vector
    w4w = w4m[range(un,len(w4m),num_units)]  #unwrap vector
    # (slope[un],b)= np.polyfit(xxx,w4w,1)
    # if slope[un]<.12: 
    #            uth[un]= .02 
    #            print('Unit', un, 'had a low slope', slope[un])   
    # else:
    #            uth[un]= .3  

    
    ndex = 0
    odex = 0
    pdex = 0
       
    for i in range(un,len(weight1_multi),num_units):  #wrap back into vector
        w1[i] = w1w[ndex]
        w2[i] = w2w[ndex]
        w3[i] = w3w[ndex]
        ndex = ndex+1


buf_size = num_neurons * num_units



   

#     if slope[un]<.12:#.00025, .09, .11
#         th[un]= 0  #.065, .1,0, -.2 works well for Monk C
#         print('Unit', un, 'had a low slope', slope[un])   
#     else:
#         th[un]= .6   #.1,.2,.6   .4 works well for Monk C
# #    th[un]= .1


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
for direction in dir_range:
    SpkMonr=[] 
    SpkCntr=[]
    StMonr= []
    inp1_group=[]
    inp2_group=[]
    inp3_group=[]
    inp4_group=[]
    
    
    rep_start = 21 
    rep_end = 41
    # for rep in range(rep_cnt[ndex]):
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
        
        
        
   
        SS4 = Synapses (inp4_group,H,'w : 1', on_pre= 'v_post += 0.001*w') #.05
        SS1 = Synapses (inp1_group,H,'w : 1', on_pre= 'v_post += w')  #.1
        SS2 = Synapses (inp2_group,H,'w : 1', on_pre= 'v_post += w') #.1
        SS3 = Synapses (inp3_group,H,'w : 1', on_pre= 'v_post += w') #.1
    
        
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
    
            

            pspikes = np.array(pred_spikes[target][repcounter],dtype=float)*1000  #Note this is for using predictions from one particular unit (e.g. 57)
          
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



def make_STA(unit,target,reps,events,spk_pot,pred_spikes,epoch,gauss_center):
    #Find the 'buildup' spike accross all input groups
     width = np.zeros(3)
     center = np.zeros(3)
     width[0]= 42
     width[1]= 60 #was 200, narrow was 60,wide = 150
     width[2]= 60
 
   
     mev = np.mean(events,1)
     mtarget_show = mev[target][2]
     mstart_movement = mev[target][6]
     mpk_speed_time= mev[target][11]
     mreward = mev[target][5]
     mend_movement = mev[target][9]
     menter_target = mev[target][8]
     int1 = (mstart_movement-gauss_center[0])*1000
     int2 = (mpk_speed_time-gauss_center[1])*1000
     int3 = (mend_movement-gauss_center[2])*1000
    
    
     
     window_size = 20 #msec
     win_accum = np.zeros(window_size*10) #10th of a msec resolution
     win_pot = np.zeros(window_size*10)
     
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
         #print('IT',it,'rsingle\n')
     #print('Num Reps=',num_reps) 
     t_events = []
     for irep in range(r_start,r_end):
         try:
             len(events[0][0])
             t_events.append(events[target][irep])
         except:
             print('Problem with events')

     


    
    
     repcounter = 0
     num_spikes = 0
     
     
     for rep in range(r_start,r_end):
    
           
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
            

            pspikes = np.array(pred_spikes[target][repcounter],dtype=float)*1000  #Note this is for using predictions from one particular unit (e.g. 57)
          
            num_pspk = len(pspikes)  #number of predicted spikes in the sample
       
           
            
            for ps in range(num_pspk):   # cycle through all the predicted spikes
                t_point = pspikes[ps]  #look at each spike of the modeled unit 
                if t_point > epoch_start and t_point <= epoch_end:   #find the predicted spikes in the epoch of interest
                    potential = spk_pot[target][rep].flatten()
                    w_start = int(np.round((t_point-window_size)*10))+1
                    w_end = int(np.round(t_point*10))+1
                    win_pot= potential[w_start:w_end]
                    win_accum += win_pot
                    num_spikes +=1
                   

              
                
            repcounter += 1
     print ('Total spikes= ',num_spikes)
     STA = win_accum/num_spikes
     return(STA)
 
#%% Display input spike rates for each epoch,  A specific input can be specified

# # sspikes = []
# # inp_spike_number = 74
# # mode = 'predicted'
# # num_reps= 5
# # for epoch in range(3):
# #     star = []
# #     for dtarget in range(16):
# #         rspks = []
# #         for drep in range(num_reps):
# #                 rspks.append(inp_spikes[epoch][dtarget][drep][inp_indices[epoch][dtarget][drep]==inp_spike_number])
# #         star.append(rspks)
# #     sspikes.append(star)
        
# # from matplotlib import rc   #this is needed for CorelDraw to read the fonts correctly
# # rc("pdf", fonttype=42)
# # custom_colormap= sio.loadmat('C:/Users/Andrew/Documents/MATLAB/work/Programs/rgbColorMap.mat')
# # colors = custom_colormap["rgbColors"]

# # plt.figure()
   

# ax= plt.gca()
# ax.set_prop_cycle(plt.cycler('color', colors)) 
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# [histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[0],events,num_reps,mode)
# plt.plot(histo, lw = 2)
 
# # [histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[1],events,num_reps,mode)
# # plt.plot(histo,lw = 2)
 
# # [histo,xax_labels,mean_ev] =  make_norm_histos(sspikes[2],events,num_reps,mode)
# # plt.plot(histo,lw = 2)

# plt.ylabel ('Firing Rate   ')
# plt.xlabel ('Time (ms)')
# ymax =71
# plt.axis((0,35,0,ymax))
# ylabels = np.arange(0,ymax+1, 5)
# plt.yticks(ylabels) 
# xlabels= np.rint(xax_labels[0:len(xax_labels)+1:5]*1000).astype(int)
# xtplaces= np.arange(0,len(xax_labels),5)

# plt.xticks(ticks=xtplaces,labels=xlabels)
# mvmt_onset_bin = ((mean_ev[6]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1]
# #print ('movement_onset',(mean_ev[6]-mean_ev[2])*1000,'onset bin=',mvmt_onset_bin)
# peak_speed_bin= ((mean_ev[11]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1]
# target_acq_bin= ((mean_ev[9]-mean_ev[2])*1000/xlabels[-1])*xtplaces[-1] 
# plt.plot(mvmt_onset_bin,.5,'*')
# plt.plot(peak_speed_bin,.5,'*')
# plt.plot(target_acq_bin,.5,'*')
# # plt.savefig('Input spike rates 3 epochs',format = 'pdf', dpi= 600)  
def magnitude(vector): 
    import math
    return math.sqrt(sum(pow(element, 2) for element in vector))    
def permute(in_array,order):
    out_array = np.squeeze(np.transpose(np.expand_dims(np.array(in_array,dtype=object),axis=2,),(order)))
    return(out_array) 

#%% Example of how to go through the Epochs to get causal percents
EPer3_0 = []
EPer3_1 = []
EPer3_2 = []
EPer3_3 = []
for t in range(16):
   (accum0,accum1,accum2,accum3,percents0,percents1,percents2,percents3,out_spikes,win0_accum,win1_accum,win2_accum,win3_accum)= spike_cause_count_v3(t,reps,events,predicted_spikes,inp_spikes,inp_indices,1,gauss_center) 
   EPer3_0.append(percents0)
   EPer3_1.append(percents1)
   EPer3_2.append(percents2)
   EPer3_3.append(percents3)
sdata = {'Epoch1_0_winpercents':Ewin1_0, 'Epoch1_1_winpercents':Ewin1_1, 'Epoch1_2_winpercents':Ewin1_2,'Epoch1_3_winpercents':Ewin1_3 }
savemat("C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkC_causal_winpercents_epoch1[Jan26].mat",sdata)
def simple_regress(x, y):
    # c.f. geeks for geeks
    n = np.size(x)
    
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    return (b_0, b_1)