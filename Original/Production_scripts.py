# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 11:11:29 2023

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

#%%Get Data  for Monkey C or N

#spkstruct = sio.loadmat('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkCDataSelected.mat')
spkstruct = sio.loadmat('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkN359Selected.mat')

all_units = spkstruct['spk_all']

r_all = all_units
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


#%%%%   Read or Save input spikes (See Below)



#%% Save input spike array

with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkN_input_spikes[Aug3].npy', 'wb') as f:    
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkC_input_spikes[July30].npy', 'wb') as f:

    num_sources = len(inp_spikes)
    num_targets = len(inp_spikes[0])
    num_reps = rep_cnt
    np.save(f,num_sources)
    np.save(f,num_targets)
    np.save(f,num_reps)
    for i in range(num_sources):
        for j in range(num_targets):
            #np.save(f,len(inp_spikes[i][j]))
            for k in range(num_reps[j]):                
                np.save(f,np.array(inp_spikes[i][j][k]))
print('Finished')

          
#%% Save input indices array
num_sources = len(inp_spikes)
num_targets = len(inp_spikes[0])
num_reps = rep_cnt

with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkN_input_indices[July29].npy', 'wb') as f:
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkC_input_indices[July30].npy', 'wb') as f:

    np.save(f,num_sources)
    np.save(f,num_targets) 
    np.save(f,num_reps)
    for i in range(num_sources):
        for j in range(num_targets):
            for k in range(num_reps[j]):                
                np.save(f,np.array(inp_indices[i][j][k]))
                
                
#%%  Read input indices array
input_indices=[[[]for j in range(16)] for i in range(1)]
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkC_input_indices[July30].npy', 'rb') as f:
with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkN_input_indices[July29].npy', 'rb') as f:
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
            



#%%  Read input spike array

rep_num = []
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs//MonkC_input_spikes[July30].npy', 'rb') as f:
with open('C:/Users/Andrew/Documents/MATLAB/work/Programs//MonkN_input_spikes[July29].npy', 'rb') as f:

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
                
#%%  Read weights
with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkC_weights[Aug1].npy', 'rb') as f:
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkN_weights[Aug3].npy','rb') as f:    
         
    weight1_multi = np.load(f)
    weight2_multi = np.load(f)
    weight3_multi = np.load(f)    
    weight4_multi = np.load(f)


#%%  A script that does the training with parameters in the environment- uses individual offset to train on all units from monkey C or N
#Get stored offsets that came from auto_off_new and simrun
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/final_offsets_MonkC_July30_20reps.npy', 'rb') as f:

    with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/final_offsets_MonkN_July25_20reps.npy', 'rb') as f:

     final_offsets = np.load(f)
 
 
       
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




weight1_multi= []
weight2_multi= []
weight3_multi= []
weight4_multi= []

W1_previous = .002
W2_previous=  .002
W3_previous = .002
W4_previous =.002   #initial weights






start_scope() 

for rep in range(20):   #Train on the first 20 reps

    for target in range(16):

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
       
        run(duration*ms)
        
        W1_previous = S1.w
        W2_previous = S2.w
        W3_previous = S3.w
        W4_previous = S4.w
    
        
    print('Rep ',rep)    
    


weight1_multi= S1.w
weight2_multi= S2.w
weight3_multi= S3.w
weight4_multi= S4.w

    
print('Finished')     




#%% Read Best Values from server batch-    Needed to optimize testing phase
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkC_MinimumRMSE_Scale_Threshold_All_Units_Sept28_Rep19.npy', 'rb') as f:
with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkN_MinimumRMSE_Scale_Threshold_All_Units_Oct2_Rep19.npy', 'rb') as f:
    BestValues = np.load(f)
    
steps=BestValues[:,0]
thresh= BestValues[:,1]

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

#%% calculate predicted-actual correlation for all units

correlation=[]
for unit in range(num_units):
 
    print("Unit", unit)
    [pred_hist,b,c] =  make_norm_histos_short(out_all_spikes[unit],events,reps,'predicted')

    actual_spikes= np.squeeze(r_all[unit])
    [actual_hist,d,e] =  make_norm_histos_short(actual_spikes,events,reps,'actual')
    pred_hist = np.transpose(pred_hist)

    actual_hist = np.transpose(actual_hist)
    predicted = []
    actual = [] 
    pred = []
  
    num_bins = len(pred_hist[0])
    num_targets = len(actual_spikes)
    for target in range(num_targets):
            pred = pred_hist[target][5:num_bins-4] 
            act = actual_hist[target][5:num_bins-4]
            predicted=np.concatenate((predicted,pred),axis = 0)
            actual=np.concatenate((actual,act),axis = 0)
    mag_prod = magnitude(actual)*magnitude(predicted)
    if mag_prod != 0:   
        correlation.append(np.dot(actual,predicted)/mag_prod)
    else:
        correlation.append(0)

    print('Finished Unit ',unit, 'r= ',correlation[unit])        

correlation = np.array(correlation,dtype= float).flatten()


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
 


