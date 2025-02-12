# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:59:08 2023

@author: Andrew
"""

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
from scipy.signal import savgol_filter

#%%  Mini test     with specific events using one unit
#%%  Mini test     with specific events using one unit
# This is useful for experimenting with different parameters to see how they
# affect the predicted firing rates of a particular neuron.
# Also useful for making predictions using all reps.
# The weights and their coefficients can be found with the iterative routines and then
# plugged in manually here to match the predicted algorithm used for the batch routine
start_scope()

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
    
    # example:  to plot histogram of rep 21:
    #    plthist(SpkTimes,events,[21,22],'predicted')
    
 #%% Build a raster for one epoch and one target and one rep of all input units 

plt.figure()
ttimes = []  
for tt in range(90):
    epochs = []
    #epochs= np.concatenate((tinp_spikes[0][ttarget][trep][tinp_indices[0][ttarget][trep]==tt],tinp_spikes[1][ttarget][trep][tinp_indices[1][ttarget][trep]==tt],tinp_spikes[2][ttarget][trep][tinp_indices[2][ttarget][trep]==tt]), axis = 0)
    #epochs =tinp_spikes[1][ttarget][trep][tinp_indices[1][ttarget][trep]==tt] 
    epochs= np.concatenate((tinp_spikes[0][ttarget][trep][tinp_indices[0][ttarget][trep]==tt],tinp_spikes[1][ttarget][trep][tinp_indices[1][ttarget][trep]==tt],tinp_spikes[2][ttarget][trep][tinp_indices[2][ttarget][trep]==tt],tinp_spikes[3][ttarget][trep][tinp_indices[3][ttarget][trep]==tt]), axis = 0)


    ttimes.append(epochs)
    #ttimes.append(tinp_spikes[tinp_indices==tt])
    num_spikes = len(ttimes[tt]) 
    y = np.ones(num_spikes)*(tt+1)
    plt.plot(ttimes[tt],y,'.')

 #   example:
 #[tinp_indices,tinp_spikes]=make_individual_input_spikes(16,rep_in,events,duration,speed_all,speed_times,gauss_center,gauss_sigma,event_landmarks)
 #trep = 0
 #ttarget = 0
 
 #%%quick weight plot
num_units = len(r_all)   
unit = 41
w1 = weight1_multi[range(unit,len(weight1_multi),num_units)]  #unwrap vector
w2 = weight2_multi[range(unit,len(weight2_multi),num_units)]  #unwrap vector
w3 = weight3_multi[range(unit,len(weight3_multi),num_units)]  #unwrap vector
w4 = weight4_multi[range(unit,len(weight4_multi),num_units)]  #unwrap vector
plt.figure()
plt.plot(w1)
plt.plot(w2)
plt.plot(w3)   
plt.plot(w4)
#%% Read Best Values from server batch
#with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkC_MinimumRMSE_Scale_Threshold_All_Units_Aug1.npy', 'rb') as f:
with open('C:/Users/Andrew/Documents/MATLAB/work/Programs/MonkN_MinimumRMSE_Scale_Threshold_All_Units_v2.npy', 'rb') as f:
    BestValues = np.load(f)
    
steps=BestValues[:,0]
thresh= BestValues[:,1]

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