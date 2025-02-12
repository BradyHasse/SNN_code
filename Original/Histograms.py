# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:43:22 2023

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
        rates = np.zeros(num_bins)#if no spikes, the rate is 0
    else:
        spikes = np.hstack((np.min(spikes) - 500, spikes, np.max(spikes) + 500))#add spikes far in the past and future.
        
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

def smooth(in_array,num_pts):
    s_array = savgol_filter(in_array,num_pts,3)
    return(s_array)

