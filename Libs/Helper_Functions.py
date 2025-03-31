# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:41:59 2024

@author: BAH150
"""

"""
Helper_Functions Module
-------------------------
This module contains user-defined functions used by production scripts,
input generation, and utility routines. Functions include numerical operations,
data smoothing, histogram creation, differential evolution, and more.
"""
#%%   Initialize the enviornment-  Make sure you are in the correct directory
import os
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from brian2 import ms
#%% magnitude
def magnitude(vector): 
    """
Compute the Euclidean norm of a vector.

Parameters:
    vector (iterable): A list or array of numbers.

Returns:
    float: The Euclidean norm of the vector.
"""
    return math.sqrt(sum(pow(element, 2) for element in vector))   

#%% simple_regress
def simple_regress(x, y):
    """
Perform a simple linear regression using least squares.

Parameters:
    x (array-like): Independent variable values.
    y (array-like): Dependent variable values.
    
Returns:
    tuple: (intercept, slope) of the regression line.
"""
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

#%% permute
def permute(in_array,order):
    """
Permute the axes of an array and squeeze extra dimensions.

Parameters:
    in_array (array-like): Input array.
    order (tuple or list): Order of axes to transpose.
    
Returns:
    np.ndarray: Permuted and squeezed array.
"""
    return np.squeeze(
        np.transpose(
            np.expand_dims(
                np.array(
                    in_array,dtype=object
                    ),axis=2,
                ), (order)
            )
        )

#%% smooth
def smooth(in_array, span):
    """
Smooth a 1D array using the Savitzky-Golay filter.

Parameters:
    in_array (array-like): The input data to smooth.
    span (int): The length of the filter window (must be odd).
    
Returns:
    np.ndarray: Smoothed array.
"""
    try:
        return savgol_filter(in_array, span, 2)
    except:
        return in_array

#%% ABSERROR
def ABSERROR(v1, v2):
    """
    Calculate the absolute error between two vectors.
    
    Parameters:
        v1, v2 (array-like): Input arrays.
    
    Returns:
        float: Mean absolute error between v1 and v2.
    """
    return np.mean(abs(v1 - v2)) 

#%% RMSE
def RMSE(v1, v2):
    """
Compute the Root Mean Squared Error (RMSE) between two arrays.

Parameters:
    v1, v2 (np.ndarray): Input arrays (must be broadcastable).
    
Returns:
    np.ndarray: RMSE computed along specified axes.
"""
    return np.sqrt(np.mean(np.square(v1 - v2),axis = (1, 2)))

#%% make_weights
def make_weights(weights1_multi, weights2_multi, weights3_multi, weights4_multi):
    """
    Combine four sets of weights into a list.
    
    Parameters:
        weights1_multi, weights2_multi, weights3_multi, weights4_multi: 
            Arrays representing different weight sets.
    
    Returns:
        list: A list containing all weight arrays.
    """
    return [weights1_multi, weights2_multi, weights3_multi, weights4_multi]

#%% bin_frac2
def bin_frac2(spikes, a_time, a_b_time, bin_width): 
    """
    Calculate the spiking rate using fractional intervals.
    
    This function computes the firing rate in bins given a vector of spike times.
    It handles fractional contributions from intervals that do not align perfectly
    with the bin boundaries.
    
    Parameters:
        spikes (array-like): Sorted spike times (in seconds).
        a_time (float): Start time of the analysis period.
        a_b_time (float): End time of the analysis period.
        bin_width (float): Width of each bin.
    
    Returns:
        np.ndarray: Spike rate in each bin.
    """
    
    num_bins= int(np.floor(((a_b_time - a_time)/bin_width)+.0001))#sets the number of bins
    bin_width = (a_b_time - a_time)/num_bins#if it was not exact, makes bin_width correct.
    spikes=np.unique(np.sort(spikes.flatten()))#sort the spikes in order and remove any duplicates.
    
    if spikes.size == 0:
        return np.zeros(num_bins) #if no spikes, the rate is 0

    spikes = np.hstack((np.min(spikes) - 500, spikes, np.max(spikes) + 500)) #add spikes far in the past and future.
    
    spikes = spikes - a_time
    a_b_time = a_b_time - a_time
    a_time=0
    spikes = spikes*(num_bins/a_b_time)
    a_b_time = num_bins
    
    ind_s = np.nonzero(spikes < a_time)[0][-1]#first spike index
    ind_e = np.nonzero(spikes > a_b_time)[0][0] + 1
    spikes = spikes[ind_s:ind_e]#from spike preceeding a_time to spike after a_b_time
    dspikes = np.diff(spikes)#intervals between spikes
    intervals = np.zeros(num_bins)
    
    for i in  range(dspikes.size):#assign each interval between spikes to bins.
        j = np.arange(np.floor(spikes[i]) + 1, np.floor(spikes[i + 1]) + 0.1)#added 0.1 to get last number included
        divs = np.sort(np.hstack((j, spikes[np.array([i,i + 1])])))#time of two spikes creating dspikes and the whole numbers  between them
        int_share = np.diff(divs) * (1/dspikes[i])#each full unit is worth (1/dspikes(i))
        int_inds = np.ceil(divs[1:]) - 1#indcies of bin in interval where int_shares get assigned to
        # Remove indices outside valid range or zero shares
        rm_inds = (int_inds<0) + (int_inds>(a_b_time - 1)) + (int_share==0)#indicies to remove because they belong to unassined bins
        int_share = int_share[~rm_inds]#remove those shares
        int_inds = int_inds[~rm_inds]#remove those shares
        intervals[(np.rint(int_inds)).astype(int)] = intervals[(np.rint(int_inds)).astype(int)] + int_share#assign portion of spike to the correct bin(s)
     
    rates = intervals/bin_width
    return rates

#%% plthist
def plthist(*args):
    """
    Plot contributions of input units using percent contributions.
    
    Parameters:
        args: A variable number of arguments. The first three arguments
              should be the firing rates, events, and repetition indices.
              Additional arguments may specify display targets or output filename.
              
    Note:
        The function saves the plot as a PDF if an output filename is provided.
    """
    from matplotlib import rc  
    rc("pdf", fonttype=42) # Ensure proper font embedding for PDFs
    
    # Load a custom colormap from a MAT file.
    code_dir = os.getcwd()
    custom_colormap = sio.loadmat(os.path.join(code_dir, 'Data', 'rgbColorMap.mat'))
    colors = custom_colormap["rgbColors"]
    
    num_args = len(args)   
    plt.figure()
    ax= plt.gca()
    ax.set_prop_cycle(plt.cycler('color', colors)) 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    rates= args[0]
    events = args[1]
    reps = args[2]
    num_segs = args[3] 
    
    Sfig = False
    num_targets = len(rates)
    y_max = 71
    plt.axis((0, 35, 0, y_max))
    [histo, xax_labels, mean_ev] =  make_norm_histos(rates, events, reps, num_segs)
    
    #print('Number of arguments= ',num_args)
    
    if len(histo[0]) != num_targets:
        try:
            histo = np.transpose(histo)
        except:
            histo = histo
      
    if num_args == 4:
        plt.plot(histo, lw = 2)
    else:
        try:
            disp_target = int(args[4])
            plt.plot(histo[:, disp_target],'k')
        except:
            plt.plot(histo,lw = 2)
            if num_args == 5:
                out_file = args[4]
                if np.char.find(out_file,'.') < 1:
                    out_file = str(np.char.add(out_file, '.pdf'))  
                Sfig = True
                
    plt.ylabel ('Firing Rate   ')
    plt.xlabel ('Time (ms)')
    y_labels = np.arange(0, y_max + 1, 5)
    plt.yticks(y_labels) 
    x_labels= np.rint(xax_labels[0:len(xax_labels) + 1:5] * 1000).astype(int)
    xt_places= np.arange(0, len(xax_labels), 5)
    plt.xticks(ticks = xt_places, labels = x_labels)
    mvmt_onset_bin = ((mean_ev[6] -mean_ev[2])*1000/x_labels[-1])*xt_places[-1]
    peak_speed_bin = ((mean_ev[11]-mean_ev[2])*1000/x_labels[-1])*xt_places[-1]
    target_acq_bin = ((mean_ev[9] -mean_ev[2])*1000/x_labels[-1])*xt_places[-1] 
    plt.plot(mvmt_onset_bin,.5,'*')
    plt.plot(peak_speed_bin,.5,'*')
    plt.plot(target_acq_bin,.5,'*')
    
    if num_args==6:
      
      try:
          disp_target = int(args[5])
          plt.plot(histo[:,disp_target],'r',lw = 2)
      except:
          out_file= args[5]
          if np.char.find(out_file,'.')< 1:
              out_file = str(np.char.add(out_file, '.pdf'))
          Sfig = True
    
    if num_args== 7:
        try:
            disp_target = int(args[6])
            print('Target= ',disp_target,'\n',len(histo),len(histo[0]),len(histo[:, disp_target]))
            plt.plot(histo[:,disp_target],'b',lw =2 )
        except:
            out_file= args[6]
            if np.char.find(out_file,'.')< 1:
                out_file = str(np.char.add(out_file, '.pdf'))
            Sfig = True
    if num_args== 8:
        out_file= args[7]
        if np.char.find(out_file,'.')< 1:
            out_file = str(np.char.add(out_file, '.pdf'))
        Sfig = True
        
    if Sfig:
        plt.savefig(out_file,format = 'pdf', dpi= 600)  
        
#%% make_norm_histos
def make_norm_histos(in_array,events,reps,numsegs,o_binwidth=0.02,gauss_win = 0.025):    
# Use first three intervals only.  Stopping at end of movement    
    """
    in_array: list of spike times. list of targets, list of trials, np array of times in ms fo 1 particular neuron. 
    ex) 16list x 49list x (29x0)nparray
    events: np array of events targets x trials x events
    ex) 16x49x12 np array
    reps: reps you want to analyze from "events" should correspond to "in_array" 2-long list, saying starting and ending rep.
    ex) [20 40] 
    
    """
    try:
        in_array_ = np.array(in_array, dtype=object)
    except:
        for i in range(len(in_array_)):
            for j in range(len(in_array_[i])):
                in_array_[i][j] = np.squeeze(in_array_[i][j])
        
    num_targets = in_array_.shape[0]
    max_num_reps = in_array_.shape[1]
    
    if events.shape[1] != max_num_reps:
        raise Exception("dimensions mismatch between ""in_array"" and ""events")
    
    r_start = reps[0]
    r_end = reps[1]
    if reps[0] > reps[1]:
        r_start = reps[1]
        r_end = reps[0]
        reps = np.flip(reps)
    if r_end > max_num_reps:
        raise Exception("reps requested out of range for in_array")
        
    eventsOI = events[:, r_start:r_end, :]#events of interest
    mean_ev = np.mean(eventsOI,(0, 1))
    eventInds = [2, 6, 11, 9, 5]#target_show (2) start_movement (6) pk_speed (11) end_movement (9) reward (5)
    eventInds = eventInds[:(numsegs+1)]
    intervals = np.empty([np.size(eventsOI, 0), np.size(eventsOI, 1), len(eventInds)])
    nbins = np.empty([len(eventInds) - 1, 1])
    for i in range(len(eventInds) - 1):
        intervals[:,:,i] = eventsOI[:, :, eventInds[i+1]] - eventsOI[:, :, eventInds[i]]
        nbins[i] = np.round(np.mean(intervals[:, :, i])/o_binwidth)#calculate the average number of 20 ms bins in each interval

    total_bins = int(np.sum(nbins))#only interested in the first nbins bins
    # total_bins = int(np.sum(nbins))
    
    srates=np.zeros((num_targets, r_end - r_start, total_bins))
    
    for t in range(num_targets):
        for rep in range(r_start, r_end): 
            rates = np.empty(0)
            rr= in_array[t][rep]
            stimes = np.array(rr, dtype=float) #converts from brain units (s) to unitless
            if stimes.size > 2:
                e_spk = np.squeeze(np.diff(stimes[-2:]))+stimes[-1]
                if events[t, rep, eventInds[len(nbins)]] > e_spk:
                    e_spk = events[t, rep, eventInds[len(nbins)]]
                stimes = np.append(stimes, e_spk)
            for i in range(len(nbins)):
                #target_show (2) start_movement (6) pk_speed (11) end_movement (9) reward (5)
                delta = intervals[t, rep - r_start, i]
                binwidth = delta/nbins[i]
                rates = np.append(
                    rates, bin_frac2(
                    stimes, events[t, rep, eventInds[i]],
                    events[t, rep, eventInds[i+1]],
                    binwidth))
            # sr = smooth(rates,9)
            sr = gaussian_filter(rates, gauss_win/o_binwidth, mode = 'nearest')
            # sr = rates#if you dont want to smooth
            srates[t][rep-r_start][:] = sr
  
    mr = np.transpose(np.mean(srates, 1))
    ev_edges = mean_ev[eventInds]-mean_ev[eventInds[0]]
    binwidths = np.diff(ev_edges)/np.transpose(nbins)
    timax = np.empty(0)
    for i in range(np.size(nbins)):
        timax = np.append(timax, np.tile(binwidths[0, i], int(nbins[i])))
    timax = np.cumsum(timax) - timax[0]
    #timax = np.interp(range(total_bins), np.append(0, np.cumsum(nbins[:3])), ev_edges)
    
    return(mr,timax,mean_ev) 

#%% make_norm_histos
def make_norm_histos_nbins(in_array, events, reps, numsegs, nbins=[], o_binwidth=0.02, gauss_win=0.025):    
    """
    Generates normalized histograms of spike rates based on event timing.

    Parameters:
    - in_array: list of spike times (targets x trials x numpy array of times in ms for one neuron).
      Example: 16 lists x 49 lists x (29x0) numpy array.
    - events: numpy array of event timings (targets x trials x events).
      Example: 16x49x12 numpy array.
    - reps: List specifying the range of repetitions to analyze from "events".
      Example: [20, 40].
    - numsegs: Number of segments to analyze.
    - nbins: Number of bins per segment (optional, defaults to empty list).
    - o_binwidth: Bin width in seconds (default: 0.02s).
    - gauss_win: Gaussian smoothing window size in seconds (default: 0.025s).

    Returns:
    - mr: Mean spike rate across trials.
    - timax: Time axis for plotting.
    - mean_ev: Mean event timings.
    - nbins: Number of bins per segment.
    """

    makebins = isinstance(nbins, list)
    
    # Convert input array to a numpy array (object type to allow variable-length lists)
    try:
        in_array_ = np.array(in_array, dtype=object)
    except:
        for i in range(len(in_array_)):
            for j in range(len(in_array_[i])):
                in_array_[i][j] = np.squeeze(in_array_[i][j])
    
    num_targets, max_num_reps = in_array_.shape[:2]

    if events.shape[1] != max_num_reps:
        raise ValueError("Dimension mismatch between 'in_array' and 'events'.")

    # Ensure correct order of reps
    r_start, r_end = sorted(reps)
    if r_end > max_num_reps:
        raise ValueError("Requested reps are out of range for 'in_array'.")

    # Extract events of interest
    eventsOI = events[:, r_start:r_end, :]
    mean_ev = np.mean(eventsOI, axis=(0, 1))

    # Event indices: target_show, start_movement, peak_speed, end_movement, reward
    eventInds = [2, 6, 11, 9, 5][:numsegs + 1]

    # Compute intervals
    intervals = np.empty((eventsOI.shape[0], eventsOI.shape[1], len(eventInds) - 1))
    if makebins:
        nbins = np.empty((len(eventInds) - 1, 1))

    for i in range(len(eventInds) - 1): 
        intervals[:, :, i] = eventsOI[:, :, eventInds[i + 1]] - eventsOI[:, :, eventInds[i]]
        if makebins:
            nbins[i] = np.round(np.mean(intervals[:, :, i]) / o_binwidth)  # Average number of 20ms bins

    total_bins = int(np.sum(nbins))  # Total number of bins
    srates = np.zeros((num_targets, r_end - r_start, total_bins))

    for t in range(num_targets):
        for rep in range(r_start, r_end): 
            rates = np.empty(0)
            stimes = np.array(in_array[t][rep], dtype=float)  # Convert from brain units to unitless

            if stimes.size > 2:
                e_spk = np.squeeze(np.diff(stimes[-2:])) + stimes[-1]
                if events[t, rep, eventInds[len(nbins)]] > e_spk:
                    e_spk = events[t, rep, eventInds[len(nbins)]]
                stimes = np.append(stimes, e_spk)

            for i in range(len(nbins)):
                delta = intervals[t, rep - r_start, i]
                binwidth = delta / nbins[i]
                rates = np.append(
                    rates,
                    bin_frac2(
                        stimes, 
                        events[t, rep, eventInds[i]],
                        events[t, rep, eventInds[i + 1]],
                        binwidth
                    )
                )

            sr = gaussian_filter(rates, gauss_win / o_binwidth, mode='nearest')
            srates[t, rep - r_start, :] = sr

    mr = np.transpose(np.mean(srates, axis=1))
    ev_edges = mean_ev[eventInds] - mean_ev[eventInds[0]]
    binwidths = np.diff(ev_edges) / np.transpose(nbins)

    timax = np.concatenate([np.full(int(nbins[i]), binwidths[0, i]) for i in range(len(nbins))])
    timax = np.cumsum(timax) - timax[0]

    return mr, timax, mean_ev, nbins

#%% make_histos
def make_histos(in_array,ev,mode):    
#simple one-trial histogram aligned to movement onset 
# events need to be specified events[target][rep]

    if mode == 'predicted':
        rr= in_array
        stimes = np.array(rr,dtype=float)
    else:                       #Actual
        rr = (np.squeeze(in_array))
        stimes= rr
    
    target_show = ev[2]
    reward = ev[5]      
    binwidth1 = float(.02)
    rates = bin_frac2(stimes,target_show,reward,binwidth1)
    # sm_rates = smooth(rates,9)
    sm_rates = gaussian_filter(rates,1, mode = 'nearest')
    return(sm_rates)
#%% w_steps_gen
def w_steps_gen(num_units, bounds, popsize):
    """
Generate an initial population of weight steps.

Parameters:
    num_units (int): Number of neurons/units.
    bounds (np.ndarray): Array of shape (num_params, 2) with lower and upper bounds.
    popsize (int): Population size per parameter.
    
Returns:
    np.ndarray: Population array of shape (popsize, num_units, num_params).
"""
    population = np.zeros([num_units, bounds.shape[0], popsize])
    for samp in range(num_units):
        n, d = (popsize,bounds.shape[0])
        rng = np.random.default_rng()
        samples = rng.uniform(size=(n, d))
        perms = np.tile(np.arange(1, n + 1), (d, 1))
        for i in range(d):
            rng.shuffle(perms[i, :])
        perms = perms.T
        population[samp,:,:] = (perms - samples).T / n
        
    scale = np.tile(np.expand_dims(np.transpose(np.diff(bounds)), 2), [num_units,1,popsize])
    offset = np.tile(np.transpose(np.expand_dims(bounds[:,0], [1,2]), [1,0,2]), [num_units,1,popsize])
    population = (population*scale)+offset
    population = np.transpose(population,[2,0,1])
    return population

#%% differential_evolution
def differential_evolution(results, Av_FR,bounds, prev_par, gamma):
    """
Perform one iteration of differential evolution.

Parameters:
    results (list): List of results from current population evaluation.
    av_fr (np.ndarray): Average firing rate used for fitness scaling.
    bounds (np.ndarray): Bounds for the parameters.
    prev_par (list): List of previous parent parameter sets.
    gamma (float): Weighting factor between 0 (random target) and 1 (best target).
    
Returns:
    tuple: (Updated previous parameters, List of new children parameters)
"""
    gamma = max(0, min(gamma, 1))
    rng = np.random.default_rng()
    beta = rng.random() * 0.5 + 0.5 # Mutation constant between 0.5 and 1
    CR = 0.7  # The recombination constant
    n_parnts = len(results)
    n_out =   results[0][1].shape[0]
    n_param = results[0][1].shape[1]
    bounds_range = np.diff(bounds)/2#the futher away the parameters are from 0, the higher the cost. 
    bounds_range = np.transpose(np.expand_dims(bounds_range,2),[2,1,0])
    bounds_range = np.tile(bounds_range,[n_out, n_parnts,1])
    
    # Evaluate fitness for new and previous generations
    fitness = []
    for j in range(2):
        par_set = results if j == 0 else prev_par
        rmse_neur = np.zeros([n_out, n_parnts])
        wsteps = np.zeros([n_out, n_parnts,bounds_range.shape[2]])
        RScoreneur = np.zeros(rmse_neur.shape)
        RScoreneur2 = np.zeros(rmse_neur.shape)
        
        for i in range(n_parnts):  
            rmse_neur[:,i] = (par_set[i][0]/Av_FR)*2-1#0 RMSE -> -1, double singnal -> 1
            wsteps[:,i,:] = par_set[i][1]
            RScoreneur[:,i] = par_set[i][2]
            RScoreneur2[:,i] = par_set[i][3]
            
        wsteps = np.abs(np.divide(wsteps,bounds_range))/100#not as important as r-score or rmse. 100x less important.
        wsteps = np.mean(wsteps,axis=2)
        fitness.append(np.copy(rmse_neur-RScoreneur2+wsteps))
    
    # Update parent parameters based on fitness
    for i in range(n_parnts):
        for j in range(n_out):
            new_fit = fitness[0][j,i]
            old_fit = fitness[1][j,i]
            if new_fit < old_fit:
                prev_par[i][0][j] = results[i][0][j]
                prev_par[i][1][j,:] = results[i][1][j,:]
                prev_par[i][2][j] = results[i][2][j]
                prev_par[i][3][j] = results[i][3][j]
                
    fitness[0][fitness[0]<fitness[1]] = fitness[1][fitness[0]<fitness[1]]
    fitness = fitness[0]
    fitness_argmin = np.argmin(fitness, axis=1)
    
    parents = np.zeros([n_parnts, n_out, n_param])
    for i in range(n_parnts):#create parent matrix
        parents[i,:,:] = prev_par[i][1]
    best_parents = parents[fitness_argmin,np.arange(parents.shape[1]),:]
    
    #generate new children
    children = np.zeros(parents.shape)
    cld_mask = rng.random(children.shape)<CR#mask for recombination
    unchanged = np.invert(np.any(cld_mask,axis=2))#unchanged parameters
    backup_inds = rng.integers(low=0, high=n_param, size=unchanged.shape)#index for recombination if there were no changes
    for i in range(n_parnts):
        for j in range(n_out):
            if unchanged[i,j]:#if there is an unchanged parent, set one parameter to change
                cld_mask[i,j,backup_inds[i,j]] = True
    
    for i in range(n_parnts):
        Pinds = np.arange(n_parnts)
        Pinds = np.delete(Pinds, i)
        Pinds = rng.choice(Pinds, 3, replace=False)
        targetVector = gamma*best_parents + (gamma-1)*parents[Pinds[0],:,:]
        children[i,:,:] = targetVector+beta*(parents[Pinds[1],:,:]-parents[Pinds[2],:,:])
        
    children[np.invert(cld_mask)] = parents[np.invert(cld_mask)]#recombination
    for i in range(n_param):#limit to bounds
        children[children[:,:,i]<bounds[i,0],i] = bounds[i,0]
        children[children[:,:,i]>bounds[i,1],i] = bounds[i,1]
    
    return prev_par, list(children)

#%% ready_make_out_all_spikes_par
def ready_make_out_all_spikes_par(reps, inp_indices, inp_spikes, num_units, weight_multi_3d, duration, params):
    """
Prepare inputs for the output spike generation function.

Parameters:
    reps (list): [rep_start, rep_end] specifying the repetitions to use.
    inp_indices (list): Input indices for each group.
    inp_spikes (list): Input spike data for each group.
    num_units (int): Number of neurons.
    weight_multi_3d (np.ndarray): 3D array of weights.
    duration (array-like): Duration values.
    params: Parameters for generating outputs.

Returns:
    tuple: (List of inputs, arguments for processing)
"""
    [rep_start, rep_end] = reps
    inps = []
    for rep in range(rep_start, rep_end):
        for target in range(len(inp_indices[0])):
            inp_ind_t = []
            inp_spk_t = []
            for group in range(len(inp_indices)):
                inp_ind_t.append(inp_indices[group][target][rep])
                inp_spk_t.append(inp_spikes[group][target][rep])
            inps.append([inp_ind_t, inp_spk_t, target, rep])
                
    args = [num_units, weight_multi_3d, duration, params]
    
    return inps, args
#%% score_run
def score_run(actual_hist, num_units, out_all_spikes_, events, reps, nbins):
    """
Compute the score (RMSE and correlations) between the actual and model-generated histograms.

Parameters:
    actual_hist (np.ndarray): Actual histogram data.
    num_units (int): Number of neurons.
    out_all_spikes_ (list): Model-generated spike outputs.
    events (np.ndarray): Event markers.
    reps (list): Repetition range.
    nbins: Number of bins for histogram computation.
    
Returns:
    tuple: (RMSE per neuron, correlation per neuron, global correlation)
"""
    PH = np.zeros(actual_hist.shape)
    
    for u in range(num_units):
        [PH[u,:,:],d,e,nbins] = make_norm_histos_nbins(out_all_spikes_[u],events[:,reps[0]:reps[1],:],list(np.array(reps)-reps[0]),4,nbins,o_binwidth=0.005) 

    PH = PH[:,20:-20,:]
    actual_hist = np.copy(actual_hist[:,20:-20,:])
    rmse_neur = RMSE(PH,actual_hist)
    
    correlation = np.zeros(rmse_neur.shape)#mean target vs target correlation for each unit
    correlation2 = np.zeros(rmse_neur.shape)#mean target vs target correlation for each unit
    for unit in range(num_units):
        inmask = np.logical_and(np.std(actual_hist[unit,:,:],axis=0)>10**-10, np.std(PH[unit,:,:],axis=0)>10**-10)#which targets have an std != 0
        if any(inmask):#else it corrlation equals 0 already
            sumin = sum(inmask)#how many targets have an std != 0
            CC = np.corrcoef(actual_hist[unit,:,inmask], PH[unit,:,inmask])
            correlation[unit] = np.sum(np.diagonal(CC[:sumin,sumin:]))/len(inmask)
            correlation2[unit] = np.corrcoef(np.ndarray.flatten(actual_hist[unit,:,:].T), np.ndarray.flatten(PH[unit,:,:].T))[1,0]
    correlation = np.nan_to_num(correlation, copy=False)
    return rmse_neur, correlation, correlation2

#%% differential_evolution
def CreateBestValues(prev_par,Av_FR):

    rmse_neur = np.zeros([len(prev_par[0][0]), len(prev_par)])
    rmse_neur2 = np.zeros(rmse_neur.shape)
    correlation2 = np.zeros(rmse_neur.shape)
    for i in range(len(prev_par)):  
        rmse_neur[:,i] = (prev_par[i][0]/Av_FR)*2-1
        rmse_neur2[:,i] = prev_par[i][0]
        correlation2[:,i] = prev_par[i][3]
    fitness = np.copy(rmse_neur-correlation2)   

    fitness_min = np.min(fitness, axis=1)
    fitness_argmin = np.argmin(fitness, axis=1)
    rmse_neur_min = rmse_neur2[np.arange(rmse_neur2.shape[0]), fitness_argmin]
    RScoreneur_min = correlation2[np.arange(correlation2.shape[0]), fitness_argmin]

    BestValues = np.zeros(prev_par[0][1].shape)
    for i in range(BestValues.shape[0]):
        BestValues[i,:] = prev_par[fitness_argmin[i]][1][i,:]
    BestValues = np.hstack([BestValues, np.expand_dims(rmse_neur_min,1), np.expand_dims(RScoreneur_min,1), np.expand_dims(fitness_min, 1)])
    return BestValues

#%% Create_BV_hist2
def Create_BV_hist2(BV_hist):

    rng = np.random.default_rng()
    BVhistnp = np.array(BV_hist)
    BVunique = []
    numBVs = []
    me_std = np.zeros([BVhistnp.shape[1], 2,5])
    for unit in range(BVhistnp.shape[1]):
        BVunit = BVhistnp[:,unit,:]
        uniqueW = np.unique(BVunit, axis=0)
        uniqueW = uniqueW[uniqueW[:, -1].argsort()]
        me_std[unit,0,:] = np.mean(uniqueW[:int(uniqueW.shape[0]/2), :5],axis=0)
        me_std[unit,1,:] = np.std(uniqueW[:int(uniqueW.shape[0]/2), :5],axis=0)
        BVunique.append(uniqueW[uniqueW[:, -1].argsort()])
        numBVs.append(uniqueW.shape[0])
    numBVs = np.array(numBVs)
    for unit in range(BVhistnp.shape[1]):
        uniqueW = BVunique[unit]
        randnums = rng.standard_normal([np.max(numBVs)-numBVs[unit],5])
        new_w = np.multiply(randnums,np.squeeze(me_std[unit,0,:]))+np.squeeze(me_std[unit,1,:])
        new_w = np.append(new_w, np.zeros([new_w.shape[0], 3]),axis = 1)
        BVunique[unit] = np.append(BVunique[unit], new_w,axis = 0)
    BVhist2 = np.transpose(np.array(BVunique),[0,2,1])
    
    return BVhist2
#%% spike_cause_count_v4
def spike_cause_count_v4(target,reps,events,pred_spikes,inp_spikes,inp_indices,epoch,gauss_center, winsiz):
     #Find the input spikes in an interval before each ouput spike within a specified epoch
     #epoch is based on 1 indexing - valid epochs are 1, 2, 3
     
     reps_s = np.array([len(pred_spikes[0]), len(inp_spikes[0][0]), len(inp_indices[0][0])])
     
     if ~np.all(reps_s == events.shape[1]):
         raise Exception("#reps mismatch between ""in_array"" and ""events")
     
     if reps[0] > reps[1]:
         reps = np.flip(reps)
     if reps[1] > events.shape[1]:
         raise Exception("reps requested out of range for in_array")
     
     numreps = int(np.diff(reps))
     pred_spikes_1 = pred_spikes[target][reps[0]:reps[1]]
     width = 60
   
     eventInds = np.array([6, 11, 9])#start_movement (6) pk_speed (11) end_movement (9)

     mev = np.mean(events[np.ix_([target],np.arange(events.shape[1]),eventInds)],1)
     ints = (mev-gauss_center)*1000
     
     t_events = events[np.ix_([target],np.arange(reps[0],reps[1]),eventInds)]
     ev = np.round(t_events*1000)
     ev = ev.astype(int)
     
     centers = np.squeeze(ev - np.tile(ints, [numreps, 1]))
     
     accum = np.zeros([90,len(inp_spikes)])
     
     out_sample= []
     for rep in range(numreps):
         pred_spikes_2 = np.round((np.array(pred_spikes_1[rep])*1000)*10)/10 #0.1ms resolution in simulator
         pred_spikes_2 = pred_spikes_2[(pred_spikes_2 > centers[rep, epoch-1]-width) & (pred_spikes_2 <= centers[rep, epoch-1]+width)]
         out_sample.append(pred_spikes_2)  
         for imp_g in range(len(inp_spikes)):#for all input groups
             inp_spikes_1 = np.round((np.array(inp_spikes[imp_g][target][rep+reps[0]])*1000)*10)/10 #0.1ms resolution in simulator
             inp_indices_1 = np.array(inp_indices[imp_g][target][rep+reps[0]])
             for p_spike in range(pred_spikes_2.size):
                 inp_spikes_2 = (inp_spikes_1 < pred_spikes_2[p_spike]) & (inp_spikes_1 >= pred_spikes_2[p_spike]-winsiz)
                 [imp_ind, imp_c] = np.unique(inp_indices_1[inp_spikes_2], return_counts=True)
                 accum[imp_ind,imp_g] = accum[imp_ind,imp_g] + imp_c
                 
     percents = accum/np.sum(accum)
             
     return(accum, percents, out_sample)
 
#%% spike_cause_FR_W
def spike_cause_FR_W(target,reps,events,pred_spikes,inp_spikes,inp_indices,epoch,gauss_center, winsiz,bintimes,histin):
     #Find the input spikes in an interval before each ouput spike within a specified epoch
     #epoch is based on 1 indexing - valid epochs are 1, 2, 3
     bintimes2 = np.copy(bintimes[target,reps[0]:reps[1]])
     histin2 = np.copy(histin[target,reps[0]:reps[1],:,:])
     reps_s = np.array([len(pred_spikes[0]), len(inp_spikes[0][0]), len(inp_indices[0][0])])
     
     if ~np.all(reps_s == events.shape[1]):
         raise Exception("#reps mismatch between ""in_array"" and ""events")
     
     if reps[0] > reps[1]:
         reps = np.flip(reps)
     if reps[1] > events.shape[1]:
         raise Exception("reps requested out of range for in_array")
     
     numreps = int(np.diff(reps))
     pred_spikes_1 = pred_spikes[target][reps[0]:reps[1]]
     width = 60
   
     eventInds = np.array([6, 11, 9])#start_movement (6) pk_speed (11) end_movement (9)

     mev = np.mean(events[np.ix_([target],np.arange(events.shape[1]),eventInds)],1)
     ints = (mev-gauss_center)*1000
     
     t_events = events[np.ix_([target],np.arange(reps[0],reps[1]),eventInds)]
     ev = np.round(t_events*1000)
     ev = ev.astype(int)
     
     centers = np.squeeze(ev - np.tile(ints, [numreps, 1]))
     
     accum = np.zeros([90,len(inp_spikes)])
     FRs = [ [] for _ in range(len(inp_spikes)) ]
     out_sample= []
     for rep in range(numreps):
         pred_spikes_2 = np.round((np.array(pred_spikes_1[rep])*1000)*10)/10 #0.1ms resolution in simulator
         pred_spikes_2 = pred_spikes_2[(pred_spikes_2 > centers[rep, epoch-1]-width) & (pred_spikes_2 <= centers[rep, epoch-1]+width)]
         out_sample.append(pred_spikes_2)  
         for imp_g in range(len(inp_spikes)):#for all input groups
             inp_spikes_1 = np.round((np.array(inp_spikes[imp_g][target][rep+reps[0]])*1000)*10)/10 #0.1ms resolution in simulator
             inp_indices_1 = np.array(inp_indices[imp_g][target][rep+reps[0]])
             for p_spike in range(pred_spikes_2.size):
                 if winsiz >= 0:
                     inp_spikes_2 = (inp_spikes_1 < pred_spikes_2[p_spike]) & (inp_spikes_1 >= pred_spikes_2[p_spike]-winsiz)
                 elif winsiz < 0:#looking for input spikes AFTER output spikes.
                     inp_spikes_2 = (inp_spikes_1 > pred_spikes_2[p_spike]) & (inp_spikes_1 <= pred_spikes_2[p_spike]-winsiz)
                 [imp_ind, imp_c] = np.unique(inp_indices_1[inp_spikes_2], return_counts=True)
                 accum[imp_ind,imp_g] = accum[imp_ind,imp_g] + imp_c
                 inp_spikes_3 = inp_spikes_1[inp_spikes_2]
                 inp_indices_3 = inp_indices_1[inp_spikes_2]
                 for inp_s in range(inp_spikes_3.size):
                     FR = np.interp(inp_spikes_3[inp_s],bintimes2[rep]*1000,histin2[rep,imp_g, inp_indices_3[inp_s]])
                     FRs[imp_g].append(FR)
                 
     percents = accum/np.sum(accum)
             
     return(accum, percents, out_sample,FRs)

#%% spike_cause_FR_W_all
def spike_cause_FR_W_all(target,events,pred_spikes,inp_spikes,inp_indices, winsiz,bintimes,histin):
     #Find the input spikes in an interval before each ouput spike within a specified epoch
     #epoch is based on 1 indexing - valid epochs are 1, 2, 3
     reps = [0, events.shape[1]]
     bintimes2 = np.copy(bintimes[target,reps[0]:reps[1]])
     histin2 = np.copy(histin[target,reps[0]:reps[1],:,:])
     
     numreps = int(np.diff(reps))
     pred_spikes_1 = pred_spikes[target][reps[0]:reps[1]]
     pred_spikes_2 = []
     for rep in range(numreps):
         pred_spikes_2.append(np.round(np.array(pred_spikes_1[rep])*1000,1))#0.1ms resolution in simulator
     
     FRs = []
     infoint8 = []
     for rep in range(numreps):
         for imp_g in range(len(inp_spikes)):#for all input groups
             inp_spikes_1 = np.round(np.array(inp_spikes [imp_g][target][rep+reps[0]])*1000,1)
             inp_indices_1 =         np.array(inp_indices[imp_g][target][rep+reps[0]])
             for p_spike in range(pred_spikes_2[rep].size):
                 inp_spikes_2 = (inp_spikes_1 < pred_spikes_2[rep][p_spike]) & (inp_spikes_1 >= pred_spikes_2[rep][p_spike]-winsiz)
                 [imp_ind, imp_c] = np.unique(inp_indices_1[inp_spikes_2], return_counts=True)
                 inp_spikes_3 = inp_spikes_1[inp_spikes_2]
                 inp_indices_3 = inp_indices_1[inp_spikes_2]
                 for inp_s in range(inp_spikes_3.size):
                     FR = np.interp(inp_spikes_3[inp_s],bintimes2[rep]*1000,histin2[rep,imp_g, inp_indices_3[inp_s]])
                     FRs.append(FR)
                     infoint8.append(np.array([imp_g, inp_indices_3[inp_s],rep, target],dtype=('int8')))
                     
     FRs = np.array(FRs)                
     infoint8 = np.array(infoint8)      
     return(infoint8, FRs)

#%% spike_cause_base (events,inp_spikes,inp_indices,t,epochs[e],gauss_center)
def spike_cause_base(events,inp_spikes,inp_indices,target,epoch,gauss_center,winsiz):
     #Find the input spikes in an interval before each ouput spike within a specified epoch
     #epoch is based on 1 indexing - valid epochs are 1, 2, 3
     reps = [0,events.shape[1]]

     numreps = int(np.diff(reps))
     width = 60
   
     eventInds = np.array([6, 11, 9])#start_movement (6) pk_speed (11) end_movement (9)

     mev = np.mean(events[np.ix_([target],np.arange(events.shape[1]),eventInds)],1)
     ints = (mev-gauss_center)*1000
     
     t_events = events[np.ix_([target],np.arange(reps[0],reps[1]),eventInds)]
     ev = np.round(t_events*1000)
     ev = ev.astype(int)
     
     centers = np.squeeze(ev - np.tile(ints, [numreps, 1]))
     
     accum = np.zeros([90,len(inp_spikes)])
     for rep in range(numreps):
         cr = centers[rep, epoch-1]
         xp=np.array([0,cr-width-winsiz,cr-width,cr+width-winsiz,cr+width,cr+width+1])
         fp = np.array([0,0,1,1,0,0])
         for imp_g in range(len(inp_spikes)):#for all input groups
             inp_spikes_1 = np.round((np.array(inp_spikes[imp_g][target][rep+reps[0]])*1000)*10)/10 #0.1ms resolution in simulator
             inp_indices_1 = np.array(inp_indices[imp_g][target][rep+reps[0]])
             np.add.at(accum[:,imp_g], inp_indices_1, np.interp(inp_spikes_1,xp, fp))
     percents = accum/np.sum(accum)
     return(accum, percents)
#%% spike_cause_pot
def spike_cause_pot(events,oas,oap,gauss_center):
    #oap is a targets x reps ndarray. inside each cell is a num_units x duration (0.1ms res) array (float 64)
    
    winsiz = 200#20ms
    width = 60
    eventInds = np.array([6, 11, 9])#start_movement (6) pk_speed (11) end_movement (9)
    
    t_events = events[:,:,eventInds]*10000
    mev = np.mean(t_events,axis=(0,1))
    ints = (mev-gauss_center*10000)
    ints = np.transpose(np.expand_dims(ints,(1,2)), axes=[2,1,0])
    
    centers = np.squeeze(t_events - np.tile(ints, [t_events.shape[0], t_events.shape[1], 1]))
    centers = np.round(centers)
    centers = centers.astype(int)
    
    pot_snips = np.empty((oas.shape[0], oas.shape[1], oas.shape[2], 3), dtype =  object)
    for unit in range(oas.shape[0]):
        for target in range(oas.shape[1]):
            for rep in range(oas.shape[2]):
                uoi = ((oas[unit,target,rep]/ms)*10).astype(int)
                potentials = oap[target,rep][unit,:]
                for epoch in range(3):
                    uoi2 = uoi[(uoi > centers[target,rep, epoch]-width) & (uoi <= centers[target,rep, epoch]+width)]
                    pot_snips[unit, target,rep,epoch] = np.zeros([len(uoi2),winsiz])
                    for s in range(len(uoi2)):
                        pot_snips[unit, target,rep,epoch][s,:] = potentials[(uoi2[s]-winsiz+1):uoi2[s]+1]
    return (pot_snips, mev, centers)

#%% make_STA
def make_STA(unit,target,reps,events,spk_pot,pred_spikes,epoch,gauss_center):
    #Find the 'buildup' spike accross all input groups
     width = np.zeros(3)
     center = np.zeros(3)
     width[0]= 42
     width[1]= 60 #was 200, narrow was 60,wide = 150
     width[2]= 60
 
     mev = np.mean(events,1)
     mstart_movement = mev[target][6]
     mpk_speed_time= mev[target][11]
     mend_movement = mev[target][9]
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
            start_movement = int(ev[6])
            pk_speed= int(ev[11])
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
 