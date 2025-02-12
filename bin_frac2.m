function rates= bin_frac2(spikes,a_time,a_b_time, bin_width)
%% notes
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
% updated to bin_frac2 by Brady Hasse Jan 16, 2023
%% function body

num_bins= floor(((a_b_time-a_time)/bin_width)+.0001);%sets the number of bins
bin_width = (a_b_time-a_time)/num_bins;%if it was not exact, makes bin_width correct.
spikes=unique(sort(spikes(:)))';%sort the spikes in order and remove any duplicates.

if size(spikes,2) == 0 %|| ~any(spikes>a_time & spikes<a_b_time)%uncommenting will result in the same results as bin_frac.m
    rates(1:num_bins,1) = 0.0;%if no spikes, the rate is 0
else
    spikes = [min(spikes) - 500, spikes, max(spikes) + 500];%add spikes far in the past and future.
    
    spikes = spikes - a_time;
    a_b_time = a_b_time - a_time;
    a_time=0;
    
    spikes = spikes*(num_bins/a_b_time);
    a_b_time = num_bins;

    spikes = spikes(find(spikes<a_time,1,'last'):find(spikes>a_b_time,1));%from spike preceeding a_time to spike after a_b_time
    dspikes = diff(spikes);%intervals between spikes
    dspikes(dspikes==0) = .000001;
    intervals = zeros(num_bins,1);
    
    for i = 1:length(dspikes)%assign each interval between spikes to bins.
        j = floor(spikes(i))+1:floor(spikes(i+1));
        divs = sort([j spikes(i:i+1)]);%time of two spikes creating dspikes and the whole numbers  between them
        int_share = diff(divs)*(1/dspikes(i));%each full unit is worth (1/dspikes(i))
        int_inds = ceil(divs(2:end));%indcies of bin in interval where int_shares get assigned to
        rm_inds = (int_inds<1 | int_inds>a_b_time | int_share == 0);%indicies to remove because they belong to unassined bins
        int_share(rm_inds) = [];%remove those shares
        int_inds(rm_inds) = [];%remove those shares
        intervals(int_inds) = intervals(int_inds) + int_share';%assign portion of spike to the correct bin(s)
    end 
    
    rates = intervals/bin_width;
end%if 

end%funciton
