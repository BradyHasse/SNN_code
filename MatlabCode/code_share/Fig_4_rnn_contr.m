%% Figure 4, S8.  Extrinsic and intrinsic contributions (to dx(t)) to an example neuron
%                 in one of the three epochs


subject = 'MonkeyC';    % monkeyC or monkeyN

epoch4PDgroup = 1;  % 1, 2, or 3 for early, middle, and late epoch

epochNames = {'Early', 'Middel', 'Late'};
iepochName = epochNames{epoch4PDgroup};

%% Load data used for RNN training
addpath('.\util\');
load(['./data/' sprintf('%s_RNN_Data.mat', subject)]);

numBlock  = Data.numBlock;
numTarget = Data.numTarget;
numTrial  = numBlock * numTarget;

Inputs    = Data.inputGau;
numInput  = size(Inputs{1}, 1);
numOutput = size(Data.fr{1}, 1);

%% load RNN results
irun = 1;
netsDir = './rnn_results/';
resultsFileName = sprintf('Results_%s_Run%d_SingleTrial.mat', subject, irun);
resultsFilePath = [netsDir '/' resultsFileName];
netResults = load(resultsFilePath);

numRNNUnit = size(netResults.rnnXs{1}, 1);
numCV = length(netResults.allNetworkFileNames);
trialMask  = netResults.trialMask;

%% choose an output neuron
switch subject
    case 'MonkeyC'
        unit2plot = 59;
        gaussOffsets = [4 6 3];
    case 'MonkeyN'
        unit2plot = 48;
        gaussOffsets = [10 20 13];
end
fprintf('\t%s Unit %d.\n', subject, unit2plot);

% choose the number of groups that inputs will be divided into according to their preferred direction
numContrGroup = 18;
groupContrR = nan(numTarget, numContrGroup);
groupWeight = nan(1, numContrGroup);

groupPDBinSize = 360/numContrGroup;
groupPDTicks = (0+groupPDBinSize/2):groupPDBinSize:(360-groupPDBinSize/2);
groupPDEdges = 0:groupPDBinSize:360;


%% calculate contributions to the output neuron in individual trials
numEpoch = length(epochNames);

gaussBins = [-12:11];  % a 120 ms window around the peak of each epoch
allRate   = nan(numTrial, numRNNUnit, length(gaussBins), numEpoch);
allInput  = nan(numTrial, numInput, length(gaussBins), numEpoch);
allContrI = nan(numTrial, numInput, length(gaussBins), numEpoch);
allContrR = nan(numTrial, numRNNUnit, length(gaussBins), numEpoch);

allWrr = nan(numRNNUnit, numRNNUnit, numCV);
allWru = nan(numRNNUnit, numInput, numCV);

for icv = 1:numCV
    % load net
    net_file_name = netResults.allNetworkFileNames{icv};
    load([netsDir net_file_name]);
    fprintf('\tnet file: %s\n', net_file_name);
    
    [n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1] = unpackRNN(net, net.theta);

    % % % eval_network_rnn  = create_eval_network_rnn2(1e-5);
    % % % eval_network      = create_eval_network2(eval_network_rnn, 1e-5);

    allWrr(:, :, icv) = n_Wrr_n;
    allWru(:, :, icv) = n_Wru_v;

    % get test data set
    trialIndTest = find(trialMask{icv, 3});
    inputsTest   = Inputs(trialIndTest);
    numTrialTest = length(trialIndTest);

    rnnXs = netResults.rnnXs(1, trialIndTest);
    rnnRs = netResults.rnnRs(1, trialIndTest);

    %% choose target output neuron and epoch windows
    % events - offsets = rPCA centers; offsets according to 'gauss_mu' and 'landmarks' in original data file
    timeGaussInput = Data.landmarkTime(3:5, trialIndTest) - net.dt * gaussOffsets';  % time of Gaussian input centers

    for itrial = 1:numTrialTest
        trialNo = trialIndTest(itrial);
        trialTimes = Data.times{trialNo};
    
        X = rnnXs{itrial};
        R = rnnRs{itrial};
        I = inputsTest{itrial};

        wI = n_Wru_v(unit2plot, :)';
        wR = n_Wrr_n(unit2plot, :)';

        % contributions from all extrinsic inputs
        cI = repmat(wI, 1, length(trialTimes)) .* I;  % numInput x numBin

        % contributions from all intrinsic inputs
        cR = repmat(wR, 1, length(trialTimes)) .* [R(:, 1) R(:, 1:(end-1))] ...
           - repmat([X(unit2plot, 1) X(unit2plot, 1:(end-1))], numRNNUnit, 1) ./ numRNNUnit ...
           + n_bx_1(unit2plot) ./ numRNNUnit;
    
        gaussTimes = timeGaussInput(:, itrial);
        for igauss = 1:numEpoch
            [timeDiff, ibin] = min(abs(trialTimes - gaussTimes(igauss)));
            bins = ibin + gaussBins;

            allRate(trialNo, :, :, igauss) = R(:, bins);
            allContrI(trialNo, :, :, igauss) = cI(:, bins);
            allContrR(trialNo, :, :, igauss) = cR(:, bins);
            allInput(trialNo, :, :, igauss) = I(:, bins);
        end
    end
end


%% clac preferred direction of all RNN units in the three epochs
dirAng = [0:22.5:359];  % target for condition 1 is at 0 degree
dirX = cosd(dirAng);
dirY = sind(dirAng);
regressX = [ones(numTarget, 1) dirX' dirY'];  
% for fitting cosine tuning curve: 
%     fr = [b0 b1=MD*cosd(PD) b2=MD*sind(PD)] * [1 cosd(ang) sind(ang)]'

PDs  = nan(numRNNUnit, numEpoch, numCV);  % preferred direction;  unit x epoch
MDs  = nan(numRNNUnit, numEpoch, numCV);  % modulation depth;     unit x epoch
R2s  = nan(numRNNUnit, numEpoch, numCV);  % R^2 statistic;        unit x epoch

for iunit = 1:numRNNUnit
    for iepoch = 1:numEpoch
        rawRates = squeeze(allRate(:, iunit, :, iepoch));  % trial x time bins
        trialRates = mean(rawRates, 2);

        for icv = 1:numCV
            trialIndTest = find(trialMask{icv, 3});

            fr = reshape(trialRates(trialIndTest), numTarget, []);  % target x rep
            fr = mean(fr, 2);

            [b, bint, r, rint, stats] = regress(fr, regressX);
            [theta, rho] = cart2pol(b(2), b(3));

            PDs(iunit, iepoch, icv) = rad2deg(theta);
            MDs(iunit, iepoch, icv) = rho;
            R2s(iunit, iepoch, icv) = stats(1);
        end
    end
end
PDs(PDs < 0) = PDs(PDs < 0) + 360;  % PDs in [0 360]

unit2plotPD = squeeze(PDs(unit2plot, epoch4PDgroup, :));


inpPDs  = nan(numRNNUnit, numEpoch, numCV);  % preferred direction;  unit x epoch
inpMDs  = nan(numRNNUnit, numEpoch, numCV);  % modulation depth;     unit x epoch
inpR2s  = nan(numRNNUnit, numEpoch, numCV);  % R^2 statistic;        unit x epoch
for iinp = 1:numInput
    for iepoch = 1:numEpoch
        rawRates = squeeze(allInput(:, iinp, :, iepoch));  % trial x time bins
        trialRates = mean(rawRates, 2);

        for icv = 1:numCV
            trialIndTest = find(trialMask{icv, 3});

            fr = reshape(trialRates(trialIndTest), numTarget, []);  % target x rep
            fr = mean(fr, 2);

            [b, bint, r, rint, stats] = regress(fr, regressX);
            [theta, rho] = cart2pol(b(2), b(3));

            inpPDs(iinp, iepoch, icv) = rad2deg(theta);
            inpMDs(iinp, iepoch, icv) = rho;
            inpR2s(iinp, iepoch, icv) = stats(1);
        end
    end
end

numInputPerGroup = numInput/numEpoch;
inpPlotStyle = {'rx', 'gx', 'bx'};

%% group units according to their PDs in the chosen iepoch
for icv = 1:numCV
    trialIndTest = find(trialMask{icv, 3});

    pds = PDs(:, epoch4PDgroup, icv);
    minR2s = min(R2s(:, :, icv), [], 2);

    groupIdx = discretize(pds, groupPDEdges);
    % r2Thr = 0.5;
    % groupIdx(minR2s < r2Thr) = nan;  % exclude units that are not well cosine-tuned ======================================

    for igp = 1:numContrGroup
        unitIdx = find(groupIdx == igp);
        if isempty(unitIdx)
            continue;
        end
    
        rawContrR = squeeze(allContrR(trialIndTest, unitIdx, :, epoch4PDgroup));  % trial x unit x bin
    
        avgContrR = mean(rawContrR, 3);  % trial x unit <- average across bins in current epoch
        sumContrR = sum(avgContrR, 2);   % trial x 1    <- add contr. from all units in current PD group
        sumContrR = reshape(sumContrR, numTarget, []);  % target x rep
    
        groupContrR(:, igp, icv) = mean(sumContrR, 2);
    
        cvWeights = squeeze(allWrr(:, :, icv)); % averaging across CV partitions
        groupWeight(:, igp,icv) = mean(cvWeights(unit2plot, unitIdx));
    end
end


%% contribution from inputs
avgContrI = squeeze(mean(allContrI(:, :, :, epoch4PDgroup), 3));  % average across bins: trial x input x bin x epoch -> trial x input
numInputPerGroup = numInput/numEpoch;
contrI = nan(numTarget, numInputPerGroup, numEpoch);
for igp = 1:numEpoch
    inpInd = (igp-1)*numInputPerGroup + [1:numInputPerGroup];

    for iinp = 1:numInputPerGroup
        for icv = 1:numCV
            trialIndTest = find(trialMask{icv, 3});

            tmpContrI = reshape(avgContrI(trialIndTest, inpInd(iinp)), numTarget, []);
            contrI(:, iinp, igp, icv) = mean(tmpContrI, 2);
        end
    end
end

for icv = 1:numCV
    inputWeights(:, :, icv) = squeeze(allWru(unit2plot, :, icv));
end


%% make plots
hFig = figure('Position', [100 100 2400 800]);
tiledlayout(2, 6, 'TileSpacing', 'compact', 'Padding', 'Compact');

unit2plotPDInd = median(round(0.5 + unit2plotPD / 20));

%---------- plot weights ----------
    ax11 = nexttile;  % skip the plot in (1, 1)
    axis([0 1 0 1]); daspect([1 1 1]);

    %  1: recurrent weights, plot in (1, 2)
    ax12 = nexttile; hold on;  
    plot([0.5 numContrGroup+0.5], [0 0], '--k');
    plot(unit2plotPDInd*[1 1], 5e-2*[-1 1], '--r');
    plot(1:numContrGroup, squeeze(mean(groupWeight, 3)), '-k', 'LineWidth', 3); 
    box off; daspect([numContrGroup 0.08 1]);
    axis([0.5 numContrGroup+0.5 -1e-2 1e-2]);
    set(gca, 'XTick', 0.5:9:length(groupPDEdges), 'XTickLabel', groupPDEdges(1:9:end), ...
             'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 10);

    %  2-4: input weights, plots from (1, 3) to (1, 5)
    inputPDs = 10:(360/numInputPerGroup):359;
    inputWeightLimits = [5 5 5] * 1e-2;  % monkey C unit 59
    % inputWeightLimits = [5 5 5] * 1e-2;  % monkey N unit 48
    for igp = 1:numEpoch
        inputInd = (igp-1)*numInputPerGroup + [1:numInputPerGroup];
        
        nexttile; hold on;  
        plot([0.5 numInputPerGroup+0.5], [0 0], '--k');
        plot(unit2plotPDInd*[1 1], inputWeightLimits(igp)*[-1 1], '--r');
        plot(1:numInputPerGroup, squeeze(mean(inputWeights(1, inputInd, :), 3)), '-k', 'LineWidth', 3); 
        box off; daspect([numInputPerGroup 0.08*5 1]);
        axis([0.5 numInputPerGroup+0.5 -inputWeightLimits(igp) inputWeightLimits(igp)]);
        set(gca, 'XTick', [0.5 9.5 18.5], 'XTickLabel', {'0', '180', '360'}, ...
                 'TickDir', 'out', 'FontName', 'Arial', 'FontSize', 10);
    end


    ax16 = nexttile;  % skip plot in (1, 6)
    axis([0 1 0 1]); daspect([1 1 1]);

%---------- plot contributions ----------
    unit2plotPDDirInd = 1 + median(unit2plotPD) / (360/numTarget);

    intrinColor = [106, 81,163]/255;
    extrinColor = [ 49,163, 84]/255;

    % the sum contr from all rnn units, plot in (2, 1)
    sumContrR = squeeze(sum(groupContrR, 2));

    ax21 = nexttile; hold on;
    plot([0 0], [1 numTarget+1], '-k');
    plot([-0.2 0.2], unit2plotPDDirInd * [1 1], '--r');
    plot(mean([sumContrR; sumContrR(1, :)], 2), 1:(numTarget+1), 'Color', intrinColor, 'LineWidth', 2);
    axis([-0.2 0.2 1 numTarget+1]);
    daspect([0.2*8 numTarget 1]);
    set(gca, 'TickDir', 'out', 'TickLength', [0.02, 0.01], ...
             'YTick', 1:4:(numTarget+1), 'YTickLabel', 0:90:360);
    xlabel('Total Intrinsic Contribution (a.u.)');
    ylabel('Target Direction (deg)');


    %  5: intrinsic rnn contributions, plot in (2, 2)
    ax22 = nexttile;
    imagesc(mean(groupContrR, 3), [-0.04 0.04]);  % groupContrR: 
    % imagesc(groupContrR - mean(groupContrR, 1), [-0.01 0.01]);

    colormap turbo;
    daspect([length(groupPDTicks) numTarget 1]); box off;


    set(gca, 'YDir', 'normal', 'TickDir', 'out', 'TickLength', [0.02, 0.01], ...
             'XTick', 0.5:9:length(groupPDEdges), 'XTickLabel', groupPDEdges(1:9:end), ...
             'YTick', 0.5:4:(numTarget+0.5), 'YTickLabel', 0:90:360);
    if epoch4PDgroup == 1
        xlabel({sprintf('Preferred Dirction in %s Epoch (deg)', iepochName); 'Early Epoch'});
        ylabel('Target Direction (deg)');
    elseif epoch4PDgroup == 2
        xlabel({''; 'Middle Epoch'});
    elseif epoch4PDgroup == numEpoch
        xlabel({''; 'Late Epoch'});
    end

    %  6-8: input contributions
    for igp = 1:numEpoch
        nexttile;
        imagesc(squeeze(mean(contrI(:, :, igp), 4)), [-0.04 0.04]);
        % imagesc(squeeze(contrI(:, :, igp)) - mean(squeeze(contrI(:, :, igp)), 1), [-0.01 0.01]);

        colormap turbo;
        daspect([length(groupPDTicks) numTarget 1]); box off;
        set(gca, 'YDir', 'normal', 'TickDir', 'out', 'TickLength', [0.02, 0.01], ...
                 'XTick', 0.5:9:length(groupPDEdges), 'XTickLabel', groupPDEdges(1:9:end), ...
                 'YTick', 0.5:4:(numTarget+0.5), 'YTickLabel', 0:90:360);

        if igp == numEpoch
            colorbar;
        end
    end
    
    % the sum contr from all inputs, plot in (2, 6)
    sumContrI = sum(sum(mean(contrI, 4), 2), 3);

    ax26 = nexttile; hold on;
    plot([0 0], [1 numTarget+1], '-k');
    plot([-0.2 0.2], unit2plotPDDirInd * [1 1], '--r');
    plot([sumContrI; sumContrI(1)], 1:(numTarget+1), 'Color', extrinColor, 'LineWidth', 2);
    axis([-0.2 0.2 1 numTarget+1]);
    daspect([0.2*8 numTarget 1]);
    set(gca, 'TickDir', 'out', 'TickLength', [0.02, 0.01], ...
             'YTick', 1:4:(numTarget+1), 'YTickLabel', 0:90:360);
    xlabel('Total Extrinsic Contribution (a.u.)');
    ylabel('Target Direction (deg)');


    % fix subplot figure size
    ax11.Parent = hFig;
    ax11.InnerPosition(2) = ax12.InnerPosition(2);
    ax11.InnerPosition(4) = ax12.InnerPosition(4);

    ax16.Parent = hFig;
    ax16.InnerPosition(2) = ax12.InnerPosition(2);
    ax16.InnerPosition(4) = ax12.InnerPosition(4);

    ax21.Parent = hFig;  % ax.Parent was TiledChartLayout which doesn't allow change of InnerPosition
    ax21.InnerPosition(2) = ax22.InnerPosition(2);
    ax21.InnerPosition(4) = ax22.InnerPosition(4);

    ax26.Parent = hFig;
    ax26.InnerPosition(2) = ax22.InnerPosition(2);
    ax26.InnerPosition(4) = ax22.InnerPosition(4);


rmpath('.\util\');

