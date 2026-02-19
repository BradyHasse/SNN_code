%% Figure 1D, S4D. rPCA scores of recorded units.
%  Figure 3A-B.    RNN example unit and rPCA scores - Monkey C.
%  Figure S7A-B.   RNN example unit and rPCA scores - Monkey N.
%  Figure S10.     Rates of RNN units with input dropout.

addpath('.\util');

subject     = 'MonkeyC';        % monkeyC  monkeyN
irun        = 1;

dt          = 5;        % ms
tau         = 20;       % ms

fSaveFig    = false;

% hack: load firing rate normalization factor values for converting normalized rates back
load(['.\data\' sprintf('%s_RNN_Data.mat', subject)]);
frNormFactor = Data.frNormFactor;

%% load results from file
numNet = 2;  % 1: RNN; 2: RNN with input dropout
netOutput = cell(1, numNet);

fileName = sprintf('Results_%s_Run%d_Aligned.mat', subject, irun);
filePath = ['.\rnn_results\' fileName];
if exist(filePath, 'file');
    netOutput{1} = load(filePath);
    fprintf('File %s loaded!\n', filePath);
else
    error('File %s does not exist!', filePath);
end

fileName = sprintf('Results_%s_Run%d_Aligned2.mat', subject, irun);
filePath = ['.\rnn_results\' fileName];
if exist(filePath, 'file');
    netOutput{2} = load(filePath);
    fprintf('File %s loaded!\n', filePath);
else
    error('File %s does not exist!', filePath);
end

[numRep, numCondition, numBin, numNeuron] = size(netOutput{1}.matFR);
[~, ~, ~, numRNNUnit] = size(netOutput{1}.matR);

load('target_colormap.mat');
colors = colors([8:end 1:7], :);

%% calc. corr. of recorded vs. RNN unit firing rates
rateCorr = nan(numNeuron, numNet);
landmarkBinInd = netOutput{1}.landmarkBinInd;

for ineuron = 1:numNeuron
    rateNeuron = squeeze(mean(netOutput{1}.matFR(:, :, landmarkBinInd(1):end, ineuron), 1));    % trail-averaged rates

    for inet = 1:numNet
        if isfield(netOutput{inet}, 'matR')
            rateRNN = squeeze(mean(netOutput{inet}.matR(:, :, landmarkBinInd(1):end, ineuron), 1));   % trail-averaged rates
        elseif isfield(netOutput{inet}, 'matR2')
            rateRNN = squeeze(mean(netOutput{inet}.matR2(:, :, landmarkBinInd(1):end, ineuron), 1));   % trail-averaged rates
        end

        rateCorr(ineuron, inet) = corr(rateNeuron(:), rateRNN(:));
    end
end

% plot histogram 
figure('units', 'inches', 'Position', [1 1 2.5 1.5]); hold on;
corrDelta = 0.02;
histTicks = corrDelta/2:corrDelta:(1-corrDelta/2);
histEdges = 0:corrDelta:1;

histCounts = histcounts(rateCorr(:, 1), histEdges); 
histPrct = 100 * histCounts / size(rateCorr, 1);

histCounts2 = histcounts(rateCorr(:, 2), histEdges); 
histPrct2 = 100 * histCounts2 / size(rateCorr, 1);

bar(histTicks, histPrct, 1, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
bar(histTicks, histPrct2, 1, 'EdgeColor', 'none', 'FaceAlpha', 0.5);

if strcmp(subject, 'MonkeyC')
    box off; axis([0 1 0 25]);  daspect([1 60 1]);
elseif strcmp(subject, 'MonkeyN')
    box off; axis([0 1 0 50]);  daspect([1 120 1]);
end
xlabel('Correlation - Actual vs. RNN'); ylabel({'Percentage'; 'of neurons (%)'});
set(gca, 'FontSize', 8, 'TickDir', 'out', 'XTick', [0 0.5 1.0], 'YTick', [0 25 50]);


%% plot firing rates of example neuron
if strcmp(subject, 'MonkeyC')
    ineuron = 59;   % 59 (0.4)   32 (0.8)    65    52     28     26     13    monkeyC
    frMultiplier = frNormFactor(ineuron);
    yMax    = ceil(0.4 * frMultiplier * 5) / 5;
elseif strcmp(subject, 'MonkeyN')
    ineuron = 48;   % 21         13 (0.5)    48 (0.7)    70 (0.8)      (27 (0.5)   30    39 (0.6)    53 (0.6)     monkeyN  
    frMultiplier = frNormFactor(ineuron);
    yMax    = ceil(0.7 * frMultiplier * 5) / 5;
end


% scatter plot - comparison of corr
figure('units', 'inches', 'Position', [1 3 3 3]); hold on;
hold on; plot([-1 1], [-1 1], '--k');
plot(rateCorr(:, 1), rateCorr(:, 2), 'k.', 'MarkerSize', 12); 
plot(rateCorr(ineuron, 1), rateCorr(ineuron, 2), 'r.', 'MarkerSize', 12); 
axis([-0.2 1 -0.2 1]); daspect([1 1 1]);
set(gca, 'TickDir', 'out');
xlabel('Correlation - Actual vs. RNN');
ylabel('Actual vs. Input-dropout RNN');

if fSaveFig
    set(gcf, 'Renderer', 'Painters');
    figFileName = sprintf('.\\figures\\Rate_corr_comparison_%s_%s.pdf', subject, netTypes{1});
    print(gcf, figFileName, '-dpdf');
end


% plot PSTH of example unit
times = netOutput{1}.times;
rateNeuron = squeeze(mean(netOutput{1}.matFR(:, :, :, ineuron), 1));
timeLandmarks = times(netOutput{1}.landmarkBinInd);

figure('units', 'inches', 'Position', [1 6 2.5 5.0]);  % 2 x 1 subplots
tiledlayout(1+numNet, 1, 'TileSpacing', 'compact');

nexttile; hold on;
for icond = 1:numCondition
    plot(1000*(times - timeLandmarks(1)), frMultiplier * rateNeuron(icond, :), 'Color', colors(icond, :), 'LineWidth', 2);
end
plot(1000*(timeLandmarks(2:end) - timeLandmarks(1)), 0, 'pentagram', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');
if strcmp(subject, 'MonkeyC')
    axis([0 1000*(times(end)-timeLandmarks(1)) 0 yMax]);
    daspect([500*1000*(times(end)-timeLandmarks(1))/700 0.5*yMax/0.4 1]);
elseif strcmp(subject, 'MonkeyN')
    axis([0 1000*(times(end)-timeLandmarks(1)) 0 yMax]);
    daspect([500*1000*(times(end)-timeLandmarks(1))/700 0.5*yMax/0.4 1]);
end
ylabel({'Actual'; 'Firing rate (Hz)'});

set(gca, 'TickDir', 'Out', 'XTickLabel', {}, 'YTick', 0:10:yMax);
for inet = 1:length(netOutput)
    nexttile; hold on;

    if isfield(netOutput{inet}, 'matR')
        rateRNN = squeeze(mean(netOutput{inet}.matR(:, :, :, ineuron), 1));
        ylabel('RNN');
    elseif isfield(netOutput{inet}, 'matR2')
        rateRNN = squeeze(mean(netOutput{inet}.matR2(:, :, :, ineuron), 1));
        ylabel({'RNN'; 'Input dropout'});
    end

    for icond = 1:numCondition
        plot(1000*(times - timeLandmarks(1)), frMultiplier * rateRNN(icond, :), 'Color', colors(icond, :), 'LineWidth', 2);
    end
    plot(1000*(timeLandmarks(2:end) - timeLandmarks(1)), 0, 'pentagram', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');
    text(100, yMax-5, sprintf('Unit %d: r = %.2f', ineuron, rateCorr(ineuron, inet)));
    if strcmp(subject, 'MonkeyC')
        axis([0 1000*(times(end)-timeLandmarks(1)) 0 yMax]);
        daspect([500*1000*(times(end)-timeLandmarks(1))/700 0.5*yMax/0.4 1]);
    elseif strcmp(subject, 'MonkeyN')
        axis([0 1000*(times(end)-timeLandmarks(1)) 0 yMax]);
        daspect([500*1000*(times(end)-timeLandmarks(1))/700 0.5*yMax/0.4 1]);
    end
    
    if inet == length(netOutput)
        set(gca, 'FontSize', 8, 'TickDir', 'Out', 'XTick', 0:100:1000, ...
                 'XTickLabel', {'0', '', '200', '', '400', '', '600', '', '800', '', '1000'}, ...
                 'YTick', 0:10:yMax);
        xlabel('Time (msec)');
    else
        set(gca, 'TickDir', 'Out', 'XTickLabel', {}, 'YTick', 0:10:yMax);
    end
end

if fSaveFig
    figFileName = sprintf('.\\figures\\RNN_%s_Unit%d.pdf', subject, ineuron);
    set(gcf, 'Renderer', 'Painters');
    print(gcf, figFileName, '-dpdf');
end


%% rPCA analysis of firing rates
for inet = 1  % 1:numNet ----------------------------------------------------------------------------------------------------
    %% PCA of recorded rates
    if isfield(netOutput{inet}, 'matFR')
        rateFR = squeeze(mean(netOutput{inet}.matFR, 1));
    else
        rateFR = squeeze(mean(netOutput{inet-1}.matFR, 1));
    end

    firstBinInd = landmarkBinInd(1)-0;
    lastBinInd = min([size(rateFR, 2) landmarkBinInd(end) + 250/5]);
    binInd = firstBinInd : lastBinInd;  % choose data bins for rPCA analysis ------------------------------------------------

    rateFR = rateFR(:, binInd, :);
    frNondir = mean(rateFR, 1);
    frDir = rateFR - frNondir;

    frTmp = permute(frDir, [2 1 3]);  % -> bin x cond x unit
    longFR = reshape(frTmp, size(frTmp, 1)*numCondition, size(frTmp, 3));
    frPCA = doPCA(longFR);

    numPC = 6;    
    % rotate scores - d x m; d = numBin*numCondition, m = numPC
    d = size(frTmp, 1)*numCondition;    m = numPC;
    rpcaParam.numPC = numPC;
    rpcaParam.power = 4;
    rpcaParam.gamma = m/2;      % 1: varimax;  0: quartimax
                                % m/2: equamax
                                % d*(m - 1)/(d + m - 2): parsimax
    frRPCA = rotatePCA(frPCA, rpcaParam);
    frRPCA = sortRPCA(frRPCA, size(frTmp, 1), numCondition, numPC); 


    %% PCA of RNN unit rates
    unitIDs = 1:numRNNUnit;                 % all RNN units, including output units

    if isfield(netOutput{inet}, 'matR')
        rateRNN = squeeze(mean(netOutput{inet}.matR(:, :, binInd, unitIDs), 1));
    elseif isfield(netOutput{inet}, 'matR2')
        rateRNN = squeeze(mean(netOutput{inet}.matR2(:, :, binInd, unitIDs), 1));
    end

    rateNondir = mean(rateRNN, 1);
    rateDir = rateRNN - rateNondir;

    rateTmp = permute(rateDir, [2 1 3]);
    longRate = reshape(rateTmp, size(rateTmp, 1)*numCondition, size(rateTmp, 3));
    outPCA = doPCA(longRate);
    
    fprintf('Top %d PCs explain %.2f and %.2f data variance.\n', numPC, sum(frPCA.explained(1:numPC)), sum(outPCA.explained(1:numPC)));


    % rotate scores - d x m; d = numBin*numCondition, m = numPC
    d = size(rateTmp, 1)*numCondition;    m = numPC;
    rpcaParam.numPC = numPC;
    rpcaParam.power = 4;
    rpcaParam.gamma = m/2;      % 1: varimax;  0: quartimax
                                % m/2: equamax
                                % d*(m - 1)/(d + m - 2): parsimax
    outRPCA = rotatePCA(outPCA, rpcaParam);
    outRPCA = sortRPCA(outRPCA, size(rateTmp, 1), numCondition, numPC);


    %% plot rPCA scores for RNN units
    rpcaScores = outRPCA.matScore; % bin x target x PC

    figure;
    plotPCAScore(rpcaScores, times(binInd)-timeLandmarks(1), colors, outRPCA.rpcID);
    set(gcf, 'Units', 'inches', 'OuterPosition', [inet*4, 1, 4.5, 4]);
    for isubplot = 1:numPC
        subplot(2, 3, isubplot); hold on;
            if strcmp(subject, 'MonkeyC')
                plot(timeLandmarks-timeLandmarks(1), -1, 'pentagram', 'Color', 'k', 'MarkerSize', 6, ...
                     'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');  % monkeyC: -1.98; monkeyN: -2.48
                xlim([0 0.7]);
                daspect([0.7 2 1]);
            elseif strcmp(subject, 'MonkeyN')
                plot(timeLandmarks-timeLandmarks(1), -1.5, 'pentagram', 'Color', 'k', 'MarkerSize', 6, ...
                     'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');  % monkeyC: -1.98; monkeyN: -2.48
                xlim([0 0.85]);
                daspect([0.85 3 1]);
            end 
    end
    if fSaveFig
        set(gcf, 'Renderer', 'Painters');
        figFileName = sprintf('.\\figures\\rPCA_RNN_%s_%s.pdf', subject, modelNames{inet});
        print(gcf, figFileName, '-dpdf');
    end
    sgtitle('rPCA scores - RNN units');

    %% plot rPCA scores for recorded neurons
    rpcaScores = frRPCA.matScore; % bin x target x PC

    figure;
    plotPCAScore(rpcaScores, times(binInd)-timeLandmarks(1), colors, frRPCA.rpcID);
    set(gcf, 'Units', 'inches', 'OuterPosition', [inet*4, 5, 4.5, 4]);
    for isubplot = 1:numPC
        subplot(2, 3, isubplot); hold on;
            if strcmp(subject, 'MonkeyC')
                plot(timeLandmarks-timeLandmarks(1), -0.6, 'pentagram', 'Color', 'k', 'MarkerSize', 6, ...
                     'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');  % monkeyC: -1.98; monkeyN: -2.48
                axis([0 0.7 -0.6 0.6]);
                set(gca, 'YTick', [-0.6 0 0.6]);
                daspect([0.7 1.2 1]);
            elseif strcmp(subject, 'MonkeyN')
                plot(timeLandmarks-timeLandmarks(1), -0.8, 'pentagram', 'Color', 'k', 'MarkerSize', 6, ...
                     'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');  % monkeyC: -1.98; monkeyN: -2.48
                axis([0 0.85 -0.8 0.8]);
                daspect([0.85 1.6 1]);
            end
    end
    if fSaveFig
        set(gcf, 'Renderer', 'Painters');
        figFileName = sprintf('.\\figures\\rPCA_M1_%s.pdf', subject);
        print(gcf, figFileName, '-dpdf');
    end
    sgtitle('rPCA scores - Actual units');

    %% plot trajectories in the 3 rPCA planes
    % % % figure('Position', [100 600 600 200]); tiledlayout(1, 3);
    figure('Units', 'inches', 'OuterPosition', [3, 9, 6, 2]); tiledlayout(1, 3);
        nexttile; hold on;
        for itgt = 1:16
            plot3(zeros(1, size(rpcaScores, 1)), rpcaScores(:, itgt, 1), rpcaScores(:, itgt, 2), 'Color', colors(itgt, :), 'LineWidth', 2); 
        end
        plot3(zeros(1, 5), 0.6*[1 -1 -1 1 1], 0.6*[1 1 -1 -1 1], '-k', 'LineWidth', 1);
        axis off; view([45 15]);

        nexttile; hold on;
        for itgt = 1:16
            plot3(rpcaScores(:, itgt, 3), rpcaScores(:, itgt, 4), zeros(1, size(rpcaScores, 1)), 'Color', colors(itgt, :), 'LineWidth', 2); 
        end
        plot3(0.6*[1 -1 -1 1 1], 0.6*[1 1 -1 -1 1], zeros(1, 5), '-k', 'LineWidth', 1);
        axis off; view([25 30]);

        nexttile; hold on;
        for itgt = 1:16
            plot3(rpcaScores(:, itgt, 5), zeros(1, size(rpcaScores, 1)), rpcaScores(:, itgt, 6), 'Color', colors(itgt, :), 'LineWidth', 2); 
        end
        plot3(0.6*[1 -1 -1 1 1], zeros(1, 5), 0.6*[1 1 -1 -1 1], '-k', 'LineWidth', 1);
        axis off; view([45 15]);

    if fSaveFig
        set(gcf, 'Renderer', 'Painters');
        figFileName = sprintf('.\\figures\\rPCA_planes_M1_%s.pdf', subject);
        print(gcf, figFileName, '-dpdf');
    end

end


%% cleanup
rmpath('.\util\');
