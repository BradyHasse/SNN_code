%% Figure 5, 7. rPCA scores of SNN units.
%  Figure 5A-C.    RNN example unit and rPCA scores - Monkey C.
%  Figure 7A-B.    SNN Discrete dynamic components - Monkey C.


addpath('.\util\');
addpath('.\CircStat2012a\');
subjectName = 'MonkeyC';        % monkeyC  monkeyN
%% load data
dataDir  = '.\data\';

if ~exist('subjectName', 'var')
    subjectName = 'MonkeyC';  % MonkeyC MonkeyN
end
dataFileName = sprintf('%s_Aligned_Data.mat', subjectName);
Data = importdata([dataDir dataFileName]);
RNNData = importdata(['.\data\' sprintf('%s_RNN_Data.mat', subjectName)]);
frNormFactor = RNNData.frNormFactor;
clear RNNData

% order of conditions: counter clockwise starting from 0 degree
[numRep, numCondition, numBinAll, numNeuron] = size(Data.Rate);
colors = importdata('target_colormap.mat');  % colors correspond to condition [10:16 1:9]

dataBins   = (Data.numBin(1)+1):numBinAll;
numBin     = length(dataBins);
dataTimes  = [0:(numBin-1)] * Data.dt;
markerTimes = dataTimes(cumsum(Data.numBin(2:4)) + 1);

dataFileName = sprintf('%s_SNN_Data.mat', subjectName);
SNNData = importdata([dataDir dataFileName]);

histo_allP = SNNData.histo_allP;
weights = SNNData.weights;
input_FR = SNNData.input_FR;


%% example neuron 
switch subjectName
    case 'MonkeyC'
        neuronInd = 59;
    case 'MonkeyN'
        neuronInd = 48;
end

condInd = [10:numCondition 1:9];

trialFR = squeeze(Data.Rate(:, condInd, dataBins, neuronInd));
FR = squeeze(mean(trialFR, 1));  % trial-averaged rates
pFR = squeeze(histo_allP.histo_all(:,condInd,:));
pFR = interp1(histo_allP.xax_labels, pFR, dataTimes);
pFRu = squeeze(pFR(:,:,neuronInd))';

switch subjectName
    case 'MonkeyC'
        xtickInd = 11 + 10 * [1 3 4 6 7 9];
        xtickTF  = 11 + 10 * [2 5 8];
        ylimits  = [0 45];
        xlimits  = [0.00 0.75];
    case 'MonkeyN'
        xtickInd = 11 + 14 * [1 3 4 6 7 9];
        xtickTF  = 11 + 14 * [2 5 8];
        ylimits  = [0 70];
        xlimits  = [0.00 0.75];
end


%% plot PSTH
figure; hold on;

for i = 1:numCondition
    plot(dataTimes, FR(i, :), '-', 'Color', colors(i, :), 'LineWidth', 1);
end
for i = 1:length(markerTimes)
    plot(markerTimes(i), ylimits(1), 'pentagram', 'Color', 'k', 'MarkerSize', 8, ...
                              'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');
end
hold off;

xlim(xlimits); ylim(ylimits);
% xlabel('Time (s)'); 
ylabel('Rate (Hz)'); 
set(gca, 'XTick', 0:0.1:0.8, 'XTickLabel', {}, 'YTick', 0:10:40, ...
         'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
         'FontSize', 9, 'FontName', 'Helvetica');
set(gca, 'Units', 'inches', 'Position', [1, 1, 4, 1]);


% SNN predicted PSTH
figure; hold on;

for i = 1:numCondition
    plot(dataTimes, pFRu(i, :), '-', 'Color', colors(i, :), 'LineWidth', 1);
end
for i = 1:length(markerTimes)
    plot(markerTimes(i), ylimits(1), 'pentagram', 'Color', 'k', 'MarkerSize', 8, ...
                              'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');
end
hold off;

xlim(xlimits); ylim(ylimits);
% xlabel('Time (s)'); 
ylabel('Rate (Hz)'); 
set(gca, 'XTick', 0:0.1:0.8, 'XTickLabel', {}, 'YTick', 0:10:40, ...
         'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
         'FontSize', 9, 'FontName', 'Helvetica');
set(gca, 'Units', 'inches', 'Position', [1, 1, 4, 1]);



%% plot SNN input PSTH
inFR = input_FR.histoin(:,condInd,:);
inFR = interp1(input_FR.xax_labelsin, inFR, dataTimes, "linear",'extrap');

for j = 1:4
    figure; hold on;
    for i = 1:numCondition
        plot(dataTimes, inFR(:, i, j), '-', 'Color', colors(i, :), 'LineWidth', 1);
    end
    for i = 1:length(markerTimes)
        plot(markerTimes(i), ylimits(1), 'pentagram', 'Color', 'k', 'MarkerSize', 8, ...
                                  'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');
    end
    hold off;
    
    xlim(xlimits); ylim([0 80]);
    % xlabel('Time (s)'); 
    ylabel('Rate (Hz)'); 
    set(gca, 'XTick', 0:0.1:0.8, 'XTickLabel', {}, 'YTick', 0:20:80, ...
             'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
             'FontSize', 9, 'FontName', 'Helvetica');
    set(gca, 'Units', 'inches', 'Position', [1, 1, 4, 1]);
end

%% plot example weights

weights = permute(weights,[3,2,1]);
ylimset = squeeze([min(weights,[],[1,2]),max(weights,[],[1,2])])';
ylimset = ylimset+diff(ylimset')'*[-0.05 0.05];
figure; hold on;
for j = 1:4
plot(weights(:,j, neuronInd), '-', 'LineWidth', 1)
end




%% rPCA analysis of firing rates
% PCA of recorded rates

pFR = permute(pFR, [2,1,3]);
pFR = pFR./reshape(frNormFactor,1,1,[]);

frNondir = mean(pFR, 1);
frDir = pFR - frNondir;

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


% plot rPCA scores for recorded neurons
rpcaScores = frRPCA.matScore; % bin x target x PC

figure;
plotPCAScore(rpcaScores, dataTimes, colors, frRPCA.rpcID);
set(gcf, 'Units', 'inches', 'OuterPosition', [4, 5, 4.5, 4]);
for isubplot = 1:numPC
    subplot(2, 3, isubplot); hold on;
        if strcmp(subjectName, 'MonkeyC')
            plot(markerTimes, -0.6, 'pentagram', 'Color', 'k', 'MarkerSize', 6, ...
                 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');  % monkeyC: -1.98; monkeyN: -2.48
            axis([0 0.7 -0.7 0.7]);
            set(gca, 'YTick', [-0.6 0 0.6]);
            daspect([0.7 1.2 1]);
        elseif strcmp(subjectName, 'MonkeyN')
            plot(markerTimes, -0.8, 'pentagram', 'Color', 'k', 'MarkerSize', 6, ...
                 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');  % monkeyC: -1.98; monkeyN: -2.48
            axis([0 0.85 -0.8 0.8]);
            daspect([0.85 1.6 1]);
        end
end
sgtitle('rPCA scores - SNN units');

%% plot trajectories in the 3 rPCA planes
% % % figure('Position', [100 600 600 200]); tiledlayout(1, 3);
figure('Units', 'inches', 'OuterPosition', [3, 9, 6, 2]); tiledlayout(1, 3);
    nexttile; hold on;
    for itgt = 1:16
        plot3(zeros(1, size(rpcaScores, 1)), rpcaScores(:, itgt, 1), rpcaScores(:, itgt, 2), 'Color', colors(itgt, :), 'LineWidth', 1); 
    end
    plot3(zeros(1, 5), 0.7*[1 -1 -1 1 1], 0.7*[1 1 -1 -1 1], '-k', 'LineWidth', 1);
    axis off; view([-20 15]);

    nexttile; hold on;
    for itgt = 1:16
        plot3(rpcaScores(:, itgt, 3), rpcaScores(:, itgt, 4), zeros(1, size(rpcaScores, 1)), 'Color', colors(itgt, :), 'LineWidth', 1); 
    end
    plot3(0.7*[1 -1 -1 1 1], 0.7*[1 1 -1 -1 1], zeros(1, 5), '-k', 'LineWidth', 1);
    axis off; view([0 90]);

    nexttile; hold on;
    for itgt = 1:16
        plot3(rpcaScores(:, itgt, 5), zeros(1, size(rpcaScores, 1)), rpcaScores(:, itgt, 6), 'Color', colors(itgt, :), 'LineWidth', 1); 
    end
    plot3(0.7*[1 -1 -1 1 1], zeros(1, 5), 0.7*[1 1 -1 -1 1], '-k', 'LineWidth', 1);
    axis off; view([345 45]);



%% cleanup
rmpath('.\util\');
rmpath('.\CircStat2012a\');
