%% Figure 1A-C. PSTH and PD curve of example neuron
%  Fig_1_AC_example_neuron('MonkeyC')

function Fig_1AC_example_neuron(subjectName)

addpath('.\util\');

[s1, s2, s3] = RandStream.create('mrg32k3a', 'NumStreams', 3);
randStream = s1;
RandStream.setGlobalStream(randStream);

%% load data
dataDir  = '.\data\';

if ~exist('subjectName', 'var')
    subjectName = 'MonkeyC';  % MonkeyC MonkeyN
end
dataFileName = sprintf('%s_Aligned_Data.mat', subjectName);
load([dataDir dataFileName]);

% order of conditions: counter clockwise starting from 0 degree
[numRep, numCondition, numBinAll, numNeuron] = size(Data.Rate);
load('target_colormap.mat');  % colors correspond to condition [10:16 1:9]

dataBins   = (Data.numBin(1)+1):numBinAll;
numBin     = length(dataBins);
dataTimes  = [0:(numBin-1)] * Data.dt;
markerTimes = dataTimes(cumsum(Data.numBin(2:4)) + 1);


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

indepFR = mean(FR, 1);  % condition independent component
depFR   = FR - repmat(indepFR, numCondition, 1);  % condition dependent

switch subjectName
    case 'MonkeyC'
        xtickInd = 11 + 10 * [1 3 4 6 7 9];
        xtickTF  = 11 + 10 * [2 5 8];
        ylimits  = [-15 25];
        xlimits  = [0.05 0.55];
    case 'MonkeyN'
        xtickInd = 11 + 14 * [1 3 4 6 7 9];
        xtickTF  = 11 + 14 * [2 5 8];
        ylimits  = [-25 25];
        xlimits  = [0.05 0.75];
end


%% plot PSTH
figure; hold on;
for i = 1:length(xtickInd)
    plot(dataTimes(xtickInd(i))*[1 1], ylimits, '--', 'Color', 0.6*[1 1 1]);
end
for i = 1:length(xtickTF)
    plot(dataTimes(xtickTF(i))*[1 1], ylimits, '--', 'Color', [1 0 0]);
end
for i = 1:numCondition
    plot(dataTimes, depFR(i, :), '-', 'Color', colors(i, :), 'LineWidth', 1);
end
for i = 1:length(markerTimes)
    plot(markerTimes(i), ylimits(1), 'pentagram', 'Color', 'k', 'MarkerSize', 8, ...
                              'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'none');
end
hold off;

xlim(xlimits); ylim(ylimits);
% xlabel('Time (s)'); 
ylabel('Rate (Hz)'); 
set(gca, 'XTick', 0:0.2:0.8, 'XTickLabel', {}, 'YTick', -20:10:20, ...
         'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
         'FontSize', 9, 'FontName', 'Helvetica');
set(gca, 'Units', 'inches', 'Position', [1, 1, 4, 1]);


%% plot tuning curves
dirStep = 2*pi/numCondition;
dirAng  = (-pi+dirStep):dirStep:pi;
dirX    = cos(dirAng');
dirY    = sin(dirAng');

fitAng  = -pi:(pi/18):pi;
fitX    = cos(fitAng');
fitY    = sin(fitAng');

figure; hold on;
for i = 1:length(xtickTF)
    subplot(1, length(xtickTF), i); hold on;
    
    fr = depFR(:, xtickTF(i));
    b = regress(fr, [ones(numCondition, 1) dirX dirY]);
    pd = cart2pol(b(2), b(3));
    frHat = [ones(length(fitAng), 1) fitX fitY]*b;
    
    plot3(zeros(1, length(fitAng)), rad2deg(fitAng), frHat, '-r');
    plot3(zeros(1, length(dirAng)), rad2deg(dirAng), fr, 'k.');
    
    xlim([-eps eps]); ylim([-200 200]); zlim([-17 20]);
    daspect([180 100 10]); view([100 45]);
    if i == 1
        ylabel('Direction (deg)'); 
        zlabel('Rate (Hz)'); 
        set(gca, 'YTick', -180:90:180, 'ZTick', -10:10:20, ...
                 'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
                 'FontSize', 9, 'FontName', 'Helvetica');
    else
        set(gca, 'YTick', -180:90:180, 'YTickLabel', {}, ...
                 'ZTick', -10:10:20, 'ZTickLabel', {}, ...
                 'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1);
    end
    set(gca, 'Units', 'inches', 'Position', [1+(i-1)*1.3, 1, 1.2, 1]);
end


%% find preferred direction
numRound = 1000;
PD = nan(1, numBin);
MD = nan(1, numBin);

numRandRep = round(numRep*0.75);  % was 0.75 for MonkeyC unit 59
Reps = nan(numRandRep, numRound);
for iround = 1:numRound
    Reps(:, iround) = randperm(numRep, numRandRep);
end

parfor iround = 1:numRound
    reps = Reps(:, iround);
    fr = squeeze(mean(trialFR(reps, :, :), 1));
    
    for ibin = 1:numBin
        y = fr(:, ibin);

        [b, bint, r, rint, stats] = regress(y, [ones(numCondition, 1) dirX dirY]);

        [theta, rho] = cart2pol(b(2), b(3));

        PD(iround, ibin) = theta;
        MD(iround, ibin) = rho;
    end
end

[muPD, ulPD, llPD] = circ_mean(PD - deg2rad(0));
[sPD s0PD] = circ_std(PD);

% fix discontinuities in PD curve
newPD = fixPDflip(rad2deg(muPD));
muPD = deg2rad(newPD);

muMD = mean(MD, 1);


%% plot arrows for PD
figure; hold on;
xAxisLen = 4;
yAxisLen = 0.5;
ylimits = [-1 1]*yAxisLen*(diff(xlimits))/xAxisLen/2;

y0 = 0;
rho   = 2.5;
scaleFactorX = 0.01;
scaleFactorY = scaleFactorX*(diff(ylimits)/yAxisLen)/(diff(xlimits)/xAxisLen);

for i = 1:length(xtickInd)
    ind = xtickInd(i);
    plot(dataTimes(ind)*[1 1], ylimits, '--', 'Color', 0.6*[1 1 1]);
    
    x0 = dataTimes(ind);
    theta = muPD(ind);
    
    x = scaleFactorX*rho*cos(theta);
    y = scaleFactorY*rho*sin(theta);
    
    plot(x0+[0 x], y0+[0 y], '-k', 'LineWidth', 2);
end

for i = 1:length(xtickTF)
    ind = xtickTF(i);
    plot(dataTimes(ind)*[1 1], ylimits, '--', 'Color', [1 0 0]);
    
    x0 = dataTimes(ind);
    theta = muPD(ind);
    
    x = scaleFactorX*rho*cos(theta);
    y = scaleFactorY*rho*sin(theta);
    
    plot(x0+[0 x], y0+[0 y], '-r', 'LineWidth', 2);
end

xlim(xlimits); ylim(ylimits);
set(gca, 'XTick', 0:0.2:0.8, 'YTick', -0.1:0.1:0.1, ...
         'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
         'FontSize', 9, 'FontName', 'Helvetica');
set(gca, 'Units', 'inches', 'Position', [1, 1, xAxisLen, yAxisLen]);


%% plot PD curve
figure; hold on;
for i = 1:length(xtickInd)
    plot(dataTimes(xtickInd(i))*[1 1], [-360 360], '--', 'Color', 0.6*[1 1 1]);
end
for i = 1:length(xtickTF)
    plot(dataTimes(xtickTF(i))*[1 1], [-360 360], '--', 'Color', [1 0 0]);
end

% plot arrow for PD
y0 = 250;
ylimits = [-380 20];  % N-48
xAxisLen = 4;
yAxisLen = 1;

% plot +/- std area
patch([dataTimes dataTimes(end:-1:1)], ...
      [rad2deg(muPD+s0PD) rad2deg(muPD(end:-1:1)-s0PD(end:-1:1))], ...
      0.8*[1 1 1], 'EdgeColor', 'none');
% plot PD curve  
plot(dataTimes, rad2deg(muPD), '-k', 'LineWidth', 1);

xlim(xlimits); ylim(ylimits);
xlabel('Time (s)'); ylabel('PD (deg)'); 
set(gca, 'XTick', 0:0.2:0.8, 'YTick', -360:90:360, ...
         'TickDir', 'out', 'TickLength', [0.02, 0.02], 'LineWidth', 1, ...
         'FontSize', 9, 'FontName', 'Helvetica');
set(gca, 'Units', 'inches', 'Position', [1, 1, xAxisLen, yAxisLen]);


rmpath('.\util\');

end


function newPDs = fixPDflip(PDs)
    [flipIdx, polarities] = findPDflip(PDs);
    newPDs = PDs;

    while ~isempty(flipIdx)
        tmpPDs = newPDs;
        newPDs = shiftPDvalues(tmpPDs, flipIdx, polarities);

        [flipIdx, polarities] = findPDflip(newPDs);
    end
end

function newPDs = shiftPDvalues(PDs, flipIdx, polarities)
    i = 1;
    newPDs = PDs;
    if length(flipIdx) == 1
        sampleIdx = 1:flipIdx;
        if polarities > 0
            newPDs(sampleIdx) = PDs(sampleIdx) + 360;
        else
            newPDs(sampleIdx) = PDs(sampleIdx) - 360;
        end
    elseif length(flipIdx) > 1  &&  sum(polarities) == length(flipIdx)
        sampleIdx = 1:flipIdx(1);
        newPDs(sampleIdx) = PDs(sampleIdx) + 360;

        sampleIdx = (flipIdx(2)+1):length(PDs);
        newPDs(sampleIdx) = PDs(sampleIdx) - 360;
    elseif length(flipIdx) > 1  &&  sum(polarities) == -length(flipIdx)        
        sampleIdx = 1:flipIdx(1);
        newPDs(sampleIdx) = PDs(sampleIdx) - 360;

        sampleIdx = (flipIdx(2)+1):length(PDs);
        newPDs(sampleIdx) = PDs(sampleIdx) + 360;
    elseif length(flipIdx) > 1
        while i < length(flipIdx)
            currPolarity = polarities(i);
            nextPolarity = polarities(i+1);
    
            if currPolarity+nextPolarity == 0
                sampleIdx = (flipIdx(i)+1) : flipIdx(i+1);  % shift PD values between idx(i) and idx(i+1)
                if currPolarity > 0
                    newPDs(sampleIdx) = PDs(sampleIdx) - 360;
                else
                    newPDs(sampleIdx) = PDs(sampleIdx) + 360;
                end
    
                i = i + 2;
            else
                i = i + 1;
            end
        end
    end
end

function [flipBinIdx, flipPolarity] = findPDflip(PDs)
    posIdx = find(diff(PDs) > 180);
    negIdx = find(diff(PDs) <-180);
    tmpIdx = [posIdx negIdx];
    tmpVal = [ones(1, length(posIdx)) -1*ones(1, length(negIdx))];
    [flipBinIdx, orders] = sort(tmpIdx);
    flipPolarity = tmpVal(orders);
end


