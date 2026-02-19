%% Figure 2.  Velocity decoding performance with firing rates inferred by an autonomous dynamical
%             systems model (LFADS) - Monkey C.
%  Figure S6. Monkey C and Monkey N.

addpath('.\util\');

subject    = 'MonkeyC';  % MonkeyC  MonkeyN
alignEvent = 'Cue';

runList    = 1:5;  % 1:5
numRun     = length(runList);

ifUsedSlidingWindow = false;

modelStr = 'no_controller';

fSaveFig = false;

%% load raw data
fileName = sprintf('.\\data\\%s_RNN_Data.mat', subject);
RawData = load(fileName);
RawData = RawData.Data;
dt = RawData.dt;

%% load lfads output data
fileDir   = '.\lfads_results\';
irun      = runList(1);
fileName  = sprintf('%s_%s_lfads_data_%s_run%d.mat', subject, alignEvent, modelStr, irun);
lfadsData = load([fileDir fileName]);

colors    = lfadsData.colors;
landmarkBin = lfadsData.landmarkBin;
trialNos  = lfadsData.trialNos;
[numUnit, numBin, numTrial, numEncSeg] = size(lfadsData.lfadsRates);
numTarget = 16;
numBlock  = numTrial / numTarget;

%% plot single-trial speed profiles
timeOn   = (landmarkBin(3, :) - landmarkBin(2, :)) * dt;  % event time relative to target onset
timePeak = (landmarkBin(4, :) - landmarkBin(2, :)) * dt;
timeOff  = (landmarkBin(5, :) - landmarkBin(2, :)) * dt;
% % % truthMaxSpd = nan(numTrial, 1);

hFig = figure('Position', [50 100 600 900]); 
tiledlayout(3, 1, 'TileSpacing', 'compact');

ax1 = nexttile; hold on;
% plot distribution of event times
eventColors = repmat(0.6 * [1 1 1], 3, 1);

ydata = [timeOn(:) timePeak(:) timeOff(:)];
make_violin_plot(ydata, 0 * [1 1 1], 'Orientation', 'horizontal', ...
               'DensityDirection', 'positive', 'DensityScaleFactor', 0.002, ...
               'FaceColor', eventColors);
ylim([-0.02 0.10]);
set(gca, 'YTick', []);
xlabel('Time (s)');


%% analyze PSTH and modulation depth of directional tuning of lfads inferred rates
lfadsPSTH = nan(numUnit, numBin, numTarget, numEncSeg, numRun);
for irun = 1:numRun
    runNo = runList(irun);

    % load lfads output data
    fileName = sprintf('%s_%s_lfads_data_%s_run%d.mat', subject, alignEvent, modelStr, runNo);
    lfadsData = load([fileDir fileName]);
    lfadsRates = lfadsData.lfadsRates;

    for itgt = 1:numTarget
        trIdx = itgt:numTarget:numTrial;
        tmpFR = lfadsRates(:, :, trIdx, :);
        lfadsPSTH(:, :, itgt, :, irun) = squeeze(mean(tmpFR, 3));
    end
end

lfadsPSTH = mean(lfadsPSTH, 5) / dt;
% calc. modulation depth
psthNondir = mean(lfadsPSTH, 3);
psthDir = lfadsPSTH - psthNondir;

tgtDir = [0:360/numTarget:359]';  % target direction in degrees
psthMD = nan(numUnit, numBin, numEncSeg);
psthR2 = nan(numUnit, numBin, numEncSeg);
for iunit = 1:numUnit
    for iseg = 1:numEncSeg
        fr = squeeze(psthDir(iunit, :, :, iseg))';
        [PD, MD, b0, R2] = fit_cos_tuning(fr, tgtDir);
        psthMD(iunit, :, iseg) = MD;
        psthR2(iunit, :, iseg) = R2;
    end
end

mdPrct10 = squeeze(prctile(psthMD, 10, 1));
mdPrct25 = squeeze(prctile(psthMD, 25, 1));
mdPrct50 = squeeze(prctile(psthMD, 50, 1));  % stats across units
mdPrct75 = squeeze(prctile(psthMD, 75, 1));
mdPrct90 = squeeze(prctile(psthMD, 90, 1));

reconTimes = (lfadsData.reconDataOffset + [1:numBin] - 1) * dt;


%% analyze velocity decoding results
angErr = zeros(2, numTrial, numEncSeg, numRun);

% for the two task phases before and after peak velocity
phaseColors = [ 27,120, 55;
               118, 42,131] / 255;  % green, purple

ax2 = nexttile; hold on;
for irun = 1:numRun
    runNo = runList(irun);

    % load lfads output data
    fileName  = sprintf('%s_%s_lfads_data_%s_run%d.mat', subject, alignEvent, modelStr, runNo);
    lfadsData = load([fileDir fileName]);

    reconDataOffset   = lfadsData.reconDataOffset;   % target onset bin # = 1
    encodDataOffsets  = lfadsData.encodDataOffsets;
    encodSegEndBinIdx = encodDataOffsets + 20;       % encod_seq_len = 20

    % load decoding results
    fileName = sprintf('%s_%s_lfads_decoding_%s_run%d.mat', ...
                       subject, alignEvent, modelStr, runNo);
    Data = load([fileDir fileName]);
    truthVel = Data.truthVel;

    for itr = 1:numTrial
        % get beh. landmarks for current trial
        binCue  = landmarkBin(2, itr);
        binOn   = landmarkBin(3, itr);
        binPeak = landmarkBin(4, itr);
        binOff  = landmarkBin(5, itr);

        velBinIdx = Data.velBins + reconDataOffset + binCue - 1;

        accelMask = velBinIdx >= binOn & velBinIdx <= binPeak;
        decelMask = velBinIdx > binPeak & velBinIdx <= binOff;

        % calc. vel. decoding performance metrics
        predVel = squeeze(Data.allPredVel(:, :, itr, :));

        for iseg = 1:size(predVel, 3)  % 1:numEncSeg
            lfadsVel = squeeze(predVel(:, :, iseg));

            % ----- accel. phase -----
            accelVel = squeeze(mean(truthVel(:, accelMask, itr), 2));
            [theta, rho] = cart2pol(accelVel(1), accelVel(2));
            accelAng = rad2deg(theta);
    
            decodedVel = squeeze(mean(lfadsVel(:, accelMask), 2));
            [theta, rho] = cart2pol(decodedVel(1), decodedVel(2));
            decodedAng = rad2deg(theta);
    
            % angular error between truth and decoded from lfads
            angErr(1, itr, iseg, irun) = decodedAng - accelAng;

            % ----- decel. phase -----
            decelVel = squeeze(mean(truthVel(:, decelMask, itr), 2));
            [theta, rho] = cart2pol(decelVel(1), decelVel(2));
            decelAng = rad2deg(theta);
    
            decodedVel = squeeze(mean(lfadsVel(:, decelMask), 2));
            [theta, rho] = cart2pol(decodedVel(1), decodedVel(2));
            decodedAng = rad2deg(theta);
    
            % angular error between truth and decoded from lfads
            angErr(2, itr, iseg, irun) = decodedAng - decelAng;
        end
    end
end


%% plot angular err. distributions for all encSeg
tmpErr = permute(angErr, [1 2 4 3]);  % epoch x trial x encSeg x run --> epoch x trial x run x encSeg
tmpErr = reshape(tmpErr, size(tmpErr, 1), [], size(tmpErr, 4));

% angular errors should be in range [-180, 180]
tmpMask = tmpErr > 180;
tmpErr(tmpMask) = tmpErr(tmpMask) - 360;
tmpMask = tmpErr < -180;
tmpErr(tmpMask) = tmpErr(tmpMask) + 360;

encodSegEndTime = (encodSegEndBinIdx - 1) * dt;
if strcmp(subject, 'MonkeyC')
    DensityScaleFactor = 0.8;
elseif strcmp(subject, 'MonkeyN')
    DensityScaleFactor = 0.6;
end
[prctX, prctY] = make_violin_plot(squeeze(tmpErr(1, :, 1:numEncSeg)), encodSegEndTime-0.000, 'Orientation', 'vertical', ...
               'EvaluationPoints', -360:2:360, ...
               'DensityDirection', 'negative', 'WrappedEvalWidth', 360, ...
               'DensityScaleFactor', DensityScaleFactor, 'FaceColor', phaseColors(1, :));
PrctLineStyles = {':', '--' , '-', '--', ':'};
for iprct = 1:size(prctX, 1)
    plot(encodSegEndTime-0.000, prctX(iprct, :), PrctLineStyles{iprct}, 'Color', phaseColors(1, :))
end

[prctX, prctY] = make_violin_plot(squeeze(tmpErr(2, :, 1:numEncSeg)), encodSegEndTime+0.000, 'Orientation', 'vertical', ...
               'EvaluationPoints', -360:2:360, ...
               'DensityDirection', 'positive', 'WrappedEvalWidth', 360, ...
               'DensityScaleFactor', DensityScaleFactor, 'FaceColor', phaseColors(2, :));
for iprct = 1:size(prctX, 1)
    plot(encodSegEndTime-0.000, prctX(iprct, :), PrctLineStyles{iprct}, 'Color', phaseColors(2, :))
end

if strcmp(subject, 'MonkeyC')
    axis([0.1 0.65 -90 90]);  % for angular error
elseif strcmp(subject, 'MonkeyN')
    axis([0.1 0.65 -60 60]);  % for angular error
end

set(gca, 'TickDir', 'out');
ylabel('Angular error (deg)');


% % % for i = 1:size(tmpErr, 1)
% % %     for j = 1:size(tmpErr, 3)
% % %         err = squeeze(tmpErr(i, :, j));
% % %         normErr = (err - mean(err)) / std(err);
% % %         [h, p, ksstat, cv] = kstest(normErr);
% % %     end
% % % end


%% plot PSTH modulation depth
ax3 = nexttile; hold on;

iseg = 2;
plot(reconTimes, squeeze(mdPrct50(:, iseg)), '-k');
plot(reconTimes, squeeze(mdPrct25(:, iseg)), '--k');
plot(reconTimes, squeeze(mdPrct75(:, iseg)), '--k');
plot(reconTimes, squeeze(mdPrct10(:, iseg)), ':k');
plot(reconTimes, squeeze(mdPrct90(:, iseg)), ':k');
axis([0.1 0.65 0 15]);

box off; 
linkaxes([ax1, ax2, ax3], 'x');
switch alignEvent
    case 'Cue'
        if strcmp(subject, 'MonkeyC')
            axis([0.1 0.55 0 18]);
        elseif strcmp(subject, 'MonkeyN')
            axis([0.1 0.65 0 30]);
        end
    case 'Move'
        axis([-0.1 0.40 0.2 1.0]);
end
set(gca, 'TickDir', 'out');
xlabel('Time (s)');
ylabel('Modulation depth (spikes/s)');


%% save figure
if fSaveFig
    figFileName = sprintf('%s_%s_lfads_%s_run%d_vel_decode.pdf', subject, alignEvent, modelStr, irun);
    set(hFig, 'Renderer', 'Painters', 'PaperOrientation', 'portrait', ...
              'PaperUnits', 'inches', 'PaperSize', [8.5 11]);
    print(hFig, figFileName, '-dpdf');
end

rmpath('.\util\');
